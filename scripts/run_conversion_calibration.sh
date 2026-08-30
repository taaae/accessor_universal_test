#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit_short="$(git -C "${repo_dir}" rev-parse --short=12 HEAD)"
build_tag="${SLURM_JOB_ID:-local}"
build_dir="${BUILD_DIR:-${repo_dir}/build-conversion-calibration-${commit_short}-${build_tag}}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/028_conversion_cost_calibration}"
run_dir="${RUN_DIR:-${results_root}/run_$(date -u +%Y%m%dT%H%M%SZ)}"
mode="${MODE:-smoke}"
mode_dir="${run_dir}/${mode}"
if [[ "${mode}" == "smoke" ]]; then
  actual_n="${SMOKE_N:-1048576}"
else
  actual_n="134217728"
fi
started="$(date +%s)"
stage="initialization"

case "${mode}" in smoke|full) ;; *) echo "MODE must be smoke or full" >&2; exit 2;; esac
mkdir -p "${run_dir}"
if [[ -e "${mode_dir}" ]]; then echo "refusing to overwrite ${mode_dir}" >&2; exit 2; fi
mkdir "${mode_dir}"
trap 'status=$?; printf "finished_utc=%s\nwall_seconds=%s\nfinal_stage=%s\nexit_status=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(($(date +%s)-started))" "${stage}" "${status}" >>"${mode_dir}/run_manifest.txt"' EXIT

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  [[ "${SLURM_GPUS_ON_NODE:-1}" == "1" ]] || { echo "exactly one GPU is required" >&2; exit 2; }
fi

{
  echo "experiment=028_conversion_cost_calibration"
  echo "mode=${mode}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
  echo "git_branch=$(git -C "${repo_dir}" branch --show-current)"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "slurm_node=${SLURMD_NODENAME:-local}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "n=${actual_n}"
  echo "blocks=512"
  echo "threads=256"
  echo "storage=two_uint32_streams"
  echo "arithmetic=fp64"
  echo "access=scalar_x1"
  echo "left_seed=0x6bd87c012a53f9e1"
  echo "right_seed=0xf5ef05b8551985f4"
  echo "generator=Philox4x32-10"
  echo "compile_flags=-O3 -lineinfo -Xptxas=-v"
  echo "fast_math=false"
} >"${mode_dir}/run_manifest.txt"

stage="environment"
{
  hostname
  nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
  nvcc --version
  cmake --version | head -1
} >"${mode_dir}/environment.txt"

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo "another compute process is already present on the allocated GPU" >&2
  exit 2
fi

stage="host_validation"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "${tmp_dir}"' RETURN
python3 -m conversion_calibration.codegen --output "${tmp_dir}/cases.cuh" --manifest "${tmp_dir}/manifest.json"
cmp "${tmp_dir}/cases.cuh" "${repo_dir}/generated/conversion_calibration_cases.cuh"
python3 -m unittest discover -s "${repo_dir}/analysis/tests" -p 'test_conversion_calibration*.py' -v | tee "${mode_dir}/host_tests.txt"
rm -rf -- "${tmp_dir}"
trap - RETURN

stage="configure"
cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
stage="build"
cmake --build "${build_dir}" --parallel "${BUILD_JOBS:-2}" --target conversion_calibration_bench 2>&1 | tee "${mode_dir}/build.txt"
binary="${build_dir}/bin/conversion_calibration_bench"

stage="sass"
cuobjdump --dump-sass --demangle "${binary}" >"${mode_dir}/sass.txt"
cuobjdump --dump-resource-usage --demangle "${binary}" >"${mode_dir}/cuobjdump_resources.txt"

stage="benchmark"
benchmark_args=(--mode "${mode}" --output "${mode_dir}/timing_samples.csv" --resources "${mode_dir}/kernel_resources.csv")
if [[ "${mode}" == smoke ]]; then benchmark_args+=(--n "${SMOKE_N:-1048576}" --warmups 1 --rounds 1 --samples 1 --interval-ms 0); fi
if [[ "${mode}" == smoke ]]; then
  stage="compute_sanitizer"
  all_cases="$(python3 - <<'PY'
print(','.join(str(index) for index in range(135)))
PY
)"
  timeout --foreground "${SANITIZER_TIMEOUT:-1200}" compute-sanitizer \
    --tool memcheck --error-exitcode 99 "${binary}" --mode smoke --n 65536 \
    --resources "${mode_dir}/sanitizer_resources.csv" \
    --profile-cases "${all_cases}" >"${mode_dir}/compute_sanitizer.txt" 2>&1
fi
stage="benchmark"
timeout --foreground "${BENCHMARK_TIMEOUT:-7200}" \
  "${binary}" "${benchmark_args[@]}" | tee "${mode_dir}/stdout.txt"

stage="feature_extraction"
python3 "${repo_dir}/tools/extract_conversion_calibration_sass.py" \
  "${mode_dir}/sass.txt" "${mode_dir}/features.csv" \
  --resources "${mode_dir}/kernel_resources.csv" --warnings "${mode_dir}/sass_warnings.txt"

stage="analysis"
mkdir "${mode_dir}/analysis"
python3 "${repo_dir}/tools/analyze_conversion_calibration.py" \
  "${mode_dir}/timing_samples.csv" "${mode_dir}/features.csv" "${mode_dir}/analysis" \
  --require-timing-qc | tee "${mode_dir}/analysis_stdout.txt"
python3 "${repo_dir}/tools/build_conversion_calibration_report.py" \
  "${mode_dir}" "${mode_dir}/analysis/report.html"

stage="post_environment"
{
  nvidia-smi --query-gpu=name,uuid,power.draw,temperature.gpu,clocks.sm,clocks.mem --format=csv,noheader
  nvidia-smi --query-gpu=clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown --format=csv,noheader 2>&1 || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
} >"${mode_dir}/post_environment.txt"

stage="complete"
echo "conversion calibration ${mode} complete: ${mode_dir}"
