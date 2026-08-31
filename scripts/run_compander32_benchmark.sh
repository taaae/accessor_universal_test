#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_tag="${SLURM_JOB_ID:-local-$$}"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200-compander32-${job_tag}}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/031_compander32_conversion_cost}"
mode="${MODE:-full}"
build_jobs="${BUILD_JOBS:-8}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}_${job_tag}}"
stage_dir="${run_dir}/${mode}"
started_epoch="$(date +%s)"

if [[ "${mode}" != "smoke" && "${mode}" != "full" ]]; then
    echo "error: MODE must be smoke or full" >&2
    exit 2
fi

mkdir -p "${stage_dir}"
node_name="${HOSTNAME%%.*}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "${node_name}" in
    gpu-nvidia-h200-2|gpu-nvidia-h200-3) ;;
    *)
        echo "error: this benchmark may run only on gpu-nvidia-h200-2 or gpu-nvidia-h200-3, got ${node_name}" >&2
        exit 3
        ;;
esac
if [[ "${gpu_name}" != *H200* ]]; then
    echo "error: expected an H200 GPU, got ${gpu_name}" >&2
    exit 3
fi
{
    echo "experiment=031_compander32_conversion_cost"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "mode=${mode}"
    echo "kernel=dot"
    echo "N_values=1048576,4194304,16777216,67108864,268435456"
    echo "maximum_allocation_N=268435456"
    echo "prefix_reuse=1"
    echo "distribution=deterministic_normal_clipped_-8_8"
    echo "geometry=512_blocks_x_256_threads"
    echo "access=scalar_x1"
    echo "warmups=$([[ "${mode}" == full ]] && echo 10 || echo 1)"
    echo "samples=$([[ "${mode}" == full ]] && echo 50 || echo 3)"
    echo "one_gpu_only=1"
    echo "node=${node_name}"
    echo "gpu=${gpu_name}"
    echo "git_status_begin"
    git -C "${repo_dir}" status --short
    echo "git_status_end"
} >"${run_dir}/run_manifest.txt"

{
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,power.limit --format=csv
    echo
    nvcc --version
} >"${run_dir}/environment.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --target clean
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target compander32_core_test compander32_bench 2>&1 \
    | tee "${run_dir}/build.txt"
ctest --test-dir "${build_dir}" --output-on-failure -R compander32_core_test \
    | tee "${run_dir}/ctest.txt"

cuobjdump --dump-sass "${build_dir}/bin/compander32_bench" \
    >"${run_dir}/compander32.sass.txt"
python3 "${repo_dir}/tools/check_compander32_codegen.py" \
    --sass "${run_dir}/compander32.sass.txt" \
    --build-log "${run_dir}/build.txt" \
    --source "${repo_dir}/include/compander32_core.hpp" \
    --output "${run_dir}/compiler_checks.txt"

timeout "$([[ "${mode}" == full ]] && echo "${FULL_TIMEOUT:-45m}" || echo "${SMOKE_TIMEOUT:-10m}")" \
    "${build_dir}/bin/compander32_bench" \
    --mode "${mode}" \
    --output "${stage_dir}/timing_samples.csv" \
    --correctness-output "${stage_dir}/correctness_checks.txt" \
    | tee "${stage_dir}/stdout.txt"

grep -q '^all_passed=1$' "${stage_dir}/correctness_checks.txt"
if [[ "${mode}" == "smoke" ]]; then
    expected_lines=46
else
    expected_lines=3751
fi
actual_lines="$(wc -l <"${stage_dir}/timing_samples.csv")"
if [[ "${actual_lines}" -ne "${expected_lines}" ]]; then
    echo "error: expected ${expected_lines} timing CSV lines, got ${actual_lines}" >&2
    exit 5
fi

python3 "${repo_dir}/tools/analyze_compander32_benchmark.py" \
    --samples "${stage_dir}/timing_samples.csv" \
    --output-dir "${stage_dir}" \
    --correctness "${stage_dir}/correctness_checks.txt" \
    --compiler "${run_dir}/compiler_checks.txt" \
    --environment "${run_dir}/environment.txt" \
    | tee "${stage_dir}/analysis.txt"

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
    echo "completed_stage=${mode}"
} >>"${run_dir}/run_manifest.txt"
echo "Compander32 ${mode} complete: ${run_dir}"
