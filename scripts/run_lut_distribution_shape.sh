#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit_short="$(git -C "${repo_dir}" rev-parse --short=12 HEAD)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200-lut-distribution-${commit_short}}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/027_lut_distribution_shape}"
run_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${RUN_DIR:-${results_root}/run_${run_tag}}"
mode="${MODE:-smoke}"
build_jobs="${BUILD_JOBS:-4}"
target_timeout="${TARGET_TIMEOUT:-1200}"

case "${mode}" in
    smoke|full) ;;
    *) echo "error: MODE must be smoke or full" >&2; exit 2 ;;
esac

mkdir -p "${results_root}"
lock_file="$(dirname "${repo_dir}")/.accessor-universal-lut-distribution-${USER:-unknown}.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "error: another LUT-distribution runner holds ${lock_file}" >&2
    exit 3
fi
stage_dir="${run_dir}/${mode}"
if [[ -e "${stage_dir}" ]]; then
    echo "error: refusing to overwrite existing stage directory: ${stage_dir}" >&2
    exit 3
fi
mkdir -p "${stage_dir}"
started_epoch="$(date +%s)"
current_stage="initialization"
finish_manifest() {
    local status="$?"
    {
        echo "stage=${mode}"
        echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "wall_time_seconds=$(($(date +%s) - started_epoch))"
        echo "final_stage=${current_stage}"
        echo "exit_status=${status}"
    } >>"${run_dir}/run_manifest.txt"
}
trap finish_manifest EXIT

if [[ ! -f "${run_dir}/run_manifest.txt" ]]; then
    {
        echo "experiment=027_lut_distribution_shape"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
        echo "formats=t16,posit16_es1,lns16_r11"
        echo "kernel=dot"
        echo "storage=uint16"
        echo "arithmetic=fp32"
        echo "access=scalar_x1_global_lut"
        echo "lut_entries=65536"
        echo "lut_bytes=262144"
        echo "full_n=67108864"
        echo "q_eighths=0,1,2,3,4,5,6,7,8"
        echo "full_warmup=10"
        echo "full_samples=50"
        echo "left_seed=2611923443488327891"
        echo "right_seed=1376283091369227076"
        echo "hot_sector_base=0"
        echo "git_status_begin"
        git -C "${repo_dir}" status --short
        echo "git_status_end"
    } >"${run_dir}/run_manifest.txt"
fi

{
    echo "stage=${mode}"
    echo "hostname=$(hostname)"
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total --format=csv
    echo
    nvcc --version
} >"${stage_dir}/environment.txt"

current_stage="configure"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-90}"

current_stage="build"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target lut_distribution_core_test lut_distribution_bench

current_stage="host_test"
ctest --test-dir "${build_dir}" --output-on-failure \
    -R '^lut_distribution_core_test$' | tee "${stage_dir}/ctest.txt"

if [[ "${mode}" == "smoke" ]]; then
    benchmark_n=1048576
    benchmark_warmup=1
    benchmark_samples=3
else
    benchmark_n=67108864
    benchmark_warmup=10
    benchmark_samples=50
fi

current_stage="${mode}_benchmark"
if ! timeout --foreground "${target_timeout}" \
    "${build_dir}/bin/lut_distribution_bench" \
        --mode "${mode}" \
        --n "${benchmark_n}" \
        --warmup "${benchmark_warmup}" \
        --samples "${benchmark_samples}" \
        --output "${stage_dir}/timing_samples.csv" \
        --metrics-output "${stage_dir}/access_metrics.csv" \
        | tee "${stage_dir}/stdout.txt"; then
    echo "error: benchmark failed or exceeded ${target_timeout}s" >&2
    exit 1
fi

if [[ "${mode}" == "full" ]]; then
    current_stage="analysis"
    python3 "${repo_dir}/tools/analyze_lut_distribution_shape.py" \
        --samples "${stage_dir}/timing_samples.csv" \
        --metrics "${stage_dir}/access_metrics.csv" \
        --output-dir "${stage_dir}" \
        | tee "${stage_dir}/analysis.txt"
fi

current_stage="complete"
echo "LUT distribution ${mode} complete: ${stage_dir}"
