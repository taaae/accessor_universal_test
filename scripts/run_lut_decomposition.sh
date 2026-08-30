#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit_short="$(git -C "${repo_dir}" rev-parse --short=12 HEAD)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200-lut-decomposition-${commit_short}}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/029_lut_decomposition_shapes}"
run_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${RUN_DIR:-${results_root}/run_${run_tag}}"
raw_fp32_samples="${RAW_FP32_SAMPLES:-${repo_dir}/results/027_lut_distribution_shape/raw_fp32_followup_20260830T104804Z/full/timing_samples.csv}"
mode="${MODE:-smoke}"
build_jobs="${BUILD_JOBS:-4}"
target_timeout="${TARGET_TIMEOUT:-1800}"

case "${mode}" in
    smoke|full) ;;
    *) echo "error: MODE must be smoke or full" >&2; exit 2 ;;
esac

mkdir -p "${results_root}"
lock_file="$(dirname "${repo_dir}")/.accessor-universal-lut-decomposition-${USER:-unknown}.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "error: another LUT-decomposition runner holds ${lock_file}" >&2
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
        echo "experiment=029_lut_decomposition_shapes"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
        echo "kernel=dot"
        echo "full_n=67108864"
        echo "blocks=512"
        echo "threads=256"
        echo "access=scalar_x1"
        echo "target_x=0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1"
        echo "full_warmup=10"
        echo "full_samples=50"
        echo "shared_staging=inside_timed_first_stage_once_per_block"
        echo "raw_fp32_source=${raw_fp32_samples}"
        echo "left_seed=7640891576956012809"
        echo "right_seed=13503953896175478587"
        echo "git_status_begin"
        git -C "${repo_dir}" status --short
        echo "git_status_end"
    } >"${run_dir}/run_manifest.txt"
fi

{
    echo "stage=${mode}"
    echo "hostname=$(hostname)"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi -L
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total --format=csv
    echo
    nvcc --version
} >"${stage_dir}/environment.txt"

if [[ "$(nvidia-smi -L | wc -l)" -ne 1 ]]; then
    echo "error: expected exactly one visible GPU" >&2
    exit 4
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'H200'; then
    echo "error: visible GPU is not an H200" >&2
    exit 4
fi

current_stage="configure"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-90}"

current_stage="build"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target lut_decomposition_core_test lut_decomposition_bench

current_stage="host_test"
ctest --test-dir "${build_dir}" --output-on-failure \
    -R '^lut_decomposition_core_test$' | tee "${stage_dir}/ctest.txt"

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
    "${build_dir}/bin/lut_decomposition_bench" \
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

if [[ "${mode}" == "smoke" ]]; then
    current_stage="smoke_contract"
    if [[ "$(wc -l <"${stage_dir}/timing_samples.csv")" -ne 76 ]]; then
        echo "error: smoke timing CSV has the wrong row count" >&2
        exit 5
    fi
    if [[ "$(wc -l <"${stage_dir}/access_metrics.csv")" -ne 70 ]]; then
        echo "error: smoke metrics CSV has the wrong row count" >&2
        exit 5
    fi
else
    current_stage="analysis"
    if [[ ! -f "${raw_fp32_samples}" ]]; then
        echo "error: immutable raw FP32 baseline is missing" >&2
        exit 6
    fi
    python3 "${repo_dir}/tools/analyze_lut_decomposition.py" \
        --samples "${stage_dir}/timing_samples.csv" \
        --metrics "${stage_dir}/access_metrics.csv" \
        --raw-fp32-samples "${raw_fp32_samples}" \
        --output-dir "${stage_dir}" \
        | tee "${stage_dir}/analysis.txt"
fi

current_stage="complete"
echo "LUT decomposition ${mode} complete: ${stage_dir}"
