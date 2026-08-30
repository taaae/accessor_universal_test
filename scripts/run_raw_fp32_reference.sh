#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit_short="$(git -C "${repo_dir}" rev-parse --short=12 HEAD)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200-lut-distribution-${commit_short}}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/027_lut_distribution_shape}"
followup_tag="${FOLLOWUP_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${RUN_DIR:-${results_root}/raw_fp32_followup_${followup_tag}}"
source_lut_run="${SOURCE_LUT_RUN:-${results_root}/run_20260830T102642Z}"
mode="${MODE:-smoke}"
build_jobs="${BUILD_JOBS:-4}"
target_timeout="${TARGET_TIMEOUT:-600}"

case "${mode}" in
    smoke|full) ;;
    *) echo "error: MODE must be smoke or full" >&2; exit 2 ;;
esac

mkdir -p "${results_root}"
lock_file="$(dirname "${repo_dir}")/.accessor-universal-lut-distribution-${USER:-unknown}.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "error: another LUT-distribution build or runner holds ${lock_file}" >&2
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
        echo "experiment=027_lut_distribution_shape_raw_fp32_followup"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
        echo "source_lut_run=${source_lut_run}"
        echo "format=raw_fp32"
        echo "x_semantics=undefined_no_lut_horizontal_reference"
        echo "kernel=dot"
        echo "storage=float32"
        echo "arithmetic=fp32"
        echo "access=scalar_x1"
        echo "blocks=512"
        echo "threads=256"
        echo "full_n=67108864"
        echo "full_warmup=10"
        echo "full_samples=50"
        echo "pre_run_n2p26_linear_sanity_estimate_ms=0.25982871422400845"
        echo "sanity_source=n2p27_bare_h200_0.5196574284480169_ms_divided_by_two"
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
    --target raw_fp32_reference_bench

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
    "${build_dir}/bin/raw_fp32_reference_bench" \
        --mode "${mode}" \
        --n "${benchmark_n}" \
        --warmup "${benchmark_warmup}" \
        --samples "${benchmark_samples}" \
        --output "${stage_dir}/timing_samples.csv" \
        | tee "${stage_dir}/stdout.txt"; then
    echo "error: raw FP32 benchmark failed or exceeded ${target_timeout}s" >&2
    exit 1
fi

if [[ "${mode}" == "full" ]]; then
    source_samples="${source_lut_run}/full/timing_samples.csv"
    source_metrics="${source_lut_run}/full/access_metrics.csv"
    if [[ ! -f "${source_samples}" || ! -f "${source_metrics}" ]]; then
        echo "error: immutable source LUT run is missing" >&2
        exit 4
    fi
    combined_dir="${run_dir}/combined"
    if [[ -e "${combined_dir}" ]]; then
        echo "error: refusing to overwrite existing combined report: ${combined_dir}" >&2
        exit 3
    fi
    current_stage="combined_analysis"
    python3 "${repo_dir}/tools/analyze_lut_distribution_shape.py" \
        --samples "${source_samples}" \
        --metrics "${source_metrics}" \
        --raw-fp32-samples "${stage_dir}/timing_samples.csv" \
        --output-dir "${combined_dir}" \
        | tee "${stage_dir}/analysis.txt"
fi

current_stage="complete"
echo "Raw FP32 reference ${mode} complete: ${stage_dir}"
