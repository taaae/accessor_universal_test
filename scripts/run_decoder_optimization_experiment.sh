#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/009_e2e3_decoder_optimization}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
started_epoch="$(date +%s)"

mkdir -p "${run_dir}"

{
    echo "experiment=009_e2e3_decoder_optimization"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "git_status_begin"
    git -C "${repo_dir}" status --short
    echo "git_status_end"
    echo
    echo "nvidia_smi"
    nvidia-smi \
        --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,ecc.mode.current,clocks.max.sm,clocks.max.memory,power.limit \
        --format=csv
    echo
    echo "nvcc_version"
    nvcc --version
} >"${run_dir}/environment.txt"

{
    echo "dot_powers=${DOT_POWERS:-16,20,24,27}"
    echo "gemv_powers=${GEMV_POWERS:-10,12,14,16}"
    echo "gemv_rows=${GEMV_ROWS:-1024}"
    echo "distributions=uniform_0_1,normal_0_1"
    echo "formats=fp64_e11m52,e2m5,e3m4"
    echo "fp64_strategies=current_x1,current_x4"
    echo "e2_e3_strategies=current_x1,current_x4,branchless_x4,lut_x1,lut_x4"
    echo "lookup_table=256_entry_fp32_read_only_global_l1"
    echo "lookup_table_bytes_per_format=1024"
    echo "lookup_initialization_timed=0"
    echo "warmup=${WARMUP:-10}"
    echo "rounds=${ROUNDS:-3}"
    echo "samples_per_round=${SAMPLES:-5}"
    echo "target_sample_ms=${TARGET_SAMPLE_MS:-20}"
    echo "decode_repeats=${DECODE_REPEATS:-256}"
    echo "base_seed=${BASE_SEED:-2611923443488327891}"
    echo "timing=CUDA_events;variants_interleaved_in_rotating_order"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target decoder_optimization_bench memory_accessor_test \
        storage_formats_test accuracy_statistics_test
ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

benchmark="${build_dir}/bin/decoder_optimization_bench"
samples_csv="${run_dir}/timing_samples.csv"
validation_csv="${run_dir}/decoder_validation.csv"

"${benchmark}" \
    --dot-powers "${DOT_POWERS:-16,20,24,27}" \
    --gemv-powers "${GEMV_POWERS:-10,12,14,16}" \
    --gemv-rows "${GEMV_ROWS:-1024}" \
    --warmup "${WARMUP:-10}" \
    --rounds "${ROUNDS:-3}" \
    --samples "${SAMPLES:-5}" \
    --target-sample-ms "${TARGET_SAMPLE_MS:-20}" \
    --decode-repeats "${DECODE_REPEATS:-256}" \
    --base-seed "${BASE_SEED:-2611923443488327891}" \
    --output "${samples_csv}" \
    --validation-output "${validation_csv}" \
    | tee "${run_dir}/timing_stdout.txt"

python3 "${repo_dir}/tools/summarize_decoder_optimization.py" \
    --samples "${samples_csv}" \
    --validation "${validation_csv}" \
    --output-dir "${run_dir}" \
    | tee "${run_dir}/timing_validation.txt"

if command -v cuobjdump >/dev/null 2>&1; then
    diagnostics_status="${run_dir}/compiler_diagnostics_status.txt"
    if cuobjdump --dump-resource-usage "${benchmark}" \
        >"${run_dir}/cuda_resource_usage.txt"; then
        echo "resource_usage=complete" >>"${diagnostics_status}"
    else
        echo "warning: cuobjdump resource-usage export failed" >&2
        echo "resource_usage=failed_nonfatal" >>"${diagnostics_status}"
    fi
    if cuobjdump --dump-sass "${benchmark}" >"${run_dir}/sass.txt"; then
        echo "sass=complete" >>"${diagnostics_status}"
    else
        echo "warning: cuobjdump SASS export failed" >&2
        echo "sass=failed_nonfatal" >>"${diagnostics_status}"
    fi
else
    echo "cuobjdump=unavailable" \
        >"${run_dir}/compiler_diagnostics_status.txt"
fi

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
} >>"${run_dir}/run_manifest.txt"

echo
echo "Decoder optimization experiment complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/decoder_validation.csv"
echo "  ${run_dir}/timing_summary.csv"
echo "  ${run_dir}/strategy_speedups.csv"
echo "  ${run_dir}/plateau_comparison.csv"
