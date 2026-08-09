#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/011_e2e3_strategy_performance}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
validation_dir="${run_dir}/validation"
started_epoch="$(date +%s)"

mkdir -p "${run_dir}" "${validation_dir}"

{
    echo "experiment=011_e2e3_strategy_performance"
    echo "purpose=full DOT and GEMV timing sweep for every E2M5/E3M4 decoder strategy"
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
    echo "dot_powers=${DOT_POWERS:-12,16,20,24,27}"
    echo "gemv_powers=${GEMV_POWERS:-8,10,12,14,16}"
    echo "gemv_rows=${GEMV_ROWS:-1024}"
    echo "distributions=uniform_0_1,normal_0_1"
    echo "formats=fp64,e2m5,e3m4"
    echo "e2m5_strategies=42"
    echo "e3m4_strategies=42"
    echo "fp64_baseline=raw_pointer_x1"
    echo "variants_per_case=85"
    echo "warmup=${WARMUP:-10}"
    echo "rounds=${ROUNDS:-3}"
    echo "samples_per_round=${SAMPLES:-5}"
    echo "target_sample_ms=${TARGET_SAMPLE_MS:-15}"
    echo "measurement_order=randomized_within_each_sample"
    echo "base_seed=${BASE_SEED:-2611923443488327891}"
    echo "timing=complete_kernel_CUDA_event_time"
    echo "dot_note=map_reduce_and_final_reduction_are_both_timed"
    echo "gemv_note=one_256_thread_block_per_row"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target e2e3_strategy_bench e2e3_strategy_smoke \
        memory_accessor_test storage_formats_test accuracy_statistics_test
ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

smoke="${build_dir}/bin/e2e3_strategy_smoke"
"${smoke}" \
    --dot-powers 16 \
    --gemv-powers 10 \
    --gemv-rows 64 \
    --warmup 1 \
    --samples 1 \
    --target-sample-ms 0.5 \
    --base-seed "${BASE_SEED:-2611923443488327891}" \
    --output "${validation_dir}/timing_samples.csv" \
    --decoder-validation-output "${validation_dir}/decoder_validation.csv" \
    --kernel-validation-output "${validation_dir}/kernel_validation.csv" \
    | tee "${validation_dir}/stdout.txt"

python3 "${repo_dir}/tools/summarize_e2e3_strategy_smoke.py" \
    --samples "${validation_dir}/timing_samples.csv" \
    --decoder-validation "${validation_dir}/decoder_validation.csv" \
    --kernel-validation "${validation_dir}/kernel_validation.csv" \
    --output-dir "${validation_dir}" \
    | tee "${validation_dir}/summary_validation.txt"

benchmark="${build_dir}/bin/e2e3_strategy_bench"
samples_csv="${run_dir}/timing_samples.csv"
"${benchmark}" \
    --dot-powers "${DOT_POWERS:-12,16,20,24,27}" \
    --gemv-powers "${GEMV_POWERS:-8,10,12,14,16}" \
    --gemv-rows "${GEMV_ROWS:-1024}" \
    --warmup "${WARMUP:-10}" \
    --rounds "${ROUNDS:-3}" \
    --samples "${SAMPLES:-5}" \
    --target-sample-ms "${TARGET_SAMPLE_MS:-15}" \
    --base-seed "${BASE_SEED:-2611923443488327891}" \
    --output "${samples_csv}" \
    | tee "${run_dir}/timing_stdout.txt"

python3 "${repo_dir}/tools/summarize_e2e3_strategy_benchmark.py" \
    --samples "${samples_csv}" \
    --output-dir "${run_dir}" \
    | tee "${run_dir}/timing_validation.txt"

if command -v cuobjdump >/dev/null 2>&1; then
    if cuobjdump --dump-resource-usage "${benchmark}" \
        >"${run_dir}/cuda_resource_usage.txt"; then
        echo "resource_usage=complete" \
            >"${run_dir}/compiler_diagnostics_status.txt"
    else
        echo "resource_usage=failed_nonfatal" \
            >"${run_dir}/compiler_diagnostics_status.txt"
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
echo "E2/E3 full strategy benchmark complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/timing_samples.csv"
echo "  ${run_dir}/timing_summary.csv"
echo "  ${run_dir}/case_winners.csv"
