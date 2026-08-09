#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/015_all_format_strategy_performance}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
validation_dir="${run_dir}/validation"
started_epoch="$(date +%s)"

formats="${FORMATS:-e1m6,fp8_e4m3,fp8_e5m2,e1m14,e2m13,e3m12,fp16_e5m10,bf16_e8m7,e11m4,e1m30,e2m29,e3m28,fp32_e8m23,e11m20}"
dot_powers="${DOT_POWERS:-12,16,20,24,27}"
gemv_powers="${GEMV_POWERS:-8,10,12,14,16}"
gemv_rows="${GEMV_ROWS:-1024}"
warmup="${WARMUP:-10}"
rounds="${ROUNDS:-3}"
samples="${SAMPLES:-5}"
target_sample_ms="${TARGET_SAMPLE_MS:-15}"
base_seed="${BASE_SEED:-2611923443488327891}"

mkdir -p "${run_dir}" "${validation_dir}"

{
    echo "experiment=015_all_format_strategy_performance"
    echo "purpose=complete DOT and GEMV timing sweep for every non-E2M5/E3M4 decoder strategy"
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
    echo "formats=${formats}"
    echo "excluded_formats=e2m5,e3m4"
    echo "excluded_formats_reason=covered_by_013_e2e3_expanded_strategy_performance"
    echo "fp64_baseline=raw_pointer_x1_remeasured_with_each_format"
    echo "dot_powers=${dot_powers}"
    echo "gemv_powers=${gemv_powers}"
    echo "gemv_rows=${gemv_rows}"
    echo "distributions=uniform_0_1,normal_0_1"
    echo "warmup=${warmup}"
    echo "rounds=${rounds}"
    echo "samples_per_round=${samples}"
    echo "target_sample_ms=${target_sample_ms}"
    echo "measurement_order=randomized_within_each_format_and_sample"
    echo "base_seed=${base_seed}"
    echo "timing=complete_kernel_CUDA_event_time"
    echo "dot_note=map_reduce_and_final_reduction_are_both_timed"
    echo "gemv_note=one_256_thread_block_per_row"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target all_format_strategy_bench all_format_strategy_smoke \
        decoder_strategy_core_test memory_accessor_test storage_formats_test \
        accuracy_statistics_test
ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

smoke="${build_dir}/bin/all_format_strategy_smoke"
smoke_inventory="${validation_dir}/decoder_validation.csv"
"${smoke}" --output "${smoke_inventory}" \
    | tee "${validation_dir}/smoke_stdout.txt"

benchmark="${build_dir}/bin/all_format_strategy_bench"
validation_samples="${validation_dir}/timing_samples.csv"
"${benchmark}" \
    --formats "${formats}" \
    --dot-powers 10 \
    --gemv-powers 8 \
    --gemv-rows 64 \
    --warmup 1 \
    --rounds 1 \
    --samples 1 \
    --target-sample-ms 0.25 \
    --base-seed "${base_seed}" \
    --output "${validation_samples}" \
    | tee "${validation_dir}/timing_stdout.txt"

python3 "${repo_dir}/tools/summarize_all_format_strategy_benchmark.py" \
    --samples "${validation_samples}" \
    --validation-inventory "${smoke_inventory}" \
    --output-dir "${validation_dir}" \
    --expected-formats "${formats}" \
    --expected-dot-powers 10 \
    --expected-gemv-powers 8 \
    --expected-gemv-rows 64 \
    --expected-rounds 1 \
    --expected-samples 1 \
    | tee "${validation_dir}/summary_validation.txt"

samples_csv="${run_dir}/timing_samples.csv"
"${benchmark}" \
    --formats "${formats}" \
    --dot-powers "${dot_powers}" \
    --gemv-powers "${gemv_powers}" \
    --gemv-rows "${gemv_rows}" \
    --warmup "${warmup}" \
    --rounds "${rounds}" \
    --samples "${samples}" \
    --target-sample-ms "${target_sample_ms}" \
    --base-seed "${base_seed}" \
    --output "${samples_csv}" \
    | tee "${run_dir}/timing_stdout.txt"

python3 "${repo_dir}/tools/summarize_all_format_strategy_benchmark.py" \
    --samples "${samples_csv}" \
    --validation-inventory "${smoke_inventory}" \
    --output-dir "${run_dir}" \
    --expected-formats "${formats}" \
    --expected-dot-powers "${dot_powers}" \
    --expected-gemv-powers "${gemv_powers}" \
    --expected-gemv-rows "${gemv_rows}" \
    --expected-rounds "${rounds}" \
    --expected-samples "${samples}" \
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
echo "All-format strategy benchmark complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/timing_samples.csv"
echo "  ${run_dir}/timing_summary.csv"
echo "  ${run_dir}/case_winners.csv"
echo "  ${run_dir}/strategy_rankings.csv"
