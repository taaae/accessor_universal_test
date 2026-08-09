#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/017_expanded_format_strategy_performance}"
build_jobs="${BUILD_JOBS:-2}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
validation_dir="${run_dir}/validation"
started_epoch="$(date +%s)"

bits_groups=(2 4 8 16 32)
formats_2="e0m1,e1m0"
formats_4="e0m3,e1m2,fp4_e2m1,e3m0"
formats_8="e0m7,e6m1,e7m0"
formats_16="e0m15,e4m11,e6m9,e7m8,e9m6,e10m5"
formats_32="e0m31,e4m27,e5m26,e6m25,e7m24,e9m22,e10m21"
formats="${formats_2},${formats_4},${formats_8},${formats_16},${formats_32}"

dot_powers="${DOT_POWERS:-12,16,20,24,27}"
gemv_powers="${GEMV_POWERS:-8,10,12,14,16}"
gemv_rows="${GEMV_ROWS:-1024}"
warmup="${WARMUP:-10}"
rounds="${ROUNDS:-3}"
samples="${SAMPLES:-5}"
target_sample_ms="${TARGET_SAMPLE_MS:-15}"
base_seed="${BASE_SEED:-2611923443488327891}"

mkdir -p "${run_dir}" "${validation_dir}"

formats_for_bits() {
    local bits="$1"
    local variable="formats_${bits}"
    printf '%s' "${!variable}"
}

merge_csv() {
    local output="$1"
    shift
    awk 'FNR == 1 { if (!header++) print; next } { print }' "$@" >"${output}"
}

{
    echo "experiment=017_expanded_format_strategy_performance"
    echo "purpose=complete DOT and GEMV timing sweep for the 22 expanded formats"
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
    echo "subbyte_storage=bit_dense"
    echo "compilation=five_bit_width_specific_executables"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"

cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target decoder_strategy_core_test memory_accessor_test \
        storage_formats_test accuracy_statistics_test
for bits in "${bits_groups[@]}"; do
    cmake --build "${build_dir}" --parallel "${build_jobs}" \
        --target "expanded_format_strategy_smoke_${bits}"
    cmake --build "${build_dir}" --parallel "${build_jobs}" \
        --target "expanded_format_strategy_bench_${bits}"
done

ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

smoke_files=()
for bits in "${bits_groups[@]}"; do
    smoke_csv="${validation_dir}/decoder_validation_${bits}bit.csv"
    "${build_dir}/bin/expanded_format_strategy_smoke_${bits}" \
        --output "${smoke_csv}" \
        | tee "${validation_dir}/smoke_${bits}bit_stdout.txt"
    smoke_files+=("${smoke_csv}")
done
smoke_inventory="${validation_dir}/decoder_validation.csv"
merge_csv "${smoke_inventory}" "${smoke_files[@]}"

validation_sample_files=()
for bits in "${bits_groups[@]}"; do
    validation_samples="${validation_dir}/timing_samples_${bits}bit.csv"
    "${build_dir}/bin/expanded_format_strategy_bench_${bits}" \
        --formats "$(formats_for_bits "${bits}")" \
        --dot-powers 10 \
        --gemv-powers 8 \
        --gemv-rows 64 \
        --warmup 1 \
        --rounds 1 \
        --samples 1 \
        --target-sample-ms 0.25 \
        --base-seed "${base_seed}" \
        --output "${validation_samples}" \
        | tee "${validation_dir}/timing_${bits}bit_stdout.txt"
    validation_sample_files+=("${validation_samples}")
done
validation_samples="${validation_dir}/timing_samples.csv"
merge_csv "${validation_samples}" "${validation_sample_files[@]}"

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

sample_files=()
for bits in "${bits_groups[@]}"; do
    group_samples="${run_dir}/timing_samples_${bits}bit.csv"
    "${build_dir}/bin/expanded_format_strategy_bench_${bits}" \
        --formats "$(formats_for_bits "${bits}")" \
        --dot-powers "${dot_powers}" \
        --gemv-powers "${gemv_powers}" \
        --gemv-rows "${gemv_rows}" \
        --warmup "${warmup}" \
        --rounds "${rounds}" \
        --samples "${samples}" \
        --target-sample-ms "${target_sample_ms}" \
        --base-seed "${base_seed}" \
        --output "${group_samples}" \
        | tee "${run_dir}/timing_${bits}bit_stdout.txt"
    sample_files+=("${group_samples}")
done
samples_csv="${run_dir}/timing_samples.csv"
merge_csv "${samples_csv}" "${sample_files[@]}"

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

: >"${run_dir}/compiler_diagnostics_status.txt"
if command -v cuobjdump >/dev/null 2>&1; then
    for bits in "${bits_groups[@]}"; do
        benchmark="${build_dir}/bin/expanded_format_strategy_bench_${bits}"
        if cuobjdump --dump-resource-usage "${benchmark}" \
            >"${run_dir}/cuda_resource_usage_${bits}bit.txt"; then
            echo "resource_usage_${bits}bit=complete" \
                >>"${run_dir}/compiler_diagnostics_status.txt"
        else
            echo "resource_usage_${bits}bit=failed_nonfatal" \
                >>"${run_dir}/compiler_diagnostics_status.txt"
        fi
    done
else
    echo "cuobjdump=unavailable" \
        >>"${run_dir}/compiler_diagnostics_status.txt"
fi

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
} >>"${run_dir}/run_manifest.txt"

echo
echo "Expanded-format strategy benchmark complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/timing_samples.csv"
echo "  ${run_dir}/timing_summary.csv"
echo "  ${run_dir}/case_winners.csv"
echo "  ${run_dir}/strategy_rankings.csv"
