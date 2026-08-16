#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/022_lns_strategy_performance}"
run_dir="${RUN_DIR:-${results_root}/run_$(date -u +%Y%m%dT%H%M%SZ)}"
build_jobs="${BUILD_JOBS:-2}"
cuda_arch="${CUDA_ARCH:-90}"
stop_after="${STOP_AFTER:-full}"
seed="${BASE_SEED:-2611923443488327891}"
started_epoch="$(date +%s)"
current_stage="initialization"

# A single deadlocked kernel once consumed an entire eight-hour allocation.
# Every benchmark binary now runs under a per-target wall clock so a hang costs
# minutes and names the variant it died on instead of the whole reservation.
smoke_timeout="${SMOKE_TIMEOUT:-600}"
screen_timeout="${SCREEN_TIMEOUT:-1800}"
full_timeout="${FULL_TIMEOUT:-5400}"

# Run one benchmark binary under a wall clock, tee-ing its stdout.  Reports the
# stdout log on failure because its last line names the variant in flight.
run_bounded() {
    local limit="$1" log="$2"
    shift 2
    if timeout --foreground "${limit}" "$@" | tee "${log}"; then
        return 0
    fi
    echo "error: '${1}' failed or exceeded ${limit}s during stage ${current_stage}" >&2
    echo "       last variant reached is the final line of ${log}" >&2
    exit 1
}

targets=(
    lns4_r1 lns6_r2
    lns8_r2 lns8_r3 lns8_r4 lns8_r5
    lns10_r4 lns12_r6
    lns16_r4 lns16_r7 lns16_r10 lns16_r11 lns16_r12 lns16_r13
    lns32_r20 lns32_r23 lns32_r28
)

case "${stop_after}" in
    sanitizer|smoke|screen|full) ;;
    *)
        echo "error: STOP_AFTER must be sanitizer, smoke, screen, or full" >&2
        exit 2
        ;;
esac

mkdir -p "${run_dir}/sanitizer" "${run_dir}/smoke" "${run_dir}/screen" \
    "${run_dir}/full"

finish_manifest() {
    local status="$?"
    {
        echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "wall_time_seconds=$(($(date +%s) - started_epoch))"
        echo "final_stage=${current_stage}"
        echo "exit_status=${status}"
    } >>"${run_dir}/run_manifest.txt"
}
trap finish_manifest EXIT

merge_csv() {
    local output="$1"
    shift
    awk 'FNR == 1 { if (!header++) print; next } { print }' "$@" >"${output}"
}

{
    echo "experiment=022_lns_strategy_performance"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "formats=${targets[*]}"
    echo "arithmetic_types=fp32,fp64"
    echo "multiply_methods=ordinary,fused"
    echo "storage_layouts=dense,padded"
    echo "access_methods=scalar,thread_packet,cooperative_shuffle"
    echo "distributions=uniform_0_1,normal_0_1"
    echo "screen_dot_powers=${SCREEN_DOT_POWERS:-24}"
    echo "screen_gemv_powers=${SCREEN_GEMV_POWERS:-14}"
    echo "full_dot_powers=${DOT_POWERS:-12,16,20,24,27}"
    echo "full_gemv_powers=${GEMV_POWERS:-8,10,12,14,16}"
    echo "gemv_rows=${GEMV_ROWS:-1024}"
    echo "base_seed=${seed}"
    echo "stop_after=${stop_after}"
    echo "smoke_timeout_seconds=${smoke_timeout}"
    echo "screen_timeout_seconds=${screen_timeout}"
    echo "full_timeout_seconds=${full_timeout}"
    echo "git_status_begin"
    git -C "${repo_dir}" status --short
    echo "git_status_end"
} >"${run_dir}/run_manifest.txt"

{
    nvidia-smi \
        --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,ecc.mode.current,power.limit \
        --format=csv
    echo
    nvcc --version
} >"${run_dir}/environment.txt"

current_stage="configure"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"

current_stage="host_tests"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target lns_benchmark_core_test bitwidth_benchmark_core_test \
        bitwidth_strategy_bench_2
ctest --test-dir "${build_dir}" --output-on-failure \
    -R 'lns_benchmark_core_test|bitwidth_benchmark_core_test' \
    | tee "${run_dir}/ctest.txt"

# lns4_r1 is the only target that exercises both warp-shuffle paths: it
# supports the cooperative loader (4 bits) and the warp-register fraction LUT
# (1 fractional bit).  Build and sanitize it before the remaining sixteen so a
# synchronisation fault costs one short build rather than the whole run.
current_stage="build_sanitizer_target"
# lns4_r1 covers the warp paths; lns12_r6 is the smallest target that exercises
# the fused-sum lookup in both its global and shared forms with a table large
# enough (4096 entries) for an indexing slip to land outside the allocation.
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target lns_strategy_bench_lns4_r1
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target lns_strategy_bench_lns12_r6

current_stage="warp_synccheck"
if ! command -v compute-sanitizer >/dev/null 2>&1; then
    echo "error: compute-sanitizer is required for the LNS preflight" >&2
    exit 2
fi
compute-sanitizer --tool synccheck --error-exitcode 97 \
    "${build_dir}/bin/lns_strategy_bench_lns4_r1" \
    --mode smoke \
    --distributions uniform_0_1 \
    --variants "ordinary/dense/scalar/x1/fraction_lut_warp,ordinary/dense/cooperative_shuffle/x8/fraction_lut_global,fused/dense/cooperative_shuffle/x8/ex2_approx" \
    --seed "${seed}" \
    --validation-output "${run_dir}/sanitizer/decoder_validation.csv" \
    --output "${run_dir}/sanitizer/timing_samples.csv" \
    2>&1 | tee "${run_dir}/sanitizer/synccheck.txt"

# An unmatched --variants entry is skipped silently, which would make the
# whole check vacuous.  Require that the shuffle variants actually ran.
sanitizer_rows="$(($(wc -l <"${run_dir}/sanitizer/timing_samples.csv") - 1))"
if [[ "${sanitizer_rows}" -lt 6 ]]; then
    echo "error: synccheck produced ${sanitizer_rows} timing rows; the" >&2
    echo "       --variants ids no longer match any compiled variant" >&2
    exit 2
fi
echo "synccheck covered ${sanitizer_rows} shuffle kernel launches"

current_stage="fused_sum_memcheck"
# The fused-sum lookup is indexed by a biased sum of two log codes, so its
# failure mode is an out-of-bounds read rather than a synchronisation fault.
compute-sanitizer --tool memcheck --error-exitcode 98 \
    "${build_dir}/bin/lns_strategy_bench_lns12_r6" \
    --mode smoke \
    --distributions uniform_0_1 \
    --variants "fused/dense/scalar/x1/fused_sum_lut_global,fused/dense/scalar/x1/fused_sum_lut_shared,fused/dense/thread_packet/x4/fused_sum_lut_shared,fused/padded/thread_packet/x8/fused_sum_lut_global,fused/dense/cooperative_shuffle/x8/fused_sum_lut_global" \
    --seed "${seed}" \
    --output "${run_dir}/sanitizer/fused_sum_samples.csv" \
    2>&1 | tee "${run_dir}/sanitizer/memcheck.txt"

memcheck_rows="$(($(wc -l <"${run_dir}/sanitizer/fused_sum_samples.csv") - 1))"
if [[ "${memcheck_rows}" -lt 8 ]]; then
    echo "error: memcheck produced ${memcheck_rows} timing rows; the" >&2
    echo "       fused-sum --variants ids match no compiled variant" >&2
    exit 2
fi
echo "memcheck covered ${memcheck_rows} fused-sum kernel launches"

if [[ "${stop_after}" == "sanitizer" ]]; then
    current_stage="sanitizer_complete"
    echo "LNS warp synccheck complete: ${run_dir}"
    exit 0
fi

current_stage="build_lns_targets"
for target in "${targets[@]}"; do
    cmake --build "${build_dir}" --parallel "${build_jobs}" \
        --target "lns_strategy_bench_${target}"
done

current_stage="smoke"
smoke_files=()
validation_file="${run_dir}/smoke/decoder_validation.csv"
for target in "${targets[@]}"; do
    output="${run_dir}/smoke/timing_samples_${target}.csv"
    run_bounded "${smoke_timeout}" "${run_dir}/smoke/stdout_${target}.txt" \
        "${build_dir}/bin/lns_strategy_bench_${target}" \
        --mode smoke \
        --seed "${seed}" \
        --validation-output "${validation_file}" \
        --output "${output}"
    smoke_files+=("${output}")
done
merge_csv "${run_dir}/smoke/timing_samples.csv" "${smoke_files[@]}"
python3 "${repo_dir}/tools/validate_lns_strategy_smoke.py" \
    --samples "${run_dir}/smoke/timing_samples.csv" \
    --output "${run_dir}/smoke/kernel_validation.csv" \
    | tee "${run_dir}/smoke/kernel_validation.txt"

if [[ "${stop_after}" == "smoke" ]]; then
    current_stage="smoke_complete"
    echo "LNS smoke preflight complete: ${run_dir}"
    exit 0
fi

current_stage="screen"
screen_files=()
raw_files=()
for target in "${targets[@]}"; do
    raw_output="${run_dir}/screen/raw_anchor_before_${target}.csv"
    timeout --foreground "${screen_timeout}" \
        "${build_dir}/bin/bitwidth_strategy_bench_2" \
        --mode screen \
        --dot-powers "${SCREEN_DOT_POWERS:-24}" \
        --gemv-powers "${SCREEN_GEMV_POWERS:-14}" \
        --gemv-rows "${GEMV_ROWS:-1024}" \
        --warmup "${SCREEN_WARMUP:-3}" \
        --samples "${SCREEN_SAMPLES:-3}" \
        --target-sample-ms "${SCREEN_TARGET_SAMPLE_MS:-5}" \
        --seed "${seed}" \
        --variants "natural/scalar/x1/raw,natural/thread_packet/x2/raw,natural/thread_packet/x4/raw,natural/thread_packet/x8/raw" \
        --output "${raw_output}" \
        >"${run_dir}/screen/raw_anchor_before_${target}_stdout.txt"
    raw_files+=("${raw_output}")

    output="${run_dir}/screen/timing_samples_${target}.csv"
    run_bounded "${screen_timeout}" "${run_dir}/screen/stdout_${target}.txt" \
        "${build_dir}/bin/lns_strategy_bench_${target}" \
        --mode screen \
        --dot-powers "${SCREEN_DOT_POWERS:-24}" \
        --gemv-powers "${SCREEN_GEMV_POWERS:-14}" \
        --gemv-rows "${GEMV_ROWS:-1024}" \
        --warmup "${SCREEN_WARMUP:-3}" \
        --samples "${SCREEN_SAMPLES:-3}" \
        --target-sample-ms "${SCREEN_TARGET_SAMPLE_MS:-5}" \
        --seed "${seed}" \
        --output "${output}"
    screen_files+=("${output}")
done
merge_csv "${run_dir}/screen/timing_samples.csv" "${screen_files[@]}"
merge_csv "${run_dir}/screen/raw_anchor_samples.csv" "${raw_files[@]}"
python3 "${repo_dir}/tools/summarize_lns_strategy_benchmark.py" \
    --samples "${run_dir}/screen/timing_samples.csv" \
    --output "${run_dir}/screen/timing_summary.csv"
python3 "${repo_dir}/tools/select_lns_strategy_finalists.py" \
    --samples "${run_dir}/screen/timing_samples.csv" \
    --finalists "${run_dir}/screen/finalists.txt" \
    --ranking "${run_dir}/screen/strategy_ranking.csv" \
    --top "${FINALISTS_PER_GROUP:-2}" \
    | tee "${run_dir}/screen/selection.txt"

if [[ "${stop_after}" == "screen" ]]; then
    current_stage="screen_complete"
    echo "LNS screening complete: ${run_dir}"
    exit 0
fi

current_stage="full"
full_files=()
raw_files=()
for target in "${targets[@]}"; do
    raw_output="${run_dir}/full/raw_anchor_before_${target}.csv"
    timeout --foreground "${full_timeout}" \
        "${build_dir}/bin/bitwidth_strategy_bench_2" \
        --mode full \
        --dot-powers "${DOT_POWERS:-12,16,20,24,27}" \
        --gemv-powers "${GEMV_POWERS:-8,10,12,14,16}" \
        --gemv-rows "${GEMV_ROWS:-1024}" \
        --warmup "${WARMUP:-10}" \
        --samples "${SAMPLES:-5}" \
        --target-sample-ms "${TARGET_SAMPLE_MS:-15}" \
        --seed "${seed}" \
        --variants "natural/scalar/x1/raw,natural/thread_packet/x2/raw,natural/thread_packet/x4/raw,natural/thread_packet/x8/raw" \
        --output "${raw_output}" \
        >"${run_dir}/full/raw_anchor_before_${target}_stdout.txt"
    raw_files+=("${raw_output}")

    output="${run_dir}/full/timing_samples_${target}.csv"
    run_bounded "${full_timeout}" "${run_dir}/full/stdout_${target}.txt" \
        "${build_dir}/bin/lns_strategy_bench_${target}" \
        --mode full \
        --dot-powers "${DOT_POWERS:-12,16,20,24,27}" \
        --gemv-powers "${GEMV_POWERS:-8,10,12,14,16}" \
        --gemv-rows "${GEMV_ROWS:-1024}" \
        --warmup "${WARMUP:-10}" \
        --samples "${SAMPLES:-5}" \
        --target-sample-ms "${TARGET_SAMPLE_MS:-15}" \
        --seed "${seed}" \
        --variant-file "${run_dir}/screen/finalists.txt" \
        --output "${output}"
    full_files+=("${output}")
done
merge_csv "${run_dir}/full/timing_samples.csv" "${full_files[@]}"
merge_csv "${run_dir}/full/raw_anchor_samples.csv" "${raw_files[@]}"
python3 "${repo_dir}/tools/summarize_lns_strategy_benchmark.py" \
    --samples "${run_dir}/full/timing_samples.csv" \
    --output "${run_dir}/full/timing_summary.csv"

current_stage="complete"
echo "LNS benchmark complete: ${run_dir}"
