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

targets=(
    lns4_r1 lns6_r2
    lns8_r2 lns8_r3 lns8_r4 lns8_r5
    lns10_r4 lns12_r6
    lns16_r4 lns16_r7 lns16_r10 lns16_r11 lns16_r12 lns16_r13
    lns32_r20 lns32_r23 lns32_r28
)

case "${stop_after}" in
    smoke|screen|full) ;;
    *)
        echo "error: STOP_AFTER must be smoke, screen, or full" >&2
        exit 2
        ;;
esac

mkdir -p "${run_dir}/smoke" "${run_dir}/screen" "${run_dir}/full"

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
    "${build_dir}/bin/lns_strategy_bench_${target}" \
        --mode smoke \
        --seed "${seed}" \
        --validation-output "${validation_file}" \
        --output "${output}" \
        | tee "${run_dir}/smoke/stdout_${target}.txt"
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
    "${build_dir}/bin/lns_strategy_bench_${target}" \
        --mode screen \
        --dot-powers "${SCREEN_DOT_POWERS:-24}" \
        --gemv-powers "${SCREEN_GEMV_POWERS:-14}" \
        --gemv-rows "${GEMV_ROWS:-1024}" \
        --warmup "${SCREEN_WARMUP:-3}" \
        --samples "${SCREEN_SAMPLES:-3}" \
        --target-sample-ms "${SCREEN_TARGET_SAMPLE_MS:-5}" \
        --seed "${seed}" \
        --output "${output}" \
        | tee "${run_dir}/screen/stdout_${target}.txt"
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
        --output "${output}" \
        | tee "${run_dir}/full/stdout_${target}.txt"
    full_files+=("${output}")
done
merge_csv "${run_dir}/full/timing_samples.csv" "${full_files[@]}"
merge_csv "${run_dir}/full/raw_anchor_samples.csv" "${raw_files[@]}"
python3 "${repo_dir}/tools/summarize_lns_strategy_benchmark.py" \
    --samples "${run_dir}/full/timing_samples.csv" \
    --output "${run_dir}/full/timing_summary.csv"

current_stage="complete"
echo "LNS benchmark complete: ${run_dir}"
