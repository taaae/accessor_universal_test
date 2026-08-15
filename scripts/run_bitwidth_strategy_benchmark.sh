#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/020_bitwidth_strategy_performance}"
build_jobs="${BUILD_JOBS:-2}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
started_epoch="$(date +%s)"

read -r -a widths <<<"${WIDTHS:-2 3 4 5 6 7 9 10 12 14 17 20 24 28}"
dot_powers="${DOT_POWERS:-12,16,20,24,27}"
gemv_powers="${GEMV_POWERS:-8,10,12,14,16}"
gemv_rows="${GEMV_ROWS:-1024}"
warmup="${WARMUP:-10}"
samples="${SAMPLES:-5}"
target_sample_ms="${TARGET_SAMPLE_MS:-15}"
seed="${BASE_SEED:-2611923443488327891}"

mkdir -p "${run_dir}/smoke" "${run_dir}/screen" "${run_dir}/full"

merge_csv() {
    local output="$1"
    shift
    awk 'FNR == 1 { if (!header++) print; next } { print }' "$@" >"${output}"
}

{
    echo "experiment=020_bitwidth_strategy_performance"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "widths=${widths[*]}"
    echo "arithmetic_types=fp32,fp64"
    echo "storage_layouts=dense,padded"
    echo "access_methods=scalar,thread_packet,cooperative_shuffle"
    echo "screen_dot_powers=${SCREEN_DOT_POWERS:-24}"
    echo "screen_gemv_powers=${SCREEN_GEMV_POWERS:-14}"
    echo "full_dot_powers=${dot_powers}"
    echo "full_gemv_powers=${gemv_powers}"
    echo "gemv_rows=${gemv_rows}"
    echo "distributions=uniform_0_1,normal_0_1"
    echo "full_warmup=${warmup}"
    echo "full_samples=${samples}"
    echo "full_target_sample_ms=${target_sample_ms}"
    echo "base_seed=${seed}"
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

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"

cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target bitwidth_benchmark_core_test storage_formats_test \
        decoder_strategy_core_test
for width in "${widths[@]}"; do
    cmake --build "${build_dir}" --parallel "${build_jobs}" \
        --target "bitwidth_strategy_bench_${width}"
done

ctest --test-dir "${build_dir}" --output-on-failure \
    -R 'bitwidth_benchmark_core_test|storage_formats_test|decoder_strategy_core_test' \
    | tee "${run_dir}/ctest.txt"

smoke_files=()
for width in "${widths[@]}"; do
    output="${run_dir}/smoke/timing_samples_${width}bit.csv"
    "${build_dir}/bin/bitwidth_strategy_bench_${width}" \
        --mode smoke \
        --seed "${seed}" \
        --output "${output}" \
        | tee "${run_dir}/smoke/stdout_${width}bit.txt"
    smoke_files+=("${output}")
done
merge_csv "${run_dir}/smoke/timing_samples.csv" "${smoke_files[@]}"
python3 "${repo_dir}/tools/validate_bitwidth_strategy_smoke.py" \
    --samples "${run_dir}/smoke/timing_samples.csv" \
    --output "${run_dir}/smoke/validation.csv" \
    | tee "${run_dir}/smoke/validation.txt"

screen_files=()
for width in "${widths[@]}"; do
    output="${run_dir}/screen/timing_samples_${width}bit.csv"
    "${build_dir}/bin/bitwidth_strategy_bench_${width}" \
        --mode screen \
        --dot-powers "${SCREEN_DOT_POWERS:-24}" \
        --gemv-powers "${SCREEN_GEMV_POWERS:-14}" \
        --gemv-rows "${gemv_rows}" \
        --warmup "${SCREEN_WARMUP:-3}" \
        --samples "${SCREEN_SAMPLES:-3}" \
        --target-sample-ms "${SCREEN_TARGET_SAMPLE_MS:-5}" \
        --seed "${seed}" \
        --output "${output}" \
        | tee "${run_dir}/screen/stdout_${width}bit.txt"
    screen_files+=("${output}")
done
merge_csv "${run_dir}/screen/timing_samples.csv" "${screen_files[@]}"
python3 "${repo_dir}/tools/summarize_bitwidth_strategy_benchmark.py" \
    --samples "${run_dir}/screen/timing_samples.csv" \
    --output "${run_dir}/screen/timing_summary.csv"
python3 "${repo_dir}/tools/select_bitwidth_strategy_finalists.py" \
    --samples "${run_dir}/screen/timing_samples.csv" \
    --finalists "${run_dir}/screen/finalists.txt" \
    --ranking "${run_dir}/screen/strategy_ranking.csv" \
    --top "${FINALISTS_PER_GROUP:-2}" \
    | tee "${run_dir}/screen/selection.txt"

full_files=()
for width in "${widths[@]}"; do
    output="${run_dir}/full/timing_samples_${width}bit.csv"
    "${build_dir}/bin/bitwidth_strategy_bench_${width}" \
        --mode full \
        --dot-powers "${dot_powers}" \
        --gemv-powers "${gemv_powers}" \
        --gemv-rows "${gemv_rows}" \
        --warmup "${warmup}" \
        --samples "${samples}" \
        --target-sample-ms "${target_sample_ms}" \
        --seed "${seed}" \
        --variant-file "${run_dir}/screen/finalists.txt" \
        --output "${output}" \
        | tee "${run_dir}/full/stdout_${width}bit.txt"
    full_files+=("${output}")
done
merge_csv "${run_dir}/full/timing_samples.csv" "${full_files[@]}"
python3 "${repo_dir}/tools/summarize_bitwidth_strategy_benchmark.py" \
    --samples "${run_dir}/full/timing_samples.csv" \
    --output "${run_dir}/full/timing_summary.csv" \
    | tee "${run_dir}/full/validation.txt"

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
} >>"${run_dir}/run_manifest.txt"

echo
echo "Arbitrary-width benchmark complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/screen/strategy_ranking.csv"
echo "  ${run_dir}/full/timing_samples.csv"
echo "  ${run_dir}/full/timing_summary.csv"
