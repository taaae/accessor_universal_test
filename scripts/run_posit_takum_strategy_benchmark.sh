#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/024_posit_takum_strategy_performance}"
run_dir="${RUN_DIR:-${results_root}/run_$(date -u +%Y%m%dT%H%M%SZ)}"
mode="${MODE:-smoke}"
build_jobs="${BUILD_JOBS:-2}"
seed="${BASE_SEED:-7762632251762735585}"
target_timeout="${TARGET_TIMEOUT:-600}"
started_epoch="$(date +%s)"
current_stage="initialization"

case "${mode}" in
    smoke|full) ;;
    *) echo "error: MODE must be smoke or full" >&2; exit 2 ;;
esac

alt_targets=(
    posit8_es0 posit14_es1 posit16_es1 posit32_es2
    takum8 takum14 takum16 takum32
    takum_log8 takum_log14 takum_log16 takum_log32
)
ieee_targets=(
    fp8_e4m3_fp32 fp8_e4m3_fp64 fp8_e5m2_fp32 fp8_e5m2_fp64
    e3m4_fp32 e3m4_fp64 e6m1_fp32 e6m1_fp64
    e8m5_fp32 e11m2_fp64 e2m11_fp32 e2m11_fp64 e5m8_fp32 e5m8_fp64
    fp16_e5m10_fp32 fp16_e5m10_fp64 bf16_e8m7_fp32 bf16_e8m7_fp64
    e11m4_fp64 e3m12_fp32 e3m12_fp64 e6m9_fp32 e6m9_fp64
    fp32_e8m23_fp32 fp32_e8m23_fp64 e11m20_fp64 e4m27_fp64 e10m21_fp64
)
control_targets=(8_fp32 8_fp64 14_fp32 14_fp64)
targets=("${alt_targets[@]}" "${ieee_targets[@]}" "${control_targets[@]}")

executable_for_target() {
    local target="$1"
    if [[ " ${alt_targets[*]} " == *" ${target} "* ]]; then
        echo "posit_takum_strategy_bench_${target}"
    elif [[ " ${ieee_targets[*]} " == *" ${target} "* ]]; then
        echo "ieee_scalar_strategy_bench_${target}"
    else
        echo "lut_content_control_bench_${target}"
    fi
}

mkdir -p "${run_dir}/${mode}"
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

{
    echo "experiment=024_posit_takum_strategy_performance"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "mode=${mode}"
    echo "formats=${targets[*]}"
    echo "arithmetic_types=fp32,fp64"
    echo "main_distributions=field_balanced_finite,paired_log_uniform_finite"
    echo "lut_control_traces=lut_scattered_control,lut_concentrated_control"
    echo "storage_layout=dense"
    echo "access_method=scalar"
    echo "packet_values=1"
    echo "threads_per_block=256"
    echo "dot_blocks=512"
    echo "cuda_arch=sm_${CUDA_ARCH:-90}"
    echo "cuda_compile_flags=-O3 -lineinfo -Xptxas=-v --ftz=false"
    echo "dot_n=${DOT_N:-134217728}"
    echo "gemv_m=${GEMV_M:-1024}"
    echo "gemv_n=${GEMV_N:-65536}"
    echo "warmup=${WARMUP:-10}"
    echo "samples=${SAMPLES:-30}"
    echo "base_seed=${seed}"
    echo "target_timeout_seconds=${target_timeout}"
    echo "git_status_begin"
    git -C "${repo_dir}" status --short
    echo "git_status_end"
} >"${run_dir}/run_manifest.txt"

{
    nvidia-smi --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,ecc.mode.current,power.limit --format=csv
    echo
    nvcc --version
} >"${run_dir}/environment.txt"

current_stage="configure"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-90}"

current_stage="host_test"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target posit_takum_core_test
ctest --test-dir "${build_dir}" --output-on-failure \
    -R '^posit_takum_core_test$' | tee "${run_dir}/${mode}/ctest.txt"
"${repo_dir}/scripts/check_posit_takum_host.sh" \
    | tee "${run_dir}/${mode}/universal_reference.txt"

current_stage="build_targets"
for target in "${targets[@]}"; do
    executable="$(executable_for_target "${target}")"
    cmake --build "${build_dir}" --parallel "${build_jobs}" --target "${executable}"
done

run_bounded() {
    local target="$1" executable
    executable="$(executable_for_target "${target}")"
    local prefix="${run_dir}/${mode}/${target}"
    if timeout --foreground "${target_timeout}" \
        "${build_dir}/bin/${executable}" \
        --mode "${mode}" \
        --seed "${seed}" \
        --dot-n "${DOT_N:-134217728}" \
        --gemv-m "${GEMV_M:-1024}" \
        --gemv-n "${GEMV_N:-65536}" \
        --warmup "${WARMUP:-10}" \
        --samples "${SAMPLES:-30}" \
        --output "${prefix}_timing_samples.csv" \
        --validation-output "${prefix}_decoder_validation.csv" \
        --histogram-output "${prefix}_histograms.csv" \
        | tee "${prefix}_stdout.txt"; then
        return 0
    fi
    echo "error: ${target} failed or exceeded ${target_timeout}s" >&2
    exit 1
}

current_stage="${mode}_benchmark"
for target in "${targets[@]}"; do
    run_bounded "${target}"
done

merge_csv() {
    local suffix="$1" output="$2"
    local files=()
    for target in "${targets[@]}"; do
        files+=("${run_dir}/${mode}/${target}_${suffix}.csv")
    done
    awk 'FNR == 1 { if (!header++) print; next } { print }' "${files[@]}" >"${output}"
}

current_stage="merge_and_validate"
merge_csv timing_samples "${run_dir}/${mode}/timing_samples.csv"
merge_csv decoder_validation "${run_dir}/${mode}/decoder_validation.csv"
merge_csv histograms "${run_dir}/${mode}/histograms.csv"
python3 "${repo_dir}/tools/validate_posit_takum_strategy_run.py" \
    --mode "${mode}" \
    --samples "${run_dir}/${mode}/timing_samples.csv" \
    --validation "${run_dir}/${mode}/decoder_validation.csv" \
    --histograms "${run_dir}/${mode}/histograms.csv" \
    --output "${run_dir}/${mode}/coverage.json" \
    | tee "${run_dir}/${mode}/coverage.txt"

current_stage="complete"
echo "posit/takum ${mode} complete: ${run_dir}"
