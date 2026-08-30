#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit_short="$(git -C "${repo_dir}" rev-parse --short=12 HEAD)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200-dyadic-normal32-${commit_short}}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/030_dyadic_normal32_dispersion}"
run_tag="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${RUN_DIR:-${results_root}/run_${run_tag}}"
mode="${MODE:-smoke}"
build_jobs="${BUILD_JOBS:-4}"
target_timeout="${TARGET_TIMEOUT:-1800}"

case "${mode}" in
    smoke|full) ;;
    *) echo "error: MODE must be smoke or full" >&2; exit 2 ;;
esac

mkdir -p "${results_root}"
lock_file="$(dirname "${repo_dir}")/.accessor-universal-dyadic-normal32-${USER:-unknown}.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "error: another DyadicNormal32 runner holds ${lock_file}" >&2
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
        echo "experiment=030_dyadic_normal32_dispersion"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
        echo "kernel=dot"
        echo "full_n=67108864"
        echo "blocks=512"
        echo "threads=256"
        echo "access=scalar_x1"
        echo "arithmetic=fp64"
        echo "target_x=0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1"
        echo "additional_distribution=genuine_n01_segment_probabilities"
        echo "full_warmup=10"
        echo "full_warmups_per_half_batch=5"
        echo "full_samples=50"
        echo "sample_order=raw_fp64_before,raw_fp32_before,fp32_to_fp64_before,ascending_targets_with_forward_strategy_order,genuine_forward,genuine_reverse,descending_targets_with_reverse_strategy_order,fp32_to_fp64_after,raw_fp32_after,raw_fp64_after"
        echo "baselines=raw_fp64,raw_fp32,fp32_to_fp64"
        echo "dyadic_strategies=dyadic_normal32,dyadic_sign_fused,dyadic_bitcast_shared,dyadic_bitcast_constant"
        echo "forward_strategy_order=dyadic_normal32,dyadic_sign_fused,dyadic_bitcast_shared,dyadic_bitcast_constant"
        echo "reverse_strategy_order=dyadic_bitcast_constant,dyadic_bitcast_shared,dyadic_sign_fused,dyadic_normal32"
        echo "dyadic_storage_bits=32"
        echo "coefficient_entries=32"
        echo "coefficient_bytes=512"
        echo "coefficient_locations=shared,constant"
        echo "shared_staging=inside_timed_first_stage_once_per_block_for_three_shared_variants"
        echo "source_sigma=1"
        echo "density_sigma=sqrt(3)"
        echo "terminal_convention=h31_maps_to_half_normal_tail_boundary_2^-32"
        echo "left_seed=7640891576956012809"
        echo "right_seed=13503953896175478587"
        echo "slurm_job_id=${SLURM_JOB_ID:-none}"
        echo "slurm_job_name=${SLURM_JOB_NAME:-none}"
        echo "slurm_node_list=${SLURM_JOB_NODELIST:-none}"
        echo "git_status_begin"
        git -C "${repo_dir}" status --short
        echo "git_status_end"
    } >"${run_dir}/run_manifest.txt"
fi

{
    echo "stage=${mode}"
    echo "hostname=$(hostname)"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-unset}"
    echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi -L
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
    --target dyadic_normal32_core_test dyadic_normal32_bench

current_stage="host_test"
ctest --test-dir "${build_dir}" --output-on-failure \
    -R '^dyadic_normal32_core_test$' | tee "${stage_dir}/ctest.txt"

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
    "${build_dir}/bin/dyadic_normal32_bench" \
        --mode "${mode}" \
        --n "${benchmark_n}" \
        --warmup "${benchmark_warmup}" \
        --samples "${benchmark_samples}" \
        --output "${stage_dir}/timing_samples.csv" \
        --metrics-output "${stage_dir}/access_metrics.csv" \
        --correctness-output "${stage_dir}/correctness_checks.txt" \
        --coefficients-output "${stage_dir}/coefficient_table.csv" \
        | tee "${stage_dir}/stdout.txt"; then
    echo "error: benchmark failed or exceeded ${target_timeout}s" >&2
    exit 1
fi

current_stage="artifact_contract"
if [[ ! -s "${stage_dir}/correctness_checks.txt" ]] || \
   ! grep -q '^current_cpu_gpu_bit_mismatches=0$' "${stage_dir}/correctness_checks.txt" || \
   ! grep -q '^bitcast_shared_cpu_gpu_bit_mismatches=0$' "${stage_dir}/correctness_checks.txt" || \
   ! grep -q '^bitcast_constant_cpu_gpu_bit_mismatches=0$' "${stage_dir}/correctness_checks.txt" || \
   ! grep -q '^dot_current_passed=1$' "${stage_dir}/correctness_checks.txt" || \
   ! grep -q '^dot_sign_fused_passed=1$' "${stage_dir}/correctness_checks.txt" || \
   ! grep -q '^dot_bitcast_shared_passed=1$' "${stage_dir}/correctness_checks.txt" || \
   ! grep -q '^dot_bitcast_constant_passed=1$' "${stage_dir}/correctness_checks.txt" || \
   ! grep -q '^timed_current_sign_fused_bit_mismatches=0$' "${stage_dir}/correctness_checks.txt" || \
   ! grep -q '^timed_bitcast_shared_constant_bit_mismatches=0$' "${stage_dir}/correctness_checks.txt" || \
   ! grep -q '^timed_result_validation_passed=1$' "${stage_dir}/correctness_checks.txt"; then
    echo "error: CPU/GPU correctness contract failed" >&2
    exit 5
fi
if [[ "$(wc -l <"${stage_dir}/coefficient_table.csv")" -ne 33 ]]; then
    echo "error: coefficient table has the wrong row count" >&2
    exit 5
fi
if [[ "${mode}" == "smoke" ]]; then
    if [[ "$(wc -l <"${stage_dir}/timing_samples.csv")" -ne 58 ]]; then
        echo "error: smoke timing CSV has the wrong row count" >&2
        exit 5
    fi
    if [[ "$(wc -l <"${stage_dir}/access_metrics.csv")" -ne 5 ]]; then
        echo "error: smoke metrics CSV has the wrong row count" >&2
        exit 5
    fi
    current_stage="smoke_analysis"
    python3 "${repo_dir}/tools/validate_dyadic_normal32_smoke.py" \
        --samples "${stage_dir}/timing_samples.csv" \
        --metrics "${stage_dir}/access_metrics.csv" \
        --correctness "${stage_dir}/correctness_checks.txt" \
        --coefficients "${stage_dir}/coefficient_table.csv" \
        --analyzer "${repo_dir}/tools/analyze_dyadic_normal32.py" \
        | tee "${stage_dir}/analysis.txt"
else
    if [[ "$(wc -l <"${stage_dir}/timing_samples.csv")" -ne 2151 ]]; then
        echo "error: full timing CSV has the wrong row count" >&2
        exit 5
    fi
    if [[ "$(wc -l <"${stage_dir}/access_metrics.csv")" -ne 11 ]]; then
        echo "error: full metrics CSV has the wrong row count" >&2
        exit 5
    fi
    current_stage="analysis"
    python3 "${repo_dir}/tools/analyze_dyadic_normal32.py" \
        --samples "${stage_dir}/timing_samples.csv" \
        --metrics "${stage_dir}/access_metrics.csv" \
        --correctness "${stage_dir}/correctness_checks.txt" \
        --coefficients "${stage_dir}/coefficient_table.csv" \
        --output-dir "${stage_dir}" \
        | tee "${stage_dir}/analysis.txt"
fi

current_stage="complete"
echo "DyadicNormal32 ${mode} complete: ${stage_dir}"
