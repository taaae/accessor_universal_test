#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/021_unified_strategy_performance}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
build_jobs="${BUILD_JOBS:-2}"
cuda_arch="${CUDA_ARCH:-90}"
stop_after="${STOP_AFTER:-full}"
started_epoch="$(date +%s)"
current_stage="initialization"

case "${stop_after}" in
    sanitizer|smoke|unified|full) ;;
    *)
        echo "error: STOP_AFTER must be sanitizer, smoke, unified, or full" >&2
        exit 2
        ;;
esac

mkdir -p "${run_dir}/sanitizer"

finish_manifest() {
    local status="$?"
    {
        echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "wall_time_seconds=$(($(date +%s) - started_epoch))"
        echo "final_stage=${current_stage}"
        echo "exit_status=${status}"
    } >>"${run_dir}/run_manifest.txt"
    if ((status != 0)); then
        echo "error: unified benchmark failed in stage '${current_stage}' (exit ${status})" >&2
    fi
}
trap finish_manifest EXIT

gpu_snapshot() {
    local label="$1"
    {
        echo
        echo "snapshot=${label}"
        echo "snapshot_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi \
            --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,ecc.mode.current,mig.mode.current \
            --format=csv
    } >>"${run_dir}/gpu_snapshots.txt"
}

{
    echo "experiment=021_unified_strategy_performance"
    echo "purpose=rerun old and new conversion strategies on one physical H200 allocation"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "stop_after=${stop_after}"
    echo "core_widths=2,3,4,5,6,7,8,9,10,12,14,16,17,20,24,28,32"
    echo "core_arithmetic=fp32,fp64"
    echo "core_policy=screen_all_then_full_selected_per_access_group"
    echo "legacy_e2e3=all_historical_fp64_strategies"
    echo "legacy_primary=all_historical_fp64_strategies_for_primary_power_of_two_formats"
    echo "legacy_expanded=all_historical_fp64_strategies_for_expanded_power_of_two_formats"
    echo "timing=CUDA_events_complete_DOT_or_GEMV"
    echo "profiling=none"
    echo "git_status_begin"
    git -C "${repo_dir}" status --short
    echo "git_status_end"
} >"${run_dir}/run_manifest.txt"

{
    nvcc --version
    echo
    # MIG is not a portable -d selector, even on drivers that expose MIG query
    # fields.  Keep the environment capture compatible with physical H200s.
    nvidia-smi -q -d COMPUTE,CLOCK,POWER,ECC
} >"${run_dir}/environment.txt"
gpu_snapshot start

current_stage="build_synccheck_target"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target bitwidth_benchmark_core_test storage_formats_test \
        decoder_strategy_core_test bitwidth_strategy_bench_3
ctest --test-dir "${build_dir}" --output-on-failure \
    -R 'bitwidth_benchmark_core_test|storage_formats_test|decoder_strategy_core_test' \
    | tee "${run_dir}/sanitizer/ctest.txt"

current_stage="cooperative_synccheck"
if ! command -v compute-sanitizer >/dev/null 2>&1; then
    echo "error: compute-sanitizer is required for the unified preflight" >&2
    exit 2
fi
compute-sanitizer --tool synccheck --error-exitcode 97 \
    "${build_dir}/bin/bitwidth_strategy_bench_3" \
    --mode smoke \
    --distributions uniform_0_1 \
    --variants dense/cooperative_shuffle/x32/direct_branchy \
    --seed "${BASE_SEED:-2611923443488327891}" \
    --output "${run_dir}/sanitizer/timing_samples.csv" \
    2>&1 | tee "${run_dir}/sanitizer/synccheck.txt"
gpu_snapshot after_synccheck

if [[ "${stop_after}" == "sanitizer" ]]; then
    current_stage="sanitizer_complete"
    exit 0
fi

current_stage="unified_core"
core_stop_after="full"
if [[ "${stop_after}" == "smoke" ]]; then
    core_stop_after="smoke"
fi
env \
    BUILD_DIR="${build_dir}" \
    BUILD_JOBS="${build_jobs}" \
    CUDA_ARCH="${cuda_arch}" \
    RUN_DIR="${run_dir}/unified_core" \
    RESULTS_ROOT="${results_root}" \
    STOP_AFTER="${core_stop_after}" \
    "${repo_dir}/scripts/run_bitwidth_strategy_benchmark.sh"
gpu_snapshot after_unified_core

if [[ "${stop_after}" == "smoke" ]]; then
    current_stage="smoke_complete"
    exit 0
fi
if [[ "${stop_after}" == "unified" ]]; then
    current_stage="unified_complete"
    exit 0
fi

current_stage="legacy_e2e3"
env \
    BUILD_DIR="${build_dir}" \
    BUILD_JOBS="${build_jobs}" \
    CUDA_ARCH="${cuda_arch}" \
    RUN_DIR="${run_dir}/legacy_e2e3" \
    RESULTS_ROOT="${results_root}" \
    EXPERIMENT_ID=021_legacy_e2e3_same_gpu \
    "${repo_dir}/scripts/run_e2e3_strategy_benchmark.sh"
gpu_snapshot after_legacy_e2e3

current_stage="legacy_primary_power_widths"
env \
    BUILD_DIR="${build_dir}" \
    BUILD_JOBS="${build_jobs}" \
    CUDA_ARCH="${cuda_arch}" \
    RUN_DIR="${run_dir}/legacy_primary" \
    RESULTS_ROOT="${results_root}" \
    "${repo_dir}/scripts/run_all_format_strategy_benchmark.sh"
gpu_snapshot after_legacy_primary

current_stage="legacy_expanded_power_widths"
env \
    BUILD_DIR="${build_dir}" \
    BUILD_JOBS="${build_jobs}" \
    CUDA_ARCH="${cuda_arch}" \
    RUN_DIR="${run_dir}/legacy_expanded" \
    RESULTS_ROOT="${results_root}" \
    "${repo_dir}/scripts/run_expanded_format_strategy_benchmark.sh"
gpu_snapshot finish

current_stage="complete"
{
    echo -e "suite\tdirectory\trole"
    echo -e "unified_core\t${run_dir}/unified_core\tcommon FP32/FP64 comparison harness"
    echo -e "legacy_e2e3\t${run_dir}/legacy_e2e3\thistorical specialized E2M5/E3M4 FP64 strategies"
    echo -e "legacy_primary\t${run_dir}/legacy_primary\thistorical primary 8/16/32-bit FP64 strategies"
    echo -e "legacy_expanded\t${run_dir}/legacy_expanded\thistorical expanded 2/4/8/16/32-bit FP64 strategies"
} >"${run_dir}/suite_index.tsv"

echo "Unified same-GPU benchmark complete: ${run_dir}"
