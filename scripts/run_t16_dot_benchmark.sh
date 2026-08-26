#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/025_t16_dot_performance}"
build_jobs="${BUILD_JOBS:-2}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
stop_after="${STOP_AFTER:-full}"
seed="${BASE_SEED:-2611923443488327891}"
started_epoch="$(date +%s)"

if [[ "${stop_after}" != "smoke" && "${stop_after}" != "full" ]]; then
    echo "error: STOP_AFTER must be smoke or full" >&2
    exit 2
fi

mkdir -p "${run_dir}/smoke" "${run_dir}/full"

{
    echo "experiment=025_t16_dot_performance"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "formats=t16,fp16_e5m10,e6m9,e8m15,raw_fp32"
    echo "strategies=global_lut_x1,native_scalar_x1,direct_branchy_x1,direct_shift_x1,raw_x1"
    echo "kernel=dot"
    echo "N=134217728"
    echo "arithmetic_type=fp32"
    echo "distribution=truncated_normal"
    echo "sigma=16376"
    echo "cutoff_sigma=4"
    echo "warmup=10"
    echo "samples=30"
    echo "target_sample_ms=20"
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

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target t16_codebook_test t16_dot_bench
ctest --test-dir "${build_dir}" --output-on-failure -R t16_codebook_test \
    | tee "${run_dir}/ctest.txt"

timeout "${SMOKE_TIMEOUT:-5m}" "${build_dir}/bin/t16_dot_bench" \
    --mode smoke \
    --seed "${seed}" \
    --output "${run_dir}/smoke/timing_samples.csv" \
    | tee "${run_dir}/smoke/stdout.txt"
python3 "${repo_dir}/tools/summarize_t16_dot_benchmark.py" \
    --samples "${run_dir}/smoke/timing_samples.csv" \
    --output "${run_dir}/smoke/timing_summary.csv" \
    | tee "${run_dir}/smoke/summary.txt"

if [[ "${stop_after}" == "smoke" ]]; then
    finished_epoch="$(date +%s)"
    {
        echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "wall_time_seconds=$((finished_epoch - started_epoch))"
        echo "completed_stage=smoke"
    } >>"${run_dir}/run_manifest.txt"
    echo "T16 DOT smoke test complete: ${run_dir}"
    exit 0
fi

timeout "${FULL_TIMEOUT:-15m}" "${build_dir}/bin/t16_dot_bench" \
    --mode full \
    --n 134217728 \
    --warmup 10 \
    --samples 30 \
    --target-sample-ms 20 \
    --seed "${seed}" \
    --output "${run_dir}/full/timing_samples.csv" \
    | tee "${run_dir}/full/stdout.txt"
python3 "${repo_dir}/tools/summarize_t16_dot_benchmark.py" \
    --samples "${run_dir}/full/timing_samples.csv" \
    --output "${run_dir}/full/timing_summary.csv" \
    | tee "${run_dir}/full/summary.txt"

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
    echo "completed_stage=full"
} >>"${run_dir}/run_manifest.txt"

echo "T16 DOT benchmark complete: ${run_dir}"
