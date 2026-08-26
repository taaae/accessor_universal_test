#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200-normal32}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/026_normal32_dot_performance}"
build_jobs="${BUILD_JOBS:-2}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
stop_after="${STOP_AFTER:-full}"
seed="${BASE_SEED:-1376283091369227076}"
started_epoch="$(date +%s)"

if [[ "${stop_after}" != "smoke" && "${stop_after}" != "full" ]]; then
    echo "error: STOP_AFTER must be smoke or full" >&2
    exit 2
fi

mkdir -p "${run_dir}/smoke" "${run_dir}/full"

{
    echo "experiment=026_normal32_dot_performance"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "formats=pwl_normal32_16_16,pwq_normal32_8_24,qn32,fp32_e8m23,e11m20,e9m22,e8m29,e8m30,e11m36,raw_fp64"
    echo "strategies=pwl_global_x1,pwq_shared_x1,qn_direct_x1,native_f64_x1,direct_shift_x1,prefix_global_x1,word_branchy_x1,raw_pointer_x1"
    echo "kernel=dot"
    echo "N=134217728"
    echo "arithmetic_type=fp64"
    echo "access=scalar_x1"
    echo "distribution=truncated_normal"
    echo "sigma=fp32_max/4"
    echo "cutoff_sigma=4"
    echo "encoding=offline_untimed"
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
    --target normal32_formats_test normal32_dot_bench
ctest --test-dir "${build_dir}" --output-on-failure -R normal32_formats_test \
    | tee "${run_dir}/ctest.txt"

timeout "${SMOKE_TIMEOUT:-8m}" "${build_dir}/bin/normal32_dot_bench" \
    --mode smoke \
    --seed "${seed}" \
    --output "${run_dir}/smoke/timing_samples.csv" \
    | tee "${run_dir}/smoke/stdout.txt"
python3 "${repo_dir}/tools/summarize_normal32_dot_benchmark.py" \
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
    echo "Normal32 DOT smoke test complete: ${run_dir}"
    exit 0
fi

timeout "${FULL_TIMEOUT:-25m}" "${build_dir}/bin/normal32_dot_bench" \
    --mode full \
    --n 134217728 \
    --warmup 10 \
    --samples 30 \
    --target-sample-ms 20 \
    --seed "${seed}" \
    --output "${run_dir}/full/timing_samples.csv" \
    | tee "${run_dir}/full/stdout.txt"
python3 "${repo_dir}/tools/summarize_normal32_dot_benchmark.py" \
    --samples "${run_dir}/full/timing_samples.csv" \
    --output "${run_dir}/full/timing_summary.csv" \
    | tee "${run_dir}/full/summary.txt"

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
    echo "completed_stage=full"
} >>"${run_dir}/run_manifest.txt"

echo "Normal32 DOT benchmark complete: ${run_dir}"
