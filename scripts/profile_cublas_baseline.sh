#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build}"
results_dir="${RESULTS_DIR:-${repo_dir}/results/002_cublas_sdot_n2p27_profile}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
operation="${OPERATION:-sdot}"
vector_size="${N:-134217728}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

if ! command -v ncu >/dev/null 2>&1; then
    echo "error: Nsight Compute CLI (ncu) is not available" >&2
    exit 1
fi

mkdir -p "${results_dir}"

echo "GPU allocation"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
echo
ncu --version
echo

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}"

benchmark="${build_dir}/bin/dot_bench"
report_base="${results_dir}/cublas_${operation}_n${vector_size}_${timestamp}"
benchmark_csv="${report_base}_benchmark.csv"
details_txt="${report_base}_ncu_details.txt"
metrics_csv="${report_base}_ncu_raw.csv"

ncu \
    --profile-from-start off \
    --target-processes all \
    --section SpeedOfLight \
    --section SpeedOfLight_RooflineChart \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section SchedulerStats \
    --force-overwrite \
    --export "${report_base}" \
    "${benchmark}" \
        --mode profile \
        --operation "${operation}" \
        --dataset signed \
        --n "${vector_size}" \
        --warmup "${WARMUP:-5}" \
        --output "${benchmark_csv}"

ncu --import "${report_base}.ncu-rep" --page details \
    --print-details all >"${details_txt}"
ncu --import "${report_base}.ncu-rep" --page raw --csv \
    --print-units base >"${metrics_csv}"

echo
echo "Profile outputs:"
echo "  ${report_base}.ncu-rep"
echo "  ${details_txt}"
echo "  ${metrics_csv}"
echo "  ${benchmark_csv}"
