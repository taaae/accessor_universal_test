#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build}"
results_dir="${RESULTS_DIR:-${repo_dir}/results}"
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

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}"

benchmark="${build_dir}/bin/dot_bench"
report_base="${results_dir}/cublas_${operation}_n${vector_size}_${timestamp}"
profile_csv="${report_base}.csv"

ncu \
    --profile-from-start off \
    --target-processes all \
    --set full \
    --force-overwrite \
    --export "${report_base}" \
    "${benchmark}" \
        --mode profile \
        --operation "${operation}" \
        --dataset signed \
        --n "${vector_size}" \
        --warmup "${WARMUP:-5}" \
        --output "${profile_csv}"

echo
echo "Profile report: ${report_base}.ncu-rep"
echo "Operation CSV:  ${profile_csv}"
