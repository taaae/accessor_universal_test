#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build}"
results_dir="${RESULTS_DIR:-${repo_dir}/results/001_cublas_baseline}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${results_dir}"

echo "GPU allocation"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
echo

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}"

benchmark="${build_dir}/bin/dot_bench"
performance_csv="${results_dir}/cublas_performance_${timestamp}.csv"
accuracy_positive_csv="${results_dir}/cublas_accuracy_positive_${timestamp}.csv"
accuracy_signed_csv="${results_dir}/cublas_accuracy_signed_${timestamp}.csv"

"${benchmark}" \
    --mode performance \
    --operation "${OPERATION:-both}" \
    --dataset signed \
    --powers "${PERF_POWERS:-10,16,20,23,26,27}" \
    --warmup "${WARMUP:-5}" \
    --samples "${SAMPLES:-20}" \
    --output "${performance_csv}"

"${benchmark}" \
    --mode accuracy \
    --operation "${OPERATION:-both}" \
    --dataset positive \
    --powers "${ACCURACY_POWERS:-10,16,20}" \
    --warmup "${WARMUP:-5}" \
    --samples "${ACCURACY_SAMPLES:-10}" \
    --output "${accuracy_positive_csv}"

"${benchmark}" \
    --mode accuracy \
    --operation "${OPERATION:-both}" \
    --dataset signed \
    --powers "${ACCURACY_POWERS:-10,16,20}" \
    --warmup "${WARMUP:-5}" \
    --samples "${ACCURACY_SAMPLES:-10}" \
    --output "${accuracy_signed_csv}"

echo
echo "Results:"
echo "  ${performance_csv}"
echo "  ${accuracy_positive_csv}"
echo "  ${accuracy_signed_csv}"
