#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/016_expanded_format_strategy_smoke}"
build_jobs="${BUILD_JOBS:-2}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"

mkdir -p "${run_dir}"

{
    echo "experiment=016_expanded_format_strategy_smoke"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "git_status_begin"
    git -C "${repo_dir}" status --short
    echo "git_status_end"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
        --format=csv
    nvcc --version
} >"${run_dir}/environment.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"

# Build the split translation units sequentially. This avoids making nvcc
# instantiate every candidate family in one large process.
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target storage_formats_test decoder_strategy_core_test
for bits in 2 4 8 16 32; do
    cmake --build "${build_dir}" --parallel "${build_jobs}" \
        --target "expanded_format_strategy_smoke_${bits}"
done

ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

for bits in 2 4 8 16 32; do
    "${build_dir}/bin/expanded_format_strategy_smoke_${bits}" \
        --output "${run_dir}/decoder_validation_${bits}bit.csv" \
        | tee "${run_dir}/smoke_${bits}bit_stdout.txt"
done

if command -v cuobjdump >/dev/null 2>&1; then
    for bits in 2 4 8 16 32; do
        cuobjdump --dump-resource-usage \
            "${build_dir}/bin/expanded_format_strategy_smoke_${bits}" \
            >"${run_dir}/cuda_resource_usage_${bits}bit.txt" || true
    done
fi

echo "Expanded-format strategy smoke results: ${run_dir}"
