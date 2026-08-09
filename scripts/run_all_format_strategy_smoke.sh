#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/014_all_format_strategy_smoke}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"

mkdir -p "${run_dir}"

{
    echo "experiment=014_all_format_strategy_smoke"
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
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target all_format_strategy_smoke decoder_strategy_core_test \
        storage_formats_test
ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

"${build_dir}/bin/all_format_strategy_smoke" \
    --output "${run_dir}/decoder_validation.csv" \
    | tee "${run_dir}/smoke_stdout.txt"

if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-resource-usage \
        "${build_dir}/bin/all_format_strategy_smoke" \
        >"${run_dir}/cuda_resource_usage.txt" || true
fi

echo "All-format strategy smoke results: ${run_dir}"
