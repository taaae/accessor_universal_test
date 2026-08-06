#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/004_storage_formats}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
count="${COUNT:-262144}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
started_epoch="$(date +%s)"

mkdir -p "${run_dir}"

{
    echo "experiment=004_storage_formats"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "git_status_begin"
    git -C "${repo_dir}" status --short
    echo "git_status_end"
    echo
    echo "nvidia_smi"
    nvidia-smi \
        --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total \
        --format=csv
    echo
    echo "nvcc_version"
    nvcc --version
} >"${run_dir}/environment.txt"

{
    echo "count_per_distribution=${count}"
    echo "distributions=uniform_-1_1,normal_0_1"
    echo "decode_lanes=1,2,4"
    echo "arithmetic=fp64"
    echo "cuda_arch=${cuda_arch}"
    echo "dump_sass=${DUMP_SASS:-1}"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}"
ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

validator="${build_dir}/bin/storage_formats_cuda_test"
"${validator}" \
    --count "${count}" \
    --output "${run_dir}/validation.csv" \
    | tee "${run_dir}/validation_stdout.txt"

if command -v cuobjdump >/dev/null 2>&1; then
    cuobjdump --dump-resource-usage "${validator}" \
        >"${run_dir}/cuda_resource_usage.txt"
    if [[ "${DUMP_SASS:-1}" == "1" ]]; then
        cuobjdump --dump-sass "${validator}" >"${run_dir}/sass.txt"
    fi
fi

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
} >>"${run_dir}/run_manifest.txt"

echo
echo "Storage-format validation complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/validation.csv"
