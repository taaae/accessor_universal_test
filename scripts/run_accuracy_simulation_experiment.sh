#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/007_gpu_accuracy_simulation}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
dot_powers="${DOT_POWERS:-10,14,18,22}"
gemv_powers="${GEMV_POWERS:-8,10,12,14,16}"
dot_samples="${DOT_SAMPLES:-8192}"
dot_batches="${DOT_STATISTICAL_BATCHES:-32}"
gemv_rows="${GEMV_ROWS:-1024}"
gemv_replicates="${GEMV_REPLICATES:-16}"
base_seed="${BASE_SEED:-2611923443488327891}"
workspace_gib="${WORKSPACE_GIB:-12}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
started_epoch="$(date +%s)"

mkdir -p "${run_dir}"

{
    echo "experiment=007_gpu_accuracy_simulation"
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
    echo "dot_powers=${dot_powers}"
    echo "dot_samples=${dot_samples}"
    echo "dot_statistical_batches=${dot_batches}"
    echo "gemv_powers=${gemv_powers}"
    echo "gemv_rows=${gemv_rows}"
    echo "gemv_replicates=${gemv_replicates}"
    echo "distributions=uniform_0_1,normal_0_1"
    echo "formats=all_17"
    echo "load_lanes=1,2,4"
    echo "arithmetic=fp64_fma"
    echo "reference=gpu_double_double"
    echo "rng=curand_philox4x32_10"
    echo "base_seed=${base_seed}"
    echo "workspace_gib=${workspace_gib}"
    echo "cuda_arch=${cuda_arch}"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target accuracy_simulation memory_accessor_test storage_formats_test \
        accuracy_statistics_test
ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

"${build_dir}/bin/accuracy_simulation" \
    --dot-powers "${dot_powers}" \
    --gemv-powers "${gemv_powers}" \
    --dot-samples "${dot_samples}" \
    --dot-statistical-batches "${dot_batches}" \
    --gemv-rows "${gemv_rows}" \
    --gemv-replicates "${gemv_replicates}" \
    --base-seed "${base_seed}" \
    --workspace-gib "${workspace_gib}" \
    --output-dir "${run_dir}" \
    | tee "${run_dir}/simulation_stdout.txt"

python3 "${repo_dir}/tools/check_accuracy_simulation.py" "${run_dir}" \
    | tee "${run_dir}/output_validation.txt"

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
} >>"${run_dir}/run_manifest.txt"

echo
echo "GPU accuracy simulation complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/simulation_summary.csv"
echo "  ${run_dir}/empirical_quantiles.csv"
echo "  ${run_dir}/batch_estimates.csv"
echo "  ${run_dir}/convergence_report.csv"
