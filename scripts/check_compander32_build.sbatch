#!/usr/bin/env bash
#SBATCH --job-name=comp32-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/031_compander32_conversion_cost/slurm-build-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from the repository root}"
if [[ ! -f "${repo_dir}/CMakeLists.txt" ]]; then
    echo "error: submit from the repository root" >&2
    exit 2
fi
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cd "${repo_dir}"

build_dir="${repo_dir}/build-h200-compander32-build-${SLURM_JOB_ID}"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "${build_dir}" --parallel "${SLURM_CPUS_PER_TASK}" \
    --target compander32_core_test compander32_bench
ctest --test-dir "${build_dir}" --output-on-failure -R compander32_core_test
echo "Compander32 targets compiled successfully."
