#!/usr/bin/env bash
#SBATCH --job-name=t16-dot-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/025_t16_dot_performance/slurm-build-%j.out

set -eo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
if [[ ! -f "${repo_dir}/CMakeLists.txt" ]]; then
    echo "error: submit from the accessor_universal_test repository root" >&2
    exit 2
fi

source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"

build_dir="${repo_dir}/build-h200-t16"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "${build_dir}" --parallel "${SLURM_CPUS_PER_TASK}" \
    --target t16_codebook_test t16_dot_bench
ctest --test-dir "${build_dir}" --output-on-failure -R t16_codebook_test

echo "T16 DOT targets compiled successfully."
