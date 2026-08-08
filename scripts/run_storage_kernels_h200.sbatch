#!/usr/bin/env bash
#SBATCH --job-name=storage-kernels
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=00:30:00
#SBATCH --output=results/005_storage_kernels/slurm-%j.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"

if [[ ! -f "${repo_dir}/CMakeLists.txt" ||
      ! -x "${repo_dir}/scripts/run_storage_kernels_experiment.sh" ]]; then
    echo "error: submit this job from the accessor_universal_test repository root" >&2
    exit 2
fi

cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env \
        BUILD_DIR="${repo_dir}/build-h200" \
        RESULTS_ROOT="${repo_dir}/results/005_storage_kernels" \
        BUILD_JOBS="${SLURM_CPUS_PER_TASK}" \
        CUDA_ARCH=90 \
        DOT_COUNT=262147 \
        GEMV_ROWS=257 \
        GEMV_COLUMNS=1027 \
        "${repo_dir}/scripts/run_storage_kernels_experiment.sh"
