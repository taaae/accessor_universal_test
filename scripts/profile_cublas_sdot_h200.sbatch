#!/usr/bin/env bash
#SBATCH --job-name=cublas-sdot-ncu
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-nvidia-h200-[1-3]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"

if [[ ! -f "${repo_dir}/CMakeLists.txt" ||
      ! -x "${repo_dir}/scripts/profile_cublas_baseline.sh" ]]; then
    echo "error: submit this job from the accessor_universal_test repository root" >&2
    exit 2
fi

cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env \
        BUILD_DIR="${repo_dir}/build-h200" \
        RESULTS_DIR="${repo_dir}/results" \
        BUILD_JOBS="${SLURM_CPUS_PER_TASK}" \
        CUDA_ARCH=90 \
        OPERATION=sdot \
        N=134217728 \
        WARMUP=5 \
        "${repo_dir}/scripts/profile_cublas_baseline.sh"
