#!/usr/bin/env bash
#SBATCH --job-name=storage-formats
#SBATCH --partition=compute
#SBATCH --nodelist=gpu-nvidia-h200-[1-3]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=00:15:00
#SBATCH --output=results/004_storage_formats/slurm-%j.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"

if [[ ! -f "${repo_dir}/CMakeLists.txt" ||
      ! -x "${repo_dir}/scripts/run_storage_formats_experiment.sh" ]]; then
    echo "error: submit this job from the accessor_universal_test repository root" >&2
    exit 2
fi

cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env \
        BUILD_DIR="${repo_dir}/build-h200" \
        RESULTS_ROOT="${repo_dir}/results/004_storage_formats" \
        BUILD_JOBS="${SLURM_CPUS_PER_TASK}" \
        CUDA_ARCH=90 \
        "${repo_dir}/scripts/run_storage_formats_experiment.sh"
