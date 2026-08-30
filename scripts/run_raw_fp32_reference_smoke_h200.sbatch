#!/usr/bin/env bash
#SBATCH --job-name=raw-fp32-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=00:10:00
#SBATCH --output=results/027_lut_distribution_shape/slurm-raw-fp32-smoke-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from the repository root}"
if [[ ! -x "${repo_dir}/scripts/run_raw_fp32_reference.sh" ]]; then
    echo "error: submit from the LUT-distribution worktree root" >&2
    exit 2
fi
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env MODE=smoke RESULTS_ROOT="${repo_dir}/results/027_lut_distribution_shape" \
    BUILD_JOBS=4 CUDA_ARCH=90 TMPDIR=/tmp TARGET_TIMEOUT=300 \
    "${repo_dir}/scripts/run_raw_fp32_reference.sh"
