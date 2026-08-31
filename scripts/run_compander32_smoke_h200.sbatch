#!/usr/bin/env bash
#SBATCH --job-name=comp32-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=00:25:00
#SBATCH --output=results/031_compander32_conversion_cost/slurm-smoke-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

# Submit with --nodelist=gpu-nvidia-h200-2 or gpu-nvidia-h200-3. The runner
# rejects every other host and verifies that CUDA exposes an H200.

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from the repository root}"
if [[ ! -x "${repo_dir}/scripts/run_compander32_benchmark.sh" ]]; then
    echo "error: submit from the repository root" >&2
    exit 2
fi
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env MODE=smoke BUILD_JOBS=8 CUDA_ARCH=90 TMPDIR=/tmp \
    "${repo_dir}/scripts/run_compander32_benchmark.sh"
