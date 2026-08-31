#!/usr/bin/env bash
#SBATCH --job-name=comp32-full
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=01:00:00
#SBATCH --output=results/031_compander32_conversion_cost/slurm-full-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

# Submit with --nodelist=gpu-nvidia-h200-2,gpu-nvidia-h200-3. Slurm chooses the
# first available node; the runner rejects any other host and non-H200 device.

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
    env MODE=full BUILD_JOBS=8 CUDA_ARCH=90 TMPDIR=/tmp \
    "${repo_dir}/scripts/run_compander32_benchmark.sh"
