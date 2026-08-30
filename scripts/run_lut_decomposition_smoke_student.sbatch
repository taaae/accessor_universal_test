#!/usr/bin/env bash
#SBATCH --job-name=lut-decomp-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=00:20:00
#SBATCH --output=results/029_lut_decomposition_shapes/slurm-smoke-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from the repository root}"
if [[ ! -x "${repo_dir}/scripts/run_lut_decomposition.sh" ]]; then
    echo "error: submit from the accessor_universal_test repository root" >&2
    exit 2
fi
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env MODE=smoke BUILD_JOBS=4 CUDA_ARCH=90 TMPDIR=/tmp TARGET_TIMEOUT=900 \
    "${repo_dir}/scripts/run_lut_decomposition.sh"
