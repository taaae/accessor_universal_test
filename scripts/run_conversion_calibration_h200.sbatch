#!/usr/bin/env bash
#SBATCH --job-name=conversion-calibration
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=04:00:00
#SBATCH --output=results/028_conversion_cost_calibration/slurm-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from the exact repository checkout}"
[[ -x "${repo_dir}/scripts/run_conversion_calibration.sh" ]] || { echo "wrong submit directory" >&2; exit 2; }
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  env MODE="${MODE:-smoke}" RUN_DIR="${RUN_DIR:?set one protected run directory}" \
  BUILD_DIR="${repo_dir}/build-conversion-calibration-$(git rev-parse --short=12 HEAD)" BUILD_JOBS=2 \
  "${repo_dir}/scripts/run_conversion_calibration.sh"
