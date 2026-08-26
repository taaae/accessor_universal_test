#!/usr/bin/env bash
#SBATCH --job-name=posit-takum-full
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=08:00:00
#SBATCH --output=results/024_posit_takum_strategy_performance/slurm-full-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from the repository root}"
if [[ ! -x "${repo_dir}/scripts/run_posit_takum_strategy_benchmark.sh" ]]; then
    echo "error: submit from the accessor_universal_test repository root" >&2
    exit 2
fi
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env MODE=full BUILD_DIR="${repo_dir}/build-h200" \
    RESULTS_ROOT="${repo_dir}/results/024_posit_takum_strategy_performance" \
    BUILD_JOBS=2 CUDA_ARCH=90 TMPDIR=/tmp TARGET_TIMEOUT=5400 \
    "${repo_dir}/scripts/run_posit_takum_strategy_benchmark.sh"
