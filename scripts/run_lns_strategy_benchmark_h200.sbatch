#!/usr/bin/env bash
#SBATCH --job-name=lns-strategies
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=08:00:00
#SBATCH --output=results/022_lns_strategy_performance/slurm-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
if [[ ! -f "${repo_dir}/CMakeLists.txt" ||
      ! -x "${repo_dir}/scripts/run_lns_strategy_benchmark.sh" ]]; then
    echo "error: submit from the accessor_universal_test repository root" >&2
    exit 2
fi

source /etc/profile
set -u
cuda_module="${CUDA_MODULE:-cuda/13.1.1}"
if ! type module >/dev/null 2>&1; then
    echo "error: environment-modules is unavailable" >&2
    exit 2
fi
module load "${cuda_module}"

cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env \
        BUILD_DIR="${repo_dir}/build-h200" \
        RESULTS_ROOT="${repo_dir}/results/022_lns_strategy_performance" \
        BUILD_JOBS=2 \
        CUDA_ARCH=90 \
        TMPDIR=/tmp \
        "${repo_dir}/scripts/run_lns_strategy_benchmark.sh"
