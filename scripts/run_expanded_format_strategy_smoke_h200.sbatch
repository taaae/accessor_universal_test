#!/usr/bin/env bash
#SBATCH --job-name=expanded-format-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=01:00:00
#SBATCH --output=results/016_expanded_format_strategy_smoke/slurm-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
if [[ ! -f "${repo_dir}/CMakeLists.txt" ||
      ! -x "${repo_dir}/scripts/run_expanded_format_strategy_smoke.sh" ]]; then
    echo "error: submit from the accessor_universal_test repository root" >&2
    exit 2
fi

cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env \
        BUILD_DIR="${repo_dir}/build-h200" \
        RESULTS_ROOT="${repo_dir}/results/016_expanded_format_strategy_smoke" \
        BUILD_JOBS=2 \
        CUDA_ARCH=90 \
        TMPDIR=/tmp \
        "${repo_dir}/scripts/run_expanded_format_strategy_smoke.sh"
