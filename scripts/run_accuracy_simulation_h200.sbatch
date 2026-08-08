#!/usr/bin/env bash
#SBATCH --job-name=accuracy-sim
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=02:00:00
#SBATCH --output=results/007_gpu_accuracy_simulation/slurm-%j.out

set -euo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"

if [[ ! -f "${repo_dir}/CMakeLists.txt" ||
      ! -x "${repo_dir}/scripts/run_accuracy_simulation_experiment.sh" ]]; then
    echo "error: submit this job from the accessor_universal_test repository root" >&2
    exit 2
fi

cd "${repo_dir}"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    env \
        BUILD_DIR="${repo_dir}/build-h200" \
        RESULTS_ROOT="${repo_dir}/results/007_gpu_accuracy_simulation" \
        BUILD_JOBS="${SLURM_CPUS_PER_TASK}" \
        CUDA_ARCH=90 \
        DOT_POWERS="${DOT_POWERS:-10,14,18,22}" \
        DOT_SAMPLES="${DOT_SAMPLES:-8192}" \
        DOT_STATISTICAL_BATCHES="${DOT_STATISTICAL_BATCHES:-32}" \
        GEMV_POWERS="${GEMV_POWERS:-8,10,12,14,16}" \
        GEMV_ROWS="${GEMV_ROWS:-1024}" \
        GEMV_REPLICATES="${GEMV_REPLICATES:-16}" \
        BASE_SEED="${BASE_SEED:-2611923443488327891}" \
        WORKSPACE_GIB="${WORKSPACE_GIB:-12}" \
        "${repo_dir}/scripts/run_accuracy_simulation_experiment.sh"
