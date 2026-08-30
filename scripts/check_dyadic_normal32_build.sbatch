#!/usr/bin/env bash
#SBATCH --job-name=dn32-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=results/030_dyadic_normal32_dispersion/slurm-build-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from the repository root}"
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cmake -S "${repo_dir}" -B "${repo_dir}/build-dyadic-normal32-check" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "${repo_dir}/build-dyadic-normal32-check" --parallel 4 \
    --target dyadic_normal32_core_test dyadic_normal32_bench
ctest --test-dir "${repo_dir}/build-dyadic-normal32-check" \
    --output-on-failure -R '^dyadic_normal32_core_test$'
