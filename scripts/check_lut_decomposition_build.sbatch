#!/usr/bin/env bash
#SBATCH --job-name=lut-decomp-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/029_lut_decomposition_shapes/slurm-build-%j.out
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
commit_short="$(git rev-parse --short=12 HEAD)"
build_dir="${repo_dir}/build-h200-lut-decomposition-${commit_short}"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "${build_dir}" --parallel 4 \
    --target lut_decomposition_core_test lut_decomposition_bench
ctest --test-dir "${build_dir}" --output-on-failure \
    -R '^lut_decomposition_core_test$'
