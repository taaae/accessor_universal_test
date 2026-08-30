#!/usr/bin/env bash
#SBATCH --job-name=lut-dist-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/027_lut_distribution_shape/slurm-build-%j.out

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from the repository root}"
if [[ ! -x "${repo_dir}/scripts/run_lut_distribution_shape.sh" ]]; then
    echo "error: submit from the accessor_universal_test repository root" >&2
    exit 2
fi
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cd "${repo_dir}"
lock_file="$(dirname "${repo_dir}")/.accessor-universal-lut-distribution-${USER:-unknown}.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "error: another LUT-distribution build or runner holds ${lock_file}" >&2
    exit 3
fi
commit_short="$(git -C "${repo_dir}" rev-parse --short=12 HEAD)"
build_dir="${repo_dir}/build-h200-lut-distribution-${commit_short}"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "${build_dir}" --parallel 4 \
    --target lut_distribution_core_test lut_distribution_bench
ctest --test-dir "${build_dir}" \
    --output-on-failure -R '^lut_distribution_core_test$'
python3 -m py_compile "${repo_dir}/tools/analyze_lut_distribution_shape.py"
python3 "${repo_dir}/analysis/tests/test_lut_distribution_shape_analysis.py"
