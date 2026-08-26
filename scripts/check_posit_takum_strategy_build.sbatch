#!/usr/bin/env bash
#SBATCH --job-name=posit-takum-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/024_posit_takum_strategy_performance/slurm-build-%j.out

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from the repository root}"
if [[ ! -f "${repo_dir}/CMakeLists.txt" ]]; then
    echo "error: submit from the accessor_universal_test repository root" >&2
    exit 2
fi
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
export TMPDIR=/tmp

build_dir="${repo_dir}/build-h200"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "${build_dir}" --parallel 4 --target posit_takum_core_test
ctest --test-dir "${build_dir}" --output-on-failure -R '^posit_takum_core_test$'
targets=(
    posit8_es0 posit14_es1 posit16_es1 posit32_es2
    takum8 takum14 takum16 takum32
    takum_log8 takum_log14 takum_log16 takum_log32
)
for target in "${targets[@]}"; do
    cmake --build "${build_dir}" --parallel 1 \
        --target "posit_takum_strategy_bench_${target}"
done
echo "All scalar posit/takum CUDA targets compiled successfully."
