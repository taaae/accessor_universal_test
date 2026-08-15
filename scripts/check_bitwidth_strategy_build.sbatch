#!/usr/bin/env bash
#SBATCH --job-name=bitwidth-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/020_bitwidth_strategy_performance/slurm-build-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail

repo_dir="${SLURM_SUBMIT_DIR:?submit this job from the repository root}"
if [[ ! -f "${repo_dir}/CMakeLists.txt" ]]; then
    echo "error: submit from the accessor_universal_test repository root" >&2
    exit 2
fi

source /etc/profile
set -u
cuda_module="${CUDA_MODULE:-cuda/13.1.1}"
if ! type module >/dev/null 2>&1; then
    echo "error: environment-modules is unavailable after sourcing /etc/profile" >&2
    exit 2
fi
module load "${cuda_module}"

build_dir="${repo_dir}/build-h200"
cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90

cmake --build "${build_dir}" --parallel 2 \
    --target bitwidth_benchmark_core_test storage_formats_test \
        decoder_strategy_core_test

ctest --test-dir "${build_dir}" --output-on-failure \
    -R 'bitwidth_benchmark_core_test|storage_formats_test|decoder_strategy_core_test'

widths=(2 3 4 5 6 7 9 10 12 14 17 20 24 28)
for width in "${widths[@]}"; do
    target="bitwidth_strategy_bench_${width}"
    echo "Building ${target}"
    cmake --build "${build_dir}" --parallel 2 --target "${target}"
done

echo "All arbitrary-width CUDA targets compiled successfully."
