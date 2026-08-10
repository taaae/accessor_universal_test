#!/usr/bin/env bash
#SBATCH --job-name=precision-packing-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/018_precision_packing_bottlenecks/slurm-build-%j.out

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

formats=(fp16 bf16 fp32 fp8_e4m3 fp8_e5m2 fp4_e2m1 fp64)
for format in "${formats[@]}"; do
    target="precision_packing_bench_${format}"
    echo "Building ${target}"
    cmake --build "${build_dir}" --parallel 1 --target "${target}"
done

echo "All precision-packing CUDA targets compiled successfully."
