#!/usr/bin/env bash
#SBATCH --job-name=bscale-r34-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/034_block_scale_reconstruction/slurm-build-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp
set -eo pipefail
source /etc/profile
set -u
module load cuda/13.1.1
repo="${SLURM_SUBMIT_DIR:?submit from repository root}"
cd "$repo"
test -f CMakeLists.txt
test -z "$(git status --porcelain --untracked-files=no)"
build="$repo/build-block-scale-reconstruction-${SLURM_JOB_ID}"
evidence="$repo/results/034_block_scale_reconstruction/preflight_${SLURM_JOB_ID}"
mkdir -p "$evidence"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$build" --parallel 8 --target block_scale_reconstruction_core_test block_scale_reconstruction_bench 2>&1 | tee "$evidence/build.txt"
"$build/bin/block_scale_reconstruction_core_test" | tee "$evidence/host_test.txt"
cuobjdump --dump-sass "$build/bin/block_scale_reconstruction_bench" > "$evidence/block_scale_reconstruction.sass.txt"
python3 tools/check_block_scale_reconstruction_codegen.py --sass "$evidence/block_scale_reconstruction.sass.txt" --build-log "$evidence/build.txt" --output-dir "$evidence"
git rev-parse HEAD | tee "$evidence/commit.txt"
sha256sum "$build/bin/block_scale_reconstruction_bench" | tee "$evidence/binary.sha256"
sha256sum "$evidence/block_scale_reconstruction.sass.txt" | tee "$evidence/sass.sha256"
sha256sum "$evidence/assembly_audit.json" | tee "$evidence/audit.sha256"
echo PREFLIGHT_COMPLETE
