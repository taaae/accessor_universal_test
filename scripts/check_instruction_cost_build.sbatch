#!/usr/bin/env bash
#SBATCH --job-name=icost-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/032_instruction_cost_sweep/slurm-build-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp
set -eo pipefail
source /etc/profile
set -u
module load cuda/13.1.1
repo="${SLURM_SUBMIT_DIR:?submit from repo root}";cd "$repo";test -f CMakeLists.txt
test -z "$(git status --porcelain --untracked-files=no)"
build="$repo/build-instruction-cost-${SLURM_JOB_ID}";evidence="$repo/results/032_instruction_cost_sweep/preflight_${SLURM_JOB_ID}";mkdir -p "$evidence"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$build" --parallel 8 --target instruction_cost_core_test instruction_cost_bench 2>&1 | tee "$evidence/build.txt"
"$build/bin/instruction_cost_core_test" | tee "$evidence/host_test.txt"
cuobjdump --dump-sass "$build/bin/instruction_cost_bench" > "$evidence/instruction_cost.sass.txt"
python3 tools/check_instruction_cost_codegen.py --sass "$evidence/instruction_cost.sass.txt" --build-log "$evidence/build.txt" --output-dir "$evidence"
sha256sum "$build/bin/instruction_cost_bench" | tee "$evidence/binary.sha256";git rev-parse HEAD | tee "$evidence/commit.txt";echo PREFLIGHT_COMPLETE
sha256sum "$evidence/instruction_cost.sass.txt" | tee "$evidence/sass.sha256"
sha256sum "$evidence/assembly_audit.json" | tee "$evidence/audit.sha256"
