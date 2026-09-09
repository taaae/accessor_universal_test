#!/usr/bin/env bash
#SBATCH --job-name=icost-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=00:15:00
#SBATCH --output=results/032_instruction_cost_sweep/slurm-smoke-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp
set -eo pipefail;source /etc/profile;set -u;module load cuda/13.1.1
repo="${SLURM_SUBMIT_DIR:?}";cd "$repo";node="${HOSTNAME%%.*}";case "$node" in gpu-nvidia-h200-2|gpu-nvidia-h200-3);;*)exit 3;;esac
build="${PREFLIGHT_BUILD:?set PREFLIGHT_BUILD}";bin="$build/bin/instruction_cost_bench";test -x "$bin";run="results/032_instruction_cost_sweep/run_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}";mkdir -p "$run/smoke"
sha256sum "$bin" | tee "$run/binary.sha256";timeout 4m "$bin" --mode smoke --output "$run/smoke/timing_samples.csv" --correctness-output "$run/smoke/correctness_checks.txt" | tee "$run/smoke/stdout.txt"
timeout 4m compute-sanitizer --tool memcheck --error-exitcode=9 "$bin" --mode validate --correctness-output "$run/smoke/memcheck_checks.txt" > "$run/smoke/memcheck.txt" 2>&1
timeout 4m compute-sanitizer --tool synccheck --error-exitcode=9 "$bin" --mode validate --correctness-output "$run/smoke/synccheck_checks.txt" > "$run/smoke/synccheck.txt" 2>&1
test "$(wc -l < "$run/smoke/timing_samples.csv")" -eq 217;grep -q all_passed=1 "$run/smoke/correctness_checks.txt";echo SMOKE_COMPLETE
