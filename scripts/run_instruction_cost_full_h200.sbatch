#!/usr/bin/env bash
#SBATCH --job-name=icost-full
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=00:30:00
#SBATCH --output=results/032_instruction_cost_sweep/slurm-full-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp
set -eo pipefail;source /etc/profile;set -u;module load cuda/13.1.1
repo="${SLURM_SUBMIT_DIR:?}";cd "$repo";node="${HOSTNAME%%.*}";case "$node" in gpu-nvidia-h200-2|gpu-nvidia-h200-3);;*)exit 3;;esac
build="${PREFLIGHT_BUILD:?set PREFLIGHT_BUILD}";bin="$build/bin/instruction_cost_bench";test -x "$bin";run="results/032_instruction_cost_sweep/run_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}";mkdir -p "$run/full"
git rev-parse HEAD > "$run/commit.txt";sha256sum "$bin" > "$run/binary.sha256";nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$run/environment.txt"
printf '{"job_id":"%s","node":"%s","dot_n":67108864,"gemv_rows":4096,"gemv_cols":16384,"warmups":10,"samples":50,"access":"scalar_x1","left_seed":"0x6bd87c012a53f9e1","right_seed":"0xf5ef05b8551985f4"}\n' "$SLURM_JOB_ID" "$node" > "$run/manifest.json"
timeout 20m "$bin" --mode full --output "$run/full/timing_samples.csv" --correctness-output "$run/full/correctness_checks.txt" | tee "$run/full/stdout.txt"
test "$(wc -l < "$run/full/timing_samples.csv")" -ge 3601;python3 tools/analyze_instruction_cost_sweep.py --samples "$run/full/timing_samples.csv" --output-dir "$run/full" --manifest "$run/manifest.json" | tee "$run/full/analysis.txt";echo FULL_COMPLETE="$run"
