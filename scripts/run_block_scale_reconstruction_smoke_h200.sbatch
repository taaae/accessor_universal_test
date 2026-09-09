#!/usr/bin/env bash
#SBATCH --job-name=bscale-r34-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=00:15:00
#SBATCH --output=results/034_block_scale_reconstruction/slurm-smoke-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp
set -eo pipefail
source /etc/profile
set -u
module load cuda/13.1.1
repo="${SLURM_SUBMIT_DIR:?}"
cd "$repo"
node="$(hostname)"
node_short="${node%%.*}"
case "$node_short" in gpu-nvidia-h200-2|gpu-nvidia-h200-3);;*) echo "ineligible node $node";exit 3;;esac
build="${PREFLIGHT_BUILD:?}"
evidence="${PREFLIGHT_EVIDENCE:?}"
bin="$build/bin/block_scale_reconstruction_bench"
test -x "$bin"
commit="$(git rev-parse HEAD)"
test "$(cat "$evidence/commit.txt")" = "$commit"
binary="$(cut -d' ' -f1 "$evidence/binary.sha256")"
sass="$(cut -d' ' -f1 "$evidence/sass.sha256")"
audit="$(cut -d' ' -f1 "$evidence/audit.sha256")"
test "$(sha256sum "$bin"|cut -d' ' -f1)" = "$binary"
sha256sum -c "$evidence/sass.sha256"
sha256sum -c "$evidence/audit.sha256"
run="results/034_block_scale_reconstruction/run_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}"
mkdir -p "$run/smoke"
git rev-parse HEAD > "$run/commit.txt"
sha256sum "$bin" > "$run/binary.sha256"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free,clocks.current.sm,power.draw --format=csv > "$run/environment.txt"
environment_sha="$(sha256sum "$run/environment.txt"|cut -d' ' -f1)"
printf '{"mode":"smoke","job_id":"%s","node":"%s","sizes":[16384,1048576],"variants":14,"warmups":1,"samples":3,"grid":512,"threads":256,"payload":"fp32","dot_accumulation":"fp64_except_raw_fp32","access":"scalar_x1","cache_protocol":"repeat_input_no_explicit_flush","experiment_id":"034_block_scale_reconstruction","dataset_id":"block-scale-v1","seed_set_id":"block-scale-v1","rounding_reference_cases":238,"commit":"%s","binary_sha256":"%s","sass_sha256":"%s","audit_sha256":"%s","environment_sha256":"%s"}\n' "$SLURM_JOB_ID" "$node" "$commit" "$binary" "$sass" "$audit" "$environment_sha" > "$run/manifest.json"
timeout 4m "$bin" --mode smoke --output "$run/smoke/timing_samples.csv" --correctness-output "$run/smoke/correctness_checks.txt" --source-commit "$commit" --binary-sha "$binary" --sass-sha "$sass" --audit-sha "$audit" | tee "$run/smoke/stdout.txt"
python3 tools/validate_block_scale_reconstruction_results.py --samples "$run/smoke/timing_samples.csv" --manifest "$run/manifest.json" --mode smoke | tee "$run/smoke/validation.txt"
timeout 4m compute-sanitizer --tool memcheck --error-exitcode=9 "$bin" --mode validate --correctness-output "$run/smoke/memcheck_checks.txt" > "$run/smoke/memcheck.txt" 2>&1
timeout 4m compute-sanitizer --tool synccheck --error-exitcode=9 "$bin" --mode validate --correctness-output "$run/smoke/synccheck_checks.txt" > "$run/smoke/synccheck.txt" 2>&1
grep -q 'ERROR SUMMARY: 0 errors' "$run/smoke/memcheck.txt"
grep -q 'ERROR SUMMARY: 0 errors' "$run/smoke/synccheck.txt"
grep -q 'validated_case_count=238' "$run/smoke/memcheck_checks.txt"
grep -q 'validated_case_count=238' "$run/smoke/synccheck_checks.txt"
echo SMOKE_COMPLETE="$run"
