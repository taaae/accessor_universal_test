#!/usr/bin/env bash
#SBATCH --job-name=conversion-build
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/028_conversion_cost_calibration/slurm-build-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from repository root}"
[[ -x "${repo_dir}/scripts/run_conversion_calibration.sh" ]] || { echo "wrong submit directory" >&2; exit 2; }
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cd "${repo_dir}"
commit_short="$(git rev-parse --short=12 HEAD)"
build_dir="${repo_dir}/build-conversion-calibration-${commit_short}-${SLURM_JOB_ID}"
python3 -m unittest discover -s analysis/tests -p 'test_conversion_calibration*.py' -v
cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "${build_dir}" --parallel 2 --target conversion_calibration_bench 2>&1 | tee "results/028_conversion_cost_calibration/build-${SLURM_JOB_ID}.txt"
cuobjdump --dump-sass "${build_dir}/bin/conversion_calibration_bench" >"results/028_conversion_cost_calibration/build-${SLURM_JOB_ID}.sass"
python3 tools/extract_conversion_calibration_sass.py \
  "results/028_conversion_cost_calibration/build-${SLURM_JOB_ID}.sass" \
  "results/028_conversion_cost_calibration/build-${SLURM_JOB_ID}-features.csv" \
  --warnings "results/028_conversion_cost_calibration/build-${SLURM_JOB_ID}-warnings.txt"
echo "conversion calibration compile/SASS gate passed"
