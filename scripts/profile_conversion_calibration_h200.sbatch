#!/usr/bin/env bash
#SBATCH --job-name=conversion-profile
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia:1
#SBATCH --time=02:00:00
#SBATCH --output=results/028_conversion_cost_calibration/slurm-profile-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -eo pipefail
repo_dir="${SLURM_SUBMIT_DIR:?submit from repository checkout}"
source /etc/profile
set -u
module load "${CUDA_MODULE:-cuda/13.1.1}"
cd "${repo_dir}"
build_dir="${repo_dir}/build-conversion-calibration-$(git rev-parse --short=12 HEAD)-${SLURM_JOB_ID}"
cmake -S "${repo_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "${build_dir}" --parallel 2 --target conversion_calibration_bench
binary="${build_dir}/bin/conversion_calibration_bench"
profile_dir="${RUN_DIR:?set RUN_DIR}/profile"
[[ ! -e "${profile_dir}" ]] || { echo "refusing to overwrite profile" >&2; exit 2; }
mkdir "${profile_dir}"
ncu --query-metrics >"${profile_dir}/available_metrics.txt"
metrics="${NCU_METRICS:-smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active}"
ncu --target-processes all --kernel-name-base demangled --kernel-name 'regex:dot_case_kernel' \
  --metrics "${metrics}" --csv --log-file "${profile_dir}/ncu.csv" \
  "${binary}" --mode smoke --n 1048576 --resources "${profile_dir}/resources.csv" \
  --profile-cases "1,7,31,55,68,71,85,90,102,119,123,129,134"
