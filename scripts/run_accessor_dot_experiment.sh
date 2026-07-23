#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/003_accessor_dot}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
profile_dir="${run_dir}/profile"
started_epoch="$(date +%s)"

mkdir -p "${run_dir}" "${profile_dir}"

{
    echo "experiment=003_accessor_dot"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "git_status_begin"
    git -C "${repo_dir}" status --short
    echo "git_status_end"
    echo
    echo "nvidia_smi"
    nvidia-smi \
        --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,ecc.mode.current,clocks.max.sm,clocks.max.memory,power.limit \
        --format=csv
    echo
    echo "nvcc_version"
    nvcc --version
    if command -v ncu >/dev/null 2>&1; then
        echo
        echo "ncu_version"
        ncu --version
    fi
} >"${run_dir}/environment.txt"

{
    echo "performance_powers=${PERF_POWERS:-10,16,20,23,26,27}"
    echo "accuracy_powers=${ACCURACY_POWERS:-10,16,20}"
    echo "warmup=${WARMUP:-5}"
    echo "rounds=${ROUNDS:-3}"
    echo "samples=${SAMPLES:-20}"
    echo "target_sample_ms=${TARGET_SAMPLE_MS:-50}"
    echo "profile=${PROFILE:-1}"
    echo "profile_n=${PROFILE_N:-134217728}"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}"
ctest --test-dir "${build_dir}" --output-on-failure

benchmark="${build_dir}/bin/accessor_dot_bench"
performance_csv="${run_dir}/performance_samples.csv"
accuracy_csv="${run_dir}/accuracy.csv"

"${benchmark}" \
    --mode performance \
    --variant all \
    --powers "${PERF_POWERS:-10,16,20,23,26,27}" \
    --warmup "${WARMUP:-5}" \
    --rounds "${ROUNDS:-3}" \
    --samples "${SAMPLES:-20}" \
    --target-sample-ms "${TARGET_SAMPLE_MS:-50}" \
    --output "${performance_csv}"

"${benchmark}" \
    --mode accuracy \
    --variant all \
    --powers "${ACCURACY_POWERS:-10,16,20}" \
    --warmup "${WARMUP:-5}" \
    --output "${accuracy_csv}"

python3 "${repo_dir}/tools/summarize_accessor_results.py" \
    --performance "${performance_csv}" \
    --accuracy "${accuracy_csv}" \
    --output-dir "${run_dir}"

if [[ "${PROFILE:-1}" == "1" ]]; then
    if ! command -v ncu >/dev/null 2>&1; then
        echo "error: PROFILE=1 but Nsight Compute CLI (ncu) is unavailable" >&2
        exit 1
    fi

    metrics="dram__bytes_read.sum,dram__bytes_write.sum"
    metrics+=",smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
    metrics+=",smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
    metrics+=",smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
    metrics+=",smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"
    metrics+=",smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"
    metrics+=",smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"

    variants=(
        raw_f32
        raw_f64
        accessor_f32_f32
        accessor_f64_f32
    )
    for variant in "${variants[@]}"; do
        variant_dir="${profile_dir}/${variant}"
        mkdir -p "${variant_dir}"
        report_base="${variant_dir}/${variant}_n${PROFILE_N:-134217728}_${timestamp}"
        metadata_csv="${report_base}_metadata.csv"

        ncu \
            --profile-from-start off \
            --target-processes all \
            --section SpeedOfLight \
            --section SpeedOfLight_RooflineChart \
            --section MemoryWorkloadAnalysis \
            --section LaunchStats \
            --section Occupancy \
            --section SchedulerStats \
            --metrics "${metrics}" \
            --force-overwrite \
            --export "${report_base}" \
            "${benchmark}" \
                --mode profile \
                --variant "${variant}" \
                --dataset signed_uniform_seed0 \
                --n "${PROFILE_N:-134217728}" \
                --warmup "${WARMUP:-5}" \
                --output "${metadata_csv}"

        ncu --import "${report_base}.ncu-rep" --page details \
            --print-details all >"${report_base}_ncu_details.txt"
        ncu --import "${report_base}.ncu-rep" --page raw --csv \
            --print-units base >"${report_base}_ncu_raw.csv"
    done

    python3 "${repo_dir}/tools/summarize_ncu.py" \
        --profile-dir "${profile_dir}" \
        --output "${run_dir}/profile_summary.csv"

    if command -v cuobjdump >/dev/null 2>&1; then
        cuobjdump --dump-resource-usage "${benchmark}" \
            >"${run_dir}/cuda_resource_usage.txt"
        cuobjdump --dump-sass "${benchmark}" >"${run_dir}/sass.txt"
    fi
fi

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
} >>"${run_dir}/run_manifest.txt"

echo
echo "Experiment complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/performance_summary.csv"
echo "  ${run_dir}/accuracy_comparison.csv"
if [[ "${PROFILE:-1}" == "1" ]]; then
    echo "  ${run_dir}/profile_summary.csv"
fi
