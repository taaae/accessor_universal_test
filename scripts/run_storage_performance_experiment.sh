#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/008_storage_performance}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
profile_dir="${run_dir}/profile"
started_epoch="$(date +%s)"

mkdir -p "${run_dir}" "${profile_dir}"

{
    echo "experiment=008_storage_performance"
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
    echo "dot_powers=${DOT_POWERS:-12,16,20,24,27}"
    echo "gemv_powers=${GEMV_POWERS:-8,10,12,14,16}"
    echo "gemv_rows=${GEMV_ROWS:-1024}"
    echo "distributions=uniform_0_1,normal_0_1"
    echo "formats=all_17"
    echo "load_lanes=1,2,4"
    echo "warmup=${WARMUP:-10}"
    echo "rounds=${ROUNDS:-3}"
    echo "samples_per_round=${SAMPLES:-5}"
    echo "target_sample_ms=${TARGET_SAMPLE_MS:-15}"
    echo "decode_repeats=${DECODE_REPEATS:-256}"
    echo "base_seed=${BASE_SEED:-2611923443488327891}"
    echo "profile=${PROFILE:-1}"
    echo "profile_components=${PROFILE_COMPONENTS:-register_decode,stream_decode,dot,gemv}"
    echo "profile_lanes=${PROFILE_LANES:-1,4}"
    echo "profile_distribution=${PROFILE_DISTRIBUTION:-normal_0_1}"
    echo "profile_register_n=${PROFILE_REGISTER_N:-1048576}"
    echo "profile_stream_n=${PROFILE_STREAM_N:-134217728}"
    echo "profile_dot_n=${PROFILE_DOT_N:-134217728}"
    echo "profile_gemv_m=${PROFILE_GEMV_M:-1024}"
    echo "profile_gemv_n=${PROFILE_GEMV_N:-65536}"
    echo "timing_note=CUDA_event_timings_only;Nsight_replay_timings_are_separate"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target storage_performance_bench memory_accessor_test \
        storage_formats_test accuracy_statistics_test
ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

benchmark="${build_dir}/bin/storage_performance_bench"
samples_csv="${run_dir}/timing_samples.csv"

"${benchmark}" \
    --mode sweep \
    --dot-powers "${DOT_POWERS:-12,16,20,24,27}" \
    --gemv-powers "${GEMV_POWERS:-8,10,12,14,16}" \
    --gemv-rows "${GEMV_ROWS:-1024}" \
    --warmup "${WARMUP:-10}" \
    --rounds "${ROUNDS:-3}" \
    --samples "${SAMPLES:-5}" \
    --target-sample-ms "${TARGET_SAMPLE_MS:-15}" \
    --decode-repeats "${DECODE_REPEATS:-256}" \
    --base-seed "${BASE_SEED:-2611923443488327891}" \
    --output "${samples_csv}" \
    | tee "${run_dir}/timing_stdout.txt"

python3 "${repo_dir}/tools/summarize_storage_performance.py" \
    --samples "${samples_csv}" \
    --output-dir "${run_dir}" \
    | tee "${run_dir}/timing_validation.txt"

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

    IFS=',' read -r -a components <<<"${PROFILE_COMPONENTS:-register_decode,stream_decode,dot,gemv}"
    IFS=',' read -r -a lanes_list <<<"${PROFILE_LANES:-1,4}"
    for component in "${components[@]}"; do
        case "${component}" in
            register_decode)
                profile_n="${PROFILE_REGISTER_N:-1048576}"
                profile_m=1
                profile_repeats="${PROFILE_DECODE_REPEATS:-64}"
                ;;
            stream_load|stream_decode)
                profile_n="${PROFILE_STREAM_N:-134217728}"
                profile_m=1
                profile_repeats="${DECODE_REPEATS:-256}"
                ;;
            dot)
                profile_n="${PROFILE_DOT_N:-134217728}"
                profile_m=1
                profile_repeats="${DECODE_REPEATS:-256}"
                ;;
            gemv)
                profile_n="${PROFILE_GEMV_N:-65536}"
                profile_m="${PROFILE_GEMV_M:-1024}"
                profile_repeats="${DECODE_REPEATS:-256}"
                ;;
            *)
                echo "error: unknown profile component ${component}" >&2
                exit 2
                ;;
        esac
        for lanes in "${lanes_list[@]}"; do
            case_dir="${profile_dir}/${component}_x${lanes}"
            mkdir -p "${case_dir}"
            report_base="${case_dir}/${component}_x${lanes}_${timestamp}"
            metadata_csv="${report_base}_metadata.csv"

            ncu \
                --profile-from-start off \
                --target-processes all \
                --kernel-name-base demangled \
                --section SpeedOfLight \
                --section SpeedOfLight_RooflineChart \
                --section MemoryWorkloadAnalysis \
                --section ComputeWorkloadAnalysis \
                --section InstructionStats \
                --section LaunchStats \
                --section Occupancy \
                --section SchedulerStats \
                --metrics "${metrics}" \
                --force-overwrite \
                --export "${report_base}" \
                "${benchmark}" \
                    --mode profile \
                    --component "${component}" \
                    --format all \
                    --lanes "${lanes}" \
                    --n "${profile_n}" \
                    --m "${profile_m}" \
                    --distribution "${PROFILE_DISTRIBUTION:-normal_0_1}" \
                    --warmup "${WARMUP:-10}" \
                    --decode-repeats "${profile_repeats}" \
                    --base-seed "${BASE_SEED:-2611923443488327891}" \
                    --output "${metadata_csv}" \
                | tee "${report_base}_stdout.txt"

            ncu --import "${report_base}.ncu-rep" --page details \
                --print-details all >"${report_base}_ncu_details.txt"
            ncu --import "${report_base}.ncu-rep" --page raw --csv \
                --print-units base >"${report_base}_ncu_raw.csv"
        done
    done

    python3 "${repo_dir}/tools/summarize_storage_performance.py" \
        --samples "${samples_csv}" \
        --profile-dir "${profile_dir}" \
        --output-dir "${run_dir}" \
        | tee "${run_dir}/profile_validation.txt"

    if command -v cuobjdump >/dev/null 2>&1; then
        cuobjdump --dump-resource-usage "${benchmark}" \
            >"${run_dir}/cuda_resource_usage.txt"
    fi
fi

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
} >>"${run_dir}/run_manifest.txt"

echo
echo "Storage performance experiment complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/timing_summary.csv"
echo "  ${run_dir}/packed_speedups.csv"
if [[ "${PROFILE:-1}" == "1" ]]; then
    echo "  ${run_dir}/profile_operations.csv"
    echo "  ${run_dir}/profile_kernels.csv"
fi
