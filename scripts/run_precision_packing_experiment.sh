#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/018_precision_packing_bottlenecks}"
build_jobs="${BUILD_JOBS:-2}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
validation_dir="${run_dir}/validation"
profile_dir="${run_dir}/profile"
started_epoch="$(date +%s)"

formats=(fp16 bf16 fp32 fp8_e4m3 fp8_e5m2 fp4_e2m1 fp64)
targets=()
for format in "${formats[@]}"; do
    targets+=("precision_packing_bench_${format}")
done

arithmetic_for_format() {
    case "$1" in
        fp16) printf '%s\n' fp16 fp64 ;;
        bf16) printf '%s\n' bf16 fp64 ;;
        fp32) printf '%s\n' fp32 fp64 ;;
        fp8_e4m3|fp8_e5m2|fp4_e2m1) printf '%s\n' fp16 fp32 fp64 ;;
        fp64) printf '%s\n' fp64 ;;
        *) echo "error: unknown format $1" >&2; return 2 ;;
    esac
}

merge_csv() {
    local output="$1"
    shift
    awk 'FNR == 1 { if (!header++) print; next } { print }' "$@" >"${output}"
}

mkdir -p "${run_dir}" "${validation_dir}" "${profile_dir}"

{
    echo "experiment=018_precision_packing_bottlenecks"
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
    echo "formats=${formats[*]}"
    echo "arithmetic_fp16=fp16,fp64"
    echo "arithmetic_bf16=bf16,fp64"
    echo "arithmetic_fp32=fp32,fp64"
    echo "arithmetic_fp8=fp16,fp32,fp64"
    echo "arithmetic_fp4=fp16,fp32,fp64"
    echo "arithmetic_fp64=fp64"
    echo "families=scalar_single,scalar_unrolled,vector_packet,packed_arithmetic_where_native"
    echo "lanes=1,2,4,8"
    echo "dot_powers=${DOT_POWERS:-12,16,20,24,27}"
    echo "gemv_powers=${GEMV_POWERS:-8,10,12,14,16}"
    echo "gemv_rows=${GEMV_ROWS:-1024}"
    echo "distributions=uniform_0_1,normal_0_1"
    echo "warmup=${WARMUP:-10}"
    echo "rounds=${ROUNDS:-3}"
    echo "samples_per_round=${SAMPLES:-5}"
    echo "target_sample_ms=${TARGET_SAMPLE_MS:-15}"
    echo "component_n=${COMPONENT_N:-134217728}"
    echo "register_n=${REGISTER_N:-1048576}"
    echo "decode_repeats=${DECODE_REPEATS:-256}"
    echo "arithmetic_repeats=${ARITHMETIC_REPEATS:-4096}"
    echo "base_seed=${BASE_SEED:-2611923443488327891}"
    echo "profile=${PROFILE:-1}"
    echo "profile_distribution=${PROFILE_DISTRIBUTION:-normal_0_1}"
    echo "profile_dot_n=${PROFILE_DOT_N:-134217728}"
    echo "profile_gemv_m=${PROFILE_GEMV_M:-1024}"
    echo "profile_gemv_n=${PROFILE_GEMV_N:-65536}"
    echo "profile_variants=all_control_families_and_widths"
    echo "timing=CUDA_events;profiler_replay_times_excluded"
    echo "component_floors=overlapping_bounds_do_not_sum"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target memory_accessor_test storage_formats_test \
        decoder_strategy_core_test accuracy_statistics_test \
    | tee "${run_dir}/build_host_tests.txt"
for target in "${targets[@]}"; do
    cmake --build "${build_dir}" --parallel 1 --target "${target}" \
        | tee "${run_dir}/build_${target}.txt"
done
ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

validation_files=()
for format in "${formats[@]}"; do
    benchmark="${build_dir}/bin/precision_packing_bench_${format}"
    output="${validation_dir}/validation_${format}.csv"
    "${benchmark}" \
        --mode validate \
        --base-seed "${BASE_SEED:-2611923443488327891}" \
        --output "${output}" \
        | tee "${validation_dir}/validation_${format}_stdout.txt"
    validation_files+=("${output}")
done
validation_csv="${validation_dir}/validation_samples.csv"
merge_csv "${validation_csv}" "${validation_files[@]}"

timing_files=()
for format in "${formats[@]}"; do
    benchmark="${build_dir}/bin/precision_packing_bench_${format}"
    output="${run_dir}/timing_${format}.csv"
    "${benchmark}" \
        --mode sweep \
        --dot-powers "${DOT_POWERS:-12,16,20,24,27}" \
        --gemv-powers "${GEMV_POWERS:-8,10,12,14,16}" \
        --gemv-rows "${GEMV_ROWS:-1024}" \
        --warmup "${WARMUP:-10}" \
        --rounds "${ROUNDS:-3}" \
        --samples "${SAMPLES:-5}" \
        --target-sample-ms "${TARGET_SAMPLE_MS:-15}" \
        --base-seed "${BASE_SEED:-2611923443488327891}" \
        --output "${output}" \
        | tee "${run_dir}/timing_${format}_stdout.txt"
    timing_files+=("${output}")
done
timing_csv="${run_dir}/timing_samples.csv"
merge_csv "${timing_csv}" "${timing_files[@]}"

component_files=()
for format in "${formats[@]}"; do
    benchmark="${build_dir}/bin/precision_packing_bench_${format}"
    output="${run_dir}/components_${format}.csv"
    "${benchmark}" \
        --mode components \
        --component-n "${COMPONENT_N:-134217728}" \
        --register-n "${REGISTER_N:-1048576}" \
        --decode-repeats "${DECODE_REPEATS:-256}" \
        --arithmetic-repeats "${ARITHMETIC_REPEATS:-4096}" \
        --warmup "${WARMUP:-10}" \
        --rounds "${ROUNDS:-3}" \
        --samples "${SAMPLES:-5}" \
        --target-sample-ms "${TARGET_SAMPLE_MS:-15}" \
        --base-seed "${BASE_SEED:-2611923443488327891}" \
        --output "${output}" \
        | tee "${run_dir}/components_${format}_stdout.txt"
    component_files+=("${output}")
done
component_csv="${run_dir}/component_samples.csv"
merge_csv "${component_csv}" "${component_files[@]}"

python3 "${repo_dir}/tools/summarize_precision_packing.py" \
    --timing "${timing_csv}" \
    --components "${component_csv}" \
    --validation "${validation_csv}" \
    --output-dir "${run_dir}" \
    --expected-rounds "${ROUNDS:-3}" \
    --expected-samples "${SAMPLES:-5}" \
    | tee "${run_dir}/summary_validation.txt"

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

    profile_manifest="${profile_dir}/profile_manifest.csv"
    echo "format,arithmetic,kernel,report_base,report_bytes,report_status" \
        >"${profile_manifest}"
    oversize_dir="${build_dir}/oversize_ncu_reports/${timestamp}"
    for format in "${formats[@]}"; do
        benchmark="${build_dir}/bin/precision_packing_bench_${format}"
        while IFS= read -r arithmetic; do
            for kernel in dot gemv; do
                case_dir="${profile_dir}/${format}/${arithmetic}/${kernel}"
                mkdir -p "${case_dir}"
                report_base="${case_dir}/${format}_${arithmetic}_${kernel}_${timestamp}"
                metadata_csv="${report_base}_metadata.csv"
                if [[ "${kernel}" == "dot" ]]; then
                    profile_n="${PROFILE_DOT_N:-134217728}"
                    profile_m=1
                else
                    profile_n="${PROFILE_GEMV_N:-65536}"
                    profile_m="${PROFILE_GEMV_M:-1024}"
                fi
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
                    --section WarpStateStats \
                    --metrics "${metrics}" \
                    --force-overwrite \
                    --export "${report_base}" \
                    "${benchmark}" \
                        --mode profile \
                        --kernel "${kernel}" \
                        --arithmetic "${arithmetic}" \
                        --family all \
                        --lanes 8 \
                        --n "${profile_n}" \
                        --m "${profile_m}" \
                        --distribution "${PROFILE_DISTRIBUTION:-normal_0_1}" \
                        --warmup "${WARMUP:-10}" \
                        --base-seed "${BASE_SEED:-2611923443488327891}" \
                        --output "${metadata_csv}" \
                    | tee "${report_base}_stdout.txt"

                ncu --import "${report_base}.ncu-rep" --page details \
                    --print-details all >"${report_base}_ncu_details.txt"
                ncu --import "${report_base}.ncu-rep" --page raw --csv \
                    --print-units base >"${report_base}_ncu_raw.csv"
                report_bytes="$(wc -c <"${report_base}.ncu-rep")"
                report_status=tracked
                if (( report_bytes > 95000000 )); then
                    mkdir -p "${oversize_dir}"
                    mv "${report_base}.ncu-rep" "${oversize_dir}/"
                    report_status="moved_to_${oversize_dir}"
                fi
                printf '%s,%s,%s,%s,%s,%s\n' \
                    "${format}" "${arithmetic}" "${kernel}" \
                    "${report_base#${run_dir}/}" "${report_bytes}" \
                    "${report_status}" >>"${profile_manifest}"
            done
        done < <(arithmetic_for_format "${format}")
    done

    python3 "${repo_dir}/tools/summarize_precision_packing_ncu.py" \
        --profile-dir "${profile_dir}" \
        --output-dir "${run_dir}" \
        | tee "${run_dir}/profile_validation.txt"
fi

if command -v cuobjdump >/dev/null 2>&1; then
    for format in "${formats[@]}"; do
        cuobjdump --dump-resource-usage \
            "${build_dir}/bin/precision_packing_bench_${format}" \
            >"${run_dir}/cuda_resource_usage_${format}.txt"
    done
fi

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
} >>"${run_dir}/run_manifest.txt"

echo
echo "Precision-packing bottleneck experiment complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/timing_summary.csv"
echo "  ${run_dir}/component_summary.csv"
echo "  ${run_dir}/roof_metrics.csv"
echo "  ${run_dir}/resource_floors.csv"
if [[ "${PROFILE:-1}" == "1" ]]; then
    echo "  ${run_dir}/profile_operations.csv"
fi
