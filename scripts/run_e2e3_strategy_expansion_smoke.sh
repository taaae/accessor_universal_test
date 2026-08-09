#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_dir}/build-h200}"
results_root="${RESULTS_ROOT:-${repo_dir}/results/012_e2e3_strategy_expansion_smoke}"
build_jobs="${BUILD_JOBS:-4}"
cuda_arch="${CUDA_ARCH:-90}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${RUN_DIR:-${results_root}/run_${timestamp}}"
started_epoch="$(date +%s)"

mkdir -p "${run_dir}"

{
    echo "experiment=012_e2e3_strategy_expansion_smoke"
    echo "purpose=exhaustive decoder and small DOT/GEMV validation"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "git_status_begin"
    git -C "${repo_dir}" status --short
    echo "git_status_end"
    echo
    echo "nvidia_smi"
    nvidia-smi \
        --query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,ecc.mode.current \
        --format=csv
    echo
    echo "nvcc_version"
    nvcc --version
} >"${run_dir}/environment.txt"

{
    echo "dot_powers=${DOT_POWERS:-16}"
    echo "gemv_powers=${GEMV_POWERS:-10}"
    echo "gemv_rows=${GEMV_ROWS:-256}"
    echo "distributions=uniform_0_1,normal_0_1"
    echo "formats=e2m5,e3m4"
    echo "strategies_per_format=42"
    echo "new_strategies_per_format=15"
    echo "load_widths=x1,x2,x4,x8"
    echo "warmup=${WARMUP:-2}"
    echo "samples=${SAMPLES:-2}"
    echo "target_sample_ms=${TARGET_SAMPLE_MS:-1}"
    echo "base_seed=${BASE_SEED:-2611923443488327891}"
    echo "timing=smoke_only_not_for_performance_conclusions"
} >"${run_dir}/run_manifest.txt"

cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}"
cmake --build "${build_dir}" --parallel "${build_jobs}" \
    --target e2e3_strategy_smoke memory_accessor_test storage_formats_test \
        accuracy_statistics_test
ctest --test-dir "${build_dir}" --output-on-failure \
    | tee "${run_dir}/ctest.txt"

benchmark="${build_dir}/bin/e2e3_strategy_smoke"
samples_csv="${run_dir}/strategy_smoke_samples.csv"
decoder_validation_csv="${run_dir}/decoder_validation.csv"
kernel_validation_csv="${run_dir}/kernel_validation.csv"

"${benchmark}" \
    --dot-powers "${DOT_POWERS:-16}" \
    --gemv-powers "${GEMV_POWERS:-10}" \
    --gemv-rows "${GEMV_ROWS:-256}" \
    --warmup "${WARMUP:-2}" \
    --samples "${SAMPLES:-2}" \
    --target-sample-ms "${TARGET_SAMPLE_MS:-1}" \
    --base-seed "${BASE_SEED:-2611923443488327891}" \
    --output "${samples_csv}" \
    --decoder-validation-output "${decoder_validation_csv}" \
    --kernel-validation-output "${kernel_validation_csv}" \
    | tee "${run_dir}/smoke_stdout.txt"

python3 "${repo_dir}/tools/summarize_e2e3_strategy_smoke.py" \
    --samples "${samples_csv}" \
    --decoder-validation "${decoder_validation_csv}" \
    --kernel-validation "${kernel_validation_csv}" \
    --output-dir "${run_dir}" \
    | tee "${run_dir}/summary_validation.txt"

if command -v cuobjdump >/dev/null 2>&1; then
    if cuobjdump --dump-resource-usage "${benchmark}" \
        >"${run_dir}/cuda_resource_usage.txt"; then
        echo "resource_usage=complete" \
            >"${run_dir}/compiler_diagnostics_status.txt"
    else
        echo "resource_usage=failed_nonfatal" \
            >"${run_dir}/compiler_diagnostics_status.txt"
    fi
else
    echo "cuobjdump=unavailable" \
        >"${run_dir}/compiler_diagnostics_status.txt"
fi

finished_epoch="$(date +%s)"
{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "wall_time_seconds=$((finished_epoch - started_epoch))"
} >>"${run_dir}/run_manifest.txt"

echo
echo "E2/E3 expanded-strategy smoke experiment complete:"
echo "  ${run_dir}"
echo "  ${run_dir}/decoder_validation.csv"
echo "  ${run_dir}/kernel_validation.csv"
echo "  ${run_dir}/timing_summary.csv"
