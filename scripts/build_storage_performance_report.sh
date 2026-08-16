#!/usr/bin/env bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

run_dir="${1:-}"
if [[ -z "${run_dir}" ]]; then
    while IFS= read -r candidate; do
        if [[ -f "${candidate}/timing_summary.csv" &&
              -f "${candidate}/profile_operations.csv" ]]; then
            run_dir="${candidate}"
        fi
    done < <(find results/008_storage_performance -maxdepth 1 -type d -name 'run_*' | sort)
fi

if [[ -z "${run_dir}" ||
      ! -f "${run_dir}/timing_summary.csv" ||
      ! -f "${run_dir}/profile_operations.csv" ]]; then
    echo "error: no complete storage-performance run found" >&2
    exit 2
fi

uv run python tools/summarize_storage_performance.py \
    --summary "${run_dir}/timing_summary.csv" \
    --output-dir "${run_dir}"

uv run python tools/build_storage_performance_report.py \
    --run-dir "${run_dir}" \
    --output-dir results/report

uv run python tools/build_precision_packing_report.py \
    --output-dir results/report

uv run python tools/build_ieee_question_report.py \
    --output-dir results/report

# LNS_FORMATS limits the rollout while the page design is under review; leave it
# unset to build a page for every format in the newest experiment 022 run.
uv run python tools/build_lns_question_report.py \
    --output-dir results/report \
    --formats "${LNS_FORMATS:-}"
