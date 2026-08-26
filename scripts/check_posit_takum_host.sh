#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
universal_dir="${UNIVERSAL_DIR:-$(cd "${repo_dir}/../universal" 2>/dev/null && pwd || true)}"
evidence="${repo_dir}/results/024_posit_takum_strategy_performance/reference_evidence.txt"

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{ print $1 }'
    else
        sha256sum "$1" | awk '{ print $1 }'
    fi
}

if [[ ! -f "${universal_dir}/include/sw/universal/number/posit/posit.hpp" ]]; then
    if [[ ! -f "${evidence}" ]]; then
        echo "error: Universal is unavailable and committed reference evidence is missing" >&2
        exit 2
    fi
    expected_core="$(awk -F= '$1 == "core_sha256" { print $2 }' "${evidence}")"
    expected_validator="$(awk -F= '$1 == "validator_sha256" { print $2 }' "${evidence}")"
    actual_core="$(sha256_file "${repo_dir}/include/posit_takum_core.hpp")"
    actual_validator="$(sha256_file "${repo_dir}/tools/validate_posit_takum_against_universal.cpp")"
    if [[ "${actual_core}" != "${expected_core}" ||
          "${actual_validator}" != "${expected_validator}" ]]; then
        echo "error: committed reference evidence does not match decoder sources" >&2
        exit 2
    fi
    echo "Verified source-bound committed independent reference evidence"
    cat "${evidence}"
    exit 0
fi

compiler="${CXX:-c++}"
output="${TMPDIR:-/tmp}/validate_posit_takum_against_universal"
"${compiler}" -std=c++20 -O2 \
    -I"${repo_dir}/include" \
    -I"${universal_dir}/include/sw" \
    "${repo_dir}/tools/validate_posit_takum_against_universal.cpp" \
    -o "${output}"
"${output}"
