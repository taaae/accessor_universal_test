#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
universal_dir="${UNIVERSAL_DIR:-$(cd "${repo_dir}/../universal" 2>/dev/null && pwd || true)}"
if [[ ! -f "${universal_dir}/include/sw/universal/number/posit/posit.hpp" ]]; then
    echo "error: set UNIVERSAL_DIR to a Universal checkout" >&2
    exit 2
fi

compiler="${CXX:-c++}"
output="${TMPDIR:-/tmp}/validate_posit_takum_against_universal"
"${compiler}" -std=c++20 -O2 \
    -I"${repo_dir}/include" \
    -I"${universal_dir}/include/sw" \
    "${repo_dir}/tools/validate_posit_takum_against_universal.cpp" \
    -o "${output}"
"${output}"
