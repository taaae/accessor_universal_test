#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ALT_FORMATS = {
    "posit8_es0": 8,
    "posit14_es1": 14,
    "posit16_es1": 16,
    "posit32_es2": 32,
    "takum8": 8,
    "takum14": 14,
    "takum16": 16,
    "takum32": 32,
    "takum_log8": 8,
    "takum_log14": 14,
    "takum_log16": 16,
    "takum_log32": 32,
}
IEEE_CASES = {
    ("fp8_e4m3", "fp32"): (8, {"native_scalar", "full_lut_shared", "full_lut_global"}),
    ("fp8_e4m3", "fp64"): (8, {"native_scalar", "full_lut_shared", "full_lut_global"}),
    ("fp8_e5m2", "fp32"): (8, {"native_scalar", "full_lut_shared", "full_lut_global"}),
    ("fp8_e5m2", "fp64"): (8, {"native_scalar", "full_lut_shared", "full_lut_global"}),
    ("e3m4", "fp32"): (8, {"full_lut_shared", "full_lut_global"}),
    ("e3m4", "fp64"): (8, {"full_lut_shared", "full_lut_global"}),
    ("e6m1", "fp32"): (8, {"full_lut_shared", "full_lut_global"}),
    ("e6m1", "fp64"): (8, {"full_lut_shared", "full_lut_global"}),
    ("e8m5", "fp32"): (14, {"direct_branchy", "full_lut_shared", "full_lut_global"}),
    ("e11m2", "fp64"): (14, {"direct_masked", "full_lut_shared", "full_lut_global"}),
    ("e2m11", "fp32"): (14, {"direct_branchy", "full_lut_shared", "full_lut_global"}),
    ("e2m11", "fp64"): (14, {"direct_branchy", "full_lut_shared", "full_lut_global"}),
    ("e5m8", "fp32"): (14, {"direct_branchy", "full_lut_shared", "full_lut_global"}),
    ("e5m8", "fp64"): (14, {"direct_branchy", "full_lut_shared", "full_lut_global"}),
    ("fp16_e5m10", "fp32"): (16, {"native_scalar", "full_lut_global"}),
    ("fp16_e5m10", "fp64"): (16, {"native_scalar", "full_lut_global"}),
    ("bf16_e8m7", "fp32"): (16, {"native_scalar", "full_lut_global"}),
    ("bf16_e8m7", "fp64"): (16, {"native_scalar", "full_lut_global"}),
    ("e11m4", "fp64"): (16, {"direct_masked", "full_lut_global"}),
    ("e3m12", "fp32"): (16, {"subnormal_lut_global", "full_lut_global"}),
    ("e3m12", "fp64"): (16, {"subnormal_lut_global", "full_lut_global"}),
    ("e6m9", "fp32"): (16, {"direct_branchy", "prefix_lut_global", "full_lut_global"}),
    ("e6m9", "fp64"): (16, {"direct_branchy", "prefix_lut_global", "full_lut_global"}),
    ("fp32_e8m23", "fp32"): (32, {"native_scalar"}),
    ("fp32_e8m23", "fp64"): (32, {"native_scalar"}),
    ("e11m20", "fp64"): (32, {"direct_masked"}),
    ("e4m27", "fp64"): (32, {"direct_branchy", "prefix_lut_global"}),
    ("e10m21", "fp64"): (32, {"direct_branchy", "prefix_lut_global"}),
}
DISTRIBUTIONS = {"field_balanced_finite", "paired_log_uniform_finite"}
KERNELS = {"dot", "gemv"}
ARITHMETIC = {"fp32", "fp64"}


def expected_strategies(bits: int) -> set[str]:
    if bits <= 14:
        return {"direct", "full_lut_global", "full_lut_shared"}
    if bits == 16:
        return {"direct", "full_lut_global"}
    return {"direct"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def validate(args: argparse.Namespace) -> dict[str, int | str]:
    samples = read_rows(args.samples)
    validations = read_rows(args.validation)
    histograms = read_rows(args.histograms)
    expected_samples = 2 if args.mode == "smoke" else 30

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = (
        defaultdict(list)
    )
    infeasible: set[tuple[str, str, str, str, str]] = set()
    for row in samples:
        if row["storage_layout"] != "dense":
            raise ValueError(f"non-dense row: {row}")
        if row["access_method"] != "scalar" or row["packet_values"] != "1":
            raise ValueError(f"non-scalar-x1 row: {row}")
        key = (
            row["format"],
            row["arithmetic"],
            row["distribution"],
            row["kernel"],
            row["strategy"],
        )
        if row["status"] == "infeasible_shared_memory":
            infeasible.add(key)
        elif row["status"] == "ok":
            grouped[key].append(row)
        else:
            raise ValueError(f"unexpected sample status: {row}")

    expected_keys: set[tuple[str, str, str, str, str]] = set()
    for format_name, bits in ALT_FORMATS.items():
        for arithmetic in ARITHMETIC:
            for distribution in DISTRIBUTIONS:
                for kernel in KERNELS:
                    for strategy in expected_strategies(bits):
                        expected_keys.add(
                            (format_name, arithmetic, distribution, kernel, strategy)
                        )
    for (format_name, arithmetic), (bits, strategies) in IEEE_CASES.items():
        for distribution in DISTRIBUTIONS:
            for kernel in KERNELS:
                for strategy in strategies:
                    expected_keys.add(
                        (format_name, arithmetic, distribution, kernel, strategy)
                    )
    present = set(grouped) | infeasible
    missing = expected_keys - present
    unexpected = present - expected_keys
    if missing or unexpected:
        raise ValueError(f"coverage mismatch missing={missing} unexpected={unexpected}")
    for key, rows in grouped.items():
        if len(rows) != expected_samples:
            raise ValueError(f"{key} has {len(rows)} samples, expected {expected_samples}")
        if any(float(row["kernel_ms"]) <= 0.0 for row in rows):
            raise ValueError(f"nonpositive timing for {key}")

    validation_keys = set()
    for row in validations:
        key = (row["format"], row["arithmetic"])
        validation_keys.add((row["format"], row["arithmetic"], row["strategy"]))
        if row["status"] == "infeasible_shared_memory":
            continue
        if row["status"] != "pass" or int(row["failures"]) != 0:
            raise ValueError(f"decoder validation failed: {row}")
        bits = (
            ALT_FORMATS[row["format"]]
            if row["format"] in ALT_FORMATS
            else IEEE_CASES[key][0]
        )
        expected_codes = 1 << bits if bits <= 16 else 1_000_000
        if int(row["codes_checked"]) != expected_codes:
            raise ValueError(f"wrong decoder coverage: {row}")
    expected_validation = {
        (format_name, arithmetic, "direct")
        for format_name in ALT_FORMATS
        for arithmetic in ARITHMETIC
    }
    expected_validation |= {
        (format_name, arithmetic, strategy)
        for (format_name, arithmetic), (_, strategies) in IEEE_CASES.items()
        for strategy in strategies
    }
    if validation_keys != expected_validation:
        raise ValueError("decoder validation matrix is incomplete")

    field_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    histogram_keys = set()
    for row in histograms:
        key = (row["format"], row["arithmetic"], row["distribution"])
        histogram_keys.add(key)
        if float(row["realized_q_min"]) >= float(row["realized_q_max"]):
            raise ValueError(f"degenerate realized interval: {row}")
        if row["histogram"] == "field_bucket":
            field_counts[(row["format"], row["arithmetic"])].append(
                int(row["count"])
            )
    expected_histograms = {
        (format_name, arithmetic, distribution)
        for format_name in ALT_FORMATS
        for arithmetic in ARITHMETIC
        for distribution in DISTRIBUTIONS
    }
    expected_histograms |= {
        (format_name, arithmetic, distribution)
        for format_name, arithmetic in IEEE_CASES
        for distribution in DISTRIBUTIONS
    }
    if histogram_keys != expected_histograms:
        raise ValueError("histogram matrix is incomplete")
    for key, counts in field_counts.items():
        if not counts:
            raise ValueError(f"missing field buckets for {key}")
        if key[0] in ALT_FORMATS:
            balanced = max(counts) - min(counts) <= 1
        elif len(counts) > 1 and counts[-1] >= sum(counts[:-1]) - 1:
            balanced = max(counts[:-1]) - min(counts[:-1]) <= 1
        else:
            balanced = max(counts) - min(counts) <= 1
        if not balanced:
            raise ValueError(f"unbalanced field buckets for {key}: {counts}")

    return {
        "mode": args.mode,
        "timing_rows": len(samples),
        "timed_cases": len(grouped),
        "infeasible_cases": len(infeasible),
        "decoder_rows": len(validations),
        "histogram_rows": len(histograms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--histograms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = validate(args)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
