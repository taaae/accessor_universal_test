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
CONTROL_DISTRIBUTIONS = {"lut_scattered_control", "lut_concentrated_control"}
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
        if row["format"] in ALT_FORMATS:
            bits = ALT_FORMATS[row["format"]]
        else:
            matches = [
                case_bits
                for (format_name, _), (case_bits, _) in IEEE_CASES.items()
                if format_name == row["format"]
            ]
            if not matches:
                raise ValueError(f"unknown format in timing row: {row}")
            bits = matches[0]
        n = int(row["N"])
        m = int(row["M"])
        def packed(count: int) -> int:
            raw_bytes = (count * bits + 7) // 8
            return ((raw_bytes + 3) // 4) * 4 + 4 if bits == 14 else raw_bytes
        expected_input_bytes = (
            2 * packed(n)
            if row["kernel"] == "dot"
            else packed(m * n) + packed(n)
        )
        if int(row["input_bytes"]) != expected_input_bytes:
            raise ValueError(f"wrong physical input byte count: {row}")
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
    for format_name, bits in ALT_FORMATS.items():
        if bits <= 14:
            for arithmetic in ARITHMETIC:
                for distribution in CONTROL_DISTRIBUTIONS:
                    for kernel in KERNELS:
                        for strategy in {"full_lut_global", "full_lut_shared"}:
                            expected_keys.add(
                                (format_name, arithmetic, distribution, kernel, strategy)
                            )
    for format_name, arithmetic in {
        ("fp8_e4m3", "fp32"),
        ("fp8_e4m3", "fp64"),
        ("e5m8", "fp32"),
        ("e5m8", "fp64"),
    }:
        for distribution in CONTROL_DISTRIBUTIONS:
            for kernel in KERNELS:
                for strategy in {"full_lut_global", "full_lut_shared"}:
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
        expected_reference = (
            "takum_log_paper_formula"
            if row["format"].startswith("takum_log")
            else "universal_cross_validated_core"
            if row["format"] in ALT_FORMATS
            else "cpu_format_reference"
        )
        if row["reference"] != expected_reference:
            raise ValueError(f"wrong decoder reference chain: {row}")
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
        (format_name, arithmetic, strategy)
        for format_name, bits in ALT_FORMATS.items()
        for arithmetic in ARITHMETIC
        for strategy in expected_strategies(bits)
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
    histogram_kinds: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    histogram_totals: dict[tuple[str, str, str, str], int] = defaultdict(int)
    histogram_buckets: dict[
        tuple[str, str, str, str], dict[str, int]
    ] = defaultdict(dict)
    control_ranges: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in histograms:
        key = (row["format"], row["arithmetic"], row["distribution"])
        histogram_keys.add(key)
        histogram_kinds[key].add(row["histogram"])
        histogram_totals[key + (row["histogram"],)] += int(row["count"])
        histogram_buckets[key + (row["histogram"],)][row["bucket"]] = int(
            row["count"]
        )
        if float(row["realized_q_min"]) >= float(row["realized_q_max"]):
            raise ValueError(f"degenerate realized interval: {row}")
        if row["histogram"] == "raw_index_trace":
            control_ranges[key] = (
                float(row["realized_q_min"]),
                float(row["realized_q_max"]),
            )
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
    expected_histograms |= {
        (format_name, arithmetic, distribution)
        for format_name, bits in ALT_FORMATS.items()
        if bits <= 14
        for arithmetic in ARITHMETIC
        for distribution in CONTROL_DISTRIBUTIONS
    }
    expected_histograms |= {
        (format_name, arithmetic, distribution)
        for format_name, arithmetic in {
            ("fp8_e4m3", "fp32"),
            ("fp8_e4m3", "fp64"),
            ("e5m8", "fp32"),
            ("e5m8", "fp64"),
        }
        for distribution in CONTROL_DISTRIBUTIONS
    }
    if histogram_keys != expected_histograms:
        raise ValueError("histogram matrix is incomplete")
    for key in expected_histograms:
        kinds = histogram_kinds[key]
        distribution = key[2]
        if distribution in CONTROL_DISTRIBUTIONS:
            required = {"raw_index_trace", "raw_index_distinct"}
        elif distribution == "paired_log_uniform_finite":
            required = {"sign", "log_interval"}
        elif key[0] in ALT_FORMATS:
            required = (
                {"sign", "field_bucket", "regime"}
                if key[0].startswith("posit")
                else {
                    "sign",
                    "field_bucket",
                    "direction",
                    "regime",
                    "characteristic",
                }
            )
        else:
            required = {"sign", "field_bucket", "exponent", "subnormal"}
        if not required.issubset(kinds):
            raise ValueError(f"missing histogram kinds for {key}: {required - kinds}")
        expected_total = 4096 if args.mode == "smoke" else 65536
        total_kinds = required - {"raw_index_distinct"}
        totals = {histogram_totals[key + (kind,)] for kind in total_kinds}
        if len(totals) != 1:
            raise ValueError(f"histogram totals disagree for {key}: {totals}")
        if totals != {expected_total}:
            raise ValueError(f"wrong histogram total for {key}: {totals}")
        if "sign" in required:
            sign = histogram_buckets[key + ("sign",)]
            if set(sign) != {"0", "1"} or abs(sign["0"] - sign["1"]) > 1:
                raise ValueError(f"unbalanced sign histogram for {key}: {sign}")
        if "direction" in required:
            direction = histogram_buckets[key + ("direction",)]
            if set(direction) != {"0", "1"} or abs(
                direction["0"] - direction["1"]
            ) > 1:
                raise ValueError(
                    f"unbalanced direction histogram for {key}: {direction}"
                )
        if "regime" in required:
            regime = histogram_buckets[key + ("regime",)]
            if key[0].startswith("takum") and set(regime) != {
                str(index) for index in range(8)
            }:
                raise ValueError(f"incomplete takum regime histogram for {key}")
            if key[0].startswith("takum") and max(regime.values()) - min(
                regime.values()
            ) > 1:
                raise ValueError(f"unbalanced takum regime histogram for {key}")
            if len(regime) < 2:
                raise ValueError(f"degenerate regime histogram for {key}")
        if key[0].startswith("takum") and "field_bucket" in required:
            cells = histogram_buckets[key + ("field_bucket",)]
            if set(cells) != {str(index) for index in range(16)}:
                raise ValueError(f"incomplete takum field matrix for {key}")
            if direction["0"] != sum(cells[str(index)] for index in range(8)):
                raise ValueError(f"takum D=0 cells disagree for {key}")
            if direction["1"] != sum(cells[str(index)] for index in range(8, 16)):
                raise ValueError(f"takum D=1 cells disagree for {key}")
            for index in range(8):
                if regime[str(index)] != cells[str(index)] + cells[str(8 + index)]:
                    raise ValueError(
                        f"takum regime cells disagree for {key}, regime {index}"
                    )
        if "characteristic" in required and len(
            histogram_buckets[key + ("characteristic",)]
        ) < 2:
            raise ValueError(f"degenerate characteristic histogram for {key}")
        if "exponent" in required and len(
            histogram_buckets[key + ("exponent",)]
        ) < 2:
            raise ValueError(f"degenerate exponent histogram for {key}")
        if "subnormal" in required:
            labels = set(histogram_buckets[key + ("subnormal",)])
            if not labels or not labels.issubset({"0", "1"}):
                raise ValueError(f"bad subnormal histogram for {key}")
    for key, (minimum, maximum) in control_ranges.items():
        bits = ALT_FORMATS.get(key[0])
        if bits is None:
            bits = next(
                case_bits
                for (format_name, _), (case_bits, _) in IEEE_CASES.items()
                if format_name == key[0]
            )
        if key[2] == "lut_concentrated_control":
            if minimum != 0 or maximum != 255:
                raise ValueError(f"wrong concentrated LUT trace range for {key}")
            if histogram_totals[key + ("raw_index_distinct",)] != 256:
                raise ValueError(f"wrong concentrated distinct-index count for {key}")
        elif args.mode == "full" and (minimum != 0 or maximum != (1 << bits) - 1):
            raise ValueError(f"scattered LUT trace does not cover the table for {key}")
        elif args.mode == "full" and histogram_totals[
            key + ("raw_index_distinct",)
        ] != (1 << bits):
            raise ValueError(f"scattered LUT trace misses indices for {key}")
    for key, counts in field_counts.items():
        if not counts:
            raise ValueError(f"missing field buckets for {key}")
        if key[0].startswith("takum"):
            balanced = True
        elif key[0] in ALT_FORMATS:
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
