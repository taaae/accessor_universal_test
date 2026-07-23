#!/usr/bin/env python3
"""Summarize raw timing samples and matched accessor accuracy results."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


VARIANT_ORDER = [
    "raw_f32",
    "raw_f64",
    "accessor_f32_f32",
    "accessor_f64_f32",
]


def percentile(sorted_values, fraction):
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] + weight * (
        sorted_values[upper] - sorted_values[lower]
    )


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_performance_summary(input_path, output_path):
    rows = read_rows(input_path)
    grouped = defaultdict(list)
    metadata = {}
    for row in rows:
        key = (int(row["n"]), row["variant"])
        grouped[key].append(float(row["total_time_ms"]))
        metadata[key] = row

    summaries = {}
    for key, samples in grouped.items():
        ordered = sorted(samples)
        mean = statistics.fmean(samples)
        summaries[key] = {
            "samples": len(samples),
            "min_ms": ordered[0],
            "median_ms": percentile(ordered, 0.5),
            "p10_ms": percentile(ordered, 0.1),
            "p90_ms": percentile(ordered, 0.9),
            "mean_ms": mean,
            "coefficient_of_variation": (
                statistics.pstdev(samples) / mean if mean else 0.0
            ),
        }

    fields = [
        "gpu",
        "dataset",
        "variant",
        "storage_type",
        "arithmetic_type",
        "access_kind",
        "n",
        "rounds",
        "samples",
        "min_ms",
        "median_ms",
        "p10_ms",
        "p90_ms",
        "mean_ms",
        "coefficient_of_variation",
        "giga_elements_per_s",
        "algorithmic_gb_per_s",
        "gflop_per_s",
        "speedup_vs_raw_f32",
        "accessor_overhead_percent",
        "fp64_arithmetic_cost_percent",
    ]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for n in sorted({key[0] for key in summaries}):
            for variant in VARIANT_ORDER:
                key = (n, variant)
                if key not in summaries:
                    continue
                result = summaries[key]
                meta = metadata[key]
                median_s = result["median_ms"] * 1.0e-3
                raw_f32 = summaries.get((n, "raw_f32"))
                matching_raw_name = (
                    "raw_f64" if meta["arithmetic_type"] == "float64" else "raw_f32"
                )
                matching_raw = summaries.get((n, matching_raw_name))
                matching_f32_name = (
                    "accessor_f32_f32"
                    if meta["access_kind"] == "accessor"
                    else "raw_f32"
                )
                matching_f32 = summaries.get((n, matching_f32_name))
                rounds = len(
                    {
                        row["round"]
                        for row in rows
                        if int(row["n"]) == n and row["variant"] == variant
                    }
                )
                writer.writerow(
                    {
                        "gpu": meta["gpu"],
                        "dataset": meta["dataset"],
                        "variant": variant,
                        "storage_type": meta["storage_type"],
                        "arithmetic_type": meta["arithmetic_type"],
                        "access_kind": meta["access_kind"],
                        "n": n,
                        "rounds": rounds,
                        **result,
                        "giga_elements_per_s": n / median_s / 1.0e9,
                        "algorithmic_gb_per_s": 8.0 * n / median_s / 1.0e9,
                        "gflop_per_s": 2.0 * n / median_s / 1.0e9,
                        "speedup_vs_raw_f32": (
                            raw_f32["median_ms"] / result["median_ms"]
                            if raw_f32
                            else ""
                        ),
                        "accessor_overhead_percent": (
                            100.0
                            * (result["median_ms"] / matching_raw["median_ms"] - 1.0)
                            if meta["access_kind"] == "accessor" and matching_raw
                            else ""
                        ),
                        "fp64_arithmetic_cost_percent": (
                            100.0
                            * (result["median_ms"] / matching_f32["median_ms"] - 1.0)
                            if meta["arithmetic_type"] == "float64"
                            and matching_f32
                            else ""
                        ),
                    }
                )


def write_accuracy_comparison(input_path, output_path):
    rows = read_rows(input_path)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], int(row["n"]))][row["variant"]] = row

    fields = [
        "dataset",
        "n",
        "dot_condition_number",
        "raw_f32_normalized_error",
        "accessor_f32_f32_normalized_error",
        "raw_f64_normalized_error",
        "accessor_f64_f32_normalized_error",
        "raw_fp64_error_improvement",
        "accessor_fp64_error_improvement",
        "fp32_raw_accessor_bit_identical",
        "fp64_raw_accessor_bit_identical",
        "all_results_finite",
    ]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (dataset, n), variants_by_name in sorted(grouped.items()):
            if not all(name in variants_by_name for name in VARIANT_ORDER):
                continue
            r32 = variants_by_name["raw_f32"]
            r64 = variants_by_name["raw_f64"]
            a32 = variants_by_name["accessor_f32_f32"]
            a64 = variants_by_name["accessor_f64_f32"]
            r32_error = float(r32["normalized_error"])
            r64_error = float(r64["normalized_error"])
            a32_error = float(a32["normalized_error"])
            a64_error = float(a64["normalized_error"])

            def improvement(low_precision_error, high_precision_error):
                if high_precision_error == 0.0:
                    return math.inf if low_precision_error else 1.0
                return low_precision_error / high_precision_error

            writer.writerow(
                {
                    "dataset": dataset,
                    "n": n,
                    "dot_condition_number": r32["dot_condition_number"],
                    "raw_f32_normalized_error": r32_error,
                    "accessor_f32_f32_normalized_error": a32_error,
                    "raw_f64_normalized_error": r64_error,
                    "accessor_f64_f32_normalized_error": a64_error,
                    "raw_fp64_error_improvement": improvement(
                        r32_error, r64_error
                    ),
                    "accessor_fp64_error_improvement": improvement(
                        a32_error, a64_error
                    ),
                    "fp32_raw_accessor_bit_identical": (
                        r32["result_bits"] == a32["result_bits"]
                    ),
                    "fp64_raw_accessor_bit_identical": (
                        r64["result_bits"] == a64["result_bits"]
                    ),
                    "all_results_finite": all(
                        row["finite"] == "true" for row in variants_by_name.values()
                    ),
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_performance_summary(
        args.performance, args.output_dir / "performance_summary.csv"
    )
    write_accuracy_comparison(
        args.accuracy, args.output_dir / "accuracy_comparison.csv"
    )


if __name__ == "__main__":
    main()
