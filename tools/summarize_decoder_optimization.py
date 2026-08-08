#!/usr/bin/env python3
"""Validate and summarize the focused E2/E3 decoder experiment."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


IDENTITY = [
    "gpu",
    "compute_capability",
    "distribution",
    "format",
    "strategy",
    "storage_bits",
    "component",
    "lanes",
    "n",
    "m",
    "blocks",
    "threads",
    "decode_repeats",
]

MODEL_FIELDS = [
    "decoded_values",
    "source_bytes",
    "lookup_bytes_requested",
    "useful_flops",
]
RATE_FIELDS = [
    "decoded_gvalues_per_s",
    "source_gb_per_s",
    "lookup_gb_per_s",
    "useful_gflop_per_s",
]
EXPECTED_DISTRIBUTIONS = {"uniform_0_1", "normal_0_1"}
EXPECTED_COMPONENTS = {
    "register_decode",
    "stream_load",
    "stream_decode",
    "dot",
    "gemv",
}
DECODER_STRATEGIES = {
    "fp64_e11m52": {"current_x1", "current_x4"},
    "e2m5": {"current_x1", "current_x4", "branchless_x4", "lut_x1", "lut_x4"},
    "e3m4": {"current_x1", "current_x4", "branchless_x4", "lut_x1", "lut_x4"},
}
LOAD_STRATEGIES = {
    format_name: {"load_only_x1", "load_only_x4"}
    for format_name in DECODER_STRATEGIES
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def validate_decoder_results(path: Path) -> None:
    rows = read_rows(path)
    expected = {
        (format_name, strategy)
        for format_name in ("e2m5", "e3m4")
        for strategy in DECODER_STRATEGIES[format_name]
    }
    actual = {(row["format"], row["strategy"]) for row in rows}
    if actual != expected:
        raise SystemExit(
            f"validation coverage mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for row in rows:
        if int(row["finite_bit_mismatches"]) != 0:
            raise SystemExit(f"finite decoder mismatch: {row}")
        if int(row["classification_mismatches"]) != 0:
            raise SystemExit(f"non-finite decoder mismatch: {row}")
        if float(row["max_finite_abs_error"]) != 0.0:
            raise SystemExit(f"finite decoder error: {row}")


def validate_coverage(rows: list[dict[str, str]]) -> None:
    if {row["distribution"] for row in rows} != EXPECTED_DISTRIBUTIONS:
        raise SystemExit("distribution coverage mismatch")
    if {row["component"] for row in rows} != EXPECTED_COMPONENTS:
        raise SystemExit("component coverage mismatch")
    if {row["format"] for row in rows} != set(DECODER_STRATEGIES):
        raise SystemExit("format coverage mismatch")
    observed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        observed[(row["component"], row["format"])].add(row["strategy"])
        expected_lanes = row["strategy"].rsplit("x", 1)[-1]
        if expected_lanes not in {"1", "4"} or int(expected_lanes) != int(
            row["lanes"]
        ):
            raise SystemExit(f"strategy/lane mismatch: {row}")
    for component in EXPECTED_COMPONENTS:
        strategies = LOAD_STRATEGIES if component == "stream_load" else DECODER_STRATEGIES
        for format_name, expected in strategies.items():
            actual = observed[(component, format_name)]
            if actual != expected:
                raise SystemExit(
                    f"strategy coverage mismatch for {component}/{format_name}: "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                )


def summarize(samples_path: Path, validation_path: Path, output_dir: Path) -> None:
    validate_decoder_results(validation_path)
    rows = read_rows(samples_path)
    if not rows:
        raise SystemExit(f"no timing rows in {samples_path}")
    validate_coverage(rows)
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in IDENTITY)].append(row)

    summary_rows: list[dict[str, object]] = []
    expected_samples: int | None = None
    for key, group in sorted(grouped.items()):
        times = [float(row["time_ms"]) for row in group]
        if not all(math.isfinite(value) and value > 0.0 for value in times):
            raise SystemExit(f"invalid timing for {key}")
        if expected_samples is None:
            expected_samples = len(times)
        elif len(times) != expected_samples:
            raise SystemExit(
                f"incomplete timing group {key}: {len(times)} vs {expected_samples}"
            )
        rounds = {int(row["round"]) for row in group}
        positions = {int(row["order_position"]) for row in group}
        if len(rounds) < 2 or len(positions) < 2:
            raise SystemExit(f"timing order was not interleaved for {key}")
        median = statistics.median(times)
        mean = statistics.fmean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        first = group[0]
        result: dict[str, object] = dict(zip(IDENTITY, key))
        result.update({field: float(first[field]) for field in MODEL_FIELDS})
        result.update(
            {
                "sample_count": len(times),
                "median_time_ms": median,
                "mean_time_ms": mean,
                "stdev_time_ms": stdev,
                "cv_percent": 100.0 * stdev / mean,
                "mad_time_ms": statistics.median(abs(value - median) for value in times),
                "p05_time_ms": quantile(times, 0.05),
                "p95_time_ms": quantile(times, 0.95),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
            }
        )
        for field in RATE_FIELDS:
            result[f"median_{field}"] = statistics.median(
                float(row[field]) for row in group
            )
        summary_rows.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "timing_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_speedups(summary_rows, output_dir / "strategy_speedups.csv")
    write_plateau(summary_rows, output_dir / "plateau_comparison.csv")
    print(f"Validated {len(rows)} timing samples in {len(summary_rows)} groups")
    print(f"Wrote {summary_path}")


def write_speedups(rows: list[dict[str, object]], path: Path) -> None:
    group_fields = ["distribution", "component", "n", "m"]
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(grouped.items()):
        indexed = {
            (str(row["format"]), str(row["strategy"])): row for row in group
        }
        load = key[1] == "stream_load"
        fp64_x1 = "load_only_x1" if load else "current_x1"
        fp64_x4 = "load_only_x4" if load else "current_x4"
        baseline_x1 = indexed[("fp64_e11m52", fp64_x1)]
        baseline_x4 = indexed[("fp64_e11m52", fp64_x4)]
        for row in group:
            strategy = str(row["strategy"])
            width = int(row["lanes"])
            current = "load_only_x1" if load and width == 1 else (
                "load_only_x4" if load else f"current_x{width}"
            )
            same_format = indexed[(str(row["format"]), current)]
            time_ms = float(row["median_time_ms"])
            result = dict(zip(group_fields, key))
            result.update(
                {
                    "format": row["format"],
                    "strategy": strategy,
                    "storage_bits": row["storage_bits"],
                    "lanes": width,
                    "median_time_ms": time_ms,
                    "speedup_vs_fp64_x1": float(baseline_x1["median_time_ms"])
                    / time_ms,
                    "speedup_vs_fp64_x4": float(baseline_x4["median_time_ms"])
                    / time_ms,
                    "speedup_vs_same_format_current_width": float(
                        same_format["median_time_ms"]
                    )
                    / time_ms,
                    "cv_percent": row["cv_percent"],
                }
            )
            output.append(result)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def write_plateau(rows: list[dict[str, object]], path: Path) -> None:
    maxima: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (str(row["distribution"]), str(row["component"]))
        maxima[key] = max(maxima[key], int(row["n"]))
    selected = [
        row
        for row in rows
        if int(row["n"])
        == maxima[(str(row["distribution"]), str(row["component"]))]
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    summarize(arguments.samples, arguments.validation, arguments.output_dir)


if __name__ == "__main__":
    main()
