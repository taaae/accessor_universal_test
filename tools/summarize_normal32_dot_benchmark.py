#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


EXPECTED_VARIANTS = [
    ("pwl_normal32_16_16", "pwl_global_x1"),
    ("pwq_normal32_8_24", "pwq_shared_x1"),
    ("qn32", "qn_direct_x1"),
    ("fp32_e8m23", "native_f64_x1"),
    ("e11m20", "direct_shift_x1"),
    ("e9m22", "prefix_global_x1"),
    ("e9m22", "word_branchy_x1"),
    ("e8m29", "prefix_global_x1"),
    ("e8m30", "prefix_global_x1"),
    ("e11m36", "direct_shift_x1"),
    ("raw_fp64", "raw_pointer_x1"),
]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples_path: Path) -> list[dict[str, object]]:
    with samples_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("timing sample file is empty")

    found = {(row["format"], row["strategy_id"]) for row in rows}
    expected = set(EXPECTED_VARIANTS)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(f"wrong variant inventory, missing={missing}, extra={extra}")
    if any(row["valid"] != "1" for row in rows):
        raise ValueError("at least one benchmark result is invalid")
    if {row["kernel"] for row in rows} != {"dot"}:
        raise ValueError("the Normal32 experiment must contain only DOT rows")
    if {row["arithmetic_type"] for row in rows} != {"fp64"}:
        raise ValueError("the Normal32 experiment must use FP64 arithmetic")
    if {row["access_method"] for row in rows} != {"scalar"} or {
        row["packet_values"] for row in rows
    } != {"1"}:
        raise ValueError("the Normal32 experiment must use scalar x1 access")
    counts = {int(row["N"]) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"expected one N, found {sorted(counts)}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["format"], row["strategy_id"])].append(row)
    sample_counts = {len(group) for group in grouped.values()}
    if len(sample_counts) != 1:
        raise ValueError("variants have different sample counts")

    raw_times = [
        float(row["mean_ms"])
        for row in grouped[("raw_fp64", "raw_pointer_x1")]
    ]
    raw_median = quantile(raw_times, 0.5)
    summary: list[dict[str, object]] = []
    for key in EXPECTED_VARIANTS:
        group = grouped[key]
        times = [float(row["mean_ms"]) for row in group]
        median_ms = quantile(times, 0.5)
        physical_bytes = int(group[0]["physical_input_bytes"])
        logical_bytes = float(group[0]["logical_input_bytes"])
        summary.append(
            {
                "format": key[0],
                "bits": int(group[0]["bits"]),
                "strategy_id": key[1],
                "N": int(group[0]["N"]),
                "samples": len(group),
                "median_ms": median_ms,
                "p05_ms": quantile(times, 0.05),
                "p95_ms": quantile(times, 0.95),
                "physical_gbps": physical_bytes / median_ms / 1.0e6,
                "logical_gbps": logical_bytes / median_ms / 1.0e6,
                "speedup_vs_raw_fp64": raw_median / median_ms,
            }
        )
    return summary


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = summarize(args.samples)
    write_summary(rows, args.output)
    print("format,strategy,median_ms,physical_GB/s,speedup_vs_raw_fp64")
    for row in rows:
        print(
            f"{row['format']},{row['strategy_id']},{row['median_ms']:.6f},"
            f"{row['physical_gbps']:.3f},{row['speedup_vs_raw_fp64']:.4f}"
        )


if __name__ == "__main__":
    main()
