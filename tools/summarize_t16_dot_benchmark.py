#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


EXPECTED_FORMATS = {
    "t16",
    "fp16_e5m10",
    "e6m9",
    "e8m15",
    "raw_fp32",
}


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

    formats = {row["format"] for row in rows}
    if formats != EXPECTED_FORMATS:
        missing = sorted(EXPECTED_FORMATS - formats)
        extra = sorted(formats - EXPECTED_FORMATS)
        raise ValueError(f"wrong format inventory, missing={missing}, extra={extra}")
    if any(row["valid"] != "1" for row in rows):
        raise ValueError("at least one benchmark result is invalid")
    counts = {int(row["N"]) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"expected one N, found {sorted(counts)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["format"]].append(row)
    sample_counts = {len(group) for group in grouped.values()}
    if len(sample_counts) != 1:
        raise ValueError("formats have different sample counts")

    raw_times = [float(row["mean_ms"]) for row in grouped["raw_fp32"]]
    raw_median = quantile(raw_times, 0.5)
    summary: list[dict[str, object]] = []
    for name in ["t16", "fp16_e5m10", "e6m9", "e8m15", "raw_fp32"]:
        group = grouped[name]
        times = [float(row["mean_ms"]) for row in group]
        median_ms = quantile(times, 0.5)
        physical_bytes = int(group[0]["physical_input_bytes"])
        logical_bytes = float(group[0]["logical_input_bytes"])
        summary.append(
            {
                "format": name,
                "bits": int(group[0]["bits"]),
                "strategy_id": group[0]["strategy_id"],
                "N": int(group[0]["N"]),
                "samples": len(group),
                "median_ms": median_ms,
                "p05_ms": quantile(times, 0.05),
                "p95_ms": quantile(times, 0.95),
                "physical_gbps": physical_bytes / median_ms / 1.0e6,
                "logical_gbps": logical_bytes / median_ms / 1.0e6,
                "speedup_vs_raw_fp32": raw_median / median_ms,
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
    print("format,strategy,median_ms,physical_GB/s,speedup_vs_raw_fp32")
    for row in rows:
        print(
            f"{row['format']},{row['strategy_id']},{row['median_ms']:.6f},"
            f"{row['physical_gbps']:.3f},{row['speedup_vs_raw_fp32']:.4f}"
        )


if __name__ == "__main__":
    main()
