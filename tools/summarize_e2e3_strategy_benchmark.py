#!/usr/bin/env python3
"""Validate and summarize the full E2M5/E3M4 strategy timing sweep."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


CORE_KINDS = (
    "generic_fp64",
    "branchless_fp32",
    "lut_fp32_global",
    "lut_fp64_global",
    "lut_prefix_global",
)
EXPECTED_STRATEGIES = {
    *(f"{kind}_x{lanes}" for lanes in (1, 2, 4, 8) for kind in CORE_KINDS),
    "direct_fp64_bits_x4",
    "decomposed_bits_x4",
    "lut_fp32_shared_x4",
    "lut_fp64_shared_x4",
    "lut_prefix_shared_x4",
    "lut_prefix_shared_x8",
    "lut_fp32_global_pipelined_x4",
    "lut_prefix_global_pipelined_x4",
    *(f"direct_fp64_words_branchy_x{lanes}" for lanes in (4, 8)),
    *(f"direct_fp64_words_masked_x{lanes}" for lanes in (4, 8)),
    *(f"lut_subnormal_{location}_x{lanes}"
      for lanes in (4, 8) for location in ("global", "shared")),
    *(f"lut_high_word_{location}_x{lanes}"
      for lanes in (4, 8) for location in ("global", "shared")),
    *(f"lut_high_word_swizzled_shared_x{lanes}" for lanes in (4, 8)),
}

IDENTITY = [
    "gpu",
    "compute_capability",
    "distribution",
    "format",
    "storage_bits",
    "strategy",
    "decode_kind",
    "table_location",
    "lanes",
    "lookup_entry_bytes",
    "shared_table_bytes",
    "pipelined",
    "component",
    "n",
    "m",
    "blocks",
    "threads",
    "warmup",
    "main_array_unique_bytes",
    "main_array_requested_bytes",
    "useful_flops",
]

CASE = ["gpu", "compute_capability", "distribution", "component", "n", "m"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"error: {message}")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_coverage(rows: list[dict[str, str]]) -> None:
    require({row["distribution"] for row in rows} == {"uniform_0_1", "normal_0_1"},
            "expected both input distributions")
    require({row["component"] for row in rows} == {"dot", "gemv"},
            "expected both DOT and GEMV")
    require({row["format"] for row in rows} == {"fp64", "e2m5", "e3m4"},
            "expected fp64, e2m5, and e3m4")

    inventory: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        inventory[row["format"]].add(row["strategy"])
    require(inventory["fp64"] == {"raw_pointer_x1"},
            f"unexpected FP64 inventory: {sorted(inventory['fp64'])}")
    for format_name in ("e2m5", "e3m4"):
        require(
            inventory[format_name] == EXPECTED_STRATEGIES,
            f"{format_name} strategy mismatch: "
            f"missing={sorted(EXPECTED_STRATEGIES - inventory[format_name])}, "
            f"extra={sorted(inventory[format_name] - EXPECTED_STRATEGIES)}",
        )

    by_case: dict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        by_case[tuple(row[field] for field in CASE)].add(
            (row["format"], row["strategy"])
        )
    expected_variants = {
        ("fp64", "raw_pointer_x1"),
        *((format_name, strategy) for format_name in ("e2m5", "e3m4")
          for strategy in EXPECTED_STRATEGIES),
    }
    for case, variants in by_case.items():
        require(
            variants == expected_variants,
            f"case {case} has {len(variants)} variants, "
            f"expected {len(expected_variants)}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.samples)
    require(bool(rows), "timing sample CSV is empty")
    validate_coverage(rows)

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in IDENTITY)].append(row)

    sample_counts = {len(group) for group in grouped.values()}
    require(len(sample_counts) == 1, f"unequal sample counts: {sorted(sample_counts)}")
    for key, group in grouped.items():
        coordinates = {(row["round"], row["sample"]) for row in group}
        require(len(coordinates) == len(group), f"duplicate sample coordinate for {key}")

    summary: list[dict[str, object]] = []
    for key, group in grouped.items():
        times = [float(row["time_ms"]) for row in group]
        mean = statistics.fmean(times)
        median_time = statistics.median(times)
        median_seconds = median_time * 1.0e-3
        identity = dict(zip(IDENTITY, key))
        cv = (
            100.0 * statistics.stdev(times) / mean
            if len(times) > 1 and mean != 0.0
            else 0.0
        )
        summary.append(
            identity
            | {
                "sample_count": len(times),
                "median_time_ms": median_time,
                "p05_time_ms": quantile(times, 0.05),
                "p95_time_ms": quantile(times, 0.95),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "cv_percent": cv,
                "median_main_array_unique_gb_per_s": (
                    float(identity["main_array_unique_bytes"])
                    / median_seconds
                    / 1.0e9
                ),
                "median_main_array_requested_gb_per_s": (
                    float(identity["main_array_requested_bytes"])
                    / median_seconds
                    / 1.0e9
                ),
                "median_useful_gflop_per_s": (
                    float(identity["useful_flops"]) / median_seconds / 1.0e9
                ),
            }
        )

    baseline: dict[tuple[str, ...], float] = {}
    case_fields = [field for field in CASE if field not in {"gpu", "compute_capability"}]
    for row in summary:
        if row["format"] == "fp64":
            baseline[tuple(str(row[field]) for field in CASE)] = float(
                row["median_time_ms"]
            )

    format_groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        case_key = tuple(str(row[field]) for field in CASE)
        require(case_key in baseline, f"missing FP64 baseline for {case_key}")
        row["speedup_vs_fp64"] = baseline[case_key] / float(row["median_time_ms"])
        rank_key = case_key + (str(row["format"]),)
        format_groups[rank_key].append(row)
    for group in format_groups.values():
        ordered = sorted(group, key=lambda row: float(row["median_time_ms"]))
        for rank, row in enumerate(ordered, start=1):
            row["rank_within_format"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.sort(
        key=lambda row: (
            str(row["component"]),
            str(row["distribution"]),
            int(str(row["n"])),
            str(row["format"]),
            str(row["strategy"]),
        )
    )
    summary_path = args.output_dir / "timing_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    inventory_fields = [
        "format", "storage_bits", "strategy", "decode_kind", "table_location",
        "lanes", "lookup_entry_bytes", "shared_table_bytes", "pipelined",
    ]
    inventory = {
        tuple(row[field] for field in inventory_fields)
        for row in rows
    }
    inventory_path = args.output_dir / "strategy_inventory.csv"
    with inventory_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(inventory_fields)
        writer.writerows(sorted(inventory))

    winners = [
        row for row in summary
        if row["format"] in {"e2m5", "e3m4"} and row["rank_within_format"] == 1
    ]
    winner_fields = case_fields + [
        "format", "strategy", "lanes", "median_time_ms", "speedup_vs_fp64",
        "cv_percent",
    ]
    winners_path = args.output_dir / "case_winners.csv"
    with winners_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=winner_fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(winners)

    print(
        f"Validated {len(rows)} samples in {len(grouped)} groups: "
        f"{1 + 2 * len(EXPECTED_STRATEGIES)} variants per case, "
        f"{sample_counts.pop()} samples per variant"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {inventory_path}")
    print(f"Wrote {winners_path}")


if __name__ == "__main__":
    main()
