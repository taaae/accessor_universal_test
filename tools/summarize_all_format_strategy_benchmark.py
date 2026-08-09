#!/usr/bin/env python3
"""Validate and summarize the all-format decoder strategy timing sweep."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


BASELINE_FORMAT = "fp64_e11m52"
BASELINE_STRATEGY = "raw_pointer_x1"

IDENTITY = [
    "gpu",
    "compute_capability",
    "distribution",
    "benchmark_format",
    "format",
    "storage_bits",
    "strategy",
    "decode_kind",
    "table_location",
    "unpack",
    "lanes",
    "lookup_entry_bytes",
    "shared_table_bytes",
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

CASE = [
    "gpu",
    "compute_capability",
    "distribution",
    "benchmark_format",
    "component",
    "n",
    "m",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"error: {message}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parse_list(text: str) -> set[str]:
    values = {token for token in text.split(",") if token}
    require(bool(values), f"empty list: {text}")
    return values


def parse_powers(text: str) -> set[int]:
    try:
        powers = {int(token) for token in text.split(",")}
    except ValueError as error:
        raise SystemExit(f"error: invalid power list: {text}") from error
    require(
        bool(powers) and all(0 <= power < 63 for power in powers),
        f"invalid power list: {text}",
    )
    return {1 << power for power in powers}


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def geometric_mean(values: list[float]) -> float:
    require(bool(values) and all(value > 0 for value in values),
            "geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def expected_inventory(
    validation_rows: list[dict[str, str]], formats: set[str]
) -> dict[str, set[tuple[str, str]]]:
    inventory: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in validation_rows:
        if row["format"] not in formats:
            continue
        require(int(row["mismatches"]) == 0,
                f"smoke validation failed for {row['format']}/{row['strategy']}")
        entry = (row["strategy"], row["lanes"])
        require(entry not in inventory[row["format"]],
                f"duplicate smoke strategy {row['format']}/{entry}")
        inventory[row["format"]].add(entry)
    require(set(inventory) == formats,
            f"smoke inventory mismatch: missing={sorted(formats - set(inventory))}")
    return inventory


def validate_size_grid(
    rows: list[dict[str, str]],
    formats: set[str],
    dot_sizes: set[int],
    gemv_sizes: set[int],
    gemv_rows: int,
) -> None:
    actual = {
        (
            row["benchmark_format"],
            row["distribution"],
            row["component"],
            int(row["n"]),
            int(row["m"]),
        )
        for row in rows
    }
    expected = {
        (format_name, distribution, "dot", size, 1)
        for format_name in formats
        for distribution in ("uniform_0_1", "normal_0_1")
        for size in dot_sizes
    } | {
        (format_name, distribution, "gemv", size, gemv_rows)
        for format_name in formats
        for distribution in ("uniform_0_1", "normal_0_1")
        for size in gemv_sizes
    }
    require(
        actual == expected,
        "size-grid mismatch: "
        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )


def validate_coverage(
    rows: list[dict[str, str]],
    formats: set[str],
    inventory: dict[str, set[tuple[str, str]]],
) -> None:
    require({row["distribution"] for row in rows}
            == {"uniform_0_1", "normal_0_1"},
            "expected both input distributions")
    require({row["component"] for row in rows} == {"dot", "gemv"},
            "expected both DOT and GEMV")
    require({row["benchmark_format"] for row in rows} == formats,
            "benchmark-format coverage mismatch")

    by_case: dict[tuple[str, ...], set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        benchmark_format = row["benchmark_format"]
        require(row["format"] in {benchmark_format, BASELINE_FORMAT},
                f"case {benchmark_format} contains unrelated format {row['format']}")
        by_case[tuple(row[field] for field in CASE)].add(
            (row["format"], row["strategy"], row["lanes"])
        )

    for case, variants in by_case.items():
        benchmark_format = case[3]
        expected = {
            (BASELINE_FORMAT, BASELINE_STRATEGY, "1"),
            *((benchmark_format, strategy, lanes)
              for strategy, lanes in inventory[benchmark_format]),
        }
        require(
            variants == expected,
            f"case {case} variant mismatch: "
            f"missing={sorted(expected - variants)}, extra={sorted(variants - expected)}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--validation-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-formats", required=True)
    parser.add_argument("--expected-dot-powers", required=True)
    parser.add_argument("--expected-gemv-powers", required=True)
    parser.add_argument("--expected-gemv-rows", type=int, required=True)
    parser.add_argument("--expected-rounds", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    args = parser.parse_args()

    formats = parse_list(args.expected_formats)
    rows = read_rows(args.samples)
    validation_rows = read_rows(args.validation_inventory)
    require(bool(rows), "timing sample CSV is empty")
    require(args.expected_gemv_rows > 0, "expected GEMV rows must be positive")
    require(args.expected_rounds > 0 and args.expected_samples > 0,
            "expected sample dimensions must be positive")
    for field in IDENTITY + ["round", "sample", "time_ms"]:
        require(field in rows[0], f"timing sample CSV lacks {field}")

    inventory = expected_inventory(validation_rows, formats)
    validate_coverage(rows, formats, inventory)
    validate_size_grid(
        rows,
        formats,
        parse_powers(args.expected_dot_powers),
        parse_powers(args.expected_gemv_powers),
        args.expected_gemv_rows,
    )

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in IDENTITY)].append(row)

    expected_sample_count = args.expected_rounds * args.expected_samples
    for key, group in grouped.items():
        require(len(group) == expected_sample_count,
                f"group {key} has {len(group)} samples, expected {expected_sample_count}")
        coordinates = {(int(row["round"]), int(row["sample"])) for row in group}
        expected_coordinates = {
            (round_index, sample_index)
            for round_index in range(args.expected_rounds)
            for sample_index in range(args.expected_samples)
        }
        require(coordinates == expected_coordinates,
                f"sample-coordinate mismatch for {key}")

    summary: list[dict[str, object]] = []
    for key, group in grouped.items():
        times = [float(row["time_ms"]) for row in group]
        require(all(math.isfinite(value) and value > 0 for value in times),
                f"non-positive or non-finite timing in {key}")
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
    for row in summary:
        if row["format"] == BASELINE_FORMAT:
            key = tuple(str(row[field]) for field in CASE)
            require(key not in baseline, f"duplicate FP64 baseline for {key}")
            baseline[key] = float(row["median_time_ms"])

    target_groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        case_key = tuple(str(row[field]) for field in CASE)
        require(case_key in baseline, f"missing FP64 baseline for {case_key}")
        row["speedup_vs_fp64"] = baseline[case_key] / float(row["median_time_ms"])
        row["rank_within_format"] = 0
        if row["format"] != BASELINE_FORMAT:
            target_groups[case_key].append(row)

    for group in target_groups.values():
        for rank, row in enumerate(
            sorted(group, key=lambda item: float(item["median_time_ms"])), start=1
        ):
            row["rank_within_format"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.sort(
        key=lambda row: (
            str(row["benchmark_format"]),
            str(row["component"]),
            str(row["distribution"]),
            int(str(row["n"])),
            int(row["rank_within_format"]),
            str(row["strategy"]),
        )
    )
    summary_path = args.output_dir / "timing_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    inventory_fields = [
        "benchmark_format", "format", "storage_bits", "strategy", "decode_kind",
        "table_location", "unpack", "lanes", "lookup_entry_bytes",
        "shared_table_bytes",
    ]
    inventory_rows = {
        tuple(row[field] for field in inventory_fields)
        for row in rows
    }
    inventory_path = args.output_dir / "strategy_inventory.csv"
    with inventory_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(inventory_fields)
        writer.writerows(sorted(inventory_rows))

    winners = [
        row for row in summary
        if row["format"] != BASELINE_FORMAT and row["rank_within_format"] == 1
    ]
    winner_fields = [
        "distribution", "benchmark_format", "component", "n", "m", "strategy",
        "decode_kind", "table_location", "unpack", "lanes", "median_time_ms",
        "speedup_vs_fp64", "cv_percent",
    ]
    winners_path = args.output_dir / "case_winners.csv"
    with winners_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=winner_fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(winners)

    ranking_groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        if row["format"] != BASELINE_FORMAT:
            ranking_groups[
                (
                    str(row["benchmark_format"]),
                    str(row["component"]),
                    str(row["distribution"]),
                    str(row["strategy"]),
                )
            ].append(row)
    rankings: list[dict[str, object]] = []
    for (format_name, component, distribution_name, strategy_name), group in ranking_groups.items():
        rankings.append(
            {
                "format": format_name,
                "component": component,
                "distribution": distribution_name,
                "strategy": strategy_name,
                "lanes": group[0]["lanes"],
                "case_count": len(group),
                "geomean_speedup_vs_fp64": geometric_mean(
                    [float(row["speedup_vs_fp64"]) for row in group]
                ),
                "mean_rank_within_format": statistics.fmean(
                    int(row["rank_within_format"]) for row in group
                ),
                "wins": sum(int(row["rank_within_format"]) == 1 for row in group),
                "median_cv_percent": statistics.median(
                    float(row["cv_percent"]) for row in group
                ),
            }
        )
    rankings.sort(
        key=lambda row: (
            str(row["format"]), str(row["component"]), str(row["distribution"]),
            -float(row["geomean_speedup_vs_fp64"]),
        )
    )
    ranking_path = args.output_dir / "strategy_rankings.csv"
    with ranking_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rankings[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rankings)

    print(
        f"Validated {len(rows)} samples in {len(grouped)} groups across "
        f"{len(formats)} formats; {expected_sample_count} samples per variant"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {inventory_path}")
    print(f"Wrote {winners_path}")
    print(f"Wrote {ranking_path}")


if __name__ == "__main__":
    main()
