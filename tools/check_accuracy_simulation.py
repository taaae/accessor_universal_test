#!/usr/bin/env python3
"""Validate GPU accuracy-simulation outputs and report Monte Carlo convergence."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


COMPARISONS = {
    "storage",
    "kernel_x1",
    "kernel_x2",
    "kernel_x4",
    "total_x1",
    "total_x2",
    "total_x4",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def identity(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(
        row[key]
        for key in ("kernel", "distribution", "format", "storage_bits", "n", "m")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--relative-se-target", type=float, default=0.05)
    args = parser.parse_args()
    directory = args.output_dir

    summary = rows(directory / "simulation_summary.csv")
    quantiles = rows(directory / "empirical_quantiles.csv")
    batches = rows(directory / "batch_estimates.csv")
    encodings = rows(directory / "encoding_stats.csv")
    seeds = rows(directory / "generation_seeds.csv")
    self_test = rows(directory / "reference_self_test.csv")
    if not summary or not quantiles or not batches or not encodings or not seeds:
        raise SystemExit("one or more simulation tables are empty")
    if not self_test or any(row["status"] != "pass" for row in self_test):
        raise SystemExit("GPU double-double reference self-test did not pass")

    comparisons: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in summary:
        comparisons[identity(row)].add(row["comparison"])
    incomplete = {
        case: values for case, values in comparisons.items() if values != COMPARISONS
    }
    if incomplete:
        raise SystemExit(f"incomplete comparison sets: {incomplete}")
    expected_comparisons = {
        case + (comparison,)
        for case in comparisons
        for comparison in COMPARISONS
    }

    quantile_counts = Counter(
        identity(row) + (row["comparison"],) for row in quantiles
    )
    if set(quantile_counts) != expected_comparisons or any(
        count != 85 for count in quantile_counts.values()
    ):
        raise SystemExit("each comparison must contain 5 metrics x 17 quantiles")

    batch_counts = Counter(identity(row) + (row["comparison"],) for row in batches)
    if set(batch_counts) != expected_comparisons:
        raise SystemExit("batch table does not cover every comparison")
    for row in summary:
        key = identity(row) + (row["comparison"],)
        if batch_counts[key] != int(row["statistical_batches"]):
            raise SystemExit(f"batch count mismatch for {key}")

    encoding_counts = Counter(identity(row) for row in encodings)
    if set(encoding_counts) != set(comparisons) or any(
        count != 2 for count in encoding_counts.values()
    ):
        raise SystemExit("each case must have exactly two encoding-stat rows")

    dataset_cases = {
        (case[0], case[1], case[4], case[5]) for case in comparisons
    }
    seeded_cases = {
        (row["kernel"], row["distribution"], row["n"], row["m"])
        for row in seeds
    }
    if seeded_cases != dataset_cases:
        raise SystemExit("seed table does not cover every generated dataset")

    convergence_path = directory / "convergence_report.csv"
    review_count = 0
    applicable_count = 0
    with convergence_path.open("w", newline="", encoding="utf-8") as output:
        fieldnames = [
            "kernel",
            "distribution",
            "format",
            "storage_bits",
            "n",
            "m",
            "comparison",
            "finite_pairs",
            "mse",
            "mse_relative_cluster_standard_error",
            "target",
            "status",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            finite_pairs = int(row["finite_pairs"])
            mse = float(row["mse"])
            relative_se = float(row["mse_relative_cluster_standard_error"])
            if finite_pairs == 0 or not math.isfinite(mse):
                status = "nonfinite_case"
            elif mse == 0.0:
                status = "exact_zero"
            elif not math.isfinite(relative_se):
                status = "insufficient_finite_batches"
                review_count += 1
                applicable_count += 1
            elif relative_se <= args.relative_se_target:
                status = "pass"
                applicable_count += 1
            else:
                status = "review"
                review_count += 1
                applicable_count += 1
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames[:7]},
                    "finite_pairs": finite_pairs,
                    "mse": row["mse"],
                    "mse_relative_cluster_standard_error": row[
                        "mse_relative_cluster_standard_error"
                    ],
                    "target": args.relative_se_target,
                    "status": status,
                }
            )

    print(
        f"Validated {len(comparisons)} cases, {len(summary)} comparisons, "
        f"{len(quantiles)} quantile rows, and {len(batches)} batch rows."
    )
    print(
        f"MSE convergence target: {args.relative_se_target:.1%}; "
        f"review={review_count}/{applicable_count}."
    )
    print(f"Wrote {convergence_path}")


if __name__ == "__main__":
    main()
