#!/usr/bin/env python3
"""Validate that every smoke strategy agrees with its generic scalar result."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.samples.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("smoke CSV is empty")

    values: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["valid"] != "1":
            raise SystemExit("smoke CSV contains an incomplete kernel row")
        group = (
            row["format"],
            row["arithmetic_type"],
            row["distribution"],
            row["kernel"],
            row["N"],
        )
        values[group][row["strategy_id"]].append(float(row["result"]))
    output_rows = []
    failures = []
    for group, strategies in sorted(values.items()):
        if group[0].startswith("raw_"):
            baseline_id = "natural/scalar/x1/raw"
        else:
            preferred = "padded/scalar/x1/generic"
            fallback = "dense/scalar/x1/generic"
            baseline_id = preferred if preferred in strategies else fallback
        if baseline_id not in strategies:
            raise SystemExit(f"missing scalar baseline for {group}")
        baseline = strategies[baseline_id][0]
        arithmetic = group[1]
        tolerance = 5.0e-3 if arithmetic == "fp32" else 1.0e-9
        scale = max(abs(baseline), 1.0)
        for strategy, samples in sorted(strategies.items()):
            value = samples[0]
            if math.isnan(baseline):
                relative = math.nan
                passed = math.isnan(value)
            elif math.isinf(baseline):
                relative = 0.0 if value == baseline else math.inf
                passed = value == baseline
            elif not math.isfinite(value):
                relative = math.inf
                passed = False
            else:
                relative = abs(value - baseline) / scale
                passed = relative <= tolerance
            output_rows.append(
                {
                    "format": group[0],
                    "arithmetic_type": arithmetic,
                    "distribution": group[2],
                    "kernel": group[3],
                    "N": group[4],
                    "strategy_id": strategy,
                    "baseline_strategy": baseline_id,
                    "result": f"{value:.17g}",
                    "baseline_result": f"{baseline:.17g}",
                    "scaled_absolute_error": f"{relative:.17g}",
                    "tolerance": f"{tolerance:.17g}",
                    "passed": int(passed),
                }
            )
            if not passed:
                failures.append(output_rows[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    if failures:
        first = failures[0]
        raise SystemExit(
            f"{len(failures)} smoke comparisons failed; first: "
            f"{first['format']} {first['arithmetic_type']} "
            f"{first['kernel']} {first['strategy_id']}"
        )
    print(f"Validated {len(output_rows)} smoke strategy comparisons")


if __name__ == "__main__":
    main()
