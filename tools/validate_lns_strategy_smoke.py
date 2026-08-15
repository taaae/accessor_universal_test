#!/usr/bin/env python3
"""Check finite LNS smoke results against matching reference-exp2 kernels."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.samples.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("smoke CSV is empty")

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["valid"] != "1":
            raise SystemExit("smoke CSV contains a non-finite kernel result")
        grouped[
            (
                row["format"],
                row["arithmetic_type"],
                row["multiply_method"],
                row["distribution"],
                row["kernel"],
                row["N"],
            )
        ].append(row)

    output_rows: list[dict[str, object]] = []
    failures = 0
    for group, candidates in sorted(grouped.items()):
        references = [row for row in candidates if row["decoder"] == "reference_exp2"]
        if not references:
            raise SystemExit(f"missing reference-exp2 row for {group}")
        preferred = next(
            (row for row in references if row["storage_layout"] == "padded"),
            references[0],
        )
        baseline = float(preferred["result"])
        tolerance = 5e-3 if group[1] == "fp32" else 2e-4
        scale = max(abs(baseline), 1.0)
        for row in candidates:
            value = float(row["result"])
            error = abs(value - baseline) / scale if math.isfinite(value) else math.inf
            passed = error <= tolerance
            failures += int(not passed)
            output_rows.append(
                {
                    "format": group[0],
                    "arithmetic_type": group[1],
                    "multiply_method": group[2],
                    "distribution": group[3],
                    "kernel": group[4],
                    "N": group[5],
                    "strategy_id": row["strategy_id"],
                    "reference_strategy": preferred["strategy_id"],
                    "scaled_error": f"{error:.17g}",
                    "tolerance": f"{tolerance:.17g}",
                    "passed": int(passed),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    if failures:
        raise SystemExit(f"{failures} LNS smoke comparisons failed")
    print(f"Validated {len(output_rows)} LNS smoke comparisons")


if __name__ == "__main__":
    main()
