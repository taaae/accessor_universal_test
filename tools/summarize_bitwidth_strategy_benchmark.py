#!/usr/bin/env python3
"""Collapse raw arbitrary-width timing samples into robust summary rows."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.samples.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("timing CSV is empty")

    excluded = {"sample", "total_ms", "mean_ms", "result", "valid"}
    key_fields = [field for field in rows[0] if field not in excluded]
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)

    output_rows = []
    for key, samples in sorted(groups.items()):
        record = dict(zip(key_fields, key))
        times = [float(row["mean_ms"]) for row in samples]
        record.update(
            sample_count=len(times),
            median_ms=f"{median(times):.17g}",
            minimum_ms=f"{min(times):.17g}",
            maximum_ms=f"{max(times):.17g}",
            all_valid=int(all(row["valid"] == "1" for row in samples)),
        )
        output_rows.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    invalid = sum(row["all_valid"] != 1 for row in output_rows)
    if invalid:
        raise SystemExit(f"{invalid} timing groups contain invalid results")
    print(f"Summarized {len(rows)} samples into {len(output_rows)} groups")


if __name__ == "__main__":
    main()
