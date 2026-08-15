#!/usr/bin/env python3
"""Select LNS finalists independently by arithmetic and access family."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--finalists", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--top", type=int, default=2)
    args = parser.parse_args()

    with args.samples.open(newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["valid"] == "1"]
    if not rows:
        raise SystemExit("screen CSV has no valid rows")

    samples: dict[tuple[str, ...], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = (
            row["format"],
            row["arithmetic_type"],
            row["multiply_method"],
            row["kernel"],
            row["distribution"],
            row["strategy_id"],
        )
        samples[key].append(float(row["mean_ms"]))
        metadata[key] = row

    collapsed: dict[tuple[str, ...], list[float]] = defaultdict(list)
    representative: dict[tuple[str, ...], dict[str, str]] = {}
    for key, times in samples.items():
        without_distribution = key[:4] + (key[5],)
        collapsed[without_distribution].append(median(times))
        representative[without_distribution] = metadata[key]

    groups: dict[tuple[str, ...], list[tuple[float, tuple[str, ...]]]] = defaultdict(list)
    for key, times in collapsed.items():
        row = representative[key]
        group = (
            row["format"],
            row["arithmetic_type"],
            row["multiply_method"],
            row["kernel"],
            row["storage_layout"],
            row["access_method"],
            row["packet_values"],
        )
        groups[group].append((median(times), key))

    selected: set[str] = set()
    ranking_rows: list[dict[str, object]] = []
    for group, candidates in sorted(groups.items()):
        candidates.sort(key=lambda item: item[0])
        for rank, (time_ms, key) in enumerate(candidates, start=1):
            row = representative[key]
            qualified = f"{row['format']}/{row['arithmetic_type']}/{row['strategy_id']}"
            mandatory = (
                row["access_method"] == "scalar"
                and row["decoder"] == "reference_exp2"
            )
            keep = rank <= args.top or mandatory
            if keep:
                selected.add(qualified)
            ranking_rows.append(
                {
                    "format": group[0],
                    "arithmetic_type": group[1],
                    "multiply_method": group[2],
                    "kernel": group[3],
                    "storage_layout": group[4],
                    "access_method": group[5],
                    "packet_values": group[6],
                    "rank": rank,
                    "strategy_id": row["strategy_id"],
                    "decoder": row["decoder"],
                    "screen_median_ms": f"{time_ms:.17g}",
                    "selected": int(keep),
                }
            )

    args.finalists.parent.mkdir(parents=True, exist_ok=True)
    args.finalists.write_text(
        "# qualified format/arithmetic/strategy finalists\n"
        + "\n".join(sorted(selected))
        + "\n"
    )
    with args.ranking.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=ranking_rows[0].keys())
        writer.writeheader()
        writer.writerows(ranking_rows)
    print(
        f"Selected {len(selected)} unique variants from "
        f"{len(ranking_rows)} rankings"
    )


if __name__ == "__main__":
    main()
