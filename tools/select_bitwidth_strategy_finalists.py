#!/usr/bin/env python3
"""Rank screen measurements and select per-access-group full-sweep finalists."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--finalists", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--top", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.samples.open(newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["valid"] == "1"]
    if not rows:
        raise SystemExit("screen CSV has no valid rows")

    # First collapse repeated timing samples within each distribution, then
    # rank by the median across distributions.
    samples: dict[tuple[str, ...], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = (
            row["format"],
            row["arithmetic_type"],
            row["kernel"],
            row["distribution"],
            row["strategy_id"],
        )
        samples[key].append(float(row["mean_ms"]))
        metadata[key] = row

    distribution_medians: dict[tuple[str, ...], list[float]] = defaultdict(list)
    representative: dict[tuple[str, ...], dict[str, str]] = {}
    for key, times in samples.items():
        collapsed = key[:3] + (key[4],)
        distribution_medians[collapsed].append(median(times))
        representative[collapsed] = metadata[key]

    groups: dict[tuple[str, ...], list[tuple[float, tuple[str, ...]]]] = defaultdict(list)
    for key, times in distribution_medians.items():
        row = representative[key]
        group = (
            row["format"],
            row["arithmetic_type"],
            row["kernel"],
            row["storage_layout"],
            row["access_method"],
            row["packet_values"],
        )
        groups[group].append((median(times), key))

    selected: set[str] = set()
    ranking_rows = []
    for group, candidates in sorted(groups.items()):
        candidates.sort(key=lambda item: item[0])
        for rank, (time_ms, key) in enumerate(candidates, start=1):
            row = representative[key]
            qualified = (
                f"{row['format']}/{row['arithmetic_type']}/{row['strategy_id']}"
            )
            mandatory = (
                row["access_method"] == "scalar"
                and row["decoder"] in {"generic", "direct_branchy"}
            )
            keep = rank <= args.top or mandatory
            if keep:
                selected.add(qualified)
            ranking_rows.append(
                {
                    "format": group[0],
                    "arithmetic_type": group[1],
                    "kernel": group[2],
                    "storage_layout": group[3],
                    "access_method": group[4],
                    "packet_values": group[5],
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
        f"Selected {len(selected)} unique qualified variants from "
        f"{len(ranking_rows)} screen rankings"
    )


if __name__ == "__main__":
    main()
