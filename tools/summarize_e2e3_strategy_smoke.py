#!/usr/bin/env python3

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"error: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--decoder-validation", type=Path, required=True)
    parser.add_argument("--kernel-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    samples = read_csv(args.samples)
    decoder = read_csv(args.decoder_validation)
    kernels = read_csv(args.kernel_validation)
    require(samples, "timing sample CSV is empty")
    require(decoder, "decoder validation CSV is empty")
    require(kernels, "kernel validation CSV is empty")

    for row in decoder:
        require(
            row["finite_bit_mismatches"] == "0"
            and row["classification_mismatches"] == "0",
            f"decoder mismatch in {row['format']}/{row['strategy']}",
        )
    for row in kernels:
        require(
            row["pass"] == "1",
            f"kernel mismatch in {row['format']}/{row['strategy']}/{row['component']}",
        )

    strategies = {(row["format"], row["strategy"]) for row in decoder}
    for format_name in ("e2m5", "e3m4"):
        count = sum(fmt == format_name for fmt, _ in strategies)
        require(count == 42, f"expected 42 {format_name} strategies, found {count}")

    key_fields = [
        "gpu",
        "compute_capability",
        "distribution",
        "format",
        "strategy",
        "component",
        "lanes",
        "n",
        "m",
        "lookup_entry_bytes",
        "shared_table_bytes",
        "pipelined",
    ]
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in samples:
        grouped[tuple(row[field] for field in key_fields)].append(
            float(row["time_ms"])
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "timing_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            key_fields
            + [
                "sample_count",
                "median_time_ms",
                "min_time_ms",
                "max_time_ms",
                "cv_percent",
            ]
        )
        for key in sorted(grouped):
            values = grouped[key]
            mean = statistics.fmean(values)
            cv = (
                100.0 * statistics.stdev(values) / mean
                if len(values) > 1 and mean != 0.0
                else 0.0
            )
            writer.writerow(
                list(key)
                + [len(values), statistics.median(values), min(values), max(values), cv]
            )

    inventory_path = args.output_dir / "strategy_inventory.csv"
    with inventory_path.open("w", newline="") as stream:
        fields = [
            "format",
            "strategy",
            "lanes",
            "lookup_entry_bytes",
            "shared_table_bytes",
            "pipelined",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            decoder, key=lambda item: (item["format"], item["strategy"])
        ):
            writer.writerow({field: row[field] for field in fields})

    print(
        f"Validated {len(decoder)} decoder rows, {len(kernels)} kernel rows, "
        f"and {len(samples)} timing samples in {len(grouped)} groups"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {inventory_path}")


if __name__ == "__main__":
    main()
