import csv
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.summarize_all_format_strategy_benchmark as summary


def test_multi_format_inventory_baselines_and_rankings(tmp_path, monkeypatch):
    samples = tmp_path / "samples.csv"
    validation = tmp_path / "validation.csv"
    formats = {"e1m6": [("generic_x1", "1"), ("fast_x4", "4")],
               "e11m20": [("generic_x1", "1"), ("prefix_word_x8", "8")]}

    with validation.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["format", "strategy", "lanes", "mismatches"]
        )
        writer.writeheader()
        for format_name, strategies in formats.items():
            for strategy, lanes in strategies:
                writer.writerow(
                    {
                        "format": format_name,
                        "strategy": strategy,
                        "lanes": lanes,
                        "mismatches": "0",
                    }
                )

    fields = summary.IDENTITY + [
        "round",
        "sample",
        "order_position",
        "iterations",
        "total_time_ms",
        "time_ms",
        "main_array_unique_gb_per_s",
        "main_array_requested_gb_per_s",
        "useful_gflop_per_s",
    ]
    rows = []
    for benchmark_format, strategies in formats.items():
        variants = [
            (summary.BASELINE_FORMAT, summary.BASELINE_STRATEGY, "1", 8.0),
            *((benchmark_format, strategy, lanes, 4.0)
              for strategy, lanes in strategies),
        ]
        for distribution in ("uniform_0_1", "normal_0_1"):
            for component, n, m in (("dot", "1024", "1"),
                                    ("gemv", "256", "1024")):
                for position, (format_name, strategy, lanes, default_time) in enumerate(variants):
                    time_ms = 2.0 if strategy in {"fast_x4", "prefix_word_x8"} else default_time
                    for round_index in range(2):
                        for sample_index in range(2):
                            identity = {
                                "gpu": "test_gpu",
                                "compute_capability": "sm_90",
                                "distribution": distribution,
                                "benchmark_format": benchmark_format,
                                "format": format_name,
                                "storage_bits": "64" if format_name == summary.BASELINE_FORMAT else "8",
                                "strategy": strategy,
                                "decode_kind": "none",
                                "table_location": "none",
                                "unpack": "shift_mask",
                                "lanes": lanes,
                                "lookup_entry_bytes": "0",
                                "shared_table_bytes": "0",
                                "component": component,
                                "n": n,
                                "m": m,
                                "blocks": m if component == "gemv" else "16",
                                "threads": "256",
                                "warmup": "10",
                                "main_array_unique_bytes": "1024",
                                "main_array_requested_bytes": "2048",
                                "useful_flops": "2048",
                            }
                            rows.append(
                                identity
                                | {
                                    "round": str(round_index),
                                    "sample": str(sample_index),
                                    "order_position": str(position),
                                    "iterations": "10",
                                    "total_time_ms": str(time_ms * 10),
                                    "time_ms": str(time_ms),
                                    "main_array_unique_gb_per_s": "1",
                                    "main_array_requested_gb_per_s": "2",
                                    "useful_gflop_per_s": "2",
                                }
                            )
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summary",
            "--samples",
            str(samples),
            "--validation-inventory",
            str(validation),
            "--output-dir",
            str(tmp_path),
            "--expected-formats",
            ",".join(formats),
            "--expected-dot-powers",
            "10",
            "--expected-gemv-powers",
            "8",
            "--expected-gemv-rows",
            "1024",
            "--expected-rounds",
            "2",
            "--expected-samples",
            "2",
        ],
    )
    summary.main()

    with (tmp_path / "case_winners.csv").open(newline="") as stream:
        winners = list(csv.DictReader(stream))
    assert len(winners) == 8
    assert {row["strategy"] for row in winners} == {"fast_x4", "prefix_word_x8"}
    assert {float(row["speedup_vs_fp64"]) for row in winners} == {4.0}

    with (tmp_path / "strategy_rankings.csv").open(newline="") as stream:
        rankings = list(csv.DictReader(stream))
    fastest = [row for row in rankings if row["strategy"] in {"fast_x4", "prefix_word_x8"}]
    assert len(fastest) == 8
    assert {int(row["wins"]) for row in fastest} == {1}
