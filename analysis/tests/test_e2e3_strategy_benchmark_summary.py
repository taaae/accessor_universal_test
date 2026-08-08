import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.summarize_e2e3_strategy_benchmark as summary


def test_full_inventory_and_fp64_speedup(tmp_path, monkeypatch):
    samples = tmp_path / "samples.csv"
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
    variants = [("fp64", "raw_pointer_x1")]
    variants += [
        (format_name, strategy)
        for format_name in ("e2m5", "e3m4")
        for strategy in sorted(summary.EXPECTED_STRATEGIES)
    ]
    for distribution in ("uniform_0_1", "normal_0_1"):
        for component, n, m in (("dot", "1024", "1"), ("gemv", "256", "1024")):
            for position, (format_name, strategy) in enumerate(variants):
                lanes = strategy.rsplit("x", 1)[-1]
                time_ms = 8.0 if format_name == "fp64" else 4.0
                if strategy == "lut_prefix_shared_x4":
                    time_ms = 2.0
                for sample in range(2):
                    identity = {
                        "gpu": "test_gpu",
                        "compute_capability": "sm_90",
                        "distribution": distribution,
                        "format": format_name,
                        "storage_bits": "64" if format_name == "fp64" else "8",
                        "strategy": strategy,
                        "decode_kind": "none",
                        "table_location": "none",
                        "lanes": lanes,
                        "lookup_entry_bytes": "0",
                        "shared_table_bytes": "0",
                        "pipelined": "0",
                        "component": component,
                        "n": n,
                        "m": m,
                        "blocks": m if component == "gemv" else "16",
                        "threads": "256",
                        "warmup": "10",
                        "main_array_unique_bytes": "1024",
                        "main_array_requested_bytes": "1024",
                        "useful_flops": "2048",
                    }
                    rows.append(
                        identity
                        | {
                            "round": "0",
                            "sample": str(sample),
                            "order_position": str(position),
                            "iterations": "10",
                            "total_time_ms": str(time_ms * 10),
                            "time_ms": str(time_ms),
                            "main_array_unique_gb_per_s": "1",
                            "main_array_requested_gb_per_s": "1",
                            "useful_gflop_per_s": "1",
                        }
                    )
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(
        sys,
        "argv",
        ["summary", "--samples", str(samples), "--output-dir", str(tmp_path)],
    )
    summary.main()

    with (tmp_path / "case_winners.csv").open(newline="") as stream:
        winners = list(csv.DictReader(stream))
    assert len(winners) == 8
    assert {row["strategy"] for row in winners} == {"lut_prefix_shared_x4"}
    assert {float(row["speedup_vs_fp64"]) for row in winners} == {4.0}
