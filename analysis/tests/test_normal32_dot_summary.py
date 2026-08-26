import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.summarize_normal32_dot_benchmark as normal32_summary


def test_summary_keeps_both_e9m22_strategies(tmp_path):
    samples = tmp_path / "samples.csv"
    fields = [
        "format",
        "strategy_id",
        "valid",
        "kernel",
        "arithmetic_type",
        "access_method",
        "packet_values",
        "N",
        "mean_ms",
        "bits",
        "physical_input_bytes",
        "logical_input_bytes",
    ]
    rows = []
    for variant_index, (format_name, strategy) in enumerate(
        normal32_summary.EXPECTED_VARIANTS
    ):
        for sample in range(2):
            rows.append(
                {
                    "format": format_name,
                    "strategy_id": strategy,
                    "valid": "1",
                    "kernel": "dot",
                    "arithmetic_type": "fp64",
                    "access_method": "scalar",
                    "packet_values": "1",
                    "N": "1024",
                    "mean_ms": str(10.0 + variant_index + sample),
                    "bits": "64" if format_name == "raw_fp64" else "32",
                    "physical_input_bytes": "8192",
                    "logical_input_bytes": "8192",
                }
            )
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = normal32_summary.summarize(samples)
    keys = {(row["format"], row["strategy_id"]) for row in summary}
    assert keys == set(normal32_summary.EXPECTED_VARIANTS)
    assert ("e9m22", "prefix_global_x1") in keys
    assert ("e9m22", "word_branchy_x1") in keys
    assert all(row["samples"] == 2 for row in summary)


def test_summary_rejects_non_scalar_rows(tmp_path):
    samples = tmp_path / "samples.csv"
    fields = [
        "format",
        "strategy_id",
        "valid",
        "kernel",
        "arithmetic_type",
        "access_method",
        "packet_values",
        "N",
        "mean_ms",
        "bits",
        "physical_input_bytes",
        "logical_input_bytes",
    ]
    rows = []
    for format_name, strategy in normal32_summary.EXPECTED_VARIANTS:
        rows.append(
            {
                "format": format_name,
                "strategy_id": strategy,
                "valid": "1",
                "kernel": "dot",
                "arithmetic_type": "fp64",
                "access_method": "thread_packet",
                "packet_values": "4",
                "N": "1024",
                "mean_ms": "1.0",
                "bits": "32",
                "physical_input_bytes": "8192",
                "logical_input_bytes": "8192",
            }
        )
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    try:
        normal32_summary.summarize(samples)
    except ValueError as error:
        assert "scalar x1" in str(error)
    else:
        raise AssertionError("summary accepted a non-scalar experiment")
