from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[2]
    / "tools"
    / "analyze_posit_takum_strategy_results.py"
)
SPEC = importlib.util.spec_from_file_location("posit_takum_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def write_samples(path: Path) -> None:
    fields = [
        "format",
        "family",
        "bits",
        "arithmetic",
        "distribution",
        "kernel",
        "strategy",
        "round",
        "kernel_ms",
        "status",
    ]
    rows = []
    for round_index, jitter in enumerate((0.99, 1.0, 1.01, 1.02)):
        for strategy, time in (("direct", 2.0), ("full_lut_global", 1.0)):
            rows.append(
                {
                    "format": "posit8_es0",
                    "family": "posit",
                    "bits": "8",
                    "arithmetic": "fp32",
                    "distribution": "field_balanced_finite",
                    "kernel": "dot",
                    "strategy": strategy,
                    "round": round_index,
                    "kernel_ms": time * jitter,
                    "status": "ok",
                }
            )
        for format_name, family, time in (
            ("posit8_es0", "posit", 1.0),
            ("fp8_e4m3", "ieee", 1.01),
        ):
            rows.append(
                {
                    "format": format_name,
                    "family": family,
                    "bits": "8",
                    "arithmetic": "fp32",
                    "distribution": "lut_scattered_control",
                    "kernel": "dot",
                    "strategy": "full_lut_global",
                    "round": round_index,
                    "kernel_ms": time * jitter,
                    "status": "ok",
                }
            )
        rows.append(
            {
                "format": "fp8_e4m3",
                "family": "ieee",
                "bits": "8",
                "arithmetic": "fp32",
                "distribution": "field_balanced_finite",
                "kernel": "dot",
                "strategy": "native_scalar",
                "round": round_index,
                "kernel_ms": 0.9 * jitter,
                "status": "ok",
            }
        )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_analysis_is_deterministic_and_uses_matching_interval_methods(
    tmp_path: Path,
) -> None:
    samples = tmp_path / "samples.csv"
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_samples(samples)

    ANALYSIS.analyze(samples, first)
    ANALYSIS.analyze(samples, second)

    assert (first / "case_comparisons.csv").read_bytes() == (
        second / "case_comparisons.csv"
    ).read_bytes()
    with (first / "case_comparisons.csv").open(newline="") as source:
        comparisons = list(csv.DictReader(source))
    assert {row["question"] for row in comparisons} == {"1", "2", "4"}
    assert {
        row["interval_method"] for row in comparisons if row["question"] != "4"
    } == {"paired_round_bootstrap"}
    assert {
        row["interval_method"] for row in comparisons if row["question"] == "4"
    } == {"independent_bootstrap"}
