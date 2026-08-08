import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.summarize_decoder_optimization as summary


def test_decoder_validation_requires_every_exact_strategy(tmp_path):
    path = tmp_path / "validation.csv"
    fields = [
        "format",
        "strategy",
        "lanes",
        "finite_bit_mismatches",
        "classification_mismatches",
        "max_finite_abs_error",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for format_name in ("e2m5", "e3m4"):
            for strategy in sorted(summary.DECODER_STRATEGIES[format_name]):
                writer.writerow(
                    {
                        "format": format_name,
                        "strategy": strategy,
                        "lanes": strategy.rsplit("x", 1)[-1],
                        "finite_bit_mismatches": 0,
                        "classification_mismatches": 0,
                        "max_finite_abs_error": 0,
                    }
                )
    summary.validate_decoder_results(path)

    rows = list(csv.DictReader(path.open(newline="")))
    rows[0]["finite_bit_mismatches"] = "1"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit, match="finite decoder mismatch"):
        summary.validate_decoder_results(path)


def test_strategy_speedups_use_width_matched_current_baseline(tmp_path):
    rows = []
    times = {
        ("fp64_e11m52", "current_x1"): 8.0,
        ("fp64_e11m52", "current_x4"): 4.0,
        ("e2m5", "current_x1"): 6.0,
        ("e2m5", "current_x4"): 5.0,
        ("e2m5", "branchless_x4"): 2.5,
        ("e2m5", "lut_x1"): 3.0,
        ("e2m5", "lut_x4"): 2.0,
    }
    for (format_name, strategy), time_ms in times.items():
        rows.append(
            {
                "distribution": "normal_0_1",
                "component": "dot",
                "n": "1024",
                "m": "1",
                "format": format_name,
                "strategy": strategy,
                "storage_bits": "64" if format_name.startswith("fp64") else "8",
                "lanes": strategy.rsplit("x", 1)[-1],
                "median_time_ms": time_ms,
                "cv_percent": 0.1,
            }
        )
    output = tmp_path / "speedups.csv"
    summary.write_speedups(rows, output)
    indexed = {
        (row["format"], row["strategy"]): row
        for row in csv.DictReader(output.open(newline=""))
    }
    branchless = indexed[("e2m5", "branchless_x4")]
    assert float(branchless["speedup_vs_fp64_x4"]) == 1.6
    assert float(branchless["speedup_vs_same_format_current_width"]) == 2.0
    lut_x1 = indexed[("e2m5", "lut_x1")]
    assert float(lut_x1["speedup_vs_same_format_current_width"]) == 2.0
