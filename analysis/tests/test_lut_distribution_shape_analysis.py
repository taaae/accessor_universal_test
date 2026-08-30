from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "tools" / "analyze_lut_distribution_shape.py"
SPEC = importlib.util.spec_from_file_location("lut_distribution_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_and_svg(tmp_path: Path) -> None:
    samples = tmp_path / "timing_samples.csv"
    metrics = tmp_path / "access_metrics.csv"
    fields = [
        "mode", "kernel", "N", "storage_bits", "arithmetic_type",
        "access_method", "packet_values", "lut_entries", "lut_bytes", "q",
        "q_eighths", "mean_unique_left", "mean_unique_right", "mean_unique_both",
        "normalized_sector_dispersion", "format", "sanitized_lut_entries",
        "warmup", "sample", "execution_order", "kernel_ms", "result",
    ]
    with samples.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for q in range(9):
            x = q / 8
            mean_unique = 1 + x * (MODULE.UNIFORM_UNIQUE - 1)
            for fmt_index, fmt in enumerate(MODULE.FORMATS):
                for sample in range(50):
                    writer.writerow(
                        {
                            "mode": "full",
                            "kernel": "dot",
                            "N": MODULE.CANONICAL_N,
                            "storage_bits": 16,
                            "arithmetic_type": "fp32",
                            "access_method": "scalar",
                            "packet_values": 1,
                            "lut_entries": 65536,
                            "lut_bytes": 262144,
                            "q": q / 8,
                            "q_eighths": q,
                            "mean_unique_left": mean_unique,
                            "mean_unique_right": mean_unique,
                            "mean_unique_both": mean_unique,
                            "format": fmt,
                            "sanitized_lut_entries": MODULE.EXPECTED_SANITIZED[fmt],
                            "warmup": MODULE.CANONICAL_WARMUP,
                            "sample": sample,
                            "execution_order": fmt_index,
                            "normalized_sector_dispersion": x,
                            "kernel_ms": 1.0 + 0.1 * q + 0.001 * fmt_index + 0.0001 * sample,
                            "result": 1.0,
                        }
                    )
    metric_fields = [
        "q", "q_eighths", "N", "left_seed", "right_seed", "hot_sector_base",
        "mean_unique_left", "mean_unique_right", "mean_unique_both",
        "uniform_mean_unique", "normalized_sector_dispersion",
    ]
    with metrics.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=metric_fields)
        writer.writeheader()
        for q in range(9):
            x = q / 8
            mean_unique = 1 + x * (MODULE.UNIFORM_UNIQUE - 1)
            writer.writerow(
                {
                    "q": x,
                    "q_eighths": q,
                    "N": MODULE.CANONICAL_N,
                    "left_seed": MODULE.CANONICAL_LEFT_SEED,
                    "right_seed": MODULE.CANONICAL_RIGHT_SEED,
                    "hot_sector_base": 0,
                    "mean_unique_left": mean_unique,
                    "mean_unique_right": mean_unique,
                    "mean_unique_both": mean_unique,
                    "uniform_mean_unique": MODULE.UNIFORM_UNIQUE,
                    "normalized_sector_dispersion": x,
                }
            )
    rows = MODULE.read_rows(samples)
    metric_rows = MODULE.read_rows(metrics)
    MODULE.validate_contract(rows, metric_rows, 50)
    summary = MODULE.summarize(rows, 50)
    assert len(summary) == 27
    svg = MODULE.make_svg(summary)
    assert "Normalized LUT-sector dispersion" in svg
    assert "posit&lt;16,1&gt;" in svg
    assert "lns&lt;16,11&gt;" in svg


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_summary_and_svg(Path(directory))
    print("lut distribution analysis test passed")
