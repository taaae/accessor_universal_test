import csv
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "summarize_t16_dot_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_t16", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summarize_five_formats(tmp_path: Path) -> None:
    samples = tmp_path / "samples.csv"
    fieldnames = [
        "format",
        "valid",
        "N",
        "mean_ms",
        "physical_input_bytes",
        "logical_input_bytes",
        "bits",
        "strategy_id",
    ]
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, bits, strategy, time in [
            ("t16", 16, "global_lut_x1", 2.0),
            ("fp16_e5m10", 16, "native_scalar_x1", 1.0),
            ("e6m9", 16, "direct_branchy_x1", 3.0),
            ("e8m15", 24, "direct_shift_x1", 1.5),
            ("raw_fp32", 32, "raw_x1", 2.0),
        ]:
            for sample_time in [time, time]:
                writer.writerow(
                    {
                        "format": name,
                        "valid": 1,
                        "N": 16,
                        "mean_ms": sample_time,
                        "physical_input_bytes": 128,
                        "logical_input_bytes": 128,
                        "bits": bits,
                        "strategy_id": strategy,
                    }
                )

    summary = MODULE.summarize(samples)
    by_format = {row["format"]: row for row in summary}
    assert by_format["t16"]["median_ms"] == 2.0
    assert by_format["fp16_e5m10"]["speedup_vs_raw_fp32"] == 2.0
    assert by_format["raw_fp32"]["speedup_vs_raw_fp32"] == 1.0
