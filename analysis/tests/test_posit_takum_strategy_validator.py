from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[2] / "tools" / "validate_posit_takum_strategy_run.py"
)
SPEC = importlib.util.spec_from_file_location("posit_takum_validator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_complete_smoke_matrix_is_accepted(tmp_path: Path) -> None:
    timed_cases: set[tuple[str, str, str, str, str]] = set()
    for format_name, bits in VALIDATOR.ALT_FORMATS.items():
        for arithmetic in VALIDATOR.ARITHMETIC:
            for distribution in VALIDATOR.DISTRIBUTIONS:
                for kernel in VALIDATOR.KERNELS:
                    for strategy in VALIDATOR.expected_strategies(bits):
                        timed_cases.add(
                            (format_name, arithmetic, distribution, kernel, strategy)
                        )
            if bits <= 14:
                for distribution in VALIDATOR.CONTROL_DISTRIBUTIONS:
                    for kernel in VALIDATOR.KERNELS:
                        for strategy in {"full_lut_global", "full_lut_shared"}:
                            timed_cases.add(
                                (
                                    format_name,
                                    arithmetic,
                                    distribution,
                                    kernel,
                                    strategy,
                                )
                            )
    for (format_name, arithmetic), (_, strategies) in VALIDATOR.IEEE_CASES.items():
        for distribution in VALIDATOR.DISTRIBUTIONS:
            for kernel in VALIDATOR.KERNELS:
                for strategy in strategies:
                    timed_cases.add(
                        (format_name, arithmetic, distribution, kernel, strategy)
                    )
    for format_name, arithmetic in {
        ("fp8_e4m3", "fp32"),
        ("fp8_e4m3", "fp64"),
        ("e5m8", "fp32"),
        ("e5m8", "fp64"),
    }:
        for distribution in VALIDATOR.CONTROL_DISTRIBUTIONS:
            for kernel in VALIDATOR.KERNELS:
                for strategy in {"full_lut_global", "full_lut_shared"}:
                    timed_cases.add(
                        (format_name, arithmetic, distribution, kernel, strategy)
                    )

    sample_rows = []
    for case in sorted(timed_cases):
        if case[0] in VALIDATOR.ALT_FORMATS:
            bits = VALIDATOR.ALT_FORMATS[case[0]]
        else:
            bits = next(
                case_bits
                for (format_name, _), (case_bits, _) in VALIDATOR.IEEE_CASES.items()
                if format_name == case[0]
            )
        def packed(count: int) -> int:
            raw_bytes = (count * bits + 7) // 8
            return ((raw_bytes + 3) // 4) * 4 + 4 if bits == 14 else raw_bytes
        input_bytes = (
            2 * packed(256)
            if case[3] == "dot"
            else packed(16 * 256) + packed(256)
        )
        for round_index in range(2):
            sample_rows.append(
                {
                    "format": case[0],
                    "arithmetic": case[1],
                    "distribution": case[2],
                    "kernel": case[3],
                    "strategy": case[4],
                    "storage_layout": "dense",
                    "access_method": "scalar",
                    "packet_values": 1,
                    "N": 256,
                    "M": 1 if case[3] == "dot" else 16,
                    "input_bytes": input_bytes,
                    "round": round_index,
                    "kernel_ms": 1.0,
                    "status": "ok",
                }
            )
    samples = tmp_path / "samples.csv"
    write_csv(samples, list(sample_rows[0]), sample_rows)

    validation_rows = []
    for format_name, bits in VALIDATOR.ALT_FORMATS.items():
        for arithmetic in VALIDATOR.ARITHMETIC:
            for strategy in VALIDATOR.expected_strategies(bits):
                validation_rows.append(
                    {
                        "format": format_name,
                        "arithmetic": arithmetic,
                        "strategy": strategy,
                        "codes_checked": 1 << bits if bits <= 16 else 1_000_000,
                        "failures": 0,
                        "reference": (
                            "takum_log_paper_formula"
                            if format_name.startswith("takum_log")
                            else "universal_cross_validated_core"
                        ),
                        "status": "pass",
                    }
                )
    for (format_name, arithmetic), (bits, strategies) in VALIDATOR.IEEE_CASES.items():
        for strategy in strategies:
            validation_rows.append(
                {
                    "format": format_name,
                    "arithmetic": arithmetic,
                    "strategy": strategy,
                    "codes_checked": 1 << bits if bits <= 16 else 1_000_000,
                    "failures": 0,
                    "reference": "cpu_format_reference",
                    "status": "pass",
                }
            )
    validation = tmp_path / "validation.csv"
    write_csv(validation, list(validation_rows[0]), validation_rows)

    histogram_keys = {
        (format_name, arithmetic, distribution)
        for format_name in VALIDATOR.ALT_FORMATS
        for arithmetic in VALIDATOR.ARITHMETIC
        for distribution in VALIDATOR.DISTRIBUTIONS
    }
    histogram_keys |= {
        (format_name, arithmetic, distribution)
        for format_name, arithmetic in VALIDATOR.IEEE_CASES
        for distribution in VALIDATOR.DISTRIBUTIONS
    }
    histogram_keys |= {
        (format_name, arithmetic, distribution)
        for format_name, bits in VALIDATOR.ALT_FORMATS.items()
        if bits <= 14
        for arithmetic in VALIDATOR.ARITHMETIC
        for distribution in VALIDATOR.CONTROL_DISTRIBUTIONS
    }
    histogram_keys |= {
        (format_name, arithmetic, distribution)
        for format_name, arithmetic in {
            ("fp8_e4m3", "fp32"),
            ("fp8_e4m3", "fp64"),
            ("e5m8", "fp32"),
            ("e5m8", "fp64"),
        }
        for distribution in VALIDATOR.CONTROL_DISTRIBUTIONS
    }
    histogram_rows = []
    for key in sorted(histogram_keys):
        if key[2] in VALIDATOR.CONTROL_DISTRIBUTIONS:
            kinds = {"raw_index_trace", "raw_index_distinct"}
        elif key[2] == "paired_log_uniform_finite":
            kinds = {"sign", "log_interval"}
        elif key[0] in VALIDATOR.ALT_FORMATS:
            kinds = (
                {"sign", "field_bucket", "regime"}
                if key[0].startswith("posit")
                else {
                    "sign",
                    "field_bucket",
                    "direction",
                    "regime",
                    "characteristic",
                }
            )
        else:
            kinds = {"sign", "field_bucket", "exponent", "subnormal"}
        for kind in kinds:
            if kind == "field_bucket" and key[0].startswith("takum"):
                counts = {str(index): 256 for index in range(16)}
            elif kind in {"sign", "direction", "field_bucket", "characteristic", "exponent"}:
                counts = {"0": 2048, "1": 2048}
            elif kind == "regime" and key[0].startswith("takum"):
                counts = {str(index): 512 for index in range(8)}
            elif kind == "regime":
                counts = {"0": 2048, "1": 2048}
            elif kind == "subnormal":
                counts = {"0": 2048, "1": 2048}
            elif kind == "raw_index_distinct":
                counts = {"all": 256}
            else:
                counts = {"all": 4096}
            for bucket, count in counts.items():
                histogram_rows.append(
                    {
                        "format": key[0],
                        "arithmetic": key[1],
                        "distribution": key[2],
                        "histogram": kind,
                        "bucket": bucket,
                        "count": count,
                        "realized_q_min": (
                            0 if kind.startswith("raw_index") else -1
                        ),
                        "realized_q_max": (
                            255 if kind.startswith("raw_index") else 1
                        ),
                    }
                )
    histograms = tmp_path / "histograms.csv"
    write_csv(histograms, list(histogram_rows[0]), histogram_rows)

    summary = VALIDATOR.validate(
        argparse.Namespace(
            mode="smoke",
            samples=samples,
            validation=validation,
            histograms=histograms,
        )
    )
    assert summary["timed_cases"] == len(timed_cases)
    assert summary["infeasible_cases"] == 0
