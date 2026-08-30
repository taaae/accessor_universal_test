from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "tools" / "analyze_dyadic_normal32.py"
SPEC = importlib.util.spec_from_file_location("dyadic_normal32_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_summary() -> list[dict[str, float | str]]:
    result = []
    for strategy_index, variant in enumerate(MODULE.DYADIC_VARIANTS):
        for target in MODULE.TARGETS:
            median = 0.28 + 0.01 * strategy_index + 0.03 * target
            result.append(
                {
                    "variant": variant,
                    "actual_x": target,
                    "median_ms": median,
                    "q1_ms": median - 0.002,
                    "q3_ms": median + 0.002,
                }
            )
    return result


def test_svg_has_direct_labels_baseline_and_genuine_marker() -> None:
    baselines = {
        "raw_fp64": {
            "label": "Raw FP64",
            "median_ms": 0.31,
            "q1_ms": 0.309,
            "q3_ms": 0.311,
        },
        "raw_fp32": {
            "label": "Raw FP32",
            "median_ms": 0.24,
            "q1_ms": 0.239,
            "q3_ms": 0.241,
        },
        "fp32_to_fp64": {
            "label": "FP32 to FP64",
            "median_ms": 0.25,
            "q1_ms": 0.249,
            "q3_ms": 0.251,
        },
    }
    genuine = [
        {
            "variant": variant,
            "target_x": 0.1001849,
            "actual_x": 0.1002,
            "median_ms": 0.295 + 0.01 * index,
            "q1_ms": 0.294 + 0.01 * index,
            "q3_ms": 0.296 + 0.01 * index,
        }
        for index, variant in enumerate(MODULE.DYADIC_VARIANTS)
    ]
    svg = MODULE.make_svg(synthetic_summary(), baselines, genuine)
    assert "Current DyadicNormal32" in svg
    assert "Sign-fused decoder" in svg
    assert "BitCast, shared coefficients" in svg
    assert "BitCast, constant coefficients" in svg
    assert "Raw FP64" in svg
    assert "Raw FP32" in svg
    assert "FP32 to FP64" in svg
    assert "genuine N(0,1)" in svg
    assert "stroke-dasharray" in svg
    assert "Normalized segment-table dispersion X" in svg
def test_full_execution_order_is_contiguous() -> None:
    orders = {0, 1, 2, 83, 84, 85}
    for variant in MODULE.DYADIC_VARIANTS:
        for target_index, _ in enumerate(MODULE.TARGETS):
            orders.add(MODULE.expected_dyadic_order(
                variant, "hot_uniform", "forward", target_index
            ))
            orders.add(MODULE.expected_dyadic_order(
                variant, "hot_uniform", "reverse", target_index
            ))
        orders.add(MODULE.expected_dyadic_order(
            variant, "genuine_n01", "forward", None
        ))
        orders.add(MODULE.expected_dyadic_order(
            variant, "genuine_n01", "reverse", None
        ))
    assert orders == set(range(86))


def test_coefficient_validator_checks_terminal_and_linear_map(tmp_path: Path) -> None:
    path = tmp_path / "coefficients.csv"
    fields = [
        "segment", "lower_boundary", "upper_boundary", "payload_bits", "levels",
        "linear_start", "linear_step", "bitcast_offset", "bitcast_span",
    ]
    rows = []
    for segment in range(32):
        payload_bits = 30 - segment if segment < 30 else 0
        levels = 1 << payload_bits if segment < 31 else 1
        step = 1 / levels if segment < 31 else 0
        start = segment + 0.5 * step if segment < 31 else 32
        span = step * levels
        rows.append({
            "segment": segment,
            "lower_boundary": segment,
            "upper_boundary": segment + 1,
            "payload_bits": payload_bits,
            "levels": levels,
            "linear_start": start,
            "linear_step": step,
            "bitcast_offset": start - span if segment < 31 else start,
            "bitcast_span": span,
        })
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    MODULE.validate_coefficients(path)

    rows[-1]["bitcast_offset"] = -456
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        MODULE.validate_coefficients(path)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed terminal coefficient was accepted")
