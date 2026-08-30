from __future__ import annotations

import importlib.util
import math
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
    for target in MODULE.TARGETS:
        median = 0.28 + 0.08 * target
        result.append(
            {
                "variant": "dyadic_normal32",
                "actual_x": target,
                "median_ms": median,
                "q1_ms": median - 0.002,
                "q3_ms": median + 0.002,
            }
        )
    return result


def test_svg_has_direct_labels_baseline_and_genuine_marker() -> None:
    raw = {"median_ms": 0.31, "q1_ms": 0.309, "q3_ms": 0.311}
    genuine = {
        "target_x": 0.1001849,
        "actual_x": 0.1002,
        "median_ms": 0.295,
        "q1_ms": 0.294,
        "q3_ms": 0.296,
    }
    svg = MODULE.make_svg(synthetic_summary(), raw, genuine)
    assert "DyadicNormal32" in svg
    assert "Raw FP64" in svg
    assert "genuine N(0,1)" in svg
    assert "stroke-dasharray" in svg
    assert "Normalized segment-table dispersion X" in svg


def test_crossover_interpolation() -> None:
    crossings = MODULE.crossover_points(synthetic_summary(), 0.31)
    assert len(crossings) == 1
    assert math.isclose(crossings[0], 0.375, abs_tol=1e-12)
