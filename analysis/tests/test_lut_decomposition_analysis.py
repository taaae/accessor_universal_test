from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "tools" / "analyze_lut_decomposition.py"
SPEC = importlib.util.spec_from_file_location("lut_decomposition_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_svg_has_direct_labels_and_baseline() -> None:
    graph = "g4_u16_split_fp32"
    rows = []
    for variant_index, variant in enumerate(
        [item for item in MODULE.VARIANTS if item.graph == graph]
    ):
        for target in MODULE.TARGETS:
            median = 0.25 + 0.2 * target + 0.02 * variant_index
            rows.append(
                {
                    "graph_id": graph,
                    "variant": variant.identifier,
                    "actual_x": target,
                    "median_ms": median,
                    "q1_ms": median - 0.002,
                    "q3_ms": median + 0.002,
                }
            )
    baseline = {
        "label": "Raw FP32",
        "median_ms": 0.265,
        "q1_ms": 0.264,
        "q3_ms": 0.266,
    }
    svg = MODULE.make_svg(graph, rows, baseline)
    assert "Normalized lookup dispersion" in svg
    assert "Raw FP32" in svg
    assert "stroke-dasharray" in svg
    assert "2 x 8-bit shared LUTs" in svg
    assert "4 x 4-bit shared LUTs" in svg


def test_endpoint_ratio_uses_measured_x_order() -> None:
    rows = [
        {"variant": "case", "actual_x": 1.0, "median_ms": 0.6},
        {"variant": "case", "actual_x": 0.0, "median_ms": 0.2},
    ]
    assert math.isclose(MODULE.ratio_at_endpoints(rows, "case"), 3.0)
