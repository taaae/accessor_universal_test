from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "tools" / "analyze_compander32_benchmark.py"
SPEC = importlib.util.spec_from_file_location("compander32_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_filtered_ranking_stops_at_raw_fp64() -> None:
    target_n = 1 << 28
    rows = [
        {
            "format": "int32",
            "label": "Int32",
            "group": "new",
            "N": target_n,
            "median_ms": 1.01,
            "ratio_to_fp32_to_fp64": 1.00,
        },
        {
            "format": "raw_fp64",
            "label": "Raw FP64",
            "group": "baseline",
            "N": target_n,
            "median_ms": 1.20,
            "ratio_to_fp32_to_fp64": 1.19,
        },
        {
            "format": "pwl4_compand32",
            "label": "PWL4Compand32",
            "group": "new",
            "N": target_n,
            "median_ms": 1.38,
            "ratio_to_fp32_to_fp64": 1.36,
        },
    ]

    svg = MODULE.ranking_graph(rows, target_n, maximum_ms=1.20)

    assert "Int32" in svg
    assert "Raw FP64" in svg
    assert "PWL4Compand32" not in svg

