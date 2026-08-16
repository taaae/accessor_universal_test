#!/usr/bin/env python3
"""Build the unified multi-page H200 storage-format report."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


matplotlib.rcParams.update(
    {
        "svg.hashsalt": "storage-performance-report-v1",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#d7dce2",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


FORMAT_ORDER = (
    "e1m6",
    "e2m5",
    "e3m4",
    "fp8_e4m3",
    "fp8_e5m2",
    "e1m14",
    "e2m13",
    "e3m12",
    "fp16_e5m10",
    "bf16_e8m7",
    "e11m4",
    "e1m30",
    "e2m29",
    "e3m28",
    "fp32_e8m23",
    "e11m20",
    "fp64_e11m52",
)

FORMAT_LABELS = {
    "e0m1": "E0M1",
    "e1m0": "E1M0",
    "e0m3": "E0M3",
    "e1m2": "E1M2",
    "fp4_e2m1": "E2M1 aka FP4",
    "e3m0": "E3M0",
    "e0m7": "E0M7",
    "e1m6": "E1M6",
    "e2m5": "E2M5",
    "e3m4": "E3M4",
    "fp8_e4m3": "E4M3 aka FP8",
    "fp8_e5m2": "E5M2 aka FP8",
    "e6m1": "E6M1",
    "e7m0": "E7M0",
    "e0m15": "E0M15",
    "e1m14": "E1M14",
    "e2m13": "E2M13",
    "e3m12": "E3M12",
    "e4m11": "E4M11",
    "fp16_e5m10": "E5M10 aka FP16",
    "e6m9": "E6M9",
    "e7m8": "E7M8",
    "bf16_e8m7": "E8M7 aka BF16",
    "e9m6": "E9M6",
    "e10m5": "E10M5",
    "e11m4": "E11M4",
    "e0m31": "E0M31",
    "e1m30": "E1M30",
    "e2m29": "E2M29",
    "e3m28": "E3M28",
    "e4m27": "E4M27",
    "e5m26": "E5M26",
    "e6m25": "E6M25",
    "e7m24": "E7M24",
    "fp32_e8m23": "E8M23 aka FP32",
    "e9m22": "E9M22",
    "e10m21": "E10M21",
    "e11m20": "E11M20",
    "fp64_e11m52": "E11M52 aka FP64",
}

FORMAT_LAYOUTS = {
    "e0m1": (0, 1),
    "e1m0": (1, 0),
    "e0m3": (0, 3),
    "e1m2": (1, 2),
    "fp4_e2m1": (2, 1),
    "e3m0": (3, 0),
    "e0m7": (0, 7),
    "e1m6": (1, 6),
    "e2m5": (2, 5),
    "e3m4": (3, 4),
    "fp8_e4m3": (4, 3),
    "fp8_e5m2": (5, 2),
    "e6m1": (6, 1),
    "e7m0": (7, 0),
    "e0m15": (0, 15),
    "e1m14": (1, 14),
    "e2m13": (2, 13),
    "e3m12": (3, 12),
    "e4m11": (4, 11),
    "fp16_e5m10": (5, 10),
    "e6m9": (6, 9),
    "e7m8": (7, 8),
    "bf16_e8m7": (8, 7),
    "e9m6": (9, 6),
    "e10m5": (10, 5),
    "e11m4": (11, 4),
    "e0m31": (0, 31),
    "e1m30": (1, 30),
    "e2m29": (2, 29),
    "e3m28": (3, 28),
    "e4m27": (4, 27),
    "e5m26": (5, 26),
    "e6m25": (6, 25),
    "e7m24": (7, 24),
    "fp32_e8m23": (8, 23),
    "e9m22": (9, 22),
    "e10m21": (10, 21),
    "e11m20": (11, 20),
    "fp64_e11m52": (11, 52),
}

BASE_CONVERSION_FORMATS = FORMAT_ORDER[:-1]
CONVERSION_BIT_FORMATS = {
    2: ("e0m1", "e1m0"),
    4: ("e0m3", "e1m2", "fp4_e2m1", "e3m0"),
    8: (
        "e0m7",
        "e1m6",
        "e2m5",
        "e3m4",
        "fp8_e4m3",
        "fp8_e5m2",
        "e6m1",
        "e7m0",
    ),
    16: (
        "e0m15",
        "e1m14",
        "e2m13",
        "e3m12",
        "e4m11",
        "fp16_e5m10",
        "e6m9",
        "e7m8",
        "bf16_e8m7",
        "e9m6",
        "e10m5",
        "e11m4",
    ),
    32: (
        "e0m31",
        "e1m30",
        "e2m29",
        "e3m28",
        "e4m27",
        "e5m26",
        "e6m25",
        "e7m24",
        "fp32_e8m23",
        "e9m22",
        "e10m21",
        "e11m20",
    ),
}
CONVERSION_FORMATS = tuple(
    format_name
    for format_names in CONVERSION_BIT_FORMATS.values()
    for format_name in format_names
)
EXPANDED_CONVERSION_FORMATS = tuple(
    format_name
    for format_name in CONVERSION_FORMATS
    if format_name not in BASE_CONVERSION_FORMATS
)

# The unified benchmark deliberately uses different format inventories for
# FP32 and FP64 arithmetic.  These maps are also the report's source of truth,
# so placeholder pages exist before the corresponding H200 run is imported.
FP32_CONVERSION_BIT_FORMATS = {
    2: ("e0m1", "e1m0"),
    3: ("e0m2", "e1m1", "e2m0"),
    4: ("e0m3", "e1m2", "fp4_e2m1", "e3m0"),
    5: ("e0m4", "e2m2", "e4m0"),
    6: ("e0m5", "e1m4", "e2m3", "e3m2", "e4m1", "e5m0"),
    7: ("e0m6", "e3m3", "e5m1"),
    8: CONVERSION_BIT_FORMATS[8],
    9: ("e0m8", "e4m4", "e8m0"),
    10: ("e2m7", "e5m4", "e8m1"),
    12: ("e0m11", "e5m6", "e8m3"),
    14: ("e2m11", "e5m8", "e8m5"),
    16: CONVERSION_BIT_FORMATS[16][:9],
    17: ("e2m14", "e5m11", "e8m8"),
    20: ("e2m17", "e5m14", "e8m11"),
    24: ("e0m23", "e5m18", "e8m15"),
    28: ("e4m23", "e5m22", "e8m19"),
    32: ("fp32_e8m23",),
}

FP64_CONVERSION_BIT_FORMATS = {
    2: CONVERSION_BIT_FORMATS[2],
    3: ("e0m2", "e1m1", "e2m0"),
    4: CONVERSION_BIT_FORMATS[4],
    5: ("e0m4", "e1m3", "e2m2", "e3m1", "e4m0"),
    6: ("e0m5", "e1m4", "e2m3", "e3m2", "e4m1", "e5m0"),
    7: ("e2m4", "e3m3", "e6m0"),
    8: CONVERSION_BIT_FORMATS[8],
    9: ("e2m6", "e5m3", "e8m0"),
    10: ("e2m7", "e5m4", "e8m1"),
    12: ("e2m9", "e5m6", "e11m0"),
    14: ("e2m11", "e5m8", "e11m2"),
    16: CONVERSION_BIT_FORMATS[16],
    17: ("e2m14", "e5m11", "e11m5"),
    20: ("e2m17", "e5m14", "e11m8"),
    24: ("e2m21", "e5m18", "e11m12"),
    28: ("e2m25", "e5m22", "e11m16"),
    32: CONVERSION_BIT_FORMATS[32],
}

COMPUTE_CONVERSION_FORMATS = {
    "fp32": FP32_CONVERSION_BIT_FORMATS,
    "fp64": FP64_CONVERSION_BIT_FORMATS,
}

FAMILY_COLORS = {
    "software": "#D55E00",
    "native": "#0072B2",
    "prefix": "#009E73",
    "baseline": "#3f4650",
}

LANE_COLORS = {1: "#6c757d", 2: "#0072B2", 4: "#D55E00"}
LANE_STYLES = {1: ":", 2: "--", 4: "-"}
STRATEGY_LANE_STYLES = {1: ":", 2: "--", 4: "-", 8: "-."}
STRATEGY_LANE_ORDER = (4, 8, 2, 1)
EXPECTED_STRATEGY_COUNT = 42
STRATEGY_FAMILY_ORDER = (
    "lut_prefix",
    "lut_high_word",
    "lut_fp32",
    "lut_fp64",
    "lut_subnormal",
    "branchless",
    "direct_bits",
    "baseline",
    "decomposed",
    "generic",
)
STRATEGY_FAMILY_LABELS = {
    "generic": "Generic codec",
    "branchless": "Branchless FP32",
    "lut_fp32": "FP32 lookup",
    "lut_fp64": "FP64 lookup",
    "lut_prefix": "Prefix lookup",
    "lut_high_word": "FP64 high-word lookup",
    "lut_subnormal": "Subnormal-only lookup",
    "direct_bits": "Direct FP64 construction",
    "decomposed": "Decomposed bits",
    "baseline": "Raw FP64",
}
STRATEGY_FAMILY_COLORS = {
    "generic": "#6C757D",
    "branchless": "#0072B2",
    "lut_fp32": "#D55E00",
    "lut_fp64": "#CC79A7",
    "lut_prefix": "#009E73",
    "lut_high_word": "#7E57C2",
    "lut_subnormal": "#8C564B",
    "direct_bits": "#E69F00",
    "decomposed": "#56B4E9",
    "baseline": "#20252A",
}
CONVERSION_FAMILY_ORDER = (
    "native",
    "prefix",
    "integer",
    "full_lut",
    "subnormal_lut",
    "fp32_path",
    "fp64_lut",
    "pair_lut",
    "direct_words",
    "decomposed",
    "generic",
    "baseline",
)
CONVERSION_FAMILY_LABELS = {
    "native": "Native CUDA path",
    "prefix": "Prefix construction / lookup",
    "integer": "Scaled integer path",
    "full_lut": "Full high-word lookup",
    "subnormal_lut": "Subnormal-only lookup",
    "fp32_path": "FP32 intermediate",
    "fp64_lut": "FP64 lookup",
    "pair_lut": "Pair / quad packet lookup",
    "direct_words": "Direct FP64 words",
    "decomposed": "Decomposed fields",
    "generic": "Generic codec",
    "baseline": "Raw FP64",
}
CONVERSION_FAMILY_COLORS = {
    "native": "#0072B2",
    "prefix": "#009E73",
    "integer": "#E69F00",
    "full_lut": "#7E57C2",
    "subnormal_lut": "#8C564B",
    "fp32_path": "#D55E00",
    "fp64_lut": "#CC79A7",
    "pair_lut": "#56B4E9",
    "direct_words": "#C17C00",
    "decomposed": "#3B8ED0",
    "generic": "#6C757D",
    "baseline": "#20252A",
}
BIT_MARKERS = {8: "o", 16: "s", 32: "^", 64: "D"}
FORMAT_MARKERS = ("o", "s", "^", "D", "P", "X")
SAME_BIT_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
BIT_FORMATS = {
    8: ("e1m6", "e2m5", "e3m4", "fp8_e4m3", "fp8_e5m2"),
    16: ("e1m14", "e2m13", "e3m12", "fp16_e5m10", "bf16_e8m7", "e11m4"),
    32: ("e1m30", "e2m29", "e3m28", "fp32_e8m23", "e11m20"),
}
ALL_BIT_FORMATS = {**BIT_FORMATS, 64: ("fp64_e11m52",)}
ACCURACY_METRICS = {
    "dot": ("rms_normalized_error", "Normalized RMS error"),
    "gemv": ("relative_l2", "Relative L2 error"),
}
ACCURACY_DISTRIBUTIONS = (
    ("uniform_0_1", "U(0,1)"),
    ("normal_0_1", "N(0,1)"),
)
SELECTED_FORMATS = (
    "e1m6",
    "e2m29",
    "fp8_e4m3",
    "fp16_e5m10",
    "e11m4",
    "fp32_e8m23",
    "fp64_e11m52",
)

LABEL_OFFSETS = {
    "e1m6": (-18, 7),
    "e2m29": (4, -10),
    "fp8_e4m3": (4, 5),
    "fp16_e5m10": (4, -10),
    "e11m4": (4, 5),
    "fp32_e8m23": (4, 5),
    "fp64_e11m52": (-22, 5),
}

PAGES = (
    ("index.html", "Overview"),
    ("total-performance.html", "Total performance"),
    ("same-bit-formats.html", "Same-bit formats"),
    ("packing.html", "Packed vs unpacked"),
    ("roofline.html", "Roofline"),
    ("conversion.html", "Conversion"),
    ("bottlenecks.html", "Bottlenecks"),
    ("accuracy.html", "Accuracy"),
    ("methodology.html", "Methodology"),
    ("general-info.html", "General info"),
    ("packing-bottlenecks.html", "Packing bottlenecks"),
    ("conversion-fp32.html", "Storage → FP32 arithmetic"),
    ("conversion-strategies.html", "Storage → FP64 arithmetic"),
)

ACCURACY_PAGES = (
    ("accuracy.html", "Accuracy overview"),
    ("accuracy-dot.html", "DOT accuracy"),
    ("accuracy-gemv.html", "GEMV accuracy"),
    ("accuracy-scalar.html", "Scalar and arithmetic"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="performance run directory; default: newest complete run",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/008_storage_performance"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/report"),
    )
    parser.add_argument(
        "--accuracy-dir",
        type=Path,
        default=Path("results/006_accuracy_model/generated"),
        help="generated analytical model and H200 comparison directory",
    )
    parser.add_argument(
        "--accuracy-run-dir",
        type=Path,
        help="GPU accuracy run directory; default: newest complete run",
    )
    parser.add_argument(
        "--accuracy-results-root",
        type=Path,
        default=Path("results/007_gpu_accuracy_simulation"),
    )
    parser.add_argument(
        "--strategy-run-dir",
        type=Path,
        help="E2M5/E3M4 strategy run; default: newest complete run",
    )
    parser.add_argument(
        "--strategy-results-root",
        type=Path,
        default=Path("results/013_e2e3_expanded_strategy_performance"),
    )
    parser.add_argument(
        "--all-strategy-run-dir",
        type=Path,
        help="all-format strategy run; default: newest complete run",
    )
    parser.add_argument(
        "--all-strategy-results-root",
        type=Path,
        default=Path("results/015_all_format_strategy_performance"),
    )
    parser.add_argument(
        "--expanded-strategy-run-dir",
        type=Path,
        help="expanded-format strategy run; default: newest complete run",
    )
    parser.add_argument(
        "--expanded-strategy-results-root",
        type=Path,
        default=Path("results/017_expanded_format_strategy_performance"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def newest_complete_run(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.glob("run_*")
        if (path / "timing_summary.csv").is_file()
        and (path / "profile_operations.csv").is_file()
    )
    if not candidates:
        raise SystemExit(f"no complete performance run below {root}")
    return candidates[-1]


def newest_accuracy_run(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.glob("run_*")
        if (path / "simulation_summary.csv").is_file()
        and (path / "convergence_report.csv").is_file()
    )
    if not candidates:
        raise SystemExit(f"no complete accuracy run below {root}")
    return candidates[-1]


def newest_strategy_run(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.glob("run_*")
        if (path / "timing_summary.csv").is_file()
        and (path / "strategy_inventory.csv").is_file()
    )
    if not candidates:
        raise SystemExit(f"no complete E2M5/E3M4 strategy run below {root}")
    return candidates[-1]


def newest_all_strategy_run(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.glob("run_*")
        if (path / "timing_summary.csv").is_file()
        and (path / "strategy_inventory.csv").is_file()
        and (path / "strategy_rankings.csv").is_file()
    )
    if not candidates:
        raise SystemExit(f"no complete all-format strategy run below {root}")
    return candidates[-1]


def format_family(name: str) -> str:
    if name == "fp64_e11m52":
        return "baseline"
    if name.startswith(("fp8_", "fp16_", "bf16_", "fp32_")):
        return "native"
    if name.startswith("e11m"):
        return "prefix"
    return "software"


def label(name: str) -> str:
    if name in FORMAT_LABELS:
        return FORMAT_LABELS[name]
    match = re.search(r"e(\d+)m(\d+)$", name)
    if not match:
        raise KeyError(f"unknown storage format {name}")
    return f"E{match.group(1)}M{match.group(2)}"


def format_layout_bits(name: str) -> tuple[int, int]:
    if name in FORMAT_LAYOUTS:
        return FORMAT_LAYOUTS[name]
    match = re.search(r"e(\d+)m(\d+)$", name)
    if not match:
        raise KeyError(f"unknown storage format {name}")
    return int(match.group(1)), int(match.group(2))


def format_total_bits(name: str) -> int:
    exponent_bits, mantissa_bits = format_layout_bits(name)
    return 1 + exponent_bits + mantissa_bits


def strategy_family(name: str) -> str:
    if name == "raw_pointer_x1":
        return "baseline"
    if name.startswith("generic_fp64"):
        return "generic"
    if name.startswith("branchless_fp32"):
        return "branchless"
    if name.startswith("lut_fp32"):
        return "lut_fp32"
    if name.startswith("lut_fp64"):
        return "lut_fp64"
    if name.startswith("lut_prefix"):
        return "lut_prefix"
    if name.startswith("lut_high_word"):
        return "lut_high_word"
    if name.startswith("lut_subnormal"):
        return "lut_subnormal"
    if name.startswith(("direct_fp64_bits", "direct_fp64_words")):
        return "direct_bits"
    if name.startswith("decomposed_bits"):
        return "decomposed"
    raise ValueError(f"unknown E2M5/E3M4 strategy: {name}")


def strategy_abbreviation(name: str) -> str:
    if name == "raw_pointer_x1":
        return "F64"
    lanes = name.rsplit("_x", 1)[-1]
    if name.startswith("generic_fp64"):
        base = "GEN"
    elif name.startswith("branchless_fp32"):
        base = "BR32"
    elif name.startswith("lut_fp32_shared"):
        base = "L32-S"
    elif name.startswith("lut_fp32_global_pipelined"):
        base = "L32-G-P"
    elif name.startswith("lut_fp32_global"):
        base = "L32-G"
    elif name.startswith("lut_fp64_shared"):
        base = "L64-S"
    elif name.startswith("lut_fp64_global"):
        base = "L64-G"
    elif name.startswith("lut_prefix_shared"):
        base = "LP-S"
    elif name.startswith("lut_prefix_global_pipelined"):
        base = "LP-G-P"
    elif name.startswith("lut_prefix_global"):
        base = "LP-G"
    elif name.startswith("lut_high_word_swizzled_shared"):
        base = "LHW-SW"
    elif name.startswith("lut_high_word_shared"):
        base = "LHW-S"
    elif name.startswith("lut_high_word_global"):
        base = "LHW-G"
    elif name.startswith("lut_subnormal_shared"):
        base = "SN-S"
    elif name.startswith("lut_subnormal_global"):
        base = "SN-G"
    elif name.startswith("direct_fp64_words_branchy"):
        base = "DW-B"
    elif name.startswith("direct_fp64_words_masked"):
        base = "DW-M"
    elif name.startswith("direct_fp64_bits"):
        base = "DB64"
    elif name.startswith("decomposed_bits"):
        base = "DEC"
    else:
        raise ValueError(f"unknown E2M5/E3M4 strategy: {name}")
    return f"{base} ×{lanes}"


def strategy_sort_key(name: str) -> tuple[int, int, str]:
    family = strategy_family(name)
    lane_rank = (
        -1
        if family == "baseline"
        else STRATEGY_LANE_ORDER.index(int(name.rsplit("_x", 1)[-1]))
    )
    return STRATEGY_FAMILY_ORDER.index(family), lane_rank, name


def is_row(
    row: dict[str, str],
    *,
    component: str | None = None,
    distribution: str | None = None,
    lanes: int | None = None,
    format_name: str | None = None,
) -> bool:
    return (
        (component is None or row["component"] == component)
        and (distribution is None or row["distribution"] == distribution)
        and (lanes is None or int(row["lanes"]) == lanes)
        and (format_name is None or row["format"] == format_name)
    )


def max_size_rows(
    rows: Sequence[dict[str, str]],
    *,
    component: str,
    distribution: str = "normal_0_1",
    lanes: int = 4,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if is_row(
            row,
            component=component,
            distribution=distribution,
            lanes=lanes,
        )
    ]
    maximum = max(int(row["n"]) for row in selected)
    by_format = {row["format"]: row for row in selected if int(row["n"]) == maximum}
    if set(by_format) != set(FORMAT_ORDER):
        raise ValueError(f"incomplete {component} max-size coverage")
    return [by_format[name] for name in FORMAT_ORDER]


def profile_rows(
    rows: Sequence[dict[str, str]], component: str, lanes: int = 4
) -> list[dict[str, str]]:
    selected = {
        row["format"]: row
        for row in rows
        if is_row(
            row,
            component=component,
            distribution="normal_0_1",
            lanes=lanes,
        )
    }
    if set(selected) != set(FORMAT_ORDER):
        raise ValueError(f"incomplete {component} profile coverage")
    return [selected[name] for name in FORMAT_ORDER]


def save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "accessor_universal_test"},
    )
    plt.close(fig)
    return path


def strip_trailing_whitespace(path: Path) -> Path:
    """Normalize Matplotlib SVG whitespace for stable generated diffs."""
    text = path.read_text(encoding="utf-8")
    trailing_newline = "\n" if text.endswith("\n") else ""
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines())
        + trailing_newline,
        encoding="utf-8",
    )
    return path


def format_axis_labels(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", labelsize=9)
    axis.spines["left"].set_color("#8b939c")
    axis.spines["bottom"].set_color("#8b939c")


def add_family_legend(fig: plt.Figure, *, y: float = 0.99) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor=color,
            label=text,
        )
        for text, color in (
            ("Software codec", FAMILY_COLORS["software"]),
            ("CUDA native", FAMILY_COLORS["native"]),
            ("E11 prefix", FAMILY_COLORS["prefix"]),
            ("FP64 baseline", FAMILY_COLORS["baseline"]),
        )
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, y))


def plot_total_kernel_time(rows: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    positions = np.arange(len(FORMAT_ORDER))
    for axis, component in zip(axes, ("dot", "gemv")):
        current = max_size_rows(rows, component=component)
        values = [number(row, "median_time_ms") for row in current]
        colors = [FAMILY_COLORS[format_family(row["format"])] for row in current]
        bars = axis.barh(positions, values, color=colors, height=0.7)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
        axis.set_title(f"{component.upper()} — x4, N={int(current[0]['n']):,}")
        axis.set_xlabel("Complete kernel time (ms) — lower is better")
        axis.invert_yaxis()
        axis.set_xlim(0, max(values) * 1.22)
        format_axis_labels(axis)
    axes[0].set_yticks(positions, [label(name) for name in FORMAT_ORDER])
    add_family_legend(fig)
    fig.suptitle("Total DOT and GEMV time by storage format", y=1.03, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, path)


def plot_relative_fp64(rows: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    positions = np.arange(len(FORMAT_ORDER))
    for axis, component in zip(axes, ("dot", "gemv")):
        current = max_size_rows(rows, component=component)
        baseline = next(
            number(row, "median_time_ms")
            for row in current
            if row["format"] == "fp64_e11m52"
        )
        values = [baseline / number(row, "median_time_ms") for row in current]
        colors = [FAMILY_COLORS[format_family(row["format"])] for row in current]
        axis.axvline(1.0, color="#66707a", linewidth=1.1)
        bars = axis.barh(positions, values, color=colors, height=0.7)
        axis.bar_label(bars, labels=[f"{value:.2f}x" for value in values], padding=3, fontsize=8)
        axis.set_title(component.upper())
        axis.set_xlabel("Speed relative to FP64 — higher is better")
        axis.invert_yaxis()
        axis.set_xlim(0, max(values) * 1.22)
        format_axis_labels(axis)
    axes[0].set_yticks(positions, [label(name) for name in FORMAT_ORDER])
    add_family_legend(fig)
    fig.suptitle("Compression benefit relative to the FP64 baseline", y=1.03, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, path)


def plot_size_scaling(rows: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    styles = ("-", "--", "-.", ":", "-", "--", "-.")
    for axis, component in zip(axes, ("dot", "gemv")):
        for name, linestyle in zip(SELECTED_FORMATS, styles):
            current = sorted(
                (
                    row
                    for row in rows
                    if is_row(
                        row,
                        component=component,
                        distribution="normal_0_1",
                        lanes=4,
                        format_name=name,
                    )
                ),
                key=lambda row: int(row["n"]),
            )
            axis.plot(
                [int(row["n"]) for row in current],
                [number(row, "median_time_ms") for row in current],
                marker=BIT_MARKERS[int(current[0]["storage_bits"])],
                markersize=4,
                linewidth=1.5,
                linestyle=linestyle,
                color=FAMILY_COLORS[format_family(name)],
                label=label(name),
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(component.upper())
        axis.set_xlabel("Reduction length N")
        axis.set_ylabel("Complete kernel time (ms)")
        format_axis_labels(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Where launch overhead gives way to format-dependent throughput", y=1.12, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, path)


def plot_same_bit_kernel_time(
    rows: Sequence[dict[str, str]], storage_bits: int, path: Path
) -> Path:
    formats = BIT_FORMATS[storage_bits]
    color_by_format = dict(zip(formats, SAME_BIT_COLORS))
    marker_by_format = dict(zip(formats, FORMAT_MARKERS))
    return plot_kernel_time_overlay(
        rows,
        formats,
        path,
        title=f"{storage_bits}-bit formats: complete kernel time versus N",
        colors=color_by_format,
        markers=marker_by_format,
        figsize=(18.5, 7.8),
        include_legends=True,
    )


def series_gid(component: str, format_name: str, lanes: int) -> str:
    return f"series--{component}--{format_name}--x{lanes}"


def plot_kernel_time_overlay(
    rows: Sequence[dict[str, str]],
    formats: Sequence[str],
    path: Path,
    *,
    title: str,
    colors: dict[str, object],
    markers: dict[str, str],
    figsize: tuple[float, float],
    include_legends: bool,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for axis, component in zip(axes, ("dot", "gemv")):
        for lane in (1, 2, 4):
            for name in formats:
                current = sorted(
                    (
                        row
                        for row in rows
                        if is_row(
                            row,
                            component=component,
                            distribution="normal_0_1",
                            lanes=lane,
                            format_name=name,
                        )
                    ),
                    key=lambda row: int(row["n"]),
                )
                line = axis.plot(
                    [int(row["n"]) for row in current],
                    [number(row, "median_time_ms") for row in current],
                    color=colors[name],
                    marker=markers[name],
                    linestyle=LANE_STYLES[lane],
                    linewidth=2.0 if lane == 4 else 1.65,
                    markersize=5.0,
                    markerfacecolor=(colors[name] if lane == 4 else "white"),
                    markevery=1,
                )[0]
                line.set_gid(series_gid(component, name, lane))
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(component.upper(), fontsize=13)
        axis.set_xlabel("Reduction length N")
        axis.set_ylabel("Complete kernel time (ms)")
        format_axis_labels(axis)
    if include_legends:
        format_handles = [
            Line2D(
                [0],
                [0],
                color=colors[name],
                marker=markers[name],
                linewidth=2.0,
                label=label(name),
            )
            for name in formats
        ]
        lane_handles = [
            Line2D(
                [0],
                [0],
                color="#4c5661",
                linestyle=LANE_STYLES[lane],
                marker="o",
                markerfacecolor=("#4c5661" if lane == 4 else "white"),
                linewidth=2.0 if lane == 4 else 1.65,
                label="unpacked x1" if lane == 1 else f"packed x{lane}",
            )
            for lane in (1, 2, 4)
        ]
        fig.legend(
            handles=format_handles,
            loc="upper center",
            ncol=len(formats),
            frameon=False,
            bbox_to_anchor=(0.5, 1.025),
            title="Storage format (color and marker)",
        )
        fig.legend(
            handles=lane_handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.94),
            title="Access width (line style)",
        )
        fig.suptitle(title, y=1.14, fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.82))
    else:
        fig.suptitle(title, y=0.995, fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, path)


def all_format_colors() -> dict[str, object]:
    palette = plt.get_cmap("tab20")
    return {name: palette(index) for index, name in enumerate(FORMAT_ORDER)}


def all_format_markers() -> dict[str, str]:
    return {
        name: BIT_MARKERS[bits]
        for bits, formats in ALL_BIT_FORMATS.items()
        for name in formats
    }


def plot_interactive_same_bit_kernel_time(
    rows: Sequence[dict[str, str]], storage_bits: int, path: Path
) -> Path:
    formats = BIT_FORMATS[storage_bits]
    return plot_kernel_time_overlay(
        rows,
        formats,
        path,
        title=f"{storage_bits}-bit formats: complete kernel time versus N",
        colors=dict(zip(formats, SAME_BIT_COLORS)),
        markers=dict(zip(formats, FORMAT_MARKERS)),
        figsize=(21.0, 8.2),
        include_legends=False,
    )


def plot_all_format_kernel_time(
    rows: Sequence[dict[str, str]], path: Path
) -> Path:
    return plot_kernel_time_overlay(
        rows,
        FORMAT_ORDER,
        path,
        title="All storage formats and access widths: complete kernel time versus N",
        colors=all_format_colors(),
        markers=all_format_markers(),
        figsize=(24.0, 9.2),
        include_legends=False,
    )


def format_storage_bits() -> dict[str, int]:
    return {
        name: bits
        for bits, formats in ALL_BIT_FORMATS.items()
        for name in formats
    }


def interactive_chart_document(
    svg_path: Path,
    *,
    title: str,
    description: str,
    formats: Sequence[str],
    colors: dict[str, object],
    include_bit_filters: bool,
) -> str:
    svg = svg_path.read_text(encoding="utf-8")
    svg = svg[svg.index("<svg ") :]
    svg = svg.replace(
        "<svg ",
        '<svg class="performance-chart" role="img" '
        'aria-labelledby="performance-chart-title performance-chart-description" ',
        1,
    )
    svg = svg.replace(
        ">",
        f'><title id="performance-chart-title">{html.escape(title)}</title>'
        f'<desc id="performance-chart-description">{html.escape(description)}</desc>',
        1,
    )
    bits_by_format = format_storage_bits()
    pattern = re.compile(
        r'<g id="series--(?P<component>dot|gemv)--'
        r'(?P<format>[a-z0-9_]+)--x(?P<lanes>[124])">'
    )

    def series_attributes(match: re.Match[str]) -> str:
        component = match.group("component")
        format_name = match.group("format")
        lanes = match.group("lanes")
        bits = bits_by_format[format_name]
        series_title = f"{component.upper()}: {label(format_name)}, x{lanes}"
        group_id = series_gid(component, format_name, int(lanes))
        return (
            f'<g id="{group_id}" data-series="true" '
            f'data-component="{component}" data-format="{format_name}" '
            f'data-lanes="{lanes}" data-bits="{bits}">'
            f"<title>{html.escape(series_title)}</title>"
        )

    svg, series_count = pattern.subn(series_attributes, svg)
    expected_series = 2 * 3 * len(formats)
    if series_count != expected_series:
        raise ValueError(
            f"expected {expected_series} interactive series in {svg_path}, "
            f"found {series_count}"
        )

    lane_options = "".join(
        f"""<label class="filter-option lane-x{lane}">
<input type="checkbox" data-filter="lanes" value="{lane}" checked>
<span class="lane-sample" aria-hidden="true"></span>
<span>{'unpacked x1' if lane == 1 else f'packed x{lane}'}</span>
</label>"""
        for lane in (1, 2, 4)
    )
    format_options = "".join(
        f"""<label class="filter-option">
<input type="checkbox" data-filter="format" value="{html.escape(name)}" checked>
<span class="format-swatch" style="--series-color: {matplotlib.colors.to_hex(colors[name])}" aria-hidden="true"></span>
<span>{html.escape(label(name))}</span>
</label>"""
        for name in formats
    )
    bit_fieldset = ""
    if include_bit_filters:
        bit_options = "".join(
            f"""<label class="filter-option">
<input type="checkbox" data-filter="bits" value="{bits}" checked>
<span class="bit-marker bit-{bits}" aria-hidden="true"></span>
<span>{bits}-bit</span>
</label>"""
            for bits in ALL_BIT_FORMATS
        )
        bit_fieldset = f"""<fieldset>
<legend>Storage width</legend>
<div class="filter-options">{bit_options}</div>
</fieldset>"""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --fg: #1f252b; --muted: #5d6872; --border: #d8dee4; --focus: #075f9a; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #fff; color: var(--fg); font: 15px/1.4 system-ui, sans-serif; }}
.chart-root {{ padding: 14px 16px 2px; }}
.filters {{ display: grid; gap: 10px; margin-bottom: 8px; }}
fieldset {{ border: 0; margin: 0; padding: 0; }}
legend {{ color: var(--muted); font-size: 0.86rem; font-weight: 500; margin-bottom: 4px; }}
.filter-options {{ display: flex; flex-wrap: wrap; gap: 5px 14px; }}
.filter-option {{ align-items: center; cursor: pointer; display: inline-flex; gap: 6px; min-height: 28px; white-space: nowrap; }}
.filter-option input {{ accent-color: var(--focus); height: 16px; margin: 0; width: 16px; }}
.filter-option:has(input:focus-visible) {{ outline: 2px solid var(--focus); outline-offset: 2px; }}
.format-swatch {{ background: var(--series-color); border-radius: 50%; height: 9px; width: 18px; }}
.lane-sample {{ border-top: 2px solid #4c5661; height: 0; width: 22px; }}
.lane-x1 .lane-sample {{ border-top-style: dotted; }}
.lane-x2 .lane-sample {{ border-top-style: dashed; }}
.lane-x4 .lane-sample {{ border-top-style: solid; }}
.bit-marker {{ background: #4c5661; display: inline-block; height: 9px; width: 9px; }}
.bit-8 {{ border-radius: 50%; }}
.bit-16 {{ border-radius: 1px; }}
.bit-32 {{ clip-path: polygon(50% 0, 100% 100%, 0 100%); }}
.bit-64 {{ transform: rotate(45deg); }}
.visible-status {{ color: var(--muted); margin: 2px 0 4px; }}
.chart-wrap {{ width: 100%; }}
.performance-chart {{ display: block; height: auto; max-width: none; width: 100%; }}
.performance-chart [data-series] {{ transition: opacity 120ms ease; }}
.performance-chart .series-hidden {{ display: none; }}
@media (prefers-reduced-motion: reduce) {{ .performance-chart [data-series] {{ transition: none; }} }}
@media (max-width: 620px) {{ .chart-root {{ padding-inline: 8px; }} .filter-options {{ column-gap: 10px; }} }}
</style>
</head>
<body>
<main id="interactive-performance-chart" class="chart-root">
<section class="filters" aria-label="Chart filters">
<fieldset>
<legend>Access width</legend>
<div class="filter-options">{lane_options}</div>
</fieldset>
{bit_fieldset}
<fieldset>
<legend>Storage format</legend>
<div class="filter-options">{format_options}</div>
</fieldset>
</section>
<p id="visible-status" class="visible-status" aria-live="polite"></p>
<div class="chart-wrap">{svg}</div>
</main>
<script>
const root = document.getElementById('interactive-performance-chart');
const status = document.getElementById('visible-status');
const controls = [...root.querySelectorAll('input[data-filter]')];
const series = [...root.querySelectorAll('[data-series]')];

function filterEnabled(kind, value) {{
  const input = controls.find(control => control.dataset.filter === kind && control.value === value);
  return input ? input.checked : true;
}}

function reportHeight() {{
  if (window.parent !== window) {{
    const height = Math.ceil(root.getBoundingClientRect().bottom + 2);
    window.parent.postMessage({{ type: 'performance-chart-height', height }}, '*');
  }}
}}

function updateSeries() {{
  let visible = 0;
  for (const group of series) {{
    const show = filterEnabled('lanes', group.dataset.lanes)
      && filterEnabled('format', group.dataset.format)
      && filterEnabled('bits', group.dataset.bits);
    group.classList.toggle('series-hidden', !show);
    group.setAttribute('aria-hidden', String(!show));
    if (show) visible += 1;
  }}
  const kernels = new Set(series.map(group => group.dataset.component)).size;
  status.textContent = `${{visible / kernels}} of ${{series.length / kernels}} lines visible per kernel · lower is faster`;
  requestAnimationFrame(reportHeight);
}}

root.addEventListener('change', event => {{
  if (event.target.matches('input[data-filter]')) updateSeries();
}});
window.addEventListener('message', event => {{
  if (event.data?.type === 'request-performance-chart-height') reportHeight();
}});
window.addEventListener('load', updateSeries);
if ('ResizeObserver' in window) new ResizeObserver(reportHeight).observe(root);
</script>
</body>
</html>
"""


def write_interactive_chart(
    svg_path: Path,
    output_path: Path,
    *,
    title: str,
    description: str,
    formats: Sequence[str],
    colors: dict[str, object],
    include_bit_filters: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        interactive_chart_document(
            svg_path,
            title=title,
            description=description,
            formats=formats,
            colors=colors,
            include_bit_filters=include_bit_filters,
        ),
        encoding="utf-8",
    )
    return output_path


def strategy_series_gid(
    format_name: str,
    distribution: str,
    component: str,
    strategy: str,
) -> str:
    return (
        f"strategy-series--{format_name}--{distribution}--"
        f"{component}--{strategy}"
    )


def plot_e2e3_strategy_kernel_time(
    rows: Sequence[dict[str, str]], format_name: str, path: Path
) -> Path:
    format_strategies = sorted(
        {row["strategy"] for row in rows if row["format"] == format_name},
        key=strategy_sort_key,
    )
    if len(format_strategies) != EXPECTED_STRATEGY_COUNT:
        raise ValueError(
            f"expected {EXPECTED_STRATEGY_COUNT} {format_name} strategies, "
            f"found {len(format_strategies)}"
        )
    strategies = (*format_strategies, "raw_pointer_x1")
    fig, axes = plt.subplots(1, 2, figsize=(21.0, 8.4))
    for axis, component in zip(axes, ("dot", "gemv")):
        for distribution in ("uniform_0_1", "normal_0_1"):
            for strategy in strategies:
                source_format = "fp64" if strategy == "raw_pointer_x1" else format_name
                current = sorted(
                    (
                        row
                        for row in rows
                        if row["format"] == source_format
                        and row["strategy"] == strategy
                        and row["component"] == component
                        and row["distribution"] == distribution
                    ),
                    key=lambda row: int(row["n"]),
                )
                if len(current) != 5:
                    raise ValueError(
                        f"incomplete {format_name}/{strategy}/{component}/"
                        f"{distribution} series"
                    )
                family = strategy_family(strategy)
                is_baseline = family == "baseline"
                lanes = 1 if is_baseline else int(current[0]["lanes"])
                if is_baseline:
                    marker = "D"
                elif current[0]["table_location"] == "shared":
                    marker = "s"
                elif current[0]["pipelined"] == "1":
                    marker = "^"
                elif family == "direct_bits":
                    marker = "P"
                elif family == "decomposed":
                    marker = "X"
                else:
                    marker = "o"
                line = axis.plot(
                    [int(row["n"]) for row in current],
                    [number(row, "median_time_ms") for row in current],
                    color=STRATEGY_FAMILY_COLORS[family],
                    linestyle="-" if is_baseline else STRATEGY_LANE_STYLES[lanes],
                    linewidth=2.4 if is_baseline else 1.45,
                    marker=marker,
                    markersize=4.4 if is_baseline else 3.6,
                    markerfacecolor=(
                        STRATEGY_FAMILY_COLORS[family]
                        if is_baseline or current[0]["table_location"] == "shared"
                        else "white"
                    ),
                    alpha=1.0 if is_baseline else 0.82,
                )[0]
                line.set_gid(
                    strategy_series_gid(
                        format_name, distribution, component, strategy
                    )
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(component.upper(), fontsize=13)
        axis.set_xlabel("Reduction length N")
        axis.set_ylabel("Complete kernel time (ms)")
        format_axis_labels(axis)
    fig.suptitle(
        f"{label(format_name)} decoder strategies and raw FP64", y=0.995, fontsize=17
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, path)


def strategy_interactive_document(
    svg_path: Path,
    *,
    format_name: str,
    strategies: Sequence[str],
) -> str:
    svg = svg_path.read_text(encoding="utf-8")
    svg = svg[svg.index("<svg ") :]
    chart_title = f"{label(format_name)} strategy performance"
    chart_description = (
        f"Complete DOT and GEMV time versus N for all {label(format_name)} "
        "decoder strategies and a raw FP64 baseline. Lower is faster."
    )
    svg = svg.replace(
        "<svg ",
        '<svg class="strategy-chart" role="img" '
        'aria-labelledby="strategy-chart-title strategy-chart-description" ',
        1,
    )
    svg = svg.replace(
        ">",
        f'><title id="strategy-chart-title">{html.escape(chart_title)}</title>'
        f'<desc id="strategy-chart-description">'
        f"{html.escape(chart_description)}</desc>",
        1,
    )
    pattern = re.compile(
        rf'<g id="strategy-series--{format_name}--'
        r'(?P<distribution>uniform_0_1|normal_0_1)--'
        r'(?P<component>dot|gemv)--(?P<strategy>[a-z0-9_]+)">'
    )

    def series_attributes(match: re.Match[str]) -> str:
        strategy = match.group("strategy")
        family = strategy_family(strategy)
        lanes = "baseline" if family == "baseline" else strategy.rsplit("_x", 1)[-1]
        full_label = f"{match.group('component').upper()}: {strategy_abbreviation(strategy)}"
        return (
            f'<g id="{match.group(0).split(chr(34))[1]}" data-series="true" '
            f'data-distribution="{match.group("distribution")}" '
            f'data-component="{match.group("component")}" '
            f'data-strategy="{strategy}" data-family="{family}" '
            f'data-lanes="{lanes}"><title>{html.escape(full_label)}</title>'
        )

    svg, series_count = pattern.subn(series_attributes, svg)
    expected_series = 2 * 2 * len(strategies)
    if series_count != expected_series:
        raise ValueError(
            f"expected {expected_series} strategy series in {svg_path}, "
            f"found {series_count}"
        )

    distribution_options = "".join(
        f"""<label class="filter-option">
<input type="radio" name="distribution" value="{distribution}" {'checked' if distribution == 'normal_0_1' else ''}>
<span>{text}</span>
</label>"""
        for distribution, text in ACCURACY_DISTRIBUTIONS
    )
    lane_options = "".join(
        f"""<label class="filter-option lane-x{lane}">
<input type="checkbox" data-filter="lanes" value="{lane}" checked>
<span class="lane-sample" aria-hidden="true"></span><span>×{lane}</span>
</label>"""
        for lane in STRATEGY_LANE_ORDER
    )
    family_options = "".join(
        f"""<label class="filter-option">
<input type="checkbox" data-filter="family" value="{family}" checked>
<span class="family-swatch" style="--series-color: {STRATEGY_FAMILY_COLORS[family]}" aria-hidden="true"></span>
<span>{html.escape(STRATEGY_FAMILY_LABELS[family])}</span>
</label>"""
        for family in STRATEGY_FAMILY_ORDER
    )
    strategy_options = "".join(
        f"""<label class="filter-option strategy-option">
<input type="checkbox" data-filter="strategy" value="{strategy}" checked>
<span class="family-swatch" style="--series-color: {STRATEGY_FAMILY_COLORS[strategy_family(strategy)]}" aria-hidden="true"></span>
<code>{html.escape(strategy_abbreviation(strategy))}</code>
</label>"""
        for strategy in strategies
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(chart_title)}</title>
<style>
:root {{ color-scheme: light; --fg: #1f252b; --muted: #5d6872; --focus: #075f9a; --border: #d8dee4; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #fff; color: var(--fg); font: 15px/1.4 system-ui, sans-serif; }}
.chart-root {{ padding: 14px 16px 2px; }}
.filters {{ display: grid; gap: 10px; margin-bottom: 8px; }}
fieldset {{ border: 0; margin: 0; padding: 0; }}
legend {{ color: var(--muted); font-size: 0.86rem; font-weight: 500; margin-bottom: 4px; }}
.filter-options {{ display: flex; flex-wrap: wrap; gap: 5px 14px; }}
.filter-option {{ align-items: center; cursor: pointer; display: inline-flex; gap: 6px; min-height: 28px; white-space: nowrap; }}
.filter-option input {{ accent-color: var(--focus); height: 16px; margin: 0; width: 16px; }}
.filter-option:has(input:focus-visible) {{ outline: 2px solid var(--focus); outline-offset: 2px; }}
.family-swatch {{ background: var(--series-color); border-radius: 50%; height: 9px; width: 18px; }}
.lane-sample {{ border-top: 2px solid #4c5661; height: 0; width: 22px; }}
.lane-x1 .lane-sample {{ border-top-style: dotted; }}
.lane-x2 .lane-sample {{ border-top-style: dashed; }}
.lane-x4 .lane-sample {{ border-top-style: solid; }}
.lane-x8 .lane-sample {{ border-top-style: double; border-top-width: 3px; }}
details {{ border-top: 1px solid var(--border); padding-top: 7px; }}
summary {{ cursor: pointer; color: var(--muted); }}
.strategy-option code {{ font-size: 0.88rem; }}
.visible-status {{ color: var(--muted); margin: 2px 0 4px; }}
.chart-wrap {{ width: 100%; }}
.strategy-chart {{ display: block; height: auto; max-width: none; width: 100%; }}
.strategy-chart [data-series] {{ transition: opacity 120ms ease; }}
.strategy-chart .series-hidden {{ display: none; }}
@media (prefers-reduced-motion: reduce) {{ .strategy-chart [data-series] {{ transition: none; }} }}
@media (max-width: 620px) {{ .chart-root {{ padding-inline: 8px; }} .filter-options {{ column-gap: 10px; }} }}
</style>
</head>
<body>
<main id="interactive-{format_name}-strategies" class="chart-root">
<section class="filters" aria-label="Chart filters">
<fieldset><legend>Input distribution</legend><div class="filter-options">{distribution_options}</div></fieldset>
<fieldset><legend>Packed load width — fastest first on large problems</legend><div class="filter-options">{lane_options}</div></fieldset>
<fieldset><legend>Decoder family / reference — fastest first on large problems</legend><div class="filter-options">{family_options}</div></fieldset>
<details><summary>Individual strategy toggles</summary><div class="filter-options">{strategy_options}</div></details>
</section>
<p id="visible-status" class="visible-status" aria-live="polite"></p>
<div class="chart-wrap">{svg}</div>
</main>
<script>
const root = document.getElementById('interactive-{format_name}-strategies');
const status = root.querySelector('#visible-status');
const controls = [...root.querySelectorAll('input')];
const series = [...root.querySelectorAll('[data-series]')];

function checked(kind, value) {{
  const input = controls.find(control => control.dataset.filter === kind && control.value === value);
  return input ? input.checked : true;
}}

function reportHeight() {{
  if (window.parent !== window) {{
    const height = Math.ceil(root.getBoundingClientRect().bottom + 2);
    window.parent.postMessage({{ type: 'performance-chart-height', height }}, '*');
  }}
}}

function updateSeries() {{
  const distribution = root.querySelector('input[name="distribution"]:checked').value;
  let visible = 0;
  for (const group of series) {{
    const widthEnabled = group.dataset.lanes === 'baseline' || checked('lanes', group.dataset.lanes);
    const show = group.dataset.distribution === distribution
      && widthEnabled
      && checked('family', group.dataset.family)
      && checked('strategy', group.dataset.strategy);
    group.classList.toggle('series-hidden', !show);
    group.setAttribute('aria-hidden', String(!show));
    if (show) visible += 1;
  }}
  status.textContent = `${{visible / 2}} of ${{series.length / 4}} lines visible per kernel · lower is faster`;
  requestAnimationFrame(reportHeight);
}}

root.addEventListener('change', updateSeries);
window.addEventListener('message', event => {{
  if (event.data?.type === 'request-performance-chart-height') reportHeight();
}});
window.addEventListener('load', updateSeries);
if ('ResizeObserver' in window) new ResizeObserver(reportHeight).observe(root);
</script>
</body>
</html>
"""


def write_strategy_interactive_chart(
    svg_path: Path,
    output_path: Path,
    *,
    format_name: str,
    strategies: Sequence[str],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        strategy_interactive_document(
            svg_path,
            format_name=format_name,
            strategies=strategies,
        ),
        encoding="utf-8",
    )
    return output_path


def strategy_top_picks(rows: Sequence[dict[str, str]]) -> str:
    body = []
    for format_name in ("e2m5", "e3m4"):
        for component in ("dot", "gemv"):
            case_rows = [
                row
                for row in rows
                if row["format"] == format_name
                and row["component"] == component
                and row["strategy"] != "raw_pointer_x1"
            ]
            sizes = sorted({int(row["n"]) for row in case_rows})
            large_sizes = set(sizes[-2:])
            speedups: dict[str, list[float]] = defaultdict(list)
            for row in case_rows:
                if int(row["n"]) in large_sizes:
                    speedups[row["strategy"]].append(number(row, "speedup_vs_fp64"))
            aggregate_speedup, aggregate_strategy = max(
                (
                    math.exp(sum(math.log(value) for value in values) / len(values)),
                    strategy,
                )
                for strategy, values in speedups.items()
            )

            largest_normal_rows = [
                row
                for row in case_rows
                if row["distribution"] == "normal_0_1"
                and int(row["n"]) == sizes[-1]
            ]
            largest_normal = max(
                largest_normal_rows,
                key=lambda row: number(row, "speedup_vs_fp64"),
            )
            normal_strategy = largest_normal["strategy"]
            normal_label = strategy_abbreviation(normal_strategy)
            if normal_strategy == aggregate_strategy:
                normal_label += " (same)"

            body.append(
                "<tr>"
                f"<th>{format_name.upper()}</th>"
                f"<td>{component.upper()}</td>"
                f"<td><code>{html.escape(strategy_abbreviation(aggregate_strategy))}</code></td>"
                f"<td>{aggregate_speedup:.2f}×</td>"
                f"<td><code>{html.escape(normal_label)}</code></td>"
                "</tr>"
            )

    return f"""<section class="text-section">
<h2>Measured top picks</h2>
<p>The aggregate pick maximizes geometric-mean speedup over raw FP64 across both distributions and the two largest measured N values. The final column shows whether the winner changes for the single largest N(0,1) case.</p>
<div class="table-wrap"><table class="strategy-table"><thead><tr><th>Format</th><th>Kernel</th><th>Aggregate pick</th><th>Speedup vs F64</th><th>Largest N(0,1) pick</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>
</section>"""


def strategy_glossary() -> str:
    rows = (
        ("F64", "Raw double pointers; no storage decoding."),
        ("GEN", "Generic bitfield codec producing FP64 directly."),
        ("BR32", "Branch-free bit/arithmetic decode to FP32, then FP32 → FP64 conversion."),
        ("L32-G", "Global/read-only 2^8-entry FP32 table lookup, then FP32 → FP64 conversion."),
        ("L32-S", "The same FP32 table staged once per block in shared memory."),
        ("L32-G-P", "Global FP32 lookup with explicit next-packet software prefetching."),
        ("L64-G", "Global/read-only 2^8-entry FP64 table lookup; no numeric conversion."),
        ("L64-S", "The same FP64 table staged once per block in shared memory."),
        ("LP-G", "Global exact FP64-prefix lookup, followed by a shift into the FP64 bit layout."),
        ("LP-S", "The same exact prefix table staged once per block in shared memory."),
        ("LP-G-P", "Global prefix lookup with explicit next-packet software prefetching."),
        ("LHW-G", "Global/read-only 2^8-entry lookup of FP64's nonzero 32-bit high word."),
        ("LHW-S", "The 1 KiB FP64-high-word table staged once per block in shared memory."),
        ("LHW-SW", "Four padded shared high-word tables, selected by eight-lane warp groups."),
        ("SN-G", "Direct normal/special construction with a global subnormal-only high-word table."),
        ("SN-S", "The 128-byte E2M5 or 64-byte E3M4 subnormal table staged in shared memory."),
        ("DB64", "Construct sign, exponent, and fraction directly in the FP64 bit layout; no table."),
        ("DW-B", "Construct FP64's high word with explicit zero/subnormal/normal/special branches."),
        ("DW-M", "Construct FP64's high word with masks instead of the outer case branch."),
        ("DEC", "Lookup a sign/exponent FP64 prefix, then insert fraction bits and handle subnormals."),
        ("×1/×2/×4/×8", "Number of adjacent 8-bit codes read and decoded per packed source load."),
    )
    body = "".join(
        f"<tr><th><code>{html.escape(abbreviation)}</code></th>"
        f"<td>{html.escape(description)}</td></tr>"
        for abbreviation, description in rows
    )
    return f"""<section class="text-section">
<h2>Strategy abbreviations</h2>
<p>G means global/read-only lookup, S means shared-memory lookup, and P means explicit software pipelining. E3M4 prefix entries are exact 16-bit E11M4 prefixes; E2M5 needs 17 meaningful E11M5 prefix bits and stores each in 32 bits.</p>
<div class="table-wrap"><table class="strategy-table"><tbody>{body}</tbody></table></div>
</section>"""


def conversion_strategy_family(strategy: str, decode_kind: str) -> str:
    if strategy == "raw_pointer_x1" or decode_kind == "none":
        return "baseline"
    if decode_kind.startswith("native_"):
        return "native"
    if decode_kind in {"prefix_word", "prefix_high_lut", "lut_prefix"}:
        return "prefix"
    if decode_kind in {"e1_integer", "fixed_integer", "exponent_only"}:
        return "integer"
    if decode_kind in {
        "full_high_lut",
        "full_high_lut_swizzled",
        "warp_high_lut",
        "lut_high_word",
        "lut_high_word_swizzled",
    }:
        return "full_lut"
    if decode_kind in {"subnormal_high_lut", "lut_subnormal"}:
        return "subnormal_lut"
    if decode_kind in {"fp32_bits", "branchless_fp32", "lut_fp32"}:
        return "fp32_path"
    if decode_kind == "lut_fp64":
        return "fp64_lut"
    if decode_kind in {"pair_high_lut", "quad_high_lut"}:
        return "pair_lut"
    if decode_kind in {
        "direct_words_branchy",
        "direct_words_masked",
        "direct_fp64_bits",
        "direct_fp64_words_branchy",
        "direct_fp64_words_masked",
    }:
        return "direct_words"
    if decode_kind == "decomposed_bits":
        return "decomposed"
    if decode_kind in {"generic", "generic_fp64"}:
        return "generic"
    raise ValueError(f"unknown conversion strategy kind: {decode_kind} ({strategy})")


def conversion_strategy_abbreviation(strategy: str) -> str:
    if strategy == "raw_pointer_x1":
        return "F64"
    match = re.search(r"_x(1|2|4|8)$", strategy)
    lanes = match.group(1) if match else "?"
    base = strategy[: match.start()] if match else strategy
    abbreviations = (
        ("generic_fp64", "GEN"),
        ("generic", "GEN"),
        ("branchless_fp32", "BR32"),
        ("lut_fp32_shared", "L32-S"),
        ("lut_fp32_global_pipelined", "L32-G-P"),
        ("lut_fp32_global", "L32-G"),
        ("lut_fp64_shared", "L64-S"),
        ("lut_fp64_global", "L64-G"),
        ("lut_prefix_shared", "LP-S"),
        ("lut_prefix_global_pipelined", "LP-G-P"),
        ("lut_prefix_global", "LP-G"),
        ("lut_high_word_swizzled_shared", "LHW-SW"),
        ("lut_high_word_shared", "LHW-S"),
        ("lut_high_word_global", "LHW-G"),
        ("lut_subnormal_shared", "SN-S"),
        ("lut_subnormal_global", "SN-G"),
        ("direct_fp64_words_branchy", "DW-B"),
        ("direct_fp64_words_masked", "DW-M"),
        ("direct_fp64_bits", "DB64"),
        ("decomposed_bits", "DEC"),
        ("full_high_swizzled", "FH-SW"),
        ("full_high_warp", "FH-W"),
        ("full_high_shared", "FH-S"),
        ("full_high_global", "FH-G"),
        ("full_high_l2", "FH-L2"),
        ("subnormal_shared", "SN-S"),
        ("subnormal_global", "SN-G"),
        ("prefix_shared", "PX-S"),
        ("prefix_global", "PX-G"),
        ("prefix_word", "PW"),
        ("byte_quad_shared", "QUAD-S"),
        ("byte_quad_l2", "QUAD-L2"),
        ("pair_shared", "PAIR-S"),
        ("pair_l2", "PAIR-L2"),
        ("word_branchy_prmt", "DW-BP"),
        ("word_branchy", "DW-B"),
        ("word_masked", "DW-M"),
        ("e1_integer", "INT"),
        ("fixed_integer", "FIX"),
        ("exponent_only", "EXP"),
        ("fp32_bit_lift", "F32-LIFT"),
        ("fp32_bits", "F32-BITS"),
        ("native_bfloat162", "N-BF162"),
        ("native_half2", "N-HALF2"),
        ("native_half", "N-HALF"),
        ("native_float4", "N-FLOAT4"),
        ("native_float2", "N-FLOAT2"),
        ("native_direct", "N-DIRECT"),
        ("native_fp32", "N-F32"),
        ("native_f64", "N-F64"),
    )
    short = next((value for prefix, value in abbreviations if base.startswith(prefix)), None)
    if short is None:
        short = base.replace("_", "-").upper()
    return f"{short} ×{lanes}"


def conversion_strategy_sort_key(
    strategy: str, metadata: dict[str, dict[str, str]]
) -> tuple[int, int, str]:
    row = metadata[strategy]
    family = conversion_strategy_family(strategy, row.get("decode_kind", "none"))
    lanes = int(row.get("lanes", "1"))
    return CONVERSION_FAMILY_ORDER.index(family), -lanes, strategy


def format_conversion_rows(
    rows: Sequence[dict[str, str]], format_name: str
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        benchmark_format = row.get("benchmark_format", "")
        if row["strategy"] == "raw_pointer_x1":
            matches = benchmark_format == format_name or (
                not benchmark_format and format_name in {"e2m5", "e3m4"}
            )
        else:
            matches = row["format"] == format_name and (
                not benchmark_format or benchmark_format == format_name
            )
        if matches:
            selected.append(row)
    return selected


def conversion_strategy_metadata(
    rows: Sequence[dict[str, str]], format_name: str
) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for row in format_conversion_rows(rows, format_name):
        metadata.setdefault(row["strategy"], row)
    if "raw_pointer_x1" not in metadata:
        raise ValueError(f"missing raw FP64 baseline for {format_name}")
    return metadata


def conversion_series_gid(
    format_name: str, distribution: str, component: str, strategy: str
) -> str:
    return (
        f"conversion-series--{format_name}--{distribution}--"
        f"{component}--{strategy}"
    )


def plot_conversion_strategy_kernel_time(
    rows: Sequence[dict[str, str]], format_name: str, path: Path
) -> Path:
    current_format_rows = format_conversion_rows(rows, format_name)
    metadata = conversion_strategy_metadata(rows, format_name)
    strategies = sorted(
        metadata, key=lambda name: conversion_strategy_sort_key(name, metadata)
    )
    fig, axes = plt.subplots(1, 2, figsize=(21.0, 8.4))
    for axis, component in zip(axes, ("dot", "gemv")):
        for distribution in ("uniform_0_1", "normal_0_1"):
            for strategy in strategies:
                current = sorted(
                    (
                        row
                        for row in current_format_rows
                        if row["strategy"] == strategy
                        and row["component"] == component
                        and row["distribution"] == distribution
                    ),
                    key=lambda row: int(row["n"]),
                )
                if len(current) != 5:
                    raise ValueError(
                        f"incomplete {format_name}/{strategy}/{component}/"
                        f"{distribution} series: {len(current)} points"
                    )
                row = metadata[strategy]
                family = conversion_strategy_family(
                    strategy, row.get("decode_kind", "none")
                )
                is_baseline = family == "baseline"
                lanes = int(row.get("lanes", "1"))
                marker = (
                    "D"
                    if is_baseline
                    else "s"
                    if row.get("table_location") == "shared"
                    else "X"
                    if row.get("unpack") == "byte_permute"
                    else "P"
                    if family == "native"
                    else "^"
                    if family == "prefix"
                    else "o"
                )
                line = axis.plot(
                    [int(item["n"]) for item in current],
                    [number(item, "median_time_ms") for item in current],
                    color=CONVERSION_FAMILY_COLORS[family],
                    linestyle="-" if is_baseline else STRATEGY_LANE_STYLES[lanes],
                    linewidth=2.5 if is_baseline else 1.45,
                    marker=marker,
                    markersize=4.5 if is_baseline else 3.6,
                    markerfacecolor=(
                        CONVERSION_FAMILY_COLORS[family]
                        if is_baseline or row.get("table_location") == "shared"
                        else "white"
                    ),
                    alpha=1.0 if is_baseline else 0.82,
                )[0]
                line.set_gid(
                    conversion_series_gid(
                        format_name, distribution, component, strategy
                    )
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(component.upper(), fontsize=13)
        axis.set_xlabel("Reduction length N")
        axis.set_ylabel("Complete kernel time (ms)")
        format_axis_labels(axis)
    fig.suptitle(
        f"{label(format_name)} conversion strategies and raw FP64",
        y=0.995,
        fontsize=17,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, path)


def conversion_strategy_interactive_document(
    svg_path: Path,
    *,
    format_name: str,
    metadata: dict[str, dict[str, str]],
) -> str:
    svg = svg_path.read_text(encoding="utf-8")
    svg = svg[svg.index("<svg ") :]
    strategies = sorted(
        metadata, key=lambda name: conversion_strategy_sort_key(name, metadata)
    )
    chart_title = f"{label(format_name)} conversion strategy performance"
    chart_description = (
        f"Complete DOT and GEMV time versus N for all {label(format_name)} "
        "conversion strategies and a raw FP64 baseline. Lower is faster."
    )
    svg = svg.replace(
        "<svg ",
        '<svg class="strategy-chart" role="img" '
        'aria-labelledby="strategy-chart-title strategy-chart-description" ',
        1,
    )
    svg = svg.replace(
        ">",
        f'><title id="strategy-chart-title">{html.escape(chart_title)}</title>'
        f'<desc id="strategy-chart-description">'
        f"{html.escape(chart_description)}</desc>",
        1,
    )
    pattern = re.compile(
        rf'<g id="conversion-series--{format_name}--'
        r'(?P<distribution>uniform_0_1|normal_0_1)--'
        r'(?P<component>dot|gemv)--(?P<strategy>[a-z0-9_]+)">'
    )

    def series_attributes(match: re.Match[str]) -> str:
        strategy = match.group("strategy")
        row = metadata[strategy]
        family = conversion_strategy_family(strategy, row.get("decode_kind", "none"))
        lanes = "baseline" if family == "baseline" else row.get("lanes", "1")
        full_label = (
            f"{match.group('component').upper()}: "
            f"{conversion_strategy_abbreviation(strategy)}"
        )
        series_id = match.group(0).split(chr(34))[1]
        return (
            f'<g id="{series_id}" data-series="true" '
            f'data-distribution="{match.group("distribution")}" '
            f'data-component="{match.group("component")}" '
            f'data-strategy="{strategy}" data-family="{family}" '
            f'data-lanes="{lanes}"><title>{html.escape(full_label)}</title>'
        )

    svg, series_count = pattern.subn(series_attributes, svg)
    expected_series = 4 * len(strategies)
    if series_count != expected_series:
        raise ValueError(
            f"expected {expected_series} conversion series in {svg_path}, "
            f"found {series_count}"
        )

    distribution_options = "".join(
        f"""<label class="filter-option">
<input type="radio" name="distribution" value="{distribution}" {'checked' if distribution == 'normal_0_1' else ''}>
<span>{text}</span>
</label>"""
        for distribution, text in ACCURACY_DISTRIBUTIONS
    )
    available_lanes = sorted(
        {
            int(row.get("lanes", "1"))
            for strategy, row in metadata.items()
            if strategy != "raw_pointer_x1"
        }
    )
    lane_options = "".join(
        f"""<label class="filter-option lane-x{lane}">
<input type="checkbox" data-filter="lanes" value="{lane}" checked>
<span class="lane-sample" aria-hidden="true"></span><span>×{lane}</span>
</label>"""
        for lane in available_lanes
    )
    available_families = [
        family
        for family in CONVERSION_FAMILY_ORDER
        if any(
            conversion_strategy_family(name, row.get("decode_kind", "none"))
            == family
            for name, row in metadata.items()
        )
    ]
    family_options = "".join(
        f"""<label class="filter-option">
<input type="checkbox" data-filter="family" value="{family}" checked>
<span class="family-swatch" style="--series-color: {CONVERSION_FAMILY_COLORS[family]}" aria-hidden="true"></span>
<span>{html.escape(CONVERSION_FAMILY_LABELS[family])}</span>
</label>"""
        for family in available_families
    )
    strategy_options = "".join(
        f"""<label class="filter-option strategy-option" title="{html.escape(strategy)}">
<input type="checkbox" data-filter="strategy" value="{strategy}" checked>
<span class="family-swatch" style="--series-color: {CONVERSION_FAMILY_COLORS[conversion_strategy_family(strategy, metadata[strategy].get('decode_kind', 'none'))]}" aria-hidden="true"></span>
<code>{html.escape(conversion_strategy_abbreviation(strategy))}</code>
</label>"""
        for strategy in strategies
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(chart_title)}</title>
<style>
:root {{ color-scheme: light; --fg: #1f252b; --muted: #5d6872; --focus: #075f9a; --border: #d8dee4; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #fff; color: var(--fg); font: 15px/1.4 system-ui, sans-serif; }}
.chart-root {{ padding: 14px 16px 2px; }}
.filters {{ display: grid; gap: 10px; margin-bottom: 8px; }}
fieldset {{ border: 0; margin: 0; padding: 0; }}
legend {{ color: var(--muted); font-size: 0.86rem; font-weight: 500; margin-bottom: 4px; }}
.filter-options {{ display: flex; flex-wrap: wrap; gap: 5px 14px; }}
.filter-option {{ align-items: center; cursor: pointer; display: inline-flex; gap: 6px; min-height: 28px; white-space: nowrap; }}
.filter-option input {{ accent-color: var(--focus); height: 16px; margin: 0; width: 16px; }}
.filter-option:has(input:focus-visible) {{ outline: 2px solid var(--focus); outline-offset: 2px; }}
.family-swatch {{ background: var(--series-color); border-radius: 50%; height: 9px; width: 18px; }}
.lane-sample {{ border-top: 2px solid #4c5661; height: 0; width: 22px; }}
.lane-x1 .lane-sample {{ border-top-style: dotted; }}
.lane-x2 .lane-sample {{ border-top-style: dashed; }}
.lane-x4 .lane-sample {{ border-top-style: solid; }}
.lane-x8 .lane-sample {{ border-top-style: double; border-top-width: 3px; }}
details {{ border-top: 1px solid var(--border); padding-top: 7px; }}
summary {{ color: var(--muted); cursor: pointer; }}
.strategy-option code {{ font-size: 0.88rem; }}
.visible-status {{ color: var(--muted); margin: 2px 0 4px; }}
.chart-wrap {{ width: 100%; }}
.strategy-chart {{ display: block; height: auto; max-width: none; width: 100%; }}
.strategy-chart [data-series] {{ transition: opacity 120ms ease; }}
.strategy-chart .series-hidden {{ display: none; }}
@media (prefers-reduced-motion: reduce) {{ .strategy-chart [data-series] {{ transition: none; }} }}
@media (max-width: 620px) {{ .chart-root {{ padding-inline: 8px; }} .filter-options {{ column-gap: 10px; }} }}
</style>
</head>
<body>
<main id="interactive-{format_name}-conversion" class="chart-root">
<section class="filters" aria-label="Chart filters">
<fieldset><legend>Input distribution</legend><div class="filter-options">{distribution_options}</div></fieldset>
<fieldset><legend>Packed load width</legend><div class="filter-options">{lane_options}</div></fieldset>
<fieldset><legend>Conversion family / reference</legend><div class="filter-options">{family_options}</div></fieldset>
<details><summary>Individual strategy toggles</summary><div class="filter-options">{strategy_options}</div></details>
</section>
<p id="visible-status" class="visible-status" aria-live="polite"></p>
<div class="chart-wrap">{svg}</div>
</main>
<script>
const root = document.getElementById('interactive-{format_name}-conversion');
const status = root.querySelector('#visible-status');
const controls = [...root.querySelectorAll('input')];
const series = [...root.querySelectorAll('[data-series]')];

function checked(kind, value) {{
  const input = controls.find(control => control.dataset.filter === kind && control.value === value);
  return input ? input.checked : true;
}}

function reportHeight() {{
  if (window.parent !== window) {{
    const height = Math.ceil(root.getBoundingClientRect().bottom + 2);
    window.parent.postMessage({{ type: 'performance-chart-height', height }}, '*');
  }}
}}

function updateSeries() {{
  const distribution = root.querySelector('input[name="distribution"]:checked').value;
  let visible = 0;
  for (const group of series) {{
    const widthEnabled = group.dataset.lanes === 'baseline' || checked('lanes', group.dataset.lanes);
    const show = group.dataset.distribution === distribution
      && widthEnabled
      && checked('family', group.dataset.family)
      && checked('strategy', group.dataset.strategy);
    group.classList.toggle('series-hidden', !show);
    group.setAttribute('aria-hidden', String(!show));
    if (show) visible += 1;
  }}
  status.textContent = `${{visible / 2}} of ${{series.length / 4}} lines visible per kernel · lower is faster`;
  requestAnimationFrame(reportHeight);
}}

root.addEventListener('change', updateSeries);
window.addEventListener('message', event => {{
  if (event.data?.type === 'request-performance-chart-height') reportHeight();
}});
window.addEventListener('load', updateSeries);
if ('ResizeObserver' in window) new ResizeObserver(reportHeight).observe(root);
</script>
</body>
</html>
"""


def write_conversion_strategy_chart(
    svg_path: Path,
    output_path: Path,
    *,
    format_name: str,
    metadata: dict[str, dict[str, str]],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        conversion_strategy_interactive_document(
            svg_path, format_name=format_name, metadata=metadata
        ),
        encoding="utf-8",
    )
    return output_path


def strategy_placement(strategy: str, row: dict[str, str]) -> str:
    if strategy == "raw_pointer_x1":
        return "Raw global FP64"
    location = row.get("table_location", "none")
    if location == "shared":
        size = int(row.get("shared_table_bytes", "0") or 0)
        return f"Shared table ({size:,} B)"
    if location == "global":
        return "Global/L2 table" if "_l2_" in strategy else "Global/read-only table"
    if location == "warp_register":
        return "Warp-register table"
    return "Registers"


def strategy_explanation(format_name: str, strategy: str, row: dict[str, str]) -> str:
    kind = row.get("decode_kind", "none")
    bits = int(row.get("storage_bits", "64"))
    pair_entries = 1 << (2 * bits) if bits <= 8 else 0
    descriptions = {
        "none": "Raw FP64 load with no storage decoding.",
        "generic": "Generic format codec decodes one extracted lane directly to FP64.",
        "generic_fp64": "Generic format codec decodes one extracted lane directly to FP64.",
        "e1_integer": "Treats E1 data as a signed fixed-point integer and applies an exact power-of-two FP64 scale.",
        "fixed_integer": "Treats exponent-free data as a signed fixed-point integer and applies an exact power-of-two FP64 scale.",
        "exponent_only": "Decodes sign and exponent-only data with compile-time-specialized integer bit construction.",
        "fp32_bits": "Constructs the exact FP32 bit pattern, then performs the FP32-to-FP64 numeric conversion.",
        "direct_words_branchy": "Constructs FP64 high/low words directly, with explicit branches for zero, subnormal, normal, and special values.",
        "direct_words_masked": "Constructs FP64 words directly and selects cases with masks instead of the outer case branch.",
        "full_high_lut": f"Looks up the exact FP64 high word in a complete 2^{bits}-entry table; the low word is inserted directly.",
        "full_high_lut_swizzled": "Uses duplicated, padded shared high-word tables to reduce random-index bank conflicts.",
        "warp_high_lut": "Distributes the complete FP64-high-word table across warp registers and fetches indexed entries with warp shuffles.",
        "subnormal_high_lut": "Builds normal/special values directly and uses a compact high-word lookup only for exponent-zero values.",
        "prefix_high_lut": "Looks up the sign/exponent FP64 prefix and inserts the source fraction bits directly.",
        "prefix_word": "Moves the stored E11 prefix and fraction directly into the FP64 words; no numeric conversion or table.",
        "pair_high_lut": f"Decodes two adjacent {bits}-bit codes with one {pair_entries:,}-entry lookup that returns two FP64 high words.",
        "quad_high_lut": "Decodes a dense packet of four 2-bit or 4-bit codes with one lookup that returns four FP64 high words.",
        "native_direct": "Uses CUDA's native scalar source conversion directly to FP64.",
        "native_fp32": "Uses CUDA's native scalar conversion to FP32, followed by FP32-to-FP64.",
        "native_packed": "Uses CUDA's native packed vector conversion, then widens each decoded lane to FP64.",
        "native_half2": "Converts native packed FP8 through half2/float lanes before widening to FP64.",
        "branchless_fp32": "Branch-free integer/bit decode to FP32, followed by FP32-to-FP64.",
        "lut_fp32": "Looks up a 256-entry FP32 value and then widens it numerically to FP64.",
        "lut_fp64": "Looks up a complete 256-entry FP64 value; no numeric conversion is needed.",
        "lut_prefix": "Looks up an exact compact FP64 prefix, then shifts it into the FP64 bit layout.",
        "lut_high_word": "Looks up FP64's exact 32-bit high word; the low word is zero.",
        "lut_high_word_swizzled": "Looks up FP64's exact high word through duplicated, padded shared tables that reduce random-index bank conflicts.",
        "lut_subnormal": "Constructs common cases directly and uses a compact lookup only for source subnormals.",
        "direct_fp64_bits": "Constructs the FP64 representation directly from source sign, exponent, and fraction bits.",
        "direct_fp64_words_branchy": "Constructs FP64's 32-bit words directly with explicit case branches.",
        "direct_fp64_words_masked": "Constructs FP64's words directly with masked case selection.",
        "decomposed_bits": "Looks up the sign/exponent prefix, inserts fraction bits, and handles subnormals separately.",
    }
    description = descriptions.get(kind, f"Uses the {kind.replace('_', ' ')} decoder.")
    if row.get("unpack") == "byte_permute":
        description += " Packed bytes are extracted with byte-permute instructions."
    if row.get("pipelined") == "1":
        description += " The next packed lookup is explicitly prefetched."
    if format_name == "bf16_e8m7" and strategy.startswith("fp32_bit_lift"):
        description = "Shifts the BF16 payload into an FP32 bit pattern, then widens FP32 to FP64."
    return description


def conversion_top_picks(
    rows: Sequence[dict[str, str]],
    format_name: str,
    *,
    allowed_lanes: set[int],
    title: str,
) -> str:
    format_rows = [
        row
        for row in format_conversion_rows(rows, format_name)
        if row["strategy"] != "raw_pointer_x1"
        and int(row.get("lanes", "1")) in allowed_lanes
    ]
    table_rows = []
    for component in ("dot", "gemv"):
        component_rows = [row for row in format_rows if row["component"] == component]
        sizes = sorted({int(row["n"]) for row in component_rows})
        large_sizes = set(sizes[-2:])
        by_strategy: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in component_rows:
            by_strategy[row["strategy"]].append(row)
        wins: dict[str, int] = defaultdict(int)
        for distribution in dict(ACCURACY_DISTRIBUTIONS):
            for size in sizes:
                case = [
                    row
                    for row in component_rows
                    if row["distribution"] == distribution and int(row["n"]) == size
                ]
                wins[min(case, key=lambda row: number(row, "median_time_ms"))["strategy"]] += 1

        ranking = []
        for strategy, strategy_rows in by_strategy.items():
            values = [number(row, "speedup_vs_fp64") for row in strategy_rows]
            large_values = [
                number(row, "speedup_vs_fp64")
                for row in strategy_rows
                if int(row["n"]) in large_sizes
            ]
            ranking.append(
                (
                    math.exp(statistics.fmean(math.log(value) for value in values)),
                    math.exp(
                        statistics.fmean(math.log(value) for value in large_values)
                    ),
                    strategy,
                    wins[strategy],
                )
            )
        ranking.sort(reverse=True)
        for pick, (all_speedup, large_speedup, strategy, win_count) in enumerate(
            ranking[:3], start=1
        ):
            lanes = int(by_strategy[strategy][0].get("lanes", "1"))
            table_rows.append(
                "<tr>"
                f"<th>{component.upper()}</th><td>{pick}</td>"
                f"<td><code>{html.escape(conversion_strategy_abbreviation(strategy))}</code>"
                f"<small class=\"strategy-full-name\">{html.escape(strategy)}</small></td>"
                f"<td>×{lanes}</td><td>{all_speedup:.2f}×</td>"
                f"<td>{large_speedup:.2f}×</td><td>{win_count}/{len(sizes) * 2}</td>"
                "</tr>"
            )
    return f"""<section class="text-section">
<h2>{html.escape(title)}</h2>
<p>Up to three picks per kernel, ranked by geometric-mean speedup over raw FP64 across both distributions and all five N values. “Large N” uses the two largest values; wins are recomputed only within this table's width scope.</p>
<div class="table-wrap"><table class="strategy-table strategy-picks"><thead><tr><th>Kernel</th><th>Pick</th><th>Strategy</th><th>Width</th><th>All-N speedup</th><th>Large-N speedup</th><th>Wins</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
</section>"""


def best_large_n_conversion_score(
    rows: Sequence[dict[str, str]],
    format_name: str,
    component: str,
    allowed_lanes: set[int],
) -> tuple[float, str]:
    candidates = [
        row
        for row in format_conversion_rows(rows, format_name)
        if row["strategy"] != "raw_pointer_x1"
        and row["component"] == component
        and int(row.get("lanes", "1")) in allowed_lanes
    ]
    sizes = sorted({int(row["n"]) for row in candidates})
    large_sizes = set(sizes[-2:])
    by_strategy: dict[str, list[float]] = defaultdict(list)
    for row in candidates:
        if int(row["n"]) in large_sizes:
            by_strategy[row["strategy"]].append(
                number(row, "speedup_vs_fp64")
            )
    return max(
        (
            math.exp(statistics.fmean(math.log(value) for value in values)),
            strategy,
        )
        for strategy, values in by_strategy.items()
    )


def more_precise_non_fp64_formats(format_name: str) -> list[str]:
    exponent_bits, mantissa_bits = format_layout_bits(format_name)
    return [
        candidate
        for candidate in CONVERSION_FORMATS
        if candidate != format_name
        and format_layout_bits(candidate)[0] >= exponent_bits
        and format_layout_bits(candidate)[1] >= mantissa_bits
        and (
            format_layout_bits(candidate)[0] > exponent_bits
            or format_layout_bits(candidate)[1] > mantissa_bits
        )
    ]


def format_performance_comparison_table(
    rows: Sequence[dict[str, str]], format_name: str
) -> str:
    scopes = (("all", {1, 2, 4, 8}), ("unpacked", {1}))
    scores = {
        (component, scope): best_large_n_conversion_score(
            rows, format_name, component, lanes
        )[0]
        for component in ("dot", "gemv")
        for scope, lanes in scopes
    }
    more_precise = more_precise_non_fp64_formats(format_name)

    def competitor_cell(component: str, scope: str, lanes: set[int]) -> str:
        if not more_precise:
            return '<span class="comparison-empty">—</span>'
        competitor, competitor_score = max(
            (
                (
                    candidate,
                    best_large_n_conversion_score(
                        rows, candidate, component, lanes
                    )[0],
                )
                for candidate in more_precise
            ),
            key=lambda item: item[1],
        )
        ratio = scores[component, scope] / competitor_score
        return (
            f'<span class="ratio-value">{ratio:.3f}×</span>'
            f'<span class="ratio-peer">vs <a href="'
            f'{conversion_strategy_filename(competitor)}">'
            f'{html.escape(label(competitor))}</a></span>'
        )

    body = []
    for component in ("dot", "gemv"):
        all_lanes = {1, 2, 4, 8}
        unpacked_lanes = {1}
        body.append(
            "<tr>"
            f"<th>{component.upper()}</th>"
            f'<td><span class="ratio-value">{scores[component, "all"]:.3f}×</span></td>'
            f'<td><span class="ratio-value">{scores[component, "unpacked"]:.3f}×</span></td>'
            f"<td>{competitor_cell(component, 'all', all_lanes)}</td>"
            f"<td>{competitor_cell(component, 'unpacked', unpacked_lanes)}</td>"
            "</tr>"
        )
    return f"""<section class="text-section performance-comparison">
<h2>Historical performance summary</h2>
<p>Each value uses the best strategy in its width scope and the geometric mean over both distributions at the two largest N values. Higher-precision peers have no fewer exponent or mantissa bits and strictly more of at least one. A peer ratio above 1 means {html.escape(label(format_name))} is faster; — means FP64 is the only higher-precision format.</p>
<div class="table-wrap"><table class="strategy-table comparison-table"><thead><tr><th>Kernel</th><th>vs FP64 · %all</th><th>vs FP64 · %unpacked</th><th>vs best higher precision · %all</th><th>vs best higher precision · %unpacked</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>
</section>"""


def conversion_strategy_explanations(
    rows: Sequence[dict[str, str]], format_name: str
) -> str:
    metadata = conversion_strategy_metadata(rows, format_name)
    strategies = sorted(
        metadata, key=lambda name: conversion_strategy_sort_key(name, metadata)
    )
    body = []
    for strategy in strategies:
        row = metadata[strategy]
        family = conversion_strategy_family(strategy, row.get("decode_kind", "none"))
        body.append(
            "<tr>"
            f"<th><code>{html.escape(conversion_strategy_abbreviation(strategy))}</code>"
            f"<small class=\"strategy-full-name\">{html.escape(strategy)}</small></th>"
            f"<td>×{row.get('lanes', '1')}</td>"
            f"<td>{html.escape(CONVERSION_FAMILY_LABELS[family])}</td>"
            f"<td>{html.escape(strategy_placement(strategy, row))}</td>"
            f"<td>{html.escape(strategy_explanation(format_name, strategy, row))}</td>"
            "</tr>"
        )
    return f"""<section class="text-section">
<h2>Strategy reference</h2>
<p>Every plotted implementation is listed here. Width is the number of adjacent stored values consumed by one packed source load.</p>
<div class="table-wrap"><table class="strategy-table strategy-reference"><thead><tr><th>Label</th><th>Width</th><th>Family</th><th>Data path</th><th>What it does</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>
</section>"""


def conversion_access_scopes(format_name: str) -> tuple[str, ...]:
    if format_total_bits(format_name) in {8, 16, 32}:
        return ("Aligned x1", "Aligned best")
    return ("Dense x1", "Padded x1", "Dense best", "Padded best")


def pending_metric_table(title: str, scopes: Sequence[str]) -> str:
    headings = "".join(f"<th>{html.escape(scope)}</th>" for scope in scopes)
    cells = "".join(
        '<td><span class="pending-value">—</span></td>' for _ in scopes
    )
    return f"""<div class="comparison-block"><h3>{html.escape(title)}</h3>
<div class="table-wrap"><table class="strategy-table pending-comparison-table"><thead><tr><th>Kernel</th>{headings}</tr></thead><tbody>
<tr><th>DOT</th>{cells}</tr><tr><th>GEMV</th>{cells}</tr>
</tbody></table></div></div>"""


def pending_top_picks_table(
    title: str, scopes: Sequence[str], *, best_width: bool
) -> str:
    rows = []
    selected = [scope for scope in scopes if ("best" in scope.lower()) == best_width]
    for component in ("DOT", "GEMV"):
        for scope in selected:
            rows.append(
                "<tr>"
                f"<th>{component}</th><td>{html.escape(scope)}</td>"
                '<td class="pending-value">—</td><td>—</td><td>—</td><td>—</td>'
                "</tr>"
            )
    return f"""<section class="text-section pending-section">
<h2>{html.escape(title)}</h2>
<div class="table-wrap"><table class="strategy-table strategy-picks"><thead><tr><th>Kernel</th><th>Access scope</th><th>Strategy</th><th>Width</th><th>Large-N speedup</th><th>Wins</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
</section>"""


def pending_dense_padded_table(format_name: str) -> str:
    if format_total_bits(format_name) in {8, 16, 32}:
        return ""
    rows = "".join(
        "<tr>"
        f"<th>{html.escape(label_)}</th>"
        '<td class="pending-value">—</td><td class="pending-value">—</td>'
        '<td class="pending-value">—</td><td class="pending-value">—</td>'
        "</tr>"
        for label_ in ("DOT x1", "DOT best", "GEMV x1", "GEMV best")
    )
    return f"""<section class="text-section pending-section">
<h2>Dense versus padded</h2>
<div class="table-wrap"><table class="strategy-table layout-comparison-table"><thead><tr><th>Case</th><th>Dense winner</th><th>Padded winner</th><th>Faster layout</th><th>Speedup</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>"""


def pending_conversion_sections(compute: str, format_name: str) -> str:
    compute_label = compute.upper()
    scopes = conversion_access_scopes(format_name)
    status = f"""<section class="experiment-status pending-section" aria-label="Experiment status">
<span class="status-badge">Pending H200 data</span>
<strong>Experiment 021 · {html.escape(label(format_name))} → {compute_label}</strong>
</section>"""
    summaries = f"""<section class="text-section pending-section">
<h2>Cross-format performance</h2>
<div class="comparison-grid">
{pending_metric_table(f'Speedup versus raw {compute_label}', scopes)}
{pending_metric_table('Speedup versus best higher-precision format', scopes)}
</div>
</section>"""
    chart = f"""<section class="graph-section pending-section">
<h2>Complete kernel time versus N</h2>
<div class="chart-placeholder" role="img" aria-label="Pending DOT and GEMV strategy timing graph">
<div><strong>DOT</strong><span>N →</span></div><div><strong>GEMV</strong><span>N →</span></div>
<p>{html.escape(label(format_name))} → {compute_label} · U(0,1) and N(0,1)</p>
</div>
</section>"""
    reference = """<section class="text-section pending-section">
<h2>Strategy reference</h2>
<div class="table-wrap"><table class="strategy-table strategy-reference"><thead><tr><th>Label</th><th>Layout</th><th>Access</th><th>Width</th><th>Decoder</th></tr></thead><tbody><tr><td class="pending-value" colspan="5">Awaiting validated strategy inventory</td></tr></tbody></table></div>
</section>"""
    return (
        status
        + summaries
        + pending_dense_padded_table(format_name)
        + chart
        + pending_top_picks_table("Top picks — x1", scopes, best_width=False)
        + pending_top_picks_table("Top picks — best access width", scopes, best_width=True)
        + reference
    )


def _accuracy_rows(
    rows: Sequence[dict[str, str]],
    *,
    component: str,
    distribution: str,
    format_name: str,
) -> list[dict[str, str]]:
    return sorted(
        (
            row
            for row in rows
            if row["kernel"] == component
            and row["distribution"] == distribution
            and row["comparison"] == "total_x4"
            and row["format"] == format_name
        ),
        key=lambda row: int(row["n"]),
    )


def _accuracy_floor(
    rows: Sequence[dict[str, str]],
    formats: Sequence[str],
    component: str,
    distribution: str,
) -> float:
    metric, _ = ACCURACY_METRICS[component]
    positive = [
        number(row, metric)
        for name in formats
        for row in _accuracy_rows(
            rows,
            component=component,
            distribution=distribution,
            format_name=name,
        )
        if math.isfinite(number(row, metric)) and number(row, metric) > 0.0
    ]
    if not positive:
        raise ValueError(f"no finite positive {component} accuracy values")
    return min(positive) / 3.0


def _plot_accuracy_lines(
    axis: plt.Axes,
    rows: Sequence[dict[str, str]],
    formats: Sequence[str],
    *,
    component: str,
    distribution: str,
    colors: dict[str, object],
    markers: dict[str, str],
) -> None:
    metric, y_label = ACCURACY_METRICS[component]
    floor = _accuracy_floor(rows, formats, component, distribution)
    showed_zero = False
    for name in formats:
        current = _accuracy_rows(
            rows,
            component=component,
            distribution=distribution,
            format_name=name,
        )
        if not current:
            raise ValueError(f"missing total_x4 accuracy rows for {component} {name}")
        raw_values = [number(row, metric) for row in current]
        showed_zero = showed_zero or any(value == 0.0 for value in raw_values)
        values = [
            floor if value == 0.0 else value if math.isfinite(value) else math.nan
            for value in raw_values
        ]
        axis.plot(
            [int(row["n"]) for row in current],
            values,
            color=colors[name],
            marker=markers[name],
            linewidth=1.9,
            markersize=5.2,
            label=label(name),
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    distribution_label = dict(ACCURACY_DISTRIBUTIONS)[distribution]
    axis.set_title(
        f"{component.upper()} — {distribution_label}",
        fontsize=13,
    )
    axis.set_xlabel("Reduction length N")
    axis.set_ylabel(y_label)
    if showed_zero:
        axis.axhline(floor, color="#8b939c", linewidth=0.8, linestyle=":")
        axis.text(
            0.01,
            0.015,
            "Exact zero shown at plot floor",
            transform=axis.transAxes,
            fontsize=8.5,
            color="#59636e",
            va="bottom",
        )
    format_axis_labels(axis)


def plot_same_bit_accuracy(
    rows: Sequence[dict[str, str]], storage_bits: int, path: Path
) -> Path:
    formats = BIT_FORMATS[storage_bits]
    colors = dict(zip(formats, SAME_BIT_COLORS))
    markers = dict(zip(formats, FORMAT_MARKERS))
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 12.5))
    for row_index, (distribution, _) in enumerate(ACCURACY_DISTRIBUTIONS):
        for column_index, component in enumerate(("dot", "gemv")):
            _plot_accuracy_lines(
                axes[row_index, column_index],
                rows,
                formats,
                component=component,
                distribution=distribution,
                colors=colors,
                markers=markers,
            )
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[name],
            marker=markers[name],
            linewidth=2.0,
            label=label(name),
        )
        for name in formats
    ]
    fig.legend(
        handles,
        [label(name) for name in formats],
        loc="upper center",
        ncol=len(formats),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        f"{storage_bits}-bit formats: x4 GPU result versus source reference",
        y=1.02,
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, path)


def plot_all_format_accuracy(
    rows: Sequence[dict[str, str]], path: Path
) -> Path:
    palette = plt.get_cmap("tab20")
    colors = {name: palette(index) for index, name in enumerate(FORMAT_ORDER)}
    markers = {
        name: BIT_MARKERS[bits]
        for bits, formats in ALL_BIT_FORMATS.items()
        for name in formats
    }
    storage_bits_by_format = {
        name: bits
        for bits, formats in ALL_BIT_FORMATS.items()
        for name in formats
    }
    fig, axes = plt.subplots(2, 2, figsize=(21.0, 13.5))
    for row_index, (distribution, _) in enumerate(ACCURACY_DISTRIBUTIONS):
        for column_index, component in enumerate(("dot", "gemv")):
            _plot_accuracy_lines(
                axes[row_index, column_index],
                rows,
                FORMAT_ORDER,
                component=component,
                distribution=distribution,
                colors=colors,
                markers=markers,
            )
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[name],
            marker=markers[name],
            linewidth=2.0,
            label=f"{label(name)} ({storage_bits_by_format[name]}b)",
        )
        for name in FORMAT_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        title="Storage format; marker shape also identifies storage width",
    )
    fig.suptitle(
        "All storage formats: x4 GPU accuracy versus N",
        y=1.045,
        fontsize=17,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    return save_figure(fig, path)


def plot_accuracy_performance_tradeoff(
    timing_rows: Sequence[dict[str, str]],
    accuracy_rows: Sequence[dict[str, str]],
    path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for axis, component in zip(axes, ("dot", "gemv")):
        performance = {
            row["format"]: row
            for row in max_size_rows(timing_rows, component=component)
        }
        candidates = [
            row
            for row in accuracy_rows
            if row["kernel"] == component
            and row["distribution"] == "normal_0_1"
        ]
        accuracy_n = max(int(row["n"]) for row in candidates)
        accuracy = {
            row["format"]: row
            for row in candidates
            if int(row["n"]) == accuracy_n
        }
        positive = [
            number(row, "measured_rmse")
            for row in accuracy.values()
            if math.isfinite(number(row, "measured_rmse"))
            and number(row, "measured_rmse") > 0.0
        ]
        floor = min(positive) / 5.0
        for name in FORMAT_ORDER:
            measured = number(accuracy[name], "measured_rmse")
            if not math.isfinite(measured):
                continue
            exact_zero = measured == 0.0
            x = number(performance[name], "median_time_ms")
            y = floor if exact_zero else measured
            axis.scatter(
                x,
                y,
                s=58,
                marker=BIT_MARKERS[int(performance[name]["storage_bits"])],
                facecolors="none" if exact_zero else FAMILY_COLORS[format_family(name)],
                edgecolors=FAMILY_COLORS[format_family(name)],
                linewidth=1.2,
                zorder=3,
            )
            if name in SELECTED_FORMATS:
                axis.annotate(
                    label(name),
                    (x, y),
                    xytext=LABEL_OFFSETS[name],
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(
            f"{component.upper()} — performance N={int(next(iter(performance.values()))['n']):,}; accuracy N={accuracy_n:,}"
        )
        axis.set_xlabel("Complete x4 kernel time (ms) — lower is better")
        axis.set_ylabel("Measured storage RMSE — lower is better")
        axis.text(
            0.03,
            0.04,
            "Exact zero is shown at the plot floor",
            transform=axis.transAxes,
            fontsize=8,
        )
        format_axis_labels(axis)
    add_family_legend(fig, y=1.02)
    fig.suptitle("Performance–accuracy trade-off on N(0,1)", y=1.11, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, path)


def _lane_speedup(
    rows: Sequence[dict[str, str]], component: str, format_name: str, lane: int
) -> float:
    candidates = [
        row
        for row in rows
        if is_row(
            row,
            component=component,
            distribution="normal_0_1",
            lanes=lane,
            format_name=format_name,
        )
    ]
    maximum = max(int(row["n"]) for row in candidates)
    current = next(row for row in candidates if int(row["n"]) == maximum)
    return number(current, "speedup_vs_x1")


def plot_packed_speedup(rows: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    positions = np.arange(len(FORMAT_ORDER))
    height = 0.34
    for axis, component in zip(axes, ("dot", "gemv")):
        x2 = [_lane_speedup(rows, component, name, 2) for name in FORMAT_ORDER]
        x4 = [_lane_speedup(rows, component, name, 4) for name in FORMAT_ORDER]
        axis.axvline(1.0, color="#66707a", linewidth=1.0)
        axis.barh(positions - height / 2, x2, height, color=LANE_COLORS[2], label="x2")
        axis.barh(positions + height / 2, x4, height, color=LANE_COLORS[4], label="x4")
        axis.set_title(component.upper())
        axis.set_xlabel("Throughput vs x1 — higher is better")
        axis.set_xlim(0, max(max(x2), max(x4)) * 1.14)
        axis.invert_yaxis()
        format_axis_labels(axis)
    axes[0].set_yticks(positions, [label(name) for name in FORMAT_ORDER])
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("x2/x4 packed access versus scalar access", y=1.01, fontsize=14)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_packing_by_size(rows: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    for axis, component in zip(axes, ("dot", "gemv")):
        for name in SELECTED_FORMATS:
            scalar = {
                int(row["n"]): number(row, "median_time_ms")
                for row in rows
                if is_row(
                    row,
                    component=component,
                    distribution="normal_0_1",
                    lanes=1,
                    format_name=name,
                )
            }
            packed = sorted(
                (
                    row
                    for row in rows
                    if is_row(
                        row,
                        component=component,
                        distribution="normal_0_1",
                        lanes=4,
                        format_name=name,
                    )
                ),
                key=lambda row: int(row["n"]),
            )
            axis.plot(
                [int(row["n"]) for row in packed],
                [scalar[int(row["n"])] / number(row, "median_time_ms") for row in packed],
                marker=BIT_MARKERS[int(packed[0]["storage_bits"])],
                markersize=4,
                linewidth=1.5,
                color=FAMILY_COLORS[format_family(name)],
                label=label(name),
            )
        axis.axhline(1.0, color="#66707a", linewidth=1.0)
        axis.set_xscale("log", base=2)
        axis.set_title(component.upper())
        axis.set_xlabel("Reduction length N")
        axis.set_ylabel("x4 time speedup over x1")
        format_axis_labels(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Packing starts helping only after launch overhead is amortized", y=1.12, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(fig, path)


def roof_value(arithmetic_intensity: float, bandwidth: float, fp64_peak: float) -> float:
    return min(fp64_peak, arithmetic_intensity * bandwidth)


def memory_ceiling(profile: Sequence[dict[str, str]]) -> float:
    values = [
        number(row, "inferred_sustained_dram_gb_per_s")
        for row in profile
        if row["bottleneck_class"] == "memory_bandwidth"
        and math.isfinite(number(row, "inferred_sustained_dram_gb_per_s"))
    ]
    if not values:
        values = [number(row, "inferred_sustained_dram_gb_per_s") for row in profile]
    return statistics.median(values)


def fp64_ceiling(rows: Sequence[dict[str, str]]) -> float:
    return statistics.median(number(row, "modeled_fp64_gflop_per_s") for row in rows)


def plot_roofline(
    rows: Sequence[dict[str, str]], profile: Sequence[dict[str, str]], path: Path
) -> Path:
    bandwidth = memory_ceiling(profile)
    peak = fp64_ceiling(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for axis, component in zip(axes, ("dot", "gemv")):
        current = max_size_rows(rows, component=component)
        intensities = [number(row, "arithmetic_intensity_unique") for row in current]
        throughput = [number(row, "median_useful_gflop_per_s") for row in current]
        x_min = min(intensities) / 1.8
        x_max = max(max(intensities) * 2.0, peak / bandwidth * 1.8)
        xs = np.logspace(math.log10(x_min), math.log10(x_max), 300)
        ys = np.minimum(xs * bandwidth, peak)
        axis.plot(xs, ys, color="#39434d", linewidth=1.6, label="Measured HBM roof")
        for row, x, y in zip(current, intensities, throughput):
            name = row["format"]
            axis.scatter(
                x,
                y,
                s=54,
                marker=BIT_MARKERS[int(row["storage_bits"])],
                color=FAMILY_COLORS[format_family(name)],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            if name in SELECTED_FORMATS:
                axis.annotate(
                    label(name),
                    (x, y),
                    xytext=LABEL_OFFSETS[name],
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(min(throughput) / 1.8, peak * 1.5)
        axis.set_title(component.upper())
        axis.set_xlabel("Useful FP64 operations per unique encoded byte")
        axis.set_ylabel("Useful GFLOP/s from event timing")
        axis.text(
            0.03,
            0.95,
            f"HBM ceiling: {bandwidth / 1000:.2f} TB/s",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        format_axis_labels(axis)
    add_family_legend(fig, y=1.02)
    fig.suptitle("Compression-aware algorithmic roofline", y=1.11, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, path)


def plot_roof_efficiency(
    rows: Sequence[dict[str, str]], profile: Sequence[dict[str, str]], path: Path
) -> Path:
    bandwidth = memory_ceiling(profile)
    peak = fp64_ceiling(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    positions = np.arange(len(FORMAT_ORDER))
    for axis, component in zip(axes, ("dot", "gemv")):
        current = max_size_rows(rows, component=component)
        values = [
            100.0
            * number(row, "median_useful_gflop_per_s")
            / roof_value(number(row, "arithmetic_intensity_unique"), bandwidth, peak)
            for row in current
        ]
        colors = [FAMILY_COLORS[format_family(row["format"])] for row in current]
        bars = axis.barh(positions, values, color=colors, height=0.7)
        axis.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=3, fontsize=8)
        axis.set_title(component.upper())
        axis.set_xlabel("Fraction of compression-aware roof")
        axis.set_xlim(0, max(105.0, max(values) * 1.15))
        axis.invert_yaxis()
        format_axis_labels(axis)
    axes[0].set_yticks(positions, [label(name) for name in FORMAT_ORDER])
    add_family_legend(fig)
    fig.suptitle("How much of each format's attainable roof is reached", y=1.03, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, path)


def plot_register_conversion(rows: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(12.5, 7.4))
    positions = np.arange(len(FORMAT_ORDER))
    width = 0.23
    for offset, lane in zip((-width, 0.0, width), (1, 2, 4)):
        values = []
        for name in FORMAT_ORDER:
            row = next(
                row
                for row in rows
                if is_row(
                    row,
                    component="register_decode",
                    distribution="normal_0_1",
                    lanes=lane,
                    format_name=name,
                )
            )
            values.append(number(row, "median_decoded_gvalues_per_s"))
        axis.barh(positions + offset, values, height=width, color=LANE_COLORS[lane], label=f"x{lane}")
    axis.set_yticks(positions, [label(name) for name in FORMAT_ORDER])
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Decoded values per second (billions, log scale)")
    axis.set_title("Register-resident decode-plus-add throughput")
    axis.legend(frameon=False, loc="lower right")
    format_axis_labels(axis)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_stream_decode(rows: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    positions = np.arange(len(FORMAT_ORDER))
    for axis, component, title in zip(
        axes,
        ("stream_load", "stream_decode"),
        ("Read only", "Read + convert + FP64 add"),
    ):
        current = max_size_rows(rows, component=component)
        values = [number(row, "median_unique_storage_gb_per_s") for row in current]
        colors = [FAMILY_COLORS[format_family(row["format"])] for row in current]
        bars = axis.barh(positions, values, color=colors, height=0.7)
        axis.bar_label(bars, labels=[f"{value:.0f}" for value in values], padding=3, fontsize=8)
        axis.set_title(title)
        axis.set_xlabel("Encoded payload GB/s — higher is better")
        axis.set_xlim(0, max(values) * 1.18)
        axis.invert_yaxis()
        format_axis_labels(axis)
    axes[0].set_yticks(positions, [label(name) for name in FORMAT_ORDER])
    add_family_legend(fig)
    fig.suptitle("Cost of decoding while streaming from global memory", y=1.03, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, path)


def plot_distribution_sensitivity(rows: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), sharey=True)
    positions = np.arange(len(FORMAT_ORDER))
    components = ("register_decode", "stream_decode", "dot", "gemv")
    component_style = {
        "register_decode": ("o", "Register"),
        "stream_decode": ("s", "Stream"),
        "dot": ("^", "DOT"),
        "gemv": ("D", "GEMV"),
    }
    for axis, lane in zip(axes, (1, 4)):
        for component in components:
            ratios = []
            for name in FORMAT_ORDER:
                normal = [
                    row
                    for row in rows
                    if is_row(
                        row,
                        component=component,
                        distribution="normal_0_1",
                        lanes=lane,
                        format_name=name,
                    )
                ]
                uniform = [
                    row
                    for row in rows
                    if is_row(
                        row,
                        component=component,
                        distribution="uniform_0_1",
                        lanes=lane,
                        format_name=name,
                    )
                ]
                max_n = max(int(row["n"]) for row in normal)
                n_row = next(row for row in normal if int(row["n"]) == max_n)
                u_row = next(row for row in uniform if int(row["n"]) == max_n)
                ratios.append(number(n_row, "median_time_ms") / number(u_row, "median_time_ms"))
            marker, text = component_style[component]
            axis.scatter(ratios, positions, marker=marker, s=35, label=text)
        axis.axvline(1.0, color="#66707a", linewidth=1.0)
        axis.set_title(f"x{lane}")
        axis.set_xlabel("N(0,1) time / U(0,1) time")
        axis.invert_yaxis()
        format_axis_labels(axis)
    axes[0].set_yticks(positions, [label(name) for name in FORMAT_ORDER])
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Data-dependent decode cost", y=1.01, fontsize=14)
    fig.tight_layout()
    return save_figure(fig, path)


def plot_bottleneck_map(profile: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for axis, component in zip(axes, ("dot", "gemv")):
        current = profile_rows(profile, component)
        for row in current:
            name = row["format"]
            x = number(row, "dram_percent_peak")
            y = number(row, "sm_percent_peak")
            axis.scatter(
                x,
                y,
                s=60,
                marker=BIT_MARKERS[int(row["storage_bits"])],
                color=FAMILY_COLORS[format_family(name)],
                edgecolor="white",
                linewidth=0.7,
            )
            if name in SELECTED_FORMATS:
                axis.annotate(
                    label(name),
                    (x, y),
                    xytext=LABEL_OFFSETS[name],
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.axvline(50, color="#a0a6ad", linewidth=0.8)
        axis.axhline(50, color="#a0a6ad", linewidth=0.8)
        axis.set_xlim(0, 105)
        axis.set_ylim(0, 105)
        axis.set_title(component.upper())
        axis.set_xlabel("DRAM throughput (% of sustained peak)")
        axis.set_ylabel("SM throughput (% of sustained peak)")
        format_axis_labels(axis)
    add_family_legend(fig, y=1.02)
    fig.suptitle("Hardware bottleneck map for x4 kernels", y=1.11, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, path)


def plot_resource_heatmap(profile: Sequence[dict[str, str]], path: Path) -> Path:
    fields = (
        ("dram_percent_peak", "DRAM"),
        ("pipe_alu_percent", "ALU"),
        ("pipe_xu_percent", "XU"),
        ("pipe_fp64_percent", "FP64"),
        ("issue_active_percent", "Issue"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.5), sharey=True)
    image = None
    for axis, component in zip(axes, ("dot", "gemv")):
        current = profile_rows(profile, component)
        matrix = np.array([[number(row, field) for field, _ in fields] for row in current])
        image = axis.pcolormesh(
            np.arange(len(fields) + 1) - 0.5,
            np.arange(len(FORMAT_ORDER) + 1) - 0.5,
            matrix,
            vmin=0,
            vmax=100,
            cmap="viridis",
            shading="flat",
        )
        axis.set_xticks(range(len(fields)), [text for _, text in fields])
        axis.set_yticks(range(len(FORMAT_ORDER)), [label(name) for name in FORMAT_ORDER])
        axis.set_xlim(-0.5, len(fields) - 0.5)
        axis.set_ylim(len(FORMAT_ORDER) - 0.5, -0.5)
        axis.set_title(component.upper())
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 35 or value > 72 else "black",
                )
    assert image is not None
    fig.suptitle(
        "Overlapping resource pressure (cell values are % of sustained peak)",
        y=0.99,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.08, top=0.92, wspace=0.16)
    return save_figure(fig, path)


def plot_register_pressure(profile: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for axis, component in zip(axes, ("dot", "gemv")):
        for name in FORMAT_ORDER:
            points = [
                row
                for row in profile
                if is_row(
                    row,
                    component=component,
                    distribution="normal_0_1",
                    format_name=name,
                )
                and int(row["lanes"]) in (1, 4)
            ]
            points.sort(key=lambda row: int(row["lanes"]))
            xs = [number(row, "registers_per_thread_max") for row in points]
            ys = [number(row, "achieved_occupancy_percent") for row in points]
            color = FAMILY_COLORS[format_family(name)]
            axis.plot(xs, ys, color=color, linewidth=0.8, alpha=0.65)
            axis.scatter(xs[0], ys[0], color=color, marker="o", s=28)
            axis.scatter(xs[-1], ys[-1], color=color, marker="^", s=38)
        axis.set_title(component.upper())
        axis.set_xlabel("Registers per thread")
        axis.set_ylabel("Achieved occupancy (%)")
        axis.set_xlim(left=0)
        axis.set_ylim(0, 100)
        format_axis_labels(axis)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color="#505860", label="x1"),
        Line2D([0], [0], marker="^", linestyle="none", color="#505860", label="x4"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Packing can trade instruction count for register pressure", y=1.08, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_figure(fig, path)


def plot_timing_stability(rows: Sequence[dict[str, str]], path: Path) -> Path:
    fig, axis = plt.subplots(figsize=(11.5, 5.0))
    components = ("register_decode", "stream_load", "stream_decode", "dot", "gemv")
    values = [[number(row, "cv_percent") for row in rows if row["component"] == component] for component in components]
    boxes = axis.boxplot(values, tick_labels=[name.replace("_", " ") for name in components], patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], ("#56B4E9", "#009E73", "#0072B2", "#E69F00", "#D55E00")):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axis.set_yscale("log")
    axis.set_ylabel("Coefficient of variation across 15 samples (%)")
    axis.set_title("Timing repeatability across every measured case")
    format_axis_labels(axis)
    fig.tight_layout()
    return save_figure(fig, path)


def navigation(current: str) -> str:
    links = []
    for filename, text in PAGES:
        is_active = filename == current or (
            filename == "accuracy.html" and current.startswith("accuracy-")
        ) or (
            filename == "conversion-strategies.html"
            and current.startswith("conversion-strategies-")
        ) or (
            filename == "conversion-fp32.html"
            and current.startswith("conversion-fp32-")
        )
        active = ' aria-current="page" class="active"' if is_active else ""
        links.append(f'<a href="{filename}"{active}>{html.escape(text)}</a>')
    return "".join(links)


def accuracy_navigation(current: str) -> str:
    links = []
    for filename, text in ACCURACY_PAGES:
        active = ' aria-current="page" class="active"' if filename == current else ""
        links.append(f'<a href="{filename}"{active}>{html.escape(text)}</a>')
    return '<nav class="subnav" aria-label="Accuracy sections">' + "".join(links) + "</nav>"


def conversion_strategy_filename(format_name: str) -> str:
    return f"conversion-strategies-{format_name}.html"


def compute_conversion_overview_filename(compute: str) -> str:
    if compute == "fp32":
        return "conversion-fp32.html"
    if compute == "fp64":
        return "conversion-strategies.html"
    raise ValueError(f"unsupported compute type {compute}")


def compute_conversion_format_filename(compute: str, format_name: str) -> str:
    if compute == "fp32":
        return f"conversion-fp32-{format_name}.html"
    if compute == "fp64":
        return conversion_strategy_filename(format_name)
    raise ValueError(f"unsupported compute type {compute}")


def conversion_navigation(current: str, compute: str = "fp64") -> str:
    overview_filename = compute_conversion_overview_filename(compute)
    overview_active = (
        ' aria-current="page" class="conversion-overview-link active"'
        if current == overview_filename
        else ' class="conversion-overview-link"'
    )
    groups = []
    for storage_bits, format_names in COMPUTE_CONVERSION_FORMATS[compute].items():
        links = []
        for format_name in format_names:
            filename = compute_conversion_format_filename(compute, format_name)
            active = (
                ' aria-current="page" class="active"'
                if filename == current
                else ""
            )
            links.append(
                f'<a href="{filename}"{active}>{html.escape(label(format_name))}</a>'
            )
        group_label_id = f"conversion-{compute}-formats-{storage_bits}-bit"
        groups.append(
            '<div class="conversion-format-group" role="group" '
            f'aria-labelledby="{group_label_id}">'
            f'<span class="conversion-format-group-label" id="{group_label_id}">'
            f'{storage_bits}-bit</span>'
            f'<span class="conversion-format-links">{"".join(links)}</span></div>'
        )
    return (
        '<nav class="subnav conversion-subnav" '
        f'aria-label="Storage to {compute.upper()} arithmetic formats">'
        f'<a href="{overview_filename}"{overview_active}>Overview</a>'
        + "".join(groups)
        + "</nav>"
    )


def page_document(
    *,
    filename: str,
    title: str,
    intro: str,
    body: str,
    performance_run_name: str,
    accuracy_run_name: str,
    strategy_run_name: str,
    all_strategy_run_name: str,
    expanded_strategy_run_name: str,
) -> str:
    performance_frame_script = ""
    if "data-performance-chart" in body:
        performance_frame_script = """
<script>
window.addEventListener('message', event => {
  if (event.data?.type !== 'performance-chart-height') return;
  const frame = [...document.querySelectorAll('iframe[data-performance-chart]')]
    .find(candidate => candidate.contentWindow === event.source);
  if (!frame) return;
  const height = Math.max(520, Math.min(2200, Number(event.data.height) + 4));
  if (Number.isFinite(height)) frame.style.height = `${height}px`;
});
for (const frame of document.querySelectorAll('iframe[data-performance-chart]')) {
  frame.addEventListener('load', () => {
    frame.contentWindow?.postMessage({ type: 'request-performance-chart-height' }, '*');
  });
}
</script>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — H200 storage-format report</title>
<link rel="stylesheet" href="report.css">
</head>
<body>
<header><div class="shell"><a class="brand" href="index.html">H200 storage-format report</a><nav aria-label="Report sections">{navigation(filename)}</nav></div></header>
<main class="shell">
<h1>{html.escape(title)}</h1>
<p class="lead">{html.escape(intro)}</p>
{body}
</main>
<footer><div class="shell">Performance: <code>{html.escape(performance_run_name)}</code> · Accuracy: <code>{html.escape(accuracy_run_name)}</code> · Conversion strategies: <code>{html.escape(strategy_run_name)}</code> + <code>{html.escape(all_strategy_run_name)}</code> + <code>{html.escape(expanded_strategy_run_name)}</code></div></footer>{performance_frame_script}
</body>
</html>
"""


def graph_section(title: str, description: str, image: str, alt: str, caption: str = "") -> str:
    caption_html = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    source = image if image.startswith("../") else f"assets/{image}"
    return f"""<section class="graph-section">
<h2>{html.escape(title)}</h2>
<p>{html.escape(description)}</p>
<figure><a href="{html.escape(source)}"><img src="{html.escape(source)}" alt="{html.escape(alt)}"></a>{caption_html}</figure>
</section>"""


def interactive_graph_section(
    title: str,
    description: str,
    source: str,
    accessible_title: str,
    caption: str,
) -> str:
    return f"""<section class="graph-section">
<h2>{html.escape(title)}</h2>
<p>{html.escape(description)}</p>
<figure class="interactive-figure">
<iframe class="interactive-chart-frame" data-performance-chart src="{html.escape(source)}" title="{html.escape(accessible_title)}"></iframe>
<figcaption>{html.escape(caption)} <a href="{html.escape(source)}">Open the chart full size</a>.</figcaption>
</figure>
</section>"""


def accuracy_figure_group(
    title: str,
    description: str,
    figures: Sequence[tuple[str, str, str]],
) -> str:
    blocks = []
    for subtitle, source, alt in figures:
        blocks.append(
            f'<h3>{html.escape(subtitle)}</h3><figure><a href="{html.escape(source)}">'
            f'<img src="{html.escape(source)}" alt="{html.escape(alt)}"></a></figure>'
        )
    return f"""<section class="graph-section">
<h2>{html.escape(title)}</h2>
<p>{html.escape(description)}</p>
{''.join(blocks)}
</section>"""


def fastest(rows: Sequence[dict[str, str]], component: str) -> tuple[str, float]:
    current = max_size_rows(rows, component=component)
    winner = min(current, key=lambda row: number(row, "median_time_ms"))
    return label(winner["format"]), number(winner, "median_time_ms")


def write_report(
    output_dir: Path,
    run_dir: Path,
    accuracy_dir: Path,
    accuracy_run_dir: Path,
    strategy_run_dir: Path,
    all_strategy_run_dir: Path,
    expanded_strategy_run_dir: Path,
    rows: Sequence[dict[str, str]],
    profile: Sequence[dict[str, str]],
    accuracy_rows: Sequence[dict[str, str]],
    conversion_rows: Sequence[dict[str, str]],
) -> None:
    dot_name, dot_time = fastest(rows, "dot")
    gemv_name, gemv_time = fastest(rows, "gemv")
    bandwidth = memory_ceiling(profile)
    raw_prefix = Path(os.path.relpath(run_dir, output_dir)).as_posix()
    accuracy_asset_prefix = Path(
        os.path.relpath(accuracy_dir, output_dir)
    ).as_posix()
    accuracy_raw_prefix = Path(
        os.path.relpath(accuracy_run_dir, output_dir)
    ).as_posix()
    strategy_raw_prefix = Path(
        os.path.relpath(strategy_run_dir, output_dir)
    ).as_posix()
    all_strategy_raw_prefix = Path(
        os.path.relpath(all_strategy_run_dir, output_dir)
    ).as_posix()
    expanded_strategy_raw_prefix = Path(
        os.path.relpath(expanded_strategy_run_dir, output_dir)
    ).as_posix()

    def accuracy_asset(name: str) -> str:
        return f"{accuracy_asset_prefix}/{name}.svg"

    comparable_accuracy = [
        row
        for row in accuracy_rows
        if number(row, "predicted_mse") > 0.0
        and math.isfinite(number(row, "measured_mse"))
    ]
    within_twenty_percent = sum(
        0.8 <= number(row, "mse_ratio_measured_to_predicted") <= 1.2
        for row in comparable_accuracy
    )

    performance_cards = "".join(
        f'<a class="report-link" href="{filename}"><strong>{html.escape(text)}</strong><span>{html.escape(description)}</span></a>'
        for filename, text, description in (
            ("total-performance.html", "Total performance", "Interactive all-format x1/x2/x4 timing and size scaling."),
            ("same-bit-formats.html", "Same-bit formats", "Interactive x1/x2/x4 timing plus same-bit and all-format accuracy."),
            ("packing.html", "Packed vs unpacked", "x2 and x4 throughput benefit over x1."),
            ("roofline.html", "Roofline", "Useful work relative to the measured HBM ceiling."),
            ("conversion.html", "Conversion", "Register decode, streaming decode, and data dependence."),
            ("bottlenecks.html", "Bottlenecks", "Nsight resource pressure, registers, and occupancy."),
            ("general-info.html", "General info", "Accessor contracts and access-pattern recommendations for 2- through 32-bit storage."),
            ("packing-bottlenecks.html", "Packing bottlenecks", "Packet speedup, roofline motion, resource transitions, and mixed-arithmetic cost."),
            ("conversion-fp32.html", "Storage → FP32 arithmetic", "Dense, padded, packet, shuffle, and decoder comparisons using FP32 arithmetic."),
            ("conversion-strategies.html", "Storage → FP64 arithmetic", "Dense, padded, packet, shuffle, and decoder comparisons using FP64 arithmetic."),
        )
    )
    accuracy_cards = "".join(
        f'<a class="report-link" href="{filename}"><strong>{html.escape(text)}</strong><span>{html.escape(description)}</span></a>'
        for filename, text, description in (
            ("accuracy.html", "Accuracy overview", "Performance–accuracy trade-off and test scope."),
            ("accuracy-dot.html", "DOT accuracy", "Measured RMSE versus the analytical model."),
            ("accuracy-gemv.html", "GEMV accuracy", "Per-row RMSE across fixed M=1024."),
            ("accuracy-scalar.html", "Scalar and arithmetic", "Encoding events and FP64 arithmetic error."),
        )
    )
    overview_body = f"""
<div class="summary-grid">
<div><span>Fastest large DOT</span><strong>{dot_name}</strong><small>{dot_time:.3f} ms</small></div>
<div><span>Fastest large GEMV</span><strong>{gemv_name}</strong><small>{gemv_time:.3f} ms</small></div>
<div><span>Measured HBM ceiling</span><strong>{bandwidth / 1000:.2f} TB/s</strong><small>median memory-bound profiles</small></div>
</div>
<section><h2>Performance</h2><p>Timing, packing, roofline, conversion, and hardware bottlenecks.</p><div class="report-grid">{performance_cards}</div></section>
<section><h2>Accuracy</h2><p>H200 simulations compared with the analytical storage-error model.</p><div class="report-grid">{accuracy_cards}</div></section>
""" + graph_section(
        "All-format overview",
        "This compares complete x4 kernel time at the largest tested N; lower bars are faster.",
        "total-kernel-time.svg",
        "Horizontal bars comparing complete DOT and GEMV time for all storage formats.",
    )

    def build_conversion_overview_sections(compute: str) -> list[str]:
        sections = []
        for storage_bits, format_names in COMPUTE_CONVERSION_FORMATS[compute].items():
            cards = []
            for format_name in format_names:
                detail = "Unified H200 run pending"
                if compute == "fp64" and format_name in CONVERSION_FORMATS:
                    metadata = conversion_strategy_metadata(
                        conversion_rows, format_name
                    )
                    candidate_count = len(metadata) - 1
                    detail = f"{candidate_count} historical strategies · unified rerun pending"
                cards.append(
                    f'<a class="report-link pending-report-link" href="'
                    f'{compute_conversion_format_filename(compute, format_name)}">'
                    f'<strong>{html.escape(label(format_name))}</strong>'
                    f'<span>{html.escape(detail)}</span></a>'
                )
            sections.append(
                f'<section class="text-section"><h2>{storage_bits}-bit formats</h2>'
                f'<div class="report-grid">{"".join(cards)}</div></section>'
            )
        return sections

    fp32_conversion_overview_sections = build_conversion_overview_sections("fp32")
    fp64_conversion_overview_sections = build_conversion_overview_sections("fp64")

    pages: dict[str, tuple[str, str, str]] = {
        "index.html": (
            "Overview",
            "One entry point for the H200 performance measurements and accuracy validation.",
            overview_body,
        ),
        "total-performance.html": (
            "Total performance",
            "Complete event-timed DOT and GEMV performance, including every kernel launch.",
            interactive_graph_section(
                "All formats and access widths",
                "Every format and x1/x2/x4 implementation is plotted against N. Use access-width, storage-width, and format controls to isolate any subset.",
                "interactive/all-format-performance.html",
                "Interactive high-resolution DOT and GEMV timing chart for all storage formats and access widths",
                "All lines use N(0,1) inputs; lower time is faster.",
            )
            + graph_section(
                "Absolute kernel time",
                "This is the end-to-end comparison at the largest tested N using x4 access and N(0,1) data.",
                "total-kernel-time.svg",
                "Complete DOT and GEMV time in milliseconds for all formats.",
            )
            + graph_section(
                "Speed relative to FP64",
                "This makes the storage-bandwidth benefit easier to compare across DOT and GEMV; values above 1 are faster than FP64.",
                "relative-fp64.svg",
                "DOT and GEMV speed relative to the FP64 baseline for all formats.",
            )
            + graph_section(
                "Size regimes",
                "This shows when launch overhead stops hiding differences in load and conversion cost.",
                "size-scaling.svg",
                "Log-log kernel time versus reduction length for representative formats.",
            ),
        ),
        "same-bit-formats.html": (
            "Same-bit format comparison",
            "High-resolution performance and accuracy comparisons for formats with equal storage width.",
            interactive_graph_section(
                "8-bit performance",
                "All five formats and all x1/x2/x4 access widths share the same axes. Toggle an access width to affect every format, or toggle one format to affect all three access widths.",
                "interactive/same-bit-8.html",
                "Interactive high-resolution DOT and GEMV timing chart for every 8-bit format and access width",
                "The chart remains vector-resolution at every zoom level.",
            )
            + graph_section(
                "8-bit accuracy",
                "DOT uses normalized RMS error; GEMV uses relative L2 error. Curves are total x4 GPU error versus the original FP64 source for both distributions.",
                "same-bit-accuracy-8.svg",
                "High-resolution DOT normalized RMS and GEMV relative L2 error versus N for every 8-bit format and both distributions.",
                "Packing variants are omitted here because they encode identical values; reduction-order differences are discussed below.",
            )
            + interactive_graph_section(
                "16-bit performance",
                "E11M4, FP16, BF16, and all custom layouts are overlaid. Access-width and format toggles apply to both DOT and GEMV together.",
                "interactive/same-bit-16.html",
                "Interactive high-resolution DOT and GEMV timing chart for every 16-bit format and access width",
                "The chart remains vector-resolution at every zoom level.",
            )
            + graph_section(
                "16-bit accuracy",
                "The same primary DOT and GEMV metrics compare every 16-bit format on U(0,1) and N(0,1).",
                "same-bit-accuracy-16.svg",
                "High-resolution DOT normalized RMS and GEMV relative L2 error versus N for every 16-bit format and both distributions.",
                "Packing variants are omitted because their storage quantization is identical.",
            )
            + interactive_graph_section(
                "32-bit performance",
                "FP32, E11M20, and all custom layouts are overlaid. Access-width and format toggles apply to both DOT and GEMV together.",
                "interactive/same-bit-32.html",
                "Interactive high-resolution DOT and GEMV timing chart for every 32-bit format and access width",
                "The chart remains vector-resolution at every zoom level.",
            )
            + graph_section(
                "32-bit accuracy",
                "The same primary DOT and GEMV metrics compare every 32-bit format on U(0,1) and N(0,1).",
                "same-bit-accuracy-32.svg",
                "High-resolution DOT normalized RMS and GEMV relative L2 error versus N for every 32-bit format and both distributions.",
                "Packing variants are omitted because their storage quantization is identical.",
            )
            + graph_section(
                "All-format accuracy",
                "This puts every 8-, 16-, 32-, and 64-bit format on the same log axes. A line ending early indicates non-finite outputs at larger N.",
                "all-format-accuracy.svg",
                "Large high-resolution comparison of DOT normalized RMS and GEMV relative L2 error versus N for all 17 formats and both distributions.",
                "Curves use the x4 GPU result versus the original FP64 source. FP64 exact-zero reference points, when present, are placed at the labeled plot floor.",
            )
            + """<section class="text-section">
<h2>Packing and accuracy</h2>
<p>x1, x2, and x4 are literally identical at the storage-quantization level: they read the same encoded values. Their complete GPU results are not guaranteed bit-for-bit identical because lane grouping changes the FP64 reduction order. For 8-, 16-, and 32-bit storage that arithmetic difference is far below the quantization error, so these format-comparison plots show the x4 result once. The separate scalar and arithmetic page retains the x1/x2/x4 kernel-only validation. FP64 is the special case where reduction rounding is the remaining error.</p>
</section>""",
        ),
        "packing.html": (
            "Packed vs unpacked",
            "Direct x2/x4 versus x1 comparisons using equal logical work, so packed lanes are normalized correctly.",
            graph_section(
                "Large-problem packing benefit",
                "This compares throughput at the largest tested N; 1 means packing made no difference.",
                "packed-speedup.svg",
                "x2 and x4 kernel throughput divided by x1 throughput for all formats.",
            )
            + graph_section(
                "Packing benefit by N",
                "This reveals whether a format needs a large problem before fewer load instructions repay unpacking overhead.",
                "packing-by-size.svg",
                "x4 versus x1 time speedup by reduction length for representative formats.",
            ),
        ),
        "general-info.html": (
            "General info",
            "How solo, packed, shuffled, and padded access divide loading, decoding, indexing, and thread cooperation across 2- through 32-bit formats.",
            """<section class="text-section">
<h2>Three access contracts</h2>
<p>A solo accessor can replace a pointer without changing a kernel's element loop. Packed and shuffled accessors can hide the physical load and conversion, but the kernel must iterate over packets or cooperative groups.</p>
<div class="table-wrap"><table class="strategy-table access-contract-table">
<thead><tr><th>Contract</th><th>Work assignment</th><th>Accessor owns</th><th>Kernel owns</th></tr></thead>
<tbody>
<tr><td>Solo <code>operator[](i)</code></td><td>One thread, one value</td><td>Scalar load and decode</td><td>Element indexing</td></tr>
<tr><td>Packed xN <code>load_packet(p)</code></td><td>One thread, N values</td><td>Packet load, lane extraction, decode</td><td>Packet indexing, per-lane arithmetic, tail</td></tr>
<tr><td>Shuffled <code>load_cooperative_packet(p, lane)</code></td><td>Several threads, one dense packet</td><td>Selected loads, shuffles, extraction, decode</td><td>Group formation, lockstep calls, packet indexing, tail</td></tr>
</tbody></table></div>
</section>

<section class="text-section">
<h2>Access pattern by storage width</h2>
<p><strong>Solo</strong> means one native container per value. <strong>Packed xN</strong> means one thread loads and computes N adjacent values. <strong>Shuffled V/W/C/P</strong> means V dense values occupy W aligned 32-bit words, loaded by the first W lanes of a C-thread group, with P values computed per thread. <strong>Padded</strong> stores each awkward-width value in the next 8-, 16-, or 32-bit container.</p>
<div class="table-wrap"><table class="strategy-table access-width-table">
<thead><tr><th>Bits</th><th>Recommended access</th><th>Packet or cooperative group</th><th>Control worth measuring</th></tr></thead>
<tbody>
<tr><th>2</th><td>Packed</td><td>x4 and x8; x16 only for decode microbenchmarks</td><td>Padded uint8 solo</td></tr>
<tr><th>3</th><td>Shuffled</td><td>32/3/8/4</td><td>Padded uint8 solo</td></tr>
<tr><th>4</th><td>Packed</td><td>x2, x4, and x8; prefer x4/x8</td><td>Padded uint8 solo</td></tr>
<tr><th>5</th><td>Shuffled</td><td>32/5/8/4</td><td>Padded uint8 solo</td></tr>
<tr><th>6</th><td>Shuffled</td><td>16/3/4/4</td><td>Padded uint8 solo</td></tr>
<tr><th>7</th><td>Shuffled or padded</td><td>32/7/8/4</td><td>Padded uint8 solo</td></tr>
<tr><th>8</th><td>Solo or packed</td><td>x1; packed x4/x8</td><td>—</td></tr>
<tr><th>9</th><td>Shuffled</td><td>32/9/16/2</td><td>Padded uint16 solo</td></tr>
<tr><th>10</th><td>Shuffled</td><td>16/5/8/2</td><td>Padded uint16 solo</td></tr>
<tr><th>11</th><td>Shuffled</td><td>32/11/16/2</td><td>Padded uint16 solo</td></tr>
<tr><th>12</th><td>Shuffled</td><td>8/3/4/2</td><td>Padded uint16 solo</td></tr>
<tr><th>13</th><td>Shuffled or padded</td><td>32/13/16/2</td><td>Padded uint16 solo</td></tr>
<tr><th>14</th><td>Shuffled or padded</td><td>16/7/8/2</td><td>Padded uint16 solo</td></tr>
<tr><th>15</th><td>Padded first; shuffled exploratory</td><td>32/15/16/2</td><td>Padded uint16 solo</td></tr>
<tr><th>16</th><td>Solo or packed</td><td>x1; packed x2/x4</td><td>—</td></tr>
<tr><th>17</th><td>Shuffled</td><td>32/17/32/1</td><td>Padded uint32 solo</td></tr>
<tr><th>18</th><td>Shuffled</td><td>16/9/16/1</td><td>Padded uint32 solo</td></tr>
<tr><th>19</th><td>Shuffled</td><td>32/19/32/1</td><td>Padded uint32 solo</td></tr>
<tr><th>20</th><td>Shuffled</td><td>8/5/8/1</td><td>Padded uint32 solo</td></tr>
<tr><th>21</th><td>Shuffled</td><td>32/21/32/1</td><td>Padded uint32 solo</td></tr>
<tr><th>22</th><td>Shuffled</td><td>16/11/16/1</td><td>Padded uint32 solo</td></tr>
<tr><th>23</th><td>Shuffled</td><td>32/23/32/1</td><td>Padded uint32 solo</td></tr>
<tr><th>24</th><td>Shuffled</td><td>4/3/4/1</td><td>Padded uint32 solo and dense packed x4</td></tr>
<tr><th>25</th><td>Shuffled versus padded</td><td>32/25/32/1</td><td>Padded uint32 solo</td></tr>
<tr><th>26</th><td>Shuffled versus padded</td><td>16/13/16/1</td><td>Padded uint32 solo</td></tr>
<tr><th>27</th><td>Shuffled versus padded</td><td>32/27/32/1</td><td>Padded uint32 solo</td></tr>
<tr><th>28</th><td>Shuffled versus padded</td><td>8/7/8/1</td><td>Padded uint32 solo</td></tr>
<tr><th>29</th><td>Padded first; shuffled exploratory</td><td>32/29/32/1</td><td>Padded uint32 solo</td></tr>
<tr><th>30</th><td>Padded first; shuffled exploratory</td><td>16/15/16/1</td><td>Padded uint32 solo</td></tr>
<tr><th>31</th><td>Padded</td><td>Solo x1</td><td>Dense shuffle 32/31/32/1</td></tr>
<tr><th>32</th><td>Solo</td><td>x1; packed x2 only for additional ILP</td><td>—</td></tr>
</tbody></table></div>
<p>The shuffled layouts are benchmark candidates, not universal winners. Decoder cost, register pressure, alignment, and the fraction of bandwidth saved can make the padded control faster.</p>
</section>

<section class="text-section">
<h2>Packed access</h2>
<p>The existing x2/x4/x8 strategies use this model. One thread loads and processes every value in its own packet; neighboring threads load neighboring packets. Values are not exchanged between threads.</p>
<pre><code>template&lt;class Format, int Lanes&gt;
struct packed_accessor {
    static constexpr int threads_per_packet = 1;
    static constexpr int values_per_thread = Lanes;

    packet&lt;double, Lanes&gt; load_packet(size_t packet_index) const {
        auto raw = load_aligned_packet(storage, packet_index);
        return decode_all_lanes&lt;Format, Lanes&gt;(raw);
    }
};

template&lt;class Accessor&gt;
kernel dot_packet(Accessor a, Accessor b, size_t N) {
    constexpr int L = Accessor::values_per_thread;
    double sums[L] = {};

    for (size_t packet = global_thread_id;
         packet &lt; N / L;
         packet += total_threads) {
        auto x = a.load_packet(packet);
        auto y = b.load_packet(packet);

        unroll for (int lane = 0; lane &lt; L; ++lane)
            sums[lane] = fma(x[lane], y[lane], sums[lane]);
    }

    block_reduce(sum(sums));
    // A solo path handles N % L remaining values.
}</code></pre>
<p>For an x4 eight-bit accessor, thread 0 handles logical elements 0–3, thread 1 handles 4–7, and so on. The loop changes from element indices to packet indices, but the same template supports x1, x2, x4, and x8.</p>
</section>

<section class="text-section">
<h2>Shuffled FP6 access</h2>
<p>Dense FP6 becomes convenient at 16 values: 16 × 6 bits = 96 bits = three aligned 32-bit words. Four threads can load those three words once, exchange them through registers, and decode four values each.</p>
<p><strong>Note — current benchmark limitation.</strong> The present cooperative implementation reconstructs each of a consumer thread's four values separately and executes two shuffles per value. That is eight shuffles per input packet, or sixteen in DOT for the two input arrays, although the same two source words cover the consumer's complete 24-bit slice. A better implementation should load the three 32-bit words once per 16-value chunk, shuffle at most two words to each consumer once, assemble one local 24-bit packet, and extract its four six-bit codes in registers. This would reduce the intended shuffle count by about 4×, so the current measurements are not an upper bound on shuffled-access performance.</p>
<div class="access-flow" role="img" aria-label="Sixteen six-bit values are densely packed into 96 bits, loaded as three 32-bit words by three lanes of a four-thread group, shuffled to every lane, and divided into four decoded values per thread.">
  <div class="flow-stage"><strong>16 × 6-bit values</strong><span>v0 … v15</span></div>
  <span class="flow-arrow" aria-hidden="true">→</span>
  <div class="flow-stage"><strong>Dense 96-bit packet</strong><span>W0: 32 · W1: 32 · W2: 32</span></div>
  <span class="flow-arrow" aria-hidden="true">→</span>
  <div class="flow-stage"><strong>Three global loads</strong><span>T0→W0 · T1→W1 · T2→W2</span></div>
  <span class="flow-arrow" aria-hidden="true">→</span>
  <div class="flow-stage"><strong>Register shuffle</strong><span>Every thread receives W0, W1, W2</span></div>
  <span class="flow-arrow" aria-hidden="true">→</span>
  <div class="flow-stage"><strong>Four values per thread</strong><span>T0:v0–3 · T1:v4–7 · T2:v8–11 · T3:v12–15</span></div>
</div>
<pre><code>struct cooperative_fp6_accessor {
    static constexpr int threads_per_packet = 4;
    static constexpr int values_per_thread = 4;
    static constexpr int values_per_packet = 16;

    packet&lt;double, 4&gt;
    load_cooperative_packet(size_t packet_index, int lane_in_group) const {
        size_t word = packet_index * 3;

        // T0, T1 and T2 each load one word; T3 performs no global load.
        uint32_t mine = lane_in_group &lt; 3 ? storage[word + lane_in_group] : 0;

        uint32_t w0 = shuffle(mine, source_lane=0, width=4);
        uint32_t w1 = shuffle(mine, source_lane=1, width=4);
        uint32_t w2 = shuffle(mine, source_lane=2, width=4);

        packet&lt;double, 4&gt; result;
        unroll for (int j = 0; j &lt; 4; ++j) {
            int logical_lane = lane_in_group * 4 + j;
            uint6_t bits = extract_6_bits(w0, w1, w2, logical_lane * 6);
            result[j] = decode_fp6_to_f64(bits);
        }
        return result;
    }
};

template&lt;class Accessor&gt;
kernel dot_cooperative(Accessor a, Accessor b, size_t N) {
    constexpr int G = Accessor::threads_per_packet;
    constexpr int V = Accessor::values_per_thread;
    constexpr int P = Accessor::values_per_packet;

    int lane = global_thread_id % G;
    size_t group = global_thread_id / G;
    size_t groups = total_threads / G;
    double sums[V] = {};

    for (size_t packet = group; packet &lt; N / P; packet += groups) {
        // Every thread in the group must call both operations in lockstep.
        auto x = a.load_cooperative_packet(packet, lane);
        auto y = b.load_cooperative_packet(packet, lane);

        unroll for (int j = 0; j &lt; V; ++j)
            sums[j] = fma(x[j], y[j], sums[j]);
    }

    block_reduce(sum(sums));
    // A separate scalar or masked path handles N % P.
}</code></pre>
</section>

<section class="text-section">
<h2>Abstraction boundary</h2>
<p>The accessor can hide storage layout, physical loads, shuffles, bit extraction, and FP64 conversion. It cannot hide that four threads must execute a shuffled load together. Consequently, solo, packed, and shuffled access are three generic execution patterns—not one completely interchangeable <code>arr[i]</code> operation.</p>
</section>""",
        ),
        "roofline.html": (
            "Roofline",
            "A compression-aware roofline using useful FP64 work, unique encoded bytes, event timing, and the measured HBM ceiling.",
            graph_section(
                "Algorithmic roofline position",
                "Each point moves right when its encoding stores more useful operands per byte; distance below the roof is implementation cost.",
                "algorithmic-roofline.svg",
                "Compression-aware DOT and GEMV roofline positions for all formats.",
            )
            + graph_section(
                "Roof efficiency",
                "This divides measured useful throughput by the roof available at that format's arithmetic intensity.",
                "roof-efficiency.svg",
                "Percentage of the compression-aware roof reached by each format.",
            ),
        ),
        "conversion.html": (
            "Conversion performance",
            "Microbenchmarks isolate decode pressure in registers and while streaming, without treating them as additive kernel phases.",
            graph_section(
                "Register-resident conversion",
                "This measures decoded values per second after the packet is already in registers; it includes an FP64 add that keeps results live.",
                "register-conversion.svg",
                "Register decode-plus-add throughput for x1, x2, and x4 formats.",
            )
            + graph_section(
                "Read-only versus read-and-decode",
                "The paired panels show the payload rate each complete instruction stream sustains; their times must not be subtracted.",
                "stream-decode.svg",
                "Encoded payload bandwidth for streaming loads and streaming decode kernels.",
            )
            + graph_section(
                "Distribution sensitivity",
                "Normal-to-uniform time ratios expose branchy decoders whose cost depends on exponent patterns.",
                "distribution-sensitivity.svg",
                "N(0,1) time divided by U(0,1) time for conversion and full kernels.",
            ),
        ),
        "bottlenecks.html": (
            "Kernel bottlenecks",
            "Nsight Compute identifies simultaneous hardware pressure; utilization percentages overlap and are not a partition of elapsed time.",
            graph_section(
                "Memory versus SM pressure",
                "Upper-left points are conversion/compute limited; lower-right points are memory limited; the upper-right region is mixed.",
                "bottleneck-map.svg",
                "DRAM versus SM percentage of sustained peak for x4 DOT and GEMV.",
            )
            + graph_section(
                "Which execution resources are active",
                "This shows where conversion uses ALU or XU pipelines while FP64 accumulation and memory traffic overlap.",
                "resource-heatmap.svg",
                "Heatmaps of DRAM, ALU, XU, FP64, and issue activity for every x4 format.",
                "Columns cannot be added into time percentages because GPU pipelines execute concurrently.",
            )
            + graph_section(
                "Register and occupancy trade-off",
                "Lines connect x1 circles to x4 triangles and show when packing increases per-thread state enough to reduce occupancy.",
                "register-pressure.svg",
                "Registers per thread versus achieved occupancy for x1 and x4 DOT and GEMV.",
            ),
        ),
        "accuracy.html": (
            "Accuracy overview",
            "Measured H200 storage error, its analytical prediction, and the resulting performance–accuracy trade-off.",
            accuracy_navigation("accuracy.html")
            + graph_section(
                "Performance–accuracy trade-off",
                "This joins the large-kernel x4 timing with measured N(0,1) storage RMSE; points toward the lower-left are both faster and more accurate.",
                "performance-accuracy-tradeoff.svg",
                "DOT and GEMV kernel time versus measured storage RMSE for every format.",
                "Performance and accuracy use their own largest tested N, shown in each panel title.",
            )
            + f"""<section class="text-section"><h2>Validation scope</h2>
<p>The GPU run contains {len(accuracy_rows)} storage comparisons across DOT, GEMV, both distributions, all formats, and every tested N. {within_twenty_percent} of {len(comparable_accuracy)} finite nonzero cases have measured MSE within 20% of the analytical prediction.</p>
<p>The DOT and GEMV pages separate the results by kernel, distribution, and storage width. The scalar page covers quantization events and the smaller FP64 arithmetic-only error.</p>
<p><a href="{accuracy_asset_prefix}/accuracy_model_report.html">Open the original complete analytical report</a>.</p>
</section>""",
        ),
        "accuracy-dot.html": (
            "DOT accuracy",
            "Measured storage RMSE versus the analytical model for every format and DOT length.",
            accuracy_navigation("accuracy-dot.html")
            + accuracy_figure_group(
                "U(0,1) DOT",
                "Each figure overlays H200 RMSE and its confidence interval on the analytical prediction.",
                tuple(
                    (
                        group,
                        accuracy_asset(f"gpu_uniform_0_1_dot_{suffix}"),
                        f"Measured and analytical DOT RMSE for U(0,1), {group} formats.",
                    )
                    for group, suffix in (
                        ("8-bit", "8bit"),
                        ("16-bit", "16bit"),
                        ("32/64-bit", "32_64bit"),
                    )
                ),
            )
            + accuracy_figure_group(
                "N(0,1) DOT",
                "This distribution also exposes exponent-range failures in narrow custom layouts.",
                tuple(
                    (
                        group,
                        accuracy_asset(f"gpu_normal_0_1_dot_{suffix}"),
                        f"Measured and analytical DOT RMSE for N(0,1), {group} formats.",
                    )
                    for group, suffix in (
                        ("8-bit", "8bit"),
                        ("16-bit", "16bit"),
                        ("32/64-bit", "32_64bit"),
                    )
                ),
            ),
        ),
        "accuracy-gemv.html": (
            "GEMV accuracy",
            "Per-row storage RMSE at fixed M=1024, separated by distribution and storage width.",
            accuracy_navigation("accuracy-gemv.html")
            + accuracy_figure_group(
                "U(0,1) GEMV",
                "The same analytical row-error model is compared with 16 independently generated matrices and vectors.",
                tuple(
                    (
                        group,
                        accuracy_asset(f"gpu_uniform_0_1_gemv_{suffix}"),
                        f"Measured and analytical GEMV row RMSE for U(0,1), {group} formats.",
                    )
                    for group, suffix in (
                        ("8-bit", "8bit"),
                        ("16-bit", "16bit"),
                        ("32/64-bit", "32_64bit"),
                    )
                ),
            )
            + accuracy_figure_group(
                "N(0,1) GEMV",
                "Shared vectors reduce the number of independent rare-overflow events, so confidence is wider than for DOT.",
                tuple(
                    (
                        group,
                        accuracy_asset(f"gpu_normal_0_1_gemv_{suffix}"),
                        f"Measured and analytical GEMV row RMSE for N(0,1), {group} formats.",
                    )
                    for group, suffix in (
                        ("8-bit", "8bit"),
                        ("16-bit", "16bit"),
                        ("32/64-bit", "32_64bit"),
                    )
                ),
            ),
        ),
        "accuracy-scalar.html": (
            "Scalar and arithmetic accuracy",
            "Scalar quantization behavior, non-finite outputs, and FP64 accumulation error are kept separate.",
            accuracy_navigation("accuracy-scalar.html")
            + accuracy_figure_group(
                "Scalar quantization under U(0,1)",
                "These analytical curves isolate one-value quantization before any kernel reduction.",
                tuple(
                    (
                        group,
                        accuracy_asset(f"uniform_0_1_scalar_{suffix}"),
                        f"Scalar quantization metrics for U(0,1), {group} formats.",
                    )
                    for group, suffix in (
                        ("8-bit", "8bit"),
                        ("16-bit", "16bit"),
                        ("32/64-bit", "32_64bit"),
                    )
                ),
            )
            + accuracy_figure_group(
                "Scalar quantization under N(0,1)",
                "Zero, overflow, and saturation probabilities explain several kernel-level failure cases.",
                tuple(
                    (
                        group,
                        accuracy_asset(f"normal_0_1_scalar_{suffix}"),
                        f"Scalar quantization metrics for N(0,1), {group} formats.",
                    )
                    for group, suffix in (
                        ("8-bit", "8bit"),
                        ("16-bit", "16bit"),
                        ("32/64-bit", "32_64bit"),
                    )
                ),
            )
            + graph_section(
                "Non-finite outputs",
                "This compares predicted and measured probabilities that a DOT or GEMV output becomes non-finite.",
                accuracy_asset("gpu_normal_0_1_nonfinite_outputs"),
                "Predicted and measured non-finite output probabilities.",
            )
            + graph_section(
                "FP64 kernel arithmetic",
                "This isolates rounding introduced by x1/x2/x4 kernel evaluation after storage values have already been decoded.",
                accuracy_asset("gpu_fp64_arithmetic_validation"),
                "Measured FP64 kernel arithmetic error relative to its structural bound.",
            ),
        ),
        "methodology.html": (
            "Methodology and data",
            "How performance timing, profiler counters, analytical accuracy, and GPU simulation were collected and compared.",
            graph_section(
                "Timing stability",
                "This summarizes variability across all 2,142 cases after 10 warmups and interleaved sampling.",
                "timing-stability.svg",
                "Coefficient-of-variation distributions for each benchmark component.",
            )
            + f"""<section class="text-section"><h2>Definitions</h2>
<p><strong>Total time:</strong> CUDA-event median over 15 samples; DOT includes both reduction launches and GEMV includes its complete kernel.</p>
<p><strong>Packing speedup:</strong> logical throughput for equal decoded values or useful operations. Register decode is compared in decoded values/s, fixing the unequal-work x1/x2/x4 normalization.</p>
<p><strong>Roofline bytes:</strong> unique encoded storage. GEMV counts its vector once because Nsight confirms cache reuse; requested bytes remain available in the raw table.</p>
<p><strong>Profiler data:</strong> only hardware counters are used. Replay-contaminated Nsight durations never replace event timing.</p>
<p><strong>DOT primary accuracy:</strong> normalized RMS error, using the per-output absolute-product sum as the cancellation-safe normalization.</p>
<p><strong>GEMV primary accuracy:</strong> relative L2 error over the M=1024 output vector.</p>
<p><strong>Error separation:</strong> decoded storage versus FP64 source isolates quantization; GPU versus decoded storage isolates arithmetic order; GPU versus source is total error. The same-bit and all-format curves use total x4 error.</p>
<p><strong>Accuracy sampling:</strong> DOT uses 8,192 independent outputs per case. GEMV fixes M=1024 and uses 16 independent matrix/vector replicates.</p>
</section>
<section class="text-section"><h2>Performance data</h2><ul>
<li><a href="{raw_prefix}/timing_samples.csv">Every timing sample</a></li>
<li><a href="{raw_prefix}/timing_summary.csv">Timing summary</a></li>
<li><a href="{raw_prefix}/packed_speedups.csv">Corrected packed comparisons</a></li>
<li><a href="{raw_prefix}/profile_operations.csv">Operation-level Nsight metrics</a></li>
<li><a href="{raw_prefix}/profile_kernels.csv">Per-kernel Nsight metrics</a></li>
</ul></section>
<section class="text-section"><h2>Accuracy data</h2><ul>
<li><a href="{accuracy_raw_prefix}/simulation_summary.csv">GPU simulation summary</a></li>
<li><a href="{accuracy_raw_prefix}/batch_estimates.csv">Independent batch estimates</a></li>
<li><a href="{accuracy_raw_prefix}/convergence_report.csv">Convergence assessment</a></li>
<li><a href="{accuracy_asset_prefix}/simulation_model_comparison.csv">Joined model and simulation results</a></li>
<li><a href="{accuracy_asset_prefix}/kernel_predictions.csv">Analytical kernel predictions</a></li>
<li><a href="{accuracy_asset_prefix}/scalar_predictions.csv">Analytical scalar predictions</a></li>
</ul></section>""",
        ),
        "conversion-fp32.html": (
            "Storage → FP32 arithmetic",
            "Per-format DOT and GEMV comparisons using FP32 arithmetic.",
            conversion_navigation("conversion-fp32.html", "fp32")
            + """<section class="experiment-status pending-section">
<span class="status-badge">Pending H200 data</span><strong>Experiment 021 · unified FP32 sweep</strong>
</section>"""
            + "".join(fp32_conversion_overview_sections),
        ),
        "conversion-strategies.html": (
            "Storage → FP64 arithmetic",
            "Per-format DOT and GEMV comparisons using FP64 arithmetic.",
            conversion_navigation("conversion-strategies.html", "fp64")
            + """<section class="experiment-status pending-section">
<span class="status-badge">Pending H200 data</span><strong>Experiment 021 · unified FP64 sweep</strong>
</section>"""
            + "".join(fp64_conversion_overview_sections)
            + f"""<section class="text-section"><h2>Historical data sources</h2>
<p>E2M5 and E3M4 use experiment 013; the original fourteen remaining formats use experiment 015; the twenty-two added 2-, 4-, 8-, 16-, and 32-bit formats use experiment 017. All three use the same N grid, distributions, warmups, repetitions, and complete-kernel timing definition.</p>
<ul><li><a href="{strategy_raw_prefix}/timing_summary.csv">E2M5/E3M4 timing summary</a></li>
<li><a href="{all_strategy_raw_prefix}/timing_summary.csv">All-other-format timing summary</a></li>
<li><a href="{all_strategy_raw_prefix}/strategy_rankings.csv">Experiment 015 aggregate rankings</a></li>
<li><a href="{expanded_strategy_raw_prefix}/timing_summary.csv">Expanded-format timing summary</a></li>
<li><a href="{expanded_strategy_raw_prefix}/strategy_rankings.csv">Experiment 017 aggregate rankings</a></li></ul></section>""",
        ),
    }

    for format_names in FP32_CONVERSION_BIT_FORMATS.values():
        for format_name in format_names:
            filename = compute_conversion_format_filename("fp32", format_name)
            pages[filename] = (
                f"{label(format_name)} → FP32 arithmetic",
                f"Unified {label(format_name)} storage and FP32 arithmetic strategy comparison.",
                conversion_navigation(filename, "fp32")
                + pending_conversion_sections("fp32", format_name),
            )

    fp64_formats = tuple(
        format_name
        for format_names in FP64_CONVERSION_BIT_FORMATS.values()
        for format_name in format_names
    )
    for format_name in fp64_formats:
        filename = conversion_strategy_filename(format_name)
        if format_name not in CONVERSION_FORMATS:
            pages[filename] = (
                f"{label(format_name)} → FP64 arithmetic",
                f"Unified {label(format_name)} storage and FP64 arithmetic strategy comparison.",
                conversion_navigation(filename, "fp64")
                + pending_conversion_sections("fp64", format_name),
            )
            continue

        candidate_count = len(conversion_strategy_metadata(conversion_rows, format_name)) - 1
        if format_name in {"e2m5", "e3m4"}:
            source_prefix = strategy_raw_prefix
        elif format_name in EXPANDED_CONVERSION_FORMATS:
            source_prefix = expanded_strategy_raw_prefix
        else:
            source_prefix = all_strategy_raw_prefix
        pages[filename] = (
            f"{label(format_name)} → FP64 arithmetic",
            f"Unified comparison placeholder plus {candidate_count} historical {label(format_name)} conversion strategies.",
            conversion_navigation(filename, "fp64")
            + pending_conversion_sections("fp64", format_name)
            + """<section class="historical-divider"><h2>Historical FP64 strategy sweep</h2></section>"""
            + format_performance_comparison_table(conversion_rows, format_name)
            + interactive_graph_section(
                f"Historical {label(format_name)} complete-kernel time",
                "Use distribution, packed-width, conversion-family, and individual-strategy controls to isolate any subset. The controls apply to DOT and GEMV together.",
                f"interactive/conversion-{format_name}.html",
                f"Interactive {label(format_name)} and raw FP64 DOT and GEMV timing chart",
                "N(0,1) is selected initially; lower time is faster. F64 is the raw-pointer baseline measured in the same randomized format sweep.",
            )
            + conversion_top_picks(
                conversion_rows,
                format_name,
                allowed_lanes={1},
                title="Historical top picks — x1 only",
            )
            + conversion_top_picks(
                conversion_rows,
                format_name,
                allowed_lanes={1, 2, 4, 8},
                title="Historical top picks — all widths",
            )
            + conversion_strategy_explanations(conversion_rows, format_name)
            + f"""<section class="text-section"><h2>Raw data</h2><ul>
<li><a href="{source_prefix}/timing_samples.csv">Every timing sample</a></li>
<li><a href="{source_prefix}/timing_summary.csv">Timing summary</a></li>
<li><a href="{source_prefix}/case_winners.csv">Fastest strategy in each case</a></li>
</ul></section>""",
        )

    for filename, (title, intro, body) in pages.items():
        document = page_document(
            filename=filename,
            title=title,
            intro=intro,
            body=body,
            performance_run_name=run_dir.name,
            accuracy_run_name=accuracy_run_dir.name,
            strategy_run_name=strategy_run_dir.name,
            all_strategy_run_name=all_strategy_run_dir.name,
            expanded_strategy_run_name=expanded_strategy_run_dir.name,
        )
        (output_dir / filename).write_text(document, encoding="utf-8")


REPORT_CSS = """:root {
  color-scheme: light dark;
  --bg: #f4f6f8;
  --fg: #1f252b;
  --muted: #5d6872;
  --surface: #ffffff;
  --border: #d8dee4;
  --link: #075f9a;
  --active: #e5f2fb;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #151719; --fg: #edf0f2; --muted: #b4bdc5; --surface: #202428; --border: #3c444c; --link: #7fc8f8; --active: #18384d; }
  figure img { background: white; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.55 system-ui, sans-serif; }
.shell { width: min(1480px, calc(100% - 32px)); margin: 0 auto; }
header { background: var(--surface); border-bottom: 1px solid var(--border); }
header .shell { display: flex; align-items: center; gap: 28px; padding: 14px 0; }
.brand { color: var(--fg); font-weight: 500; text-decoration: none; white-space: nowrap; }
nav { display: flex; gap: 5px; flex-wrap: wrap; }
nav a { color: var(--muted); text-decoration: none; padding: 6px 9px; border-radius: 6px; }
nav a:hover, nav a:focus-visible, nav a.active { color: var(--fg); background: var(--active); }
main { padding-top: 30px; padding-bottom: 48px; }
h1, h2, h3, strong { font-weight: 500; }
h1 { margin: 0 0 8px; font-size: clamp(1.8rem, 4vw, 2.6rem); }
h2 { margin: 0 0 6px; font-size: 1.28rem; }
h3 { margin: 28px 0 8px; font-size: 1.05rem; }
.lead { color: var(--muted); max-width: 850px; margin: 0 0 28px; }
.subnav { border-bottom: 1px solid var(--border); margin: -10px 0 32px; padding-bottom: 12px; }
.conversion-subnav { align-items: start; display: grid; gap: 5px 14px; grid-template-columns: auto minmax(0, 1fr); }
.conversion-subnav a { font-size: 0.9rem; padding: 5px 7px; }
.conversion-overview-link { grid-row: 1 / span 17; }
.conversion-format-group { align-items: baseline; display: flex; gap: 8px; min-width: 0; }
.conversion-format-group-label { color: var(--muted); flex: 0 0 42px; font-size: 0.78rem; font-weight: 500; text-transform: uppercase; }
.conversion-format-links { display: flex; flex-wrap: wrap; gap: 3px; }
.graph-section, .text-section { margin: 34px 0 48px; }
.graph-section > p, .text-section > p { max-width: 900px; margin: 0 0 14px; }
figure { margin: 0; }
.graph-section figure + h3 { margin-top: 34px; }
figure img { display: block; width: 100%; height: auto; border: 1px solid var(--border); }
.interactive-chart-frame { background: white; border: 1px solid var(--border); display: block; height: 900px; width: 100%; }
figcaption { color: var(--muted); margin-top: 8px; }
.summary-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 26px 0 40px; }
.summary-grid > div { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; display: grid; gap: 4px; }
.summary-grid span, .summary-grid small, .report-link span { color: var(--muted); }
.summary-grid strong { font-size: 1.35rem; }
.report-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.report-link { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; color: var(--fg); padding: 15px; text-decoration: none; display: grid; gap: 4px; }
.report-link:hover, .report-link:focus-visible { border-color: var(--link); }
.pending-report-link { border-style: dashed; }
.experiment-status { align-items: center; background: var(--surface); border: 1px dashed var(--border); border-radius: 8px; display: flex; flex-wrap: wrap; gap: 10px 14px; margin: 8px 0 30px; padding: 14px 16px; }
.status-badge { background: var(--active); border-radius: 999px; color: var(--link); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.03em; padding: 4px 9px; text-transform: uppercase; }
.comparison-grid { display: grid; gap: 18px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.comparison-block { min-width: 0; }
.comparison-block h3 { margin-top: 12px; }
.pending-comparison-table { width: 100%; }
.pending-value { color: var(--muted); font-variant-numeric: tabular-nums; }
.layout-comparison-table { width: min(100%, 1080px); }
.chart-placeholder { display: grid; gap: 18px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.chart-placeholder > div { aspect-ratio: 4 / 2; background: linear-gradient(to top, transparent 49.5%, var(--border) 50%, transparent 50.5%), linear-gradient(to right, transparent 49.5%, var(--border) 50%, transparent 50.5%), var(--surface); border: 1px dashed var(--border); border-radius: 8px; display: flex; flex-direction: column; justify-content: space-between; padding: 14px; }
.chart-placeholder > div span { align-self: end; color: var(--muted); }
.chart-placeholder > p { color: var(--muted); grid-column: 1 / -1; margin: 0; }
.historical-divider { border-top: 1px solid var(--border); margin: 64px 0 28px; padding-top: 28px; }
pre { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; line-height: 1.45; margin: 16px 0; max-width: 1180px; overflow-x: auto; padding: 16px; }
pre code { font-size: 0.9rem; }
.access-flow { align-items: stretch; display: flex; gap: 10px; margin: 20px 0 28px; }
.flow-stage { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; display: flex; flex: 1 1 0; flex-direction: column; gap: 5px; justify-content: center; min-width: 0; padding: 14px; text-align: center; }
.flow-stage span { color: var(--muted); font-size: 0.88rem; }
.flow-arrow { align-self: center; color: var(--muted); font-size: 1.35rem; }
.access-contract-table { width: 100%; }
.access-width-table { width: 100%; }
.access-width-table th:first-child { text-align: right; width: 44px; }
.table-wrap { overflow-x: auto; }
.strategy-table { border-collapse: collapse; width: min(100%, 980px); }
.strategy-table th, .strategy-table td { border-bottom: 1px solid var(--border); padding: 8px 10px; text-align: left; vertical-align: top; }
.strategy-table th { white-space: nowrap; width: 120px; }
.strategy-picks { width: min(100%, 1120px); }
.strategy-reference { width: 100%; }
.strategy-full-name { color: var(--muted); display: block; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.78rem; font-weight: 400; margin-top: 2px; white-space: normal; }
.performance-comparison { margin-top: 24px; }
.comparison-table { width: min(100%, 1180px); }
.comparison-table td { font-variant-numeric: tabular-nums; }
.ratio-value { display: block; font-weight: 500; white-space: nowrap; }
.ratio-peer { color: var(--muted); display: block; font-size: 0.86rem; white-space: nowrap; }
.comparison-empty { color: var(--muted); }
a { color: var(--link); }
li { margin: 5px 0; }
footer { border-top: 1px solid var(--border); color: var(--muted); }
footer .shell { padding: 18px 0 28px; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
@media (max-width: 760px) {
  header .shell { align-items: flex-start; flex-direction: column; gap: 8px; }
  .conversion-subnav { grid-template-columns: 1fr; }
  .conversion-overview-link { grid-row: auto; justify-self: start; }
  .conversion-format-group { align-items: flex-start; flex-direction: column; gap: 2px; }
  .conversion-format-group-label { flex-basis: auto; }
  .summary-grid, .report-grid { grid-template-columns: 1fr; }
  .comparison-grid, .chart-placeholder { grid-template-columns: 1fr; }
  .chart-placeholder > p { grid-column: auto; }
  .access-flow { flex-direction: column; }
  .flow-arrow { transform: rotate(90deg); }
  .shell { width: min(100% - 20px, 1480px); }
  main { padding-top: 22px; }
}
"""


def validate_inputs(rows: Sequence[dict[str, str]], profile: Sequence[dict[str, str]]) -> None:
    if {row["format"] for row in rows} != set(FORMAT_ORDER):
        raise SystemExit("timing data does not contain the expected 17 formats")
    if {row["format"] for row in profile} != set(FORMAT_ORDER):
        raise SystemExit("profile data does not contain the expected 17 formats")
    if {row["component"] for row in rows} != {
        "register_decode",
        "stream_load",
        "stream_decode",
        "dot",
        "gemv",
    }:
        raise SystemExit("timing component coverage is incomplete")


def validate_accuracy_summary(rows: Sequence[dict[str, str]]) -> None:
    total_x4 = [row for row in rows if row["comparison"] == "total_x4"]
    if {row["format"] for row in total_x4} != set(FORMAT_ORDER):
        raise SystemExit("total_x4 accuracy data does not contain the expected 17 formats")
    if {row["kernel"] for row in total_x4} != {"dot", "gemv"}:
        raise SystemExit("total_x4 accuracy data is missing DOT or GEMV")
    if {row["distribution"] for row in total_x4} != {
        distribution for distribution, _ in ACCURACY_DISTRIBUTIONS
    }:
        raise SystemExit("total_x4 accuracy data is missing a distribution")


def validate_strategy_summary(rows: Sequence[dict[str, str]]) -> None:
    if {row["format"] for row in rows} != {"fp64", "e2m5", "e3m4"}:
        raise SystemExit("strategy timing data is missing fp64, e2m5, or e3m4")
    if {row["component"] for row in rows} != {"dot", "gemv"}:
        raise SystemExit("strategy timing data is missing DOT or GEMV")
    if {row["distribution"] for row in rows} != {
        distribution for distribution, _ in ACCURACY_DISTRIBUTIONS
    }:
        raise SystemExit("strategy timing data is missing a distribution")
    if {row["strategy"] for row in rows if row["format"] == "fp64"} != {
        "raw_pointer_x1"
    }:
        raise SystemExit("strategy timing data has an invalid FP64 baseline")
    for format_name in ("e2m5", "e3m4"):
        strategies = {row["strategy"] for row in rows if row["format"] == format_name}
        if len(strategies) != EXPECTED_STRATEGY_COUNT:
            raise SystemExit(
                f"strategy timing data contains {len(strategies)} "
                f"{format_name} strategies instead of {EXPECTED_STRATEGY_COUNT}"
            )
        for strategy in strategies:
            strategy_abbreviation(strategy)


def validate_multi_format_strategy_summary(
    rows: Sequence[dict[str, str]],
    expected_formats: set[str],
    source_name: str,
) -> None:
    benchmark_formats = {row.get("benchmark_format", "") for row in rows}
    if benchmark_formats != expected_formats:
        raise SystemExit(
            f"{source_name} strategy timing data has incomplete benchmark-format coverage"
        )
    if {row["component"] for row in rows} != {"dot", "gemv"}:
        raise SystemExit(f"{source_name} strategy timing data is missing DOT or GEMV")
    if {row["distribution"] for row in rows} != {
        distribution for distribution, _ in ACCURACY_DISTRIBUTIONS
    }:
        raise SystemExit(f"{source_name} strategy timing data is missing a distribution")
    for format_name in expected_formats:
        selected = format_conversion_rows(rows, format_name)
        metadata = conversion_strategy_metadata(rows, format_name)
        for strategy, row in metadata.items():
            conversion_strategy_family(strategy, row.get("decode_kind", "none"))
            if len([item for item in selected if item["strategy"] == strategy]) != 20:
                raise SystemExit(
                    f"{source_name} strategy timing data has incomplete {format_name}/"
                    f"{strategy} coverage"
                )


def validate_all_strategy_summary(rows: Sequence[dict[str, str]]) -> None:
    validate_multi_format_strategy_summary(
        rows,
        set(BASE_CONVERSION_FORMATS) - {"e2m5", "e3m4"},
        "all-format",
    )


def validate_expanded_strategy_summary(rows: Sequence[dict[str, str]]) -> None:
    validate_multi_format_strategy_summary(
        rows, set(EXPANDED_CONVERSION_FORMATS), "expanded-format"
    )


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else newest_complete_run(results_root)
    accuracy_dir = args.accuracy_dir.resolve()
    accuracy_results_root = args.accuracy_results_root.resolve()
    accuracy_run_dir = (
        args.accuracy_run_dir.resolve()
        if args.accuracy_run_dir
        else newest_accuracy_run(accuracy_results_root)
    )
    strategy_results_root = args.strategy_results_root.resolve()
    strategy_run_dir = (
        args.strategy_run_dir.resolve()
        if args.strategy_run_dir
        else newest_strategy_run(strategy_results_root)
    )
    all_strategy_results_root = args.all_strategy_results_root.resolve()
    all_strategy_run_dir = (
        args.all_strategy_run_dir.resolve()
        if args.all_strategy_run_dir
        else newest_all_strategy_run(all_strategy_results_root)
    )
    expanded_strategy_results_root = args.expanded_strategy_results_root.resolve()
    expanded_strategy_run_dir = (
        args.expanded_strategy_run_dir.resolve()
        if args.expanded_strategy_run_dir
        else newest_all_strategy_run(expanded_strategy_results_root)
    )
    output_dir = args.output_dir.resolve()
    assets = output_dir / "assets"
    interactive_dir = output_dir / "interactive"
    required_accuracy_files = (
        accuracy_dir / "simulation_model_comparison.csv",
        accuracy_dir / "accuracy_model_report.html",
        accuracy_run_dir / "simulation_summary.csv",
    )
    missing_accuracy_files = [
        path for path in required_accuracy_files if not path.is_file()
    ]
    if missing_accuracy_files:
        missing = ", ".join(str(path) for path in missing_accuracy_files)
        raise SystemExit(
            f"missing generated accuracy inputs: {missing}; "
            "run scripts/build_accuracy_model.sh first"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    assets.mkdir(parents=True)

    rows = read_csv(run_dir / "timing_summary.csv")
    packed = read_csv(run_dir / "packed_speedups.csv")
    profile = read_csv(run_dir / "profile_operations.csv")
    accuracy_rows = read_csv(accuracy_dir / "simulation_model_comparison.csv")
    accuracy_summary_rows = read_csv(accuracy_run_dir / "simulation_summary.csv")
    strategy_rows = read_csv(strategy_run_dir / "timing_summary.csv")
    all_strategy_rows = read_csv(all_strategy_run_dir / "timing_summary.csv")
    expanded_strategy_rows = read_csv(
        expanded_strategy_run_dir / "timing_summary.csv"
    )
    conversion_rows = [*strategy_rows, *all_strategy_rows, *expanded_strategy_rows]
    validate_inputs(rows, profile)
    validate_accuracy_summary(accuracy_summary_rows)
    validate_strategy_summary(strategy_rows)
    validate_all_strategy_summary(all_strategy_rows)
    validate_expanded_strategy_summary(expanded_strategy_rows)
    if {row["format"] for row in accuracy_rows} != set(FORMAT_ORDER):
        raise SystemExit("accuracy data does not contain the expected 17 formats")
    required_packed_fields = {"comparison_metric", "scalar_throughput", "packed_throughput"}
    if not required_packed_fields.issubset(packed[0]):
        raise SystemExit(
            "packed_speedups.csv uses the old normalization; rerun summarize_storage_performance.py"
        )

    figures: list[Path] = []
    interactive_pages: list[Path] = []
    figures.append(plot_total_kernel_time(rows, assets / "total-kernel-time.svg"))
    figures.append(plot_relative_fp64(rows, assets / "relative-fp64.svg"))
    figures.append(plot_size_scaling(rows, assets / "size-scaling.svg"))
    for storage_bits in BIT_FORMATS:
        formats = BIT_FORMATS[storage_bits]
        figures.append(
            plot_same_bit_kernel_time(
                rows, storage_bits, assets / f"same-bit-{storage_bits}.svg"
            )
        )
        interactive_svg = plot_interactive_same_bit_kernel_time(
            rows,
            storage_bits,
            assets / f"interactive-same-bit-{storage_bits}.svg",
        )
        figures.append(interactive_svg)
        interactive_pages.append(
            write_interactive_chart(
                interactive_svg,
                interactive_dir / f"same-bit-{storage_bits}.html",
                title=f"{storage_bits}-bit DOT and GEMV performance",
                description=(
                    f"Complete kernel time versus N for all {storage_bits}-bit "
                    "formats and x1, x2, and x4 access widths. Lower is faster."
                ),
                formats=formats,
                colors=dict(zip(formats, SAME_BIT_COLORS)),
                include_bit_filters=False,
            )
        )
        figures.append(
            plot_same_bit_accuracy(
                accuracy_summary_rows,
                storage_bits,
                assets / f"same-bit-accuracy-{storage_bits}.svg",
            )
        )
    all_performance_svg = plot_all_format_kernel_time(
        rows, assets / "interactive-all-format-performance.svg"
    )
    figures.append(all_performance_svg)
    interactive_pages.append(
        write_interactive_chart(
            all_performance_svg,
            interactive_dir / "all-format-performance.html",
            title="All-format DOT and GEMV performance",
            description=(
                "Complete kernel time versus N for all 17 storage formats and "
                "x1, x2, and x4 access widths. Lower is faster."
            ),
            formats=FORMAT_ORDER,
            colors=all_format_colors(),
            include_bit_filters=True,
        )
    )
    for format_name in CONVERSION_FORMATS:
        metadata = conversion_strategy_metadata(conversion_rows, format_name)
        strategy_svg = strip_trailing_whitespace(
            plot_conversion_strategy_kernel_time(
                conversion_rows,
                format_name,
                assets / f"conversion-{format_name}.svg",
            )
        )
        figures.append(strategy_svg)
        interactive_pages.append(
            write_conversion_strategy_chart(
                strategy_svg,
                interactive_dir / f"conversion-{format_name}.html",
                format_name=format_name,
                metadata=metadata,
            )
        )
    figures.append(
        plot_all_format_accuracy(
            accuracy_summary_rows, assets / "all-format-accuracy.svg"
        )
    )
    figures.append(plot_packed_speedup(packed, assets / "packed-speedup.svg"))
    figures.append(plot_packing_by_size(rows, assets / "packing-by-size.svg"))
    figures.append(plot_roofline(rows, profile, assets / "algorithmic-roofline.svg"))
    figures.append(plot_roof_efficiency(rows, profile, assets / "roof-efficiency.svg"))
    figures.append(plot_register_conversion(rows, assets / "register-conversion.svg"))
    figures.append(plot_stream_decode(rows, assets / "stream-decode.svg"))
    figures.append(plot_distribution_sensitivity(rows, assets / "distribution-sensitivity.svg"))
    figures.append(plot_bottleneck_map(profile, assets / "bottleneck-map.svg"))
    figures.append(plot_resource_heatmap(profile, assets / "resource-heatmap.svg"))
    figures.append(plot_register_pressure(profile, assets / "register-pressure.svg"))
    figures.append(plot_timing_stability(rows, assets / "timing-stability.svg"))
    figures.append(
        plot_accuracy_performance_tradeoff(
            rows, accuracy_rows, assets / "performance-accuracy-tradeoff.svg"
        )
    )

    (output_dir / "report.css").write_text(REPORT_CSS, encoding="utf-8")
    write_report(
        output_dir,
        run_dir,
        accuracy_dir,
        accuracy_run_dir,
        strategy_run_dir,
        all_strategy_run_dir,
        expanded_strategy_run_dir,
        rows,
        profile,
        accuracy_rows,
        conversion_rows,
    )
    generated_pages = [
        "index.html",
        "total-performance.html",
        "same-bit-formats.html",
        "packing.html",
        "roofline.html",
        "conversion.html",
        "bottlenecks.html",
        *(filename for filename, _ in ACCURACY_PAGES),
        "methodology.html",
        "general-info.html",
        "packing-bottlenecks.html",
        "conversion-fp32.html",
        *(
            compute_conversion_format_filename("fp32", name)
            for format_names in FP32_CONVERSION_BIT_FORMATS.values()
            for name in format_names
        ),
        "conversion-strategies.html",
        *(
            conversion_strategy_filename(name)
            for format_names in FP64_CONVERSION_BIT_FORMATS.values()
            for name in format_names
        ),
    ]
    manifest = [
        f"performance_run={run_dir.name}",
        f"accuracy_run={accuracy_run_dir.name}",
        f"strategy_run={strategy_run_dir.name}",
        f"all_strategy_run={all_strategy_run_dir.name}",
        f"expanded_strategy_run={expanded_strategy_run_dir.name}",
        f"accuracy_model={accuracy_dir.name}",
        f"timing_rows={len(rows)}",
        f"profile_operations={len(profile)}",
        f"accuracy_comparisons={len(accuracy_rows)}",
        f"accuracy_summary_rows={len(accuracy_summary_rows)}",
        f"strategy_timing_rows={len(strategy_rows)}",
        f"all_strategy_timing_rows={len(all_strategy_rows)}",
        f"expanded_strategy_timing_rows={len(expanded_strategy_rows)}",
        f"measured_hbm_gb_per_s={memory_ceiling(profile):.6f}",
        "packing_normalization=logical_throughput",
        "pages=" + ",".join(generated_pages),
        "figures=" + ",".join(path.name for path in figures),
        "interactive_pages=" + ",".join(path.name for path in interactive_pages),
    ]
    (output_dir / "report_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(generated_pages)} pages and {len(figures)} generated figures "
        f"to {output_dir}"
    )
    print(f"Open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
