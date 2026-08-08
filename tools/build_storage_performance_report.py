#!/usr/bin/env python3
"""Build the unified multi-page H200 storage-format report."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import shutil
import statistics
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
    "e1m6": "E1M6",
    "e2m5": "E2M5",
    "e3m4": "E3M4",
    "fp8_e4m3": "FP8 E4M3",
    "fp8_e5m2": "FP8 E5M2",
    "e1m14": "E1M14",
    "e2m13": "E2M13",
    "e3m12": "E3M12",
    "fp16_e5m10": "FP16",
    "bf16_e8m7": "BF16",
    "e11m4": "E11M4",
    "e1m30": "E1M30",
    "e2m29": "E2M29",
    "e3m28": "E3M28",
    "fp32_e8m23": "FP32",
    "e11m20": "E11M20",
    "fp64_e11m52": "FP64",
}

FAMILY_COLORS = {
    "software": "#D55E00",
    "native": "#0072B2",
    "prefix": "#009E73",
    "baseline": "#3f4650",
}

LANE_COLORS = {1: "#6c757d", 2: "#0072B2", 4: "#D55E00"}
LANE_STYLES = {1: ":", 2: "--", 4: "-"}
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
        default=Path("results/008_storage_performance/report"),
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


def format_family(name: str) -> str:
    if name == "fp64_e11m52":
        return "baseline"
    if name.startswith(("fp8_", "fp16_", "bf16_", "fp32_")):
        return "native"
    if name.startswith("e11m"):
        return "prefix"
    return "software"


def label(name: str) -> str:
    return FORMAT_LABELS[name]


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
    fig, axes = plt.subplots(1, 2, figsize=(18.5, 7.8))
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
                axis.plot(
                    [int(row["n"]) for row in current],
                    [number(row, "median_time_ms") for row in current],
                    color=color_by_format[name],
                    marker=marker_by_format[name],
                    linestyle=LANE_STYLES[lane],
                    linewidth=2.0 if lane == 4 else 1.65,
                    markersize=5.0,
                    markerfacecolor=(color_by_format[name] if lane == 4 else "white"),
                    markevery=1,
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(component.upper(), fontsize=13)
        axis.set_xlabel("Reduction length N")
        axis.set_ylabel("Complete kernel time (ms)")
        format_axis_labels(axis)
    format_handles = [
        Line2D(
            [0],
            [0],
            color=color_by_format[name],
            marker=marker_by_format[name],
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
    fig.suptitle(
        f"{storage_bits}-bit formats: complete kernel time versus N",
        y=1.14,
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    return save_figure(fig, path)


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


def page_document(
    *,
    filename: str,
    title: str,
    intro: str,
    body: str,
    performance_run_name: str,
    accuracy_run_name: str,
) -> str:
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
<footer><div class="shell">Performance: <code>{html.escape(performance_run_name)}</code> · Accuracy: <code>{html.escape(accuracy_run_name)}</code></div></footer>
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
    rows: Sequence[dict[str, str]],
    profile: Sequence[dict[str, str]],
    accuracy_rows: Sequence[dict[str, str]],
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
            ("total-performance.html", "Total performance", "Complete DOT/GEMV time and size scaling."),
            ("same-bit-formats.html", "Same-bit formats", "Overlaid x1/x2/x4 performance plus same-bit and all-format accuracy."),
            ("packing.html", "Packed vs unpacked", "x2 and x4 throughput benefit over x1."),
            ("roofline.html", "Roofline", "Useful work relative to the measured HBM ceiling."),
            ("conversion.html", "Conversion", "Register decode, streaming decode, and data dependence."),
            ("bottlenecks.html", "Bottlenecks", "Nsight resource pressure, registers, and occupancy."),
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

    pages: dict[str, tuple[str, str, str]] = {
        "index.html": (
            "Overview",
            "One entry point for the H200 performance measurements and accuracy validation.",
            overview_body,
        ),
        "total-performance.html": (
            "Total performance",
            "Complete event-timed DOT and GEMV performance, including every kernel launch.",
            graph_section(
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
            graph_section(
                "8-bit performance",
                "All five 8-bit formats and all x1/x2/x4 access widths share the same axes. Color and marker identify the format; line style identifies packing.",
                "same-bit-8.svg",
                "High-resolution complete DOT and GEMV time versus N with every 8-bit format and x1, x2, and x4 access overlaid.",
                "SVG is vector resolution; select the figure to open it directly.",
            )
            + graph_section(
                "8-bit accuracy",
                "DOT uses normalized RMS error; GEMV uses relative L2 error. Curves are total x4 GPU error versus the original FP64 source for both distributions.",
                "same-bit-accuracy-8.svg",
                "High-resolution DOT normalized RMS and GEMV relative L2 error versus N for every 8-bit format and both distributions.",
                "Packing variants are omitted here because they encode identical values; reduction-order differences are discussed below.",
            )
            + graph_section(
                "16-bit performance",
                "E11M4, FP16, BF16, and all custom 16-bit layouts are overlaid with their x1/x2/x4 access widths.",
                "same-bit-16.svg",
                "High-resolution complete DOT and GEMV time versus N with every 16-bit format and x1, x2, and x4 access overlaid.",
                "SVG is vector resolution; select the figure to open it directly.",
            )
            + graph_section(
                "16-bit accuracy",
                "The same primary DOT and GEMV metrics compare every 16-bit format on U(0,1) and N(0,1).",
                "same-bit-accuracy-16.svg",
                "High-resolution DOT normalized RMS and GEMV relative L2 error versus N for every 16-bit format and both distributions.",
                "Packing variants are omitted because their storage quantization is identical.",
            )
            + graph_section(
                "32-bit performance",
                "FP32, E11M20, and all custom 32-bit layouts are overlaid with x1/x2/x4 access.",
                "same-bit-32.svg",
                "High-resolution complete DOT and GEMV time versus N with every 32-bit format and x1, x2, and x4 access overlaid.",
                "SVG is vector resolution; select the figure to open it directly.",
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
    }

    for filename, (title, intro, body) in pages.items():
        document = page_document(
            filename=filename,
            title=title,
            intro=intro,
            body=body,
            performance_run_name=run_dir.name,
            accuracy_run_name=accuracy_run_dir.name,
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
.graph-section, .text-section { margin: 34px 0 48px; }
.graph-section > p, .text-section > p { max-width: 900px; margin: 0 0 14px; }
figure { margin: 0; }
.graph-section figure + h3 { margin-top: 34px; }
figure img { display: block; width: 100%; height: auto; border: 1px solid var(--border); }
figcaption { color: var(--muted); margin-top: 8px; }
.summary-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin: 26px 0 40px; }
.summary-grid > div { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; display: grid; gap: 4px; }
.summary-grid span, .summary-grid small, .report-link span { color: var(--muted); }
.summary-grid strong { font-size: 1.35rem; }
.report-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.report-link { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; color: var(--fg); padding: 15px; text-decoration: none; display: grid; gap: 4px; }
.report-link:hover, .report-link:focus-visible { border-color: var(--link); }
a { color: var(--link); }
li { margin: 5px 0; }
footer { border-top: 1px solid var(--border); color: var(--muted); }
footer .shell { padding: 18px 0 28px; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
@media (max-width: 760px) {
  header .shell { align-items: flex-start; flex-direction: column; gap: 8px; }
  .summary-grid, .report-grid { grid-template-columns: 1fr; }
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
    output_dir = args.output_dir.resolve()
    assets = output_dir / "assets"
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
    validate_inputs(rows, profile)
    validate_accuracy_summary(accuracy_summary_rows)
    if {row["format"] for row in accuracy_rows} != set(FORMAT_ORDER):
        raise SystemExit("accuracy data does not contain the expected 17 formats")
    required_packed_fields = {"comparison_metric", "scalar_throughput", "packed_throughput"}
    if not required_packed_fields.issubset(packed[0]):
        raise SystemExit(
            "packed_speedups.csv uses the old normalization; rerun summarize_storage_performance.py"
        )

    figures: list[Path] = []
    figures.append(plot_total_kernel_time(rows, assets / "total-kernel-time.svg"))
    figures.append(plot_relative_fp64(rows, assets / "relative-fp64.svg"))
    figures.append(plot_size_scaling(rows, assets / "size-scaling.svg"))
    for storage_bits in BIT_FORMATS:
        figures.append(
            plot_same_bit_kernel_time(
                rows, storage_bits, assets / f"same-bit-{storage_bits}.svg"
            )
        )
        figures.append(
            plot_same_bit_accuracy(
                accuracy_summary_rows,
                storage_bits,
                assets / f"same-bit-accuracy-{storage_bits}.svg",
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
        rows,
        profile,
        accuracy_rows,
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
    ]
    manifest = [
        f"performance_run={run_dir.name}",
        f"accuracy_run={accuracy_run_dir.name}",
        f"accuracy_model={accuracy_dir.name}",
        f"timing_rows={len(rows)}",
        f"profile_operations={len(profile)}",
        f"accuracy_comparisons={len(accuracy_rows)}",
        f"accuracy_summary_rows={len(accuracy_summary_rows)}",
        f"measured_hbm_gb_per_s={memory_ceiling(profile):.6f}",
        "packing_normalization=logical_throughput",
        "pages=" + ",".join(generated_pages),
        "figures=" + ",".join(path.name for path in figures),
    ]
    (output_dir / "report_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(generated_pages)} pages and {len(figures)} generated figures "
        f"to {output_dir}"
    )
    print(f"Open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
