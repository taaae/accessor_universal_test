#!/usr/bin/env python3
"""Generate analytical accuracy CSVs, SVG plots, and an HTML report."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from accuracy_model import (
    FORMAT_SPECS,
    Distribution,
    arithmetic_bound_rows,
    build_kernel_rows,
    build_scalar_rows,
)

matplotlib.rcParams.update(
    {
        "svg.hashsalt": "accessor-accuracy-model-v1",
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


FORMAT_GROUPS = {
    "8bit": (8,),
    "16bit": (16,),
    "32_64bit": (32, 64),
}

COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
LINESTYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")

SCALAR_METRICS = (
    ("scalar_rmse", "Scalar RMSE", "log"),
    ("abs_scalar_bias", "Absolute scalar bias", "log"),
    ("decoded_zero_probability", "P(decoded zero)", "probability"),
    ("overflow_probability", "P(scalar overflow)", "probability"),
    ("saturation_probability", "P(scalar saturation)", "probability"),
    ("finite_probability", "P(finite encoding)", "linear"),
)

DOT_METRICS = (
    ("abs_bias", "Absolute output bias", "log"),
    ("rmse", "Storage RMSE", "log"),
    ("approx_mean_absolute_error", "Approx. mean absolute error", "log"),
    ("approx_p95_absolute_error", "Approx. p95 absolute error", "log"),
    ("normalized_rmse_proxy", "Normalized RMSE proxy", "log"),
    ("relative_rms", "Relative RMS", "log"),
    ("typical_condition_proxy", "Typical condition proxy", "log"),
    ("any_nonfinite_input_probability", "P(any nonfinite input)", "probability"),
    ("any_saturated_input_probability", "P(any saturated input)", "probability"),
)

GEMV_METRICS = (
    ("abs_row_bias", "Absolute row bias", "log"),
    ("row_rmse", "Row storage RMSE", "log"),
    ("approx_row_mean_absolute_error", "Approx. row mean absolute error", "log"),
    ("approx_row_p95_absolute_error", "Approx. row p95 absolute error", "log"),
    ("normalized_row_rmse_proxy", "Normalized row RMSE proxy", "log"),
    ("rms_l2_error", "RMS vector L2 error", "log"),
    ("relative_l2_rms", "Relative L2 RMS", "log"),
    ("any_nonfinite_input_probability", "P(any nonfinite input)", "probability"),
    ("any_saturated_input_probability", "P(any saturated input)", "probability"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/006_accuracy_model/generated"),
    )
    parser.add_argument("--dot-min-power", type=int, default=0)
    parser.add_argument("--dot-max-power", type=int, default=24)
    parser.add_argument("--gemv-min-power", type=int, default=4)
    parser.add_argument("--gemv-max-power", type=int, default=16)
    parser.add_argument("--gemv-m", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--dot-block-cap", type=int, default=2112)
    parser.add_argument(
        "--simulation-dir",
        type=Path,
        help="experiment-007 run directory; default: newest available run",
    )
    result = parser.parse_args()
    if result.dot_min_power > result.dot_max_power:
        parser.error("--dot-min-power must not exceed --dot-max-power")
    if result.gemv_min_power > result.gemv_max_power:
        parser.error("--gemv-min-power must not exceed --gemv-max-power")
    if result.gemv_m <= 0:
        parser.error("--gemv-m must be positive")
    return result


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def newest_simulation_dir(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    candidates = sorted(
        path
        for path in root.glob("run_*")
        if (path / "simulation_summary.csv").is_file()
    )
    return candidates[-1] if candidates else None


def _stable_any_probability(probability: float, count: int) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    return -math.expm1(count * math.log1p(-probability))


def _sample_standard_error(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return float("nan")
    return float(np.std(finite, ddof=1) / math.sqrt(len(finite)))


def build_simulation_comparison(
    kernel_rows: Sequence[dict[str, object]],
    scalar_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, str]],
    batch_rows: Sequence[dict[str, str]],
    convergence_rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    model = {
        (
            str(row["kernel"]),
            str(row["distribution"]),
            str(row["format"]),
            int(row["n"]),
            int(row["m"]),
        ): row
        for row in kernel_rows
    }
    scalar = {
        (str(row["distribution"]), str(row["format"])): row
        for row in scalar_rows
    }
    convergence = {
        (
            row["kernel"],
            row["distribution"],
            row["format"],
            int(row["n"]),
            int(row["m"]),
            row["comparison"],
        ): row["status"]
        for row in convergence_rows
    }
    batch_finite: dict[tuple[str, str, str, int, int], list[int]] = {}
    for row in batch_rows:
        if row["comparison"] != "storage":
            continue
        key = (
            row["kernel"],
            row["distribution"],
            row["format"],
            int(row["n"]),
            int(row["m"]),
        )
        batch_finite.setdefault(key, []).append(int(row["finite_count"]))

    result: list[dict[str, object]] = []
    for measured in summary_rows:
        if measured["comparison"] != "storage":
            continue
        key = (
            measured["kernel"],
            measured["distribution"],
            measured["format"],
            int(measured["n"]),
            int(measured["m"]),
        )
        predicted = model[key]
        predicted_mse = float(
            predicted["mse"]
            if measured["kernel"] == "dot"
            else predicted["row_mse"]
        )
        measured_mse = float(measured["mse"])
        mse_se = float(measured["mse_cluster_standard_error"])
        scalar_overflow = float(
            scalar[(measured["distribution"], measured["format"])][
                "overflow_probability"
            ]
        )
        predicted_failure = _stable_any_probability(
            scalar_overflow, 2 * int(measured["n"])
        )
        measured_failure = int(measured["class_mismatches"]) / int(
            measured["total_outputs"]
        )
        cluster_count = int(measured["statistical_batches"])
        outputs_per_cluster = int(measured["total_outputs"]) / cluster_count
        failure_rates = [
            1.0 - finite_count / outputs_per_cluster
            for finite_count in batch_finite[key]
        ]
        failure_se = _sample_standard_error(failure_rates)
        result.append(
            {
                "kernel": measured["kernel"],
                "distribution": measured["distribution"],
                "format": measured["format"],
                "storage_bits": int(measured["storage_bits"]),
                "method": predicted["method"],
                "n": int(measured["n"]),
                "m": int(measured["m"]),
                "total_outputs": int(measured["total_outputs"]),
                "finite_outputs": int(measured["finite_pairs"]),
                "predicted_mse": predicted_mse,
                "measured_mse": measured_mse,
                "predicted_rmse": math.sqrt(predicted_mse)
                if predicted_mse >= 0.0 and math.isfinite(predicted_mse)
                else float("nan"),
                "measured_rmse": float(measured["rmse"]),
                "mse_ratio_measured_to_predicted": (
                    measured_mse / predicted_mse
                    if predicted_mse > 0.0 and math.isfinite(measured_mse)
                    else float("nan")
                ),
                "mse_cluster_standard_error": mse_se,
                "mse_z_score": (
                    (measured_mse - predicted_mse) / mse_se
                    if mse_se > 0.0 and math.isfinite(mse_se)
                    else float("nan")
                ),
                "predicted_nonfinite_output_probability": predicted_failure,
                "measured_nonfinite_output_probability": measured_failure,
                "nonfinite_probability_cluster_standard_error": failure_se,
                "nonfinite_probability_z_score": (
                    (measured_failure - predicted_failure) / failure_se
                    if failure_se > 0.0 and math.isfinite(failure_se)
                    else float("nan")
                ),
                "convergence_status": convergence[key + ("storage",)],
            }
        )
    return result


def build_encoding_comparison(
    scalar_rows: Sequence[dict[str, object]],
    encoding_rows: Sequence[dict[str, str]],
) -> list[dict[str, object]]:
    scalar = {
        (str(row["distribution"]), str(row["format"])): row
        for row in scalar_rows
    }
    totals: dict[tuple[str, str, int], dict[str, int]] = {}
    for row in encoding_rows:
        key = (row["distribution"], row["format"], int(row["storage_bits"]))
        entry = totals.setdefault(
            key,
            {"source": 0, "zero": 0, "infinity": 0, "nan": 0, "saturation": 0},
        )
        entry["source"] += int(row["source_values"])
        entry["zero"] += int(row["decoded_zeros"])
        entry["infinity"] += int(row["decoded_infinities"])
        entry["nan"] += int(row["decoded_nans"])
        entry["saturation"] += int(row["saturated_values"])

    result: list[dict[str, object]] = []
    for (distribution, format_name, storage_bits), counts in sorted(totals.items()):
        predicted = scalar[(distribution, format_name)]
        count = counts["source"]
        result.append(
            {
                "distribution": distribution,
                "format": format_name,
                "storage_bits": storage_bits,
                "source_values": count,
                "predicted_zero_probability": predicted[
                    "decoded_zero_probability"
                ],
                "measured_zero_probability": counts["zero"] / count,
                "predicted_overflow_probability": predicted["overflow_probability"],
                "measured_infinity_probability": counts["infinity"] / count,
                "measured_nan_probability": counts["nan"] / count,
                "predicted_saturation_probability": predicted[
                    "saturation_probability"
                ],
                "measured_saturation_probability": counts["saturation"] / count,
            }
        )
    return result


def _metric_values(rows: Sequence[dict[str, object]], key: str) -> np.ndarray:
    if key.startswith("abs_"):
        source_key = key.removeprefix("abs_")
        return np.asarray([abs(float(row[source_key])) for row in rows])
    return np.asarray([float(row[key]) for row in rows])


def _set_value_scale(ax: plt.Axes, values: np.ndarray, kind: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    positives = finite[finite > 0.0]
    if kind == "linear":
        ax.set_ylim(bottom=0.0)
        return values
    if positives.size == 0:
        ax.set_ylim(-0.05, 1.05)
        return values

    floor = max(float(np.min(positives)) / 5.0, np.finfo(float).tiny)
    plotted = np.where(values > 0.0, values, floor)
    ax.set_yscale("log")
    if kind == "probability":
        ax.set_ylim(max(floor / 1.5, np.finfo(float).tiny), 1.05)
    else:
        ax.set_ylim(bottom=max(floor / 1.5, np.finfo(float).tiny))
    if np.any(values == 0.0):
        ax.text(
            0.01,
            0.02,
            "exact zeros at plot floor",
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
        )
    return plotted


def _style_for(index: int) -> dict[str, object]:
    return {
        "color": COLORS[index % len(COLORS)],
        "linestyle": LINESTYLES[(index // len(COLORS)) % len(LINESTYLES)],
        "marker": MARKERS[index % len(MARKERS)],
        "markevery": 4,
        "markersize": 3.5,
        "linewidth": 1.7,
    }


def _finish_figure(
    fig: plt.Figure, path: Path, *, normalize_whitespace: bool = False
) -> None:
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    if normalize_whitespace:
        text = path.read_text(encoding="utf-8")
        path.write_text(
            "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
            encoding="utf-8",
        )


def plot_scalar_group(
    rows: Sequence[dict[str, object]],
    distribution: str,
    bit_values: tuple[int, ...],
    path: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["distribution"] == distribution
        and int(row["storage_bits"]) in bit_values
    ]
    labels = [str(row["format"]) for row in selected]
    x = np.arange(len(selected))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.3), constrained_layout=True)
    for ax, (key, title, scale) in zip(axes.flat, SCALAR_METRICS):
        values = _metric_values(selected, key)
        ax.set_title(title, loc="left", fontsize=10)
        if np.all(values == 0.0):
            ax.text(
                0.5,
                0.5,
                "All formats: exactly zero",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
            ax.set_axis_off()
            continue
        plotted = _set_value_scale(ax, values, scale)
        colors = [COLORS[index % len(COLORS)] for index in range(len(selected))]
        ax.bar(x, plotted, color=colors, width=0.75)
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    fig.suptitle(
        f"Scalar quantization — {distribution} — {', '.join(map(str, bit_values))}-bit group",
        fontsize=13,
        x=0.01,
        ha="left",
    )
    _finish_figure(fig, path)


def plot_kernel_group(
    rows: Sequence[dict[str, object]],
    kernel: str,
    distribution: str,
    bit_values: tuple[int, ...],
    path: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["kernel"] == kernel
        and row["distribution"] == distribution
        and int(row["storage_bits"]) in bit_values
    ]
    metrics = DOT_METRICS if kernel == "dot" else GEMV_METRICS
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.2), constrained_layout=True)
    formats = [spec.name for spec in FORMAT_SPECS if spec.total_bits in bit_values]
    for ax, (key, title, scale) in zip(axes.flat, metrics):
        all_values = _metric_values(selected, key)
        ax.set_title(title, loc="left", fontsize=10)
        if np.all(all_values == 0.0):
            ax.text(
                0.5,
                0.5,
                "All curves are exactly zero",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
            ax.set_axis_off()
            continue
        finite = all_values[np.isfinite(all_values)]
        positives = finite[finite > 0.0]
        floor = (
            max(float(np.min(positives)) / 5.0, np.finfo(float).tiny)
            if positives.size
            else 0.0
        )
        for index, format_name in enumerate(formats):
            format_rows = [row for row in selected if row["format"] == format_name]
            format_rows.sort(key=lambda row: int(row["n"]))
            n_values = np.asarray([int(row["n"]) for row in format_rows])
            values = _metric_values(format_rows, key)
            plotted = np.where(values > 0.0, values, floor) if floor else values
            ax.plot(n_values, plotted, label=format_name, **_style_for(index))
        ax.set_xscale("log", base=2)
        if scale != "linear" and positives.size:
            ax.set_yscale("log")
            bottom = max(floor / 1.5, np.finfo(float).tiny)
            ax.set_ylim(bottom=bottom, top=1.05 if scale == "probability" else None)
            if np.any(all_values == 0.0):
                ax.text(
                    0.01,
                    0.02,
                    "zeros at floor",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="#555555",
                )
        ax.set_xlabel("N")
        ax.grid(alpha=0.22, linewidth=0.7)
    handles: list[object] = []
    labels: list[str] = []
    for legend_axis in axes.flat:
        handles, labels = legend_axis.get_legend_handles_labels()
        if handles:
            break
    fig.legend(
        handles,
        labels,
        loc="outside upper right",
        ncols=min(4, len(labels)),
        frameon=False,
        fontsize=8,
    )
    qualifier = "DOT" if kernel == "dot" else f"GEMV, M={selected[0]['m']}"
    fig.suptitle(
        f"{qualifier} storage prediction — {distribution} — {', '.join(map(str, bit_values))}-bit group",
        fontsize=13,
        x=0.01,
        ha="left",
    )
    _finish_figure(fig, path)


def plot_arithmetic_bounds(rows: Sequence[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, kernel in zip(axes, ("dot", "gemv")):
        for index, lanes in enumerate((1, 2, 4)):
            lane_rows = [
                row
                for row in rows
                if row["kernel"] == kernel and int(row["lanes"]) == lanes
            ]
            lane_rows.sort(key=lambda row: int(row["n"]))
            ax.plot(
                [int(row["n"]) for row in lane_rows],
                [float(row["fp64_normalized_error_bound"]) for row in lane_rows],
                label=f"x{lanes}",
                **_style_for(index),
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("normalized FP64 bound")
        ax.set_title(kernel.upper(), loc="left")
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.legend(frameon=False)
    fig.suptitle(
        "Structural FP64 arithmetic-rounding bounds", fontsize=13, x=0.01, ha="left"
    )
    _finish_figure(fig, path)


def plot_simulation_storage_group(
    rows: Sequence[dict[str, object]],
    kernel: str,
    distribution: str,
    bit_values: tuple[int, ...],
    path: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["kernel"] == kernel
        and row["distribution"] == distribution
        and int(row["storage_bits"]) in bit_values
    ]
    formats = [spec.name for spec in FORMAT_SPECS if spec.total_bits in bit_values]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    for panel_index, (ax, format_name) in enumerate(zip(axes.flat, formats)):
        format_rows = [row for row in selected if row["format"] == format_name]
        format_rows.sort(key=lambda row: int(row["n"]))
        n_values = np.asarray([int(row["n"]) for row in format_rows])
        predicted = np.asarray([float(row["predicted_rmse"]) for row in format_rows])
        measured = np.asarray([float(row["measured_rmse"]) for row in format_rows])
        mse = np.asarray([float(row["measured_mse"]) for row in format_rows])
        mse_se = np.asarray(
            [float(row["mse_cluster_standard_error"]) for row in format_rows]
        )
        positive = np.isfinite(predicted) & (predicted > 0.0)
        measured_positive = np.isfinite(measured) & (measured > 0.0)
        if np.any(positive):
            ax.plot(
                n_values[positive],
                predicted[positive],
                color=COLORS[0],
                linewidth=1.8,
                label="analytical RMSE",
            )
        if np.any(measured_positive):
            lower_mse = np.maximum(mse - 1.96 * mse_se, 0.0)
            upper_mse = np.maximum(mse + 1.96 * mse_se, 0.0)
            lower = np.sqrt(lower_mse)
            upper = np.sqrt(upper_mse)
            yerr = np.vstack(
                (
                    np.maximum(measured - lower, 0.0),
                    np.maximum(upper - measured, 0.0),
                )
            )
            yerr[:, ~np.isfinite(yerr).all(axis=0)] = 0.0
            ax.errorbar(
                n_values[measured_positive],
                measured[measured_positive],
                yerr=yerr[:, measured_positive],
                color=COLORS[1],
                marker="o",
                linestyle="none",
                markersize=4.0,
                capsize=2.5,
                label="H200 RMSE (95% cluster CI)",
            )
        if not np.any(positive) and not np.any(measured_positive):
            ax.text(
                0.5,
                0.5,
                "Storage conversion is exact",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#555555",
            )
        else:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("N")
            if panel_index % 3 == 0:
                ax.set_ylabel("storage RMSE")
            ax.grid(alpha=0.22, linewidth=0.7)
        ax.set_title(format_name, loc="left", fontsize=10)
    for ax in axes.flat[len(formats) :]:
        ax.set_axis_off()
    handles = [
        Line2D([0], [0], color=COLORS[0], linewidth=1.8, label="analytical RMSE"),
        Line2D(
            [0],
            [0],
            color=COLORS[1],
            marker="o",
            linestyle="none",
            label="H200 RMSE (95% cluster CI)",
        ),
    ]
    fig.legend(handles=handles, loc="outside upper right", frameon=False, ncols=2)
    qualifier = "DOT" if kernel == "dot" else "GEMV, M=1024"
    fig.suptitle(
        f"{qualifier} storage error: model vs H200 — {distribution} — "
        f"{', '.join(map(str, bit_values))}-bit group",
        fontsize=13,
        x=0.01,
        ha="left",
    )
    _finish_figure(fig, path, normalize_whitespace=True)


def plot_nonfinite_output_validation(
    rows: Sequence[dict[str, object]], path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    for ax, kernel in zip(axes, ("dot", "gemv")):
        selected = [
            row
            for row in rows
            if row["kernel"] == kernel
            and row["distribution"] == Distribution.NORMAL_01.value
        ]
        detectable_floor = min(0.5 / int(row["total_outputs"]) for row in selected)
        formats = [
            spec.name
            for spec in FORMAT_SPECS
            if any(
                row["format"] == spec.name
                and (
                    float(row["predicted_nonfinite_output_probability"])
                    >= detectable_floor
                    or float(row["measured_nonfinite_output_probability"]) > 0.0
                )
                for row in selected
            )
        ]
        for index, format_name in enumerate(formats):
            format_rows = [row for row in selected if row["format"] == format_name]
            format_rows.sort(key=lambda row: int(row["n"]))
            n_values = np.asarray([int(row["n"]) for row in format_rows])
            predicted = np.asarray(
                [
                    max(
                        float(row["predicted_nonfinite_output_probability"]),
                        detectable_floor,
                    )
                    for row in format_rows
                ]
            )
            measured = np.asarray(
                [
                    max(
                        float(row["measured_nonfinite_output_probability"]),
                        detectable_floor,
                    )
                    for row in format_rows
                ]
            )
            style = _style_for(index)
            style["markevery"] = None
            ax.plot(n_values, predicted, label=format_name, **style)
            ax.scatter(
                n_values,
                measured,
                color=COLORS[index % len(COLORS)],
                marker=MARKERS[index % len(MARKERS)],
                s=31,
                facecolors="none",
                linewidths=1.2,
                label="_nolegend_",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(bottom=detectable_floor / 1.5, top=1.1)
        ax.set_xlabel("N")
        ax.set_ylabel("non-finite output probability")
        ax.set_title(kernel.upper(), loc="left")
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.text(
            0.01,
            0.02,
            "unobserved probabilities shown at sampling floor",
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
        )
        ax.legend(frameon=False, fontsize=7, ncols=2)
        ax.text(
            0.99,
            0.02,
            "solid: model   open marker: H200",
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
            ha="right",
        )
    fig.suptitle(
        "Non-finite output probability: model vs H200 — normal inputs",
        fontsize=13,
        x=0.01,
        ha="left",
    )
    _finish_figure(fig, path, normalize_whitespace=True)


def plot_arithmetic_bound_validation(
    summary_rows: Sequence[dict[str, str]],
    arithmetic_rows: Sequence[dict[str, object]],
    path: Path,
) -> None:
    observed: dict[tuple[str, int, int], float] = {}
    for row in summary_rows:
        if not row["comparison"].startswith("kernel_x"):
            continue
        value = float(row["max_normalized_absolute_error"])
        if not math.isfinite(value):
            continue
        lanes = int(row["comparison"].removeprefix("kernel_x"))
        key = (row["kernel"], int(row["n"]), lanes)
        observed[key] = max(observed.get(key, 0.0), value)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)
    for ax, kernel in zip(axes, ("dot", "gemv")):
        simulated_sizes = sorted(
            {n for candidate_kernel, n, _ in observed if candidate_kernel == kernel}
        )
        for index, lanes in enumerate((1, 2, 4)):
            bounds = {
                int(row["n"]): float(row["fp64_normalized_error_bound"])
                for row in arithmetic_rows
                if row["kernel"] == kernel and int(row["lanes"]) == lanes
            }
            n_values = np.asarray(simulated_sizes)
            ax.plot(
                n_values,
                [bounds[n] for n in simulated_sizes],
                color=COLORS[index],
                linewidth=1.7,
                label=f"x{lanes}",
            )
            ax.scatter(
                n_values,
                [max(observed.get((kernel, n, lanes), 0.0), np.finfo(float).tiny) for n in simulated_sizes],
                color=COLORS[index],
                marker=MARKERS[index],
                facecolors="none",
                s=34,
                linewidths=1.2,
                label="_nolegend_",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("normalized kernel error")
        ax.set_title(kernel.upper(), loc="left")
        ax.grid(alpha=0.22, linewidth=0.7)
        ax.legend(frameon=False, fontsize=8, ncols=3)
        ax.text(
            0.99,
            0.02,
            "solid: bound   open marker: H200 maximum",
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
            ha="right",
        )
    fig.suptitle(
        "FP64 kernel arithmetic: structural bounds vs H200 maxima",
        fontsize=13,
        x=0.01,
        ha="left",
    )
    _finish_figure(fig, path, normalize_whitespace=True)


def _format_number(value: object) -> str:
    if isinstance(value, str):
        return html.escape(value)
    number = float(value)
    if number == 0.0:
        return "0"
    return f"{number:.4e}"


def _simulation_report_section(
    comparison_rows: Sequence[dict[str, object]],
    encoding_comparison_rows: Sequence[dict[str, object]],
    figure_by_name: dict[str, str],
    simulation_dir: Path,
    summary_rows: Sequence[dict[str, str]],
    arithmetic_rows: Sequence[dict[str, object]],
) -> str:
    validation_rows: list[str] = []
    for kernel in ("dot", "gemv"):
        for distribution in (
            Distribution.UNIFORM_01.value,
            Distribution.NORMAL_01.value,
        ):
            selected = [
                row
                for row in comparison_rows
                if row["kernel"] == kernel
                and row["distribution"] == distribution
                and float(row["predicted_mse"]) > 0.0
                and math.isfinite(float(row["measured_mse"]))
            ]
            ratios = [
                float(row["mse_ratio_measured_to_predicted"]) for row in selected
            ]
            z_scores = [
                abs(float(row["mse_z_score"]))
                for row in selected
                if math.isfinite(float(row["mse_z_score"]))
            ]
            validation_rows.append(
                "<tr>"
                f"<td>{kernel.upper()}</td>"
                f"<td>{html.escape(distribution)}</td>"
                f"<td>{len(selected)}</td>"
                f"<td>{np.median(ratios):.4f}</td>"
                f"<td>{sum(0.9 <= ratio <= 1.1 for ratio in ratios)}/{len(ratios)}</td>"
                f"<td>{sum(0.8 <= ratio <= 1.2 for ratio in ratios)}/{len(ratios)}</td>"
                f"<td>{sum(score <= 2.0 for score in z_scores)}/{len(z_scores)}</td>"
                f"<td>{sum(score <= 3.0 for score in z_scores)}/{len(z_scores)}</td>"
                "</tr>"
            )

    storage_status: dict[str, int] = {}
    for row in comparison_rows:
        status = str(row["convergence_status"])
        storage_status[status] = storage_status.get(status, 0) + 1

    comparable = [
        row
        for row in comparison_rows
        if float(row["predicted_mse"]) > 0.0
        and math.isfinite(float(row["measured_mse"]))
    ]
    dot_comparable = [row for row in comparable if row["kernel"] == "dot"]
    gemv_comparable = [row for row in comparable if row["kernel"] == "gemv"]
    dot_within_ten = sum(
        0.9 <= float(row["mse_ratio_measured_to_predicted"]) <= 1.1
        for row in dot_comparable
    )
    gemv_within_three = sum(
        math.isfinite(float(row["mse_z_score"]))
        and abs(float(row["mse_z_score"])) <= 3.0
        for row in gemv_comparable
    )
    gemv_z_count = sum(
        math.isfinite(float(row["mse_z_score"])) for row in gemv_comparable
    )
    encoding_differences = []
    for row in encoding_comparison_rows:
        encoding_differences.extend(
            (
                abs(
                    float(row["measured_zero_probability"])
                    - float(row["predicted_zero_probability"])
                ),
                abs(
                    float(row["measured_infinity_probability"])
                    - float(row["predicted_overflow_probability"])
                ),
                abs(
                    float(row["measured_saturation_probability"])
                    - float(row["predicted_saturation_probability"])
                ),
            )
        )

    bounds = {
        (str(row["kernel"]), int(row["n"]), int(row["lanes"])): float(
            row["fp64_normalized_error_bound"]
        )
        for row in arithmetic_rows
    }
    bound_fractions: list[float] = []
    violations = 0
    exact_kernel_comparisons = 0
    finite_kernel_comparisons = 0
    for row in summary_rows:
        if not row["comparison"].startswith("kernel_x"):
            continue
        measured = float(row["max_normalized_absolute_error"])
        if not math.isfinite(measured):
            continue
        finite_kernel_comparisons += 1
        exact_kernel_comparisons += measured == 0.0
        lanes = int(row["comparison"].removeprefix("kernel_x"))
        bound = bounds[(row["kernel"], int(row["n"]), lanes)]
        fraction = measured / bound
        bound_fractions.append(fraction)
        violations += fraction > 1.0 + 1e-12

    graph_blocks: list[str] = []
    for distribution in (
        Distribution.UNIFORM_01.value,
        Distribution.NORMAL_01.value,
    ):
        for kernel in ("dot", "gemv"):
            images = "".join(
                f'<figure><img src="{html.escape(figure_by_name[f"gpu_{distribution}_{kernel}_{group}"])}" '
                f'alt="{kernel} storage RMSE model and H200 comparison for {distribution}, {group}"></figure>'
                for group in FORMAT_GROUPS
            )
            graph_blocks.append(
                f"<h3>{kernel.upper()} — {html.escape(distribution)}</h3>{images}"
            )

    run_label = html.escape(simulation_dir.name)
    return f"""
<section id="gpu-validation"><h2>H200 simulation validation</h2>
<p>The measured run <code>{run_label}</code> contains 8,192 independent DOT samples per case and 16 independent GEMV matrix/vector replicates. Points show measured storage RMSE; error bars are approximate 95% intervals obtained from the cluster-level MSE standard error.</p>
<p>{dot_within_ten} of {len(dot_comparable)} finite nonzero DOT storage-MSE cases are within 10% of the analytical value. Across GEMV, {gemv_within_three} of {gemv_z_count} comparable cases are within three reported standard errors. The wider GEMV spread is consistent with having only 16 independently generated shared vectors.</p>
<p>The convergence checker classifies the 306 distinct storage comparisons as {storage_status.get('pass', 0)} pass, {storage_status.get('review', 0)} review, {storage_status.get('exact_zero', 0)} exact zero, and {storage_status.get('nonfinite_case', 0)} non-finite. Review cases need more independent samples before treating small model/measurement differences as significant.</p>
<p>Measured scalar zero, overflow, and saturation rates agree with the analytical probabilities to a maximum absolute difference of {max(encoding_differences):.2e}. Joined data are available in <a href="simulation_model_comparison.csv">simulation_model_comparison.csv</a> and <a href="simulation_encoding_comparison.csv">simulation_encoding_comparison.csv</a>.</p>
<div class="table-wrap"><table><thead><tr><th>Kernel</th><th>Distribution</th><th>Comparable cases</th><th>Median measured/predicted MSE</th><th>Within 10%</th><th>Within 20%</th><th>Within 2 SE</th><th>Within 3 SE</th></tr></thead><tbody>{''.join(validation_rows)}</tbody></table></div>
{''.join(graph_blocks)}
<h3>Non-finite outputs</h3>
<p>DOT failure rates follow the per-output model. GEMV rows share one vector, so rare vector-overflow events are sampled only 16 times; at small N, an absent rare event can make the measured rate look roughly half the marginal prediction. The complete-operation failure probability remains a separate quantity in <code>kernel_predictions.csv</code>.</p>
<figure><img src="{html.escape(figure_by_name['gpu_normal_0_1_nonfinite_outputs'])}" alt="Predicted and measured non-finite DOT and GEMV output probabilities"></figure>
<h3>FP64 kernel arithmetic</h3>
<p>No observed x1/x2/x4 normalized kernel error exceeds its structural bound. The largest observed value uses {max(bound_fractions):.1%} of the bound; violations: {violations}. {exact_kernel_comparisons} of {finite_kernel_comparisons} finite kernel comparisons are bit-exact against the decoded-storage reference.</p>
<figure><img src="{html.escape(figure_by_name['gpu_fp64_arithmetic_validation'])}" alt="FP64 arithmetic bounds and measured H200 kernel errors"></figure>
</section>
"""


def write_html_report(
    path: Path,
    scalar_rows: Sequence[dict[str, object]],
    figure_paths: Sequence[Path],
    gemv_m: int,
    simulation_section: str = "",
) -> None:
    figure_by_name = {figure.stem: figure.name for figure in figure_paths}
    sections: list[str] = []
    for distribution in (Distribution.UNIFORM_01.value, Distribution.NORMAL_01.value):
        blocks: list[str] = []
        for kernel in ("scalar", "dot", "gemv"):
            figures = [
                figure_by_name[f"{distribution}_{kernel}_{group}"]
                for group in FORMAT_GROUPS
            ]
            blocks.append(
                f"<h3>{kernel.upper()}</h3>"
                + "".join(
                    f'<figure><img src="{html.escape(name)}" alt="{kernel} analytical predictions for {distribution}"></figure>'
                    for name in figures
                )
            )
        sections.append(
            f'<section id="{distribution}"><h2>{distribution}</h2>{"".join(blocks)}</section>'
        )

    table_rows = []
    for row in scalar_rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['distribution']))}</td>"
            f"<td>{html.escape(str(row['format']))}</td>"
            f"<td>{int(row['storage_bits'])}</td>"
            f"<td>{html.escape(str(row['method']))}</td>"
            f"<td>{_format_number(row['scalar_rmse'])}</td>"
            f"<td>{_format_number(row['overflow_probability'])}</td>"
            f"<td>{_format_number(row['saturation_probability'])}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DOT and GEMV accuracy model and H200 validation</title>
<style>
:root {{ color-scheme: light dark; --bg: #f7f7f5; --fg: #202124; --muted: #5f6368; --panel: #ffffff; --border: #d8d8d4; --link: #005ea8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #171819; --fg: #ececec; --muted: #b4b4b4; --panel: #222426; --border: #414346; --link: #7dc4ff; }} img {{ background: #fff; }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
h1, h2, h3 {{ font-weight: 500; }} h1 {{ margin-bottom: 8px; }}
h2 {{ margin-top: 42px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }} h3 {{ margin-top: 28px; }}
p {{ max-width: 900px; }} nav {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 20px 0; }} a {{ color: var(--link); }}
figure {{ margin: 18px 0; }} img {{ display: block; width: 100%; height: auto; border: 1px solid var(--border); }}
.note {{ color: var(--muted); }} .table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; background: var(--panel); }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
th {{ position: sticky; top: 0; background: var(--panel); font-weight: 500; }}
</style>
</head>
<body><main>
<h1>DOT and GEMV accuracy model and H200 validation</h1>
<p>Predictions for all 17 implemented storage formats under U(0,1) and N(0,1). GEMV fixes M={gemv_m}; N is the reduction length. CSV files retain exact zero values even where logarithmic plots place them at a visible floor.</p>
<p class="note">Cell integration is exact for formats with at most 14 fraction bits. Denser formats use a labeled high-resolution approximation. Absolute-error percentiles use a folded-normal approximation. Non-finite formats report finite-conditioned moments plus explicit failure probability.</p>
<nav>{'<a href="#gpu-validation">H200 validation</a>' if simulation_section else ''}<a href="#uniform_0_1">Analytical U(0,1)</a><a href="#normal_0_1">Analytical N(0,1)</a><a href="#arithmetic">FP64 arithmetic bound</a><a href="#scalar-table">Scalar table</a></nav>
{simulation_section}
{''.join(sections)}
<section id="arithmetic"><h2>FP64 arithmetic-rounding bound</h2><figure><img src="{figure_by_name['fp64_arithmetic_bounds']}" alt="FP64 DOT and GEMV arithmetic bounds"></figure></section>
<section id="scalar-table"><h2>Scalar prediction table</h2><div class="table-wrap"><table>
<thead><tr><th>Distribution</th><th>Format</th><th>Bits</th><th>Method</th><th>Scalar RMSE</th><th>Overflow probability</th><th>Saturation probability</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div></section>
</main></body></html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dot_sizes = [1 << power for power in range(args.dot_min_power, args.dot_max_power + 1)]
    gemv_sizes = [1 << power for power in range(args.gemv_min_power, args.gemv_max_power + 1)]

    moments, scalar_rows = build_scalar_rows()
    kernel_rows = build_kernel_rows(moments, dot_sizes, gemv_sizes, args.gemv_m)
    arithmetic_rows = arithmetic_bound_rows(
        dot_sizes,
        gemv_sizes,
        threads=args.threads,
        dot_block_cap=args.dot_block_cap,
    )

    write_csv(output_dir / "scalar_predictions.csv", scalar_rows)
    write_csv(output_dir / "kernel_predictions.csv", kernel_rows)
    write_csv(output_dir / "fp64_arithmetic_bounds.csv", arithmetic_rows)

    figure_paths: list[Path] = []
    for distribution in (Distribution.UNIFORM_01.value, Distribution.NORMAL_01.value):
        for group, bit_values in FORMAT_GROUPS.items():
            scalar_path = output_dir / f"{distribution}_scalar_{group}.svg"
            dot_path = output_dir / f"{distribution}_dot_{group}.svg"
            gemv_path = output_dir / f"{distribution}_gemv_{group}.svg"
            plot_scalar_group(scalar_rows, distribution, bit_values, scalar_path)
            plot_kernel_group(kernel_rows, "dot", distribution, bit_values, dot_path)
            plot_kernel_group(kernel_rows, "gemv", distribution, bit_values, gemv_path)
            figure_paths.extend((scalar_path, dot_path, gemv_path))

    bound_path = output_dir / "fp64_arithmetic_bounds.svg"
    plot_arithmetic_bounds(arithmetic_rows, bound_path)
    figure_paths.append(bound_path)

    simulation_section = ""
    simulation_dir = (
        args.simulation_dir.resolve()
        if args.simulation_dir is not None
        else newest_simulation_dir(Path("results/007_gpu_accuracy_simulation"))
    )
    if simulation_dir is not None:
        summary_rows = read_csv(simulation_dir / "simulation_summary.csv")
        batch_rows = read_csv(simulation_dir / "batch_estimates.csv")
        convergence_rows = read_csv(simulation_dir / "convergence_report.csv")
        encoding_rows = read_csv(simulation_dir / "encoding_stats.csv")
        comparison_rows = build_simulation_comparison(
            kernel_rows,
            scalar_rows,
            summary_rows,
            batch_rows,
            convergence_rows,
        )
        encoding_comparison_rows = build_encoding_comparison(
            scalar_rows, encoding_rows
        )
        write_csv(output_dir / "simulation_model_comparison.csv", comparison_rows)
        write_csv(
            output_dir / "simulation_encoding_comparison.csv",
            encoding_comparison_rows,
        )
        for distribution in (
            Distribution.UNIFORM_01.value,
            Distribution.NORMAL_01.value,
        ):
            for kernel in ("dot", "gemv"):
                for group, bit_values in FORMAT_GROUPS.items():
                    path = output_dir / f"gpu_{distribution}_{kernel}_{group}.svg"
                    plot_simulation_storage_group(
                        comparison_rows,
                        kernel,
                        distribution,
                        bit_values,
                        path,
                    )
                    figure_paths.append(path)
        nonfinite_path = output_dir / "gpu_normal_0_1_nonfinite_outputs.svg"
        plot_nonfinite_output_validation(comparison_rows, nonfinite_path)
        figure_paths.append(nonfinite_path)
        arithmetic_validation_path = (
            output_dir / "gpu_fp64_arithmetic_validation.svg"
        )
        plot_arithmetic_bound_validation(
            summary_rows, arithmetic_rows, arithmetic_validation_path
        )
        figure_paths.append(arithmetic_validation_path)
        simulation_section = _simulation_report_section(
            comparison_rows,
            encoding_comparison_rows,
            {figure.stem: figure.name for figure in figure_paths},
            simulation_dir,
            summary_rows,
            arithmetic_rows,
        )

    write_html_report(
        output_dir / "accuracy_model_report.html",
        scalar_rows,
        figure_paths,
        args.gemv_m,
        simulation_section,
    )
    print(f"Wrote analytical accuracy model to {output_dir}")
    print(f"Open {output_dir / 'accuracy_model_report.html'}")


if __name__ == "__main__":
    main()
