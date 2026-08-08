#!/usr/bin/env python3
"""Generate analytical accuracy CSVs, SVG plots, and an HTML report."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _finish_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


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


def _format_number(value: object) -> str:
    if isinstance(value, str):
        return html.escape(value)
    number = float(value)
    if number == 0.0:
        return "0"
    return f"{number:.4e}"


def write_html_report(
    path: Path,
    scalar_rows: Sequence[dict[str, object]],
    figure_paths: Sequence[Path],
    gemv_m: int,
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
<title>DOT and GEMV analytical accuracy model</title>
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
<h1>DOT and GEMV analytical accuracy model</h1>
<p>Predictions for all 17 implemented storage formats under U(0,1) and N(0,1). GEMV fixes M={gemv_m}; N is the reduction length. CSV files retain exact zero values even where logarithmic plots place them at a visible floor.</p>
<p class="note">Cell integration is exact for formats with at most 14 fraction bits. Denser formats use a labeled high-resolution approximation. Absolute-error percentiles use a folded-normal approximation. Non-finite formats report finite-conditioned moments plus explicit failure probability.</p>
<nav><a href="#uniform_0_1">U(0,1)</a><a href="#normal_0_1">N(0,1)</a><a href="#arithmetic">FP64 arithmetic bound</a><a href="#scalar-table">Scalar table</a></nav>
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
    write_html_report(
        output_dir / "accuracy_model_report.html",
        scalar_rows,
        figure_paths,
        args.gemv_m,
    )
    print(f"Wrote analytical accuracy model to {output_dir}")
    print(f"Open {output_dir / 'accuracy_model_report.html'}")


if __name__ == "__main__":
    main()
