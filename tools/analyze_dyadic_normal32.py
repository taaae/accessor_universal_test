#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import os
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


CANONICAL_N = 1 << 26
CANONICAL_WARMUP = 10
CANONICAL_SAMPLES = 50
TARGETS = tuple(index / 8 for index in range(9))
BASELINES = {
    "raw_fp64": {
        "label": "Raw FP64",
        "storage_bits": "64",
        "arithmetic_type": "fp64",
        "before_order": 0,
        "after_order": 25,
    },
    "raw_fp32": {
        "label": "Raw FP32",
        "storage_bits": "32",
        "arithmetic_type": "fp32",
        "before_order": 1,
        "after_order": 24,
    },
    "fp32_to_fp64": {
        "label": "FP32 to FP64",
        "storage_bits": "32",
        "arithmetic_type": "fp64",
        "before_order": 2,
        "after_order": 23,
    },
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def finite_positive(row: dict[str, str], field: str, context: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{context}: invalid {field}={row[field]!r}")
    return value


def validate_contract(
    rows: list[dict[str, str]], metrics: list[dict[str, str]], correctness: str
) -> None:
    required = {
        "gpu", "mode", "variant", "distribution", "phase", "kernel", "N",
        "blocks", "threads", "storage_bits", "arithmetic_type",
        "table_location", "segments", "coefficient_bytes", "target_x", "q",
        "mean_unique_segments", "actual_x", "genuine_n01_expected_x",
        "warmup", "warmups_this_batch", "sample", "execution_order",
        "kernel_ms", "result",
    }
    if required - set(rows[0]):
        raise ValueError("timing CSV is missing required columns")
    if "cpu_gpu_bit_mismatches=0" not in correctness:
        raise ValueError("CPU/GPU correctness record contains mismatches")
    if "segments_covered=32" not in correctness:
        raise ValueError("correctness record does not cover all segments")
    if "dot_validation_passed=1" not in correctness:
        raise ValueError("end-to-end DOT correctness check failed")

    for variant, specification in BASELINES.items():
        baseline_rows = [row for row in rows if row["variant"] == variant]
        if len(baseline_rows) != CANONICAL_SAMPLES:
            raise ValueError(f"{variant} baseline has the wrong sample count")
        for index, row in enumerate(baseline_rows, start=1):
            expected = {
                "mode": "full", "kernel": "dot", "N": str(CANONICAL_N),
                "blocks": "512", "threads": "256",
                "storage_bits": str(specification["storage_bits"]),
                "arithmetic_type": str(specification["arithmetic_type"]),
                "table_location": "none", "segments": "0",
                "coefficient_bytes": "0", "warmup": str(CANONICAL_WARMUP),
                "distribution": "raw",
            }
            for field, value in expected.items():
                if row[field] != value:
                    raise ValueError(
                        f"{variant} row {index}: {field} mismatch"
                    )
            finite_positive(row, "kernel_ms", f"{variant} row {index}")
            if not math.isfinite(float(row["result"])):
                raise ValueError(f"{variant} row {index}: nonfinite result")
        if {int(row["sample"]) for row in baseline_rows} != set(
            range(CANONICAL_SAMPLES)
        ):
            raise ValueError(f"{variant} baseline has missing or duplicate samples")
        phases = {
            phase: [row for row in baseline_rows if row["phase"] == phase]
            for phase in ("before", "after")
        }
        if any(
            len(group) != CANONICAL_SAMPLES // 2 for group in phases.values()
        ):
            raise ValueError(f"{variant} baseline was not split before and after")
        if any(
            {row["warmups_this_batch"] for row in group} != {"5"}
            for group in phases.values()
        ):
            raise ValueError(f"{variant} batches do not have five warmups each")
        if {int(row["execution_order"]) for row in phases["before"]} != {
            int(specification["before_order"])
        }:
            raise ValueError(f"{variant} before batch has the wrong execution order")
        if {int(row["execution_order"]) for row in phases["after"]} != {
            int(specification["after_order"])
        }:
            raise ValueError(f"{variant} after batch has the wrong execution order")

    measured = [row for row in rows if row["variant"] == "dyadic_normal32"]
    if len(measured) != (len(TARGETS) + 1) * CANONICAL_SAMPLES:
        raise ValueError("DyadicNormal32 has the wrong total sample count")
    unknown = {row["variant"] for row in rows} - {
        *BASELINES,
        "dyadic_normal32",
    }
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    artificial = [row for row in measured if row["distribution"] == "hot_uniform"]
    genuine = [row for row in measured if row["distribution"] == "genuine_n01"]
    if len(artificial) != len(TARGETS) * CANONICAL_SAMPLES:
        raise ValueError("hot-uniform sweep has the wrong sample count")
    if len(genuine) != CANONICAL_SAMPLES:
        raise ValueError("genuine N(0,1) point has the wrong sample count")
    groups: dict[float, list[dict[str, str]]] = defaultdict(list)
    genuine_values: set[str] = set()
    for index, row in enumerate(measured, start=1):
        expected = {
            "mode": "full", "kernel": "dot", "N": str(CANONICAL_N),
            "blocks": "512", "threads": "256", "storage_bits": "32",
            "arithmetic_type": "fp64", "table_location": "shared",
            "segments": "32", "coefficient_bytes": "512",
            "warmup": str(CANONICAL_WARMUP),
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ValueError(f"Dyadic row {index}: {field} mismatch")
        finite_positive(row, "kernel_ms", f"Dyadic row {index}")
        if not math.isfinite(float(row["result"])):
            raise ValueError(f"Dyadic row {index}: nonfinite result")
        genuine_values.add(row["genuine_n01_expected_x"])
        if row["distribution"] == "hot_uniform":
            target = float(row["target_x"])
            if not any(math.isclose(target, value, abs_tol=1e-15) for value in TARGETS):
                raise ValueError(f"Dyadic row {index}: unexpected target X")
            if abs(float(row["actual_x"]) - target) > 0.01:
                raise ValueError(f"Dyadic row {index}: actual X misses target")
            if not 0 <= float(row["q"]) <= 1:
                raise ValueError(f"Dyadic row {index}: q is outside [0,1]")
            groups[target].append(row)
        elif row["distribution"] == "genuine_n01":
            if row["q"] != "nan":
                raise ValueError("genuine N(0,1) rows must not have mixture q")
        else:
            raise ValueError(f"Dyadic row {index}: unknown distribution")
    if set(groups) != set(TARGETS):
        raise ValueError("Dyadic timing groups are incomplete")
    for target_index, target in enumerate(TARGETS):
        group = groups[target]
        if len(group) != CANONICAL_SAMPLES:
            raise ValueError(f"X={target}: wrong sample count")
        if {int(row["sample"]) for row in group} != set(range(CANONICAL_SAMPLES)):
            raise ValueError(f"X={target}: missing or duplicate samples")
        if len({row["actual_x"] for row in group}) != 1:
            raise ValueError(f"X={target}: actual X changed between samples")
        phases = {phase: [row for row in group if row["phase"] == phase]
                  for phase in ("forward", "reverse")}
        if any(len(batch) != CANONICAL_SAMPLES // 2 for batch in phases.values()):
            raise ValueError(f"X={target}: forward/reverse split is wrong")
        if any({row["warmups_this_batch"] for row in batch} != {"5"}
               for batch in phases.values()):
            raise ValueError(f"X={target}: batch warmup count is wrong")
        if {int(row["execution_order"]) for row in phases["forward"]} != {3 + target_index}:
            raise ValueError(f"X={target}: forward execution order is wrong")
        if {int(row["execution_order"]) for row in phases["reverse"]} != {22 - target_index}:
            raise ValueError(f"X={target}: reverse execution order is wrong")
    if len(genuine_values) != 1:
        raise ValueError("genuine N(0,1) expected X is inconsistent")
    genuine_x = float(next(iter(genuine_values)))
    if not 0 < genuine_x < 1:
        raise ValueError("genuine N(0,1) expected X is outside (0,1)")
    if {int(row["sample"]) for row in genuine} != set(range(CANONICAL_SAMPLES)):
        raise ValueError("genuine N(0,1) point has missing or duplicate samples")
    if any(abs(float(row["target_x"]) - genuine_x) > 1e-12 for row in genuine):
        raise ValueError("genuine timing rows have the wrong expected X")
    genuine_actual = {row["actual_x"] for row in genuine}
    if len(genuine_actual) != 1 or abs(float(next(iter(genuine_actual))) - genuine_x) > 0.01:
        raise ValueError("genuine timing point misses its expected X")
    genuine_phases = {phase: [row for row in genuine if row["phase"] == phase]
                      for phase in ("forward", "reverse")}
    if any(len(batch) != CANONICAL_SAMPLES // 2 for batch in genuine_phases.values()):
        raise ValueError("genuine timing point was not split forward/reverse")
    if any({row["warmups_this_batch"] for row in batch} != {"5"}
           for batch in genuine_phases.values()):
        raise ValueError("genuine timing batches do not have five warmups each")
    if {int(row["execution_order"]) for row in genuine_phases["forward"]} != {12}:
        raise ValueError("genuine forward batch has the wrong execution order")
    if {int(row["execution_order"]) for row in genuine_phases["reverse"]} != {13}:
        raise ValueError("genuine reverse batch has the wrong execution order")

    required_metrics = {
        "distribution", "target_x", "q", "N", "left_unique_segments",
        "right_unique_segments", "mean_unique_segments", "actual_x",
    }
    if required_metrics - set(metrics[0]):
        raise ValueError("metrics CSV is missing required columns")
    if len(metrics) != len(TARGETS) + 1:
        raise ValueError("metrics CSV has the wrong row count")
    for row in metrics:
        target = float(row["target_x"])
        actual = float(row["actual_x"])
        if row["distribution"] == "hot_uniform" and (
            target not in TARGETS or abs(actual - target) > 0.01
        ):
            raise ValueError("metrics CSV has an invalid X point")
        if row["distribution"] == "genuine_n01" and abs(target - genuine_x) > 1e-12:
            raise ValueError("genuine metrics row has the wrong expected X")
        if row["N"] != str(CANONICAL_N):
            raise ValueError("metrics CSV has the wrong N")
        timing_group = groups[target] if row["distribution"] == "hot_uniform" else genuine
        timing_actual = float(timing_group[0]["actual_x"])
        if not math.isclose(actual, timing_actual, abs_tol=1e-14):
            raise ValueError("timing and metrics X disagree")


def summarize(group: list[dict[str, str]]) -> dict[str, float | int]:
    values = [float(row["kernel_ms"]) for row in group]
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "q1_ms": percentile(values, 0.25),
        "q3_ms": percentile(values, 0.75),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


def build_summary(
    rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, float | int | str]],
    dict[str, dict[str, float | int | str]],
    dict[str, float | int | str],
]:
    groups: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["variant"] == "dyadic_normal32" and row["distribution"] == "hot_uniform":
            groups[float(row["target_x"])].append(row)
    result: list[dict[str, float | int | str]] = []
    for target in TARGETS:
        group = groups[target]
        result.append(
            {
                "variant": "dyadic_normal32",
                "label": "DyadicNormal32",
                "target_x": target,
                "actual_x": float(group[0]["actual_x"]),
                "q": float(group[0]["q"]),
                "mean_unique_segments": float(group[0]["mean_unique_segments"]),
                **summarize(group),
            }
        )
    baselines: dict[str, dict[str, float | int | str]] = {}
    for variant, specification in BASELINES.items():
        baselines[variant] = {
            "variant": variant,
            "label": str(specification["label"]),
            **summarize([row for row in rows if row["variant"] == variant]),
        }
    genuine_rows = [
        row
        for row in rows
        if row["variant"] == "dyadic_normal32" and row["distribution"] == "genuine_n01"
    ]
    genuine: dict[str, float | int | str] = {
        "variant": "dyadic_normal32_genuine_n01",
        "label": "Genuine N(0,1)",
        "target_x": float(genuine_rows[0]["target_x"]),
        "actual_x": float(genuine_rows[0]["actual_x"]),
        "mean_unique_segments": float(genuine_rows[0]["mean_unique_segments"]),
        **summarize(genuine_rows),
    }
    return result, baselines, genuine


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def crossover_points(summary: list[dict[str, object]], baseline: float) -> list[float]:
    rows = sorted(summary, key=lambda row: float(row["actual_x"]))
    result: list[float] = []
    for left, right in zip(rows, rows[1:]):
        x0, x1 = float(left["actual_x"]), float(right["actual_x"])
        y0 = float(left["median_ms"]) - baseline
        y1 = float(right["median_ms"]) - baseline
        if y0 == 0:
            result.append(x0)
        elif y0 * y1 < 0:
            result.append(x0 + (x1 - x0) * (-y0) / (y1 - y0))
    if float(rows[-1]["median_ms"]) == baseline:
        result.append(float(rows[-1]["actual_x"]))
    return result


def make_svg(
    summary: list[dict[str, object]],
    baselines: dict[str, dict[str, object]],
    genuine: dict[str, object],
) -> str:
    width, height = 1240, 690
    left, right, top, bottom = 98, 330, 88, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    lows = [float(row["q1_ms"]) for row in summary] + [
        *(float(row["q1_ms"]) for row in baselines.values()),
        float(genuine["q1_ms"]),
    ]
    highs = [float(row["q3_ms"]) for row in summary] + [
        *(float(row["q3_ms"]) for row in baselines.values()),
        float(genuine["q3_ms"]),
    ]
    y_min, y_max = min(lows), max(highs)
    padding = max((y_max - y_min) * 0.16, y_max * 0.012)
    y_min -= padding
    y_max += padding

    def px(value: float) -> float:
        return left + value * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">DyadicNormal32 decoding versus segment dispersion</title>',
        '<desc id="desc">Median DOT kernel time with interquartile ranges, three raw-storage arithmetic baselines, and expected dispersion for genuine standard-normal inputs.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;fill:#243238}.title{font-size:25px;font-weight:700}.subtitle{font-size:14px;fill:#60727b}.axis{stroke:#243238;stroke-width:1.5}.grid{stroke:#dfe6e9;stroke-width:1}.tick{font-size:12px;fill:#60727b}.axis-label{font-size:15px;font-weight:600}.curve{fill:none;stroke:#b5532b;stroke-width:3.2;stroke-linecap:round;stroke-linejoin:round}.iqr{stroke:#b5532b;stroke-width:1.6;opacity:.82}.direct{font-size:14px;font-weight:700}.note{font-size:12px;fill:#60727b}.genuine{font-size:12px;font-weight:650;fill:#476a76}</style>',
        f'<text class="title" x="{left}" y="35">DyadicNormal32 decoding versus segment dispersion</text>',
        f'<text class="subtitle" x="{left}" y="59">DOT, N=2²⁶, scalar x1; drift-controlled raw FP32, FP32→FP64, and raw FP64 baselines</text>',
    ]
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = py(value)
        pieces.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        pieces.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.3f}</text>')
    for index in range(9):
        value = index / 8
        x = px(value)
        pieces.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        pieces.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle">{value:.3g}</text>')
    pieces.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text class="axis-label" x="{left + plot_width / 2}" y="{height - 35}" text-anchor="middle">Normalized segment-table dispersion X</text>',
            f'<text class="axis-label" transform="translate(27 {top + plot_height / 2}) rotate(-90)" text-anchor="middle">DOT kernel time (ms)</text>',
        ]
    )

    genuine_expected_x = float(genuine["target_x"])
    genuine_actual_x = float(genuine["actual_x"])
    genuine_px = px(genuine_expected_x)
    pieces.extend(
        [
            f'<line x1="{genuine_px:.2f}" y1="{top}" x2="{genuine_px:.2f}" y2="{top + plot_height}" stroke="#476a76" stroke-width="1.8" stroke-dasharray="4 5"/>',
            f'<text class="genuine" x="{genuine_px + 7:.2f}" y="{top + 17}">expected genuine N(0,1) X={genuine_expected_x:.3f}</text>',
        ]
    )

    baseline_styles = {
        "raw_fp32": ("#18794e", "5 5"),
        "fp32_to_fp64": ("#315f9b", "7 5"),
        "raw_fp64": ("#111820", "9 6"),
    }
    baseline_y: dict[str, float] = {}
    for variant, baseline in baselines.items():
        color, dash = baseline_styles[variant]
        median_y = py(float(baseline["median_ms"]))
        q1 = py(float(baseline["q1_ms"]))
        q3 = py(float(baseline["q3_ms"]))
        baseline_y[variant] = median_y
        pieces.extend(
            [
                f'<rect x="{left}" y="{min(q1, q3):.2f}" width="{plot_width}" height="{max(abs(q3 - q1), 1.0):.2f}" fill="{color}" opacity="0.055"/>',
                f'<line x1="{left}" y1="{median_y:.2f}" x2="{left + plot_width}" y2="{median_y:.2f}" stroke="{color}" stroke-width="2.0" stroke-dasharray="{dash}"/>',
            ]
        )

    ordered = sorted(summary, key=lambda row: float(row["actual_x"]))
    points = " ".join(
        f'{px(float(row["actual_x"])):.2f},{py(float(row["median_ms"])):.2f}'
        for row in ordered
    )
    pieces.append(f'<polyline class="curve" points="{points}"/>')
    for row in ordered:
        x = px(float(row["actual_x"]))
        y = py(float(row["median_ms"]))
        q1 = py(float(row["q1_ms"]))
        q3 = py(float(row["q3_ms"]))
        pieces.extend(
            [
                f'<line class="iqr" x1="{x:.2f}" y1="{q1:.2f}" x2="{x:.2f}" y2="{q3:.2f}"/>',
                f'<line class="iqr" x1="{x - 4:.2f}" y1="{q1:.2f}" x2="{x + 4:.2f}" y2="{q1:.2f}"/>',
                f'<line class="iqr" x1="{x - 4:.2f}" y1="{q3:.2f}" x2="{x + 4:.2f}" y2="{q3:.2f}"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.4" fill="#b5532b"/>',
            ]
        )

    genuine_point_x = px(genuine_actual_x)
    genuine_point_y = py(float(genuine["median_ms"]))
    genuine_q1 = py(float(genuine["q1_ms"]))
    genuine_q3 = py(float(genuine["q3_ms"]))
    pieces.extend(
        [
            f'<line x1="{genuine_point_x:.2f}" y1="{genuine_q1:.2f}" x2="{genuine_point_x:.2f}" y2="{genuine_q3:.2f}" stroke="#476a76" stroke-width="1.7"/>',
            f'<path d="M {genuine_point_x:.2f} {genuine_point_y - 6:.2f} L {genuine_point_x + 6:.2f} {genuine_point_y:.2f} L {genuine_point_x:.2f} {genuine_point_y + 6:.2f} L {genuine_point_x - 6:.2f} {genuine_point_y:.2f} Z" fill="#476a76"/>',
            f'<text class="genuine" x="{genuine_point_x + 10:.2f}" y="{genuine_point_y - 10:.2f}">direct genuine-source timing</text>',
        ]
    )

    label_x = left + plot_width + 48
    dyadic_y = py(float(ordered[-1]["median_ms"]))
    labels = [("DyadicNormal32", "#b5532b", dyadic_y, dyadic_y, "3 4")]
    for variant, baseline in baselines.items():
        color, dash = baseline_styles[variant]
        source_y = baseline_y[variant]
        labels.append((str(baseline["label"]), color, source_y, source_y, dash))
    labels.sort(key=lambda item: item[2])
    minimum_gap = 27.0
    for index in range(1, len(labels)):
        if labels[index][2] - labels[index - 1][2] < minimum_gap:
            label, color, _, source_y, dash = labels[index]
            labels[index] = (
                label,
                color,
                labels[index - 1][2] + minimum_gap,
                source_y,
                dash,
            )
    overflow = labels[-1][2] - (top + plot_height - 8)
    if overflow > 0:
        labels = [
            (label, color, label_y - overflow, source_y, dash)
            for label, color, label_y, source_y, dash in labels
        ]
    for label, color, label_y, source_y, dash in labels:
        pieces.append(f'<path d="M {left + plot_width:.2f} {source_y:.2f} L {label_x - 12:.2f} {label_y:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}" fill="none"/>')
        pieces.append(f'<text class="direct" x="{label_x}" y="{label_y + 5:.2f}" fill="{color}">{html.escape(label)}</text>')
    pieces.append(f'<text class="note" x="{left}" y="{height - 9}">Curve points, the genuine-source diamond, and each baseline summarize 50 launches. Bars and faint bands show IQR; every baseline was split before and after.</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def make_report(
    summary: list[dict[str, object]],
    baselines: dict[str, dict[str, object]],
    genuine: dict[str, object],
    samples_href: str,
    metrics_href: str,
    correctness_href: str,
    coefficients_href: str,
) -> str:
    raw = baselines["raw_fp64"]
    raw_median = float(raw["median_ms"])
    genuine_time = float(genuine["median_ms"])
    genuine_expected_x = float(genuine["target_x"])
    genuine_actual_x = float(genuine["actual_x"])
    medians = [float(row["median_ms"]) for row in summary]
    sensitivity = max(medians) / min(medians)
    crossings = crossover_points(summary, raw_median)
    if crossings:
        crossing_text = ", ".join(f"X≈{value:.3f}" for value in crossings)
    elif all(value < raw_median for value in medians):
        crossing_text = "No crossover. DyadicNormal32 is faster at every measured X."
    elif all(value > raw_median for value in medians):
        crossing_text = "No crossover. Raw FP64 is faster at every measured X."
    else:
        crossing_text = "No interpolation crossover could be resolved."
    rows = "".join(
        f'<tr><td>{float(row["target_x"]):.3f}</td><td>{float(row["actual_x"]):.6f}</td>'
        f'<td>{float(row["mean_unique_segments"]):.4f}</td>'
        f'<td>{float(row["median_ms"]):.6f}</td><td>{float(row["q1_ms"]):.6f}</td>'
        f'<td>{float(row["q3_ms"]):.6f}</td><td>{float(row["median_ms"]) / raw_median:.3f}×</td></tr>'
        for row in summary
    )
    rows += (
        f'<tr><td>genuine N(0,1)</td><td>{genuine_actual_x:.6f}</td>'
        f'<td>{float(genuine["mean_unique_segments"]):.4f}</td>'
        f'<td>{genuine_time:.6f}</td><td>{float(genuine["q1_ms"]):.6f}</td>'
        f'<td>{float(genuine["q3_ms"]):.6f}</td><td>{genuine_time / raw_median:.3f}×</td></tr>'
    )
    baseline_rows = "".join(
        f'<tr><td>{html.escape(str(baseline["label"]))}</td>'
        f'<td>{float(baseline["median_ms"]):.6f}</td>'
        f'<td>{float(baseline["q1_ms"]):.6f}</td>'
        f'<td>{float(baseline["q3_ms"]):.6f}</td>'
        f'<td>{float(baseline["median_ms"]) / raw_median:.3f}×</td></tr>'
        for baseline in baselines.values()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DyadicNormal32 dispersion benchmark</title>
<style>
:root{{--ink:#1d292f;--muted:#60727b;--line:#dbe3e7;--surface:#fff;--accent:#b5532b;--soft:#f0e7e2}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}main{{max-width:1280px;margin:0 auto;padding:54px 30px 80px}}h1{{font-size:40px;line-height:1.08;margin:0 0 14px}}h2{{font-size:25px;margin:40px 0 14px}}.lead{{max-width:980px;font-size:18px;line-height:1.55;color:var(--muted);margin:0 0 30px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:13px;margin:0 0 34px}}.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px}}.card b{{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px}}.figure{{background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 8px 26px rgba(36,58,70,.07)}}.figure img{{display:block;width:100%;height:auto}}.finding{{margin-top:18px;padding:18px 20px;background:var(--soft);border-left:5px solid var(--accent);border-radius:8px;font-size:17px;line-height:1.55}}table{{width:100%;border-collapse:collapse;background:white;border:1px solid var(--line);font-variant-numeric:tabular-nums}}th,td{{padding:11px 12px;text-align:right;border-bottom:1px solid var(--line)}}th:first-child,td:first-child{{text-align:left}}th{{background:#eaf0f2;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}}.method{{line-height:1.65;color:#33454e}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.93em}}.artifacts{{margin-top:40px;padding:20px 22px;background:#e7f0f5;border-left:5px solid #315f80;border-radius:8px;line-height:1.7}}a{{color:#245d87}}</style></head>
<body><main><h1>DyadicNormal32 dispersion benchmark</h1>
<p class="lead">This run isolates whether a 512-byte coefficient decoder changes speed when warps spread their accesses across more dyadic segments. Raw FP32, FP32 storage converted to FP64 arithmetic, and raw FP64 all use the same DOT geometry, executable, GPU allocation, and timing method. A separately generated genuine-standard-normal point checks the bank-access pattern that the one-dimensional X metric cannot capture.</p>
<div class="cards"><div class="card"><b>Input</b><code>N=2^26</code>, scalar x1 DOT</div><div class="card"><b>Geometry</b>512 × 256 first stage, then one reduction block</div><div class="card"><b>Raw FP32</b>{float(baselines["raw_fp32"]["median_ms"]):.6f} ms median</div><div class="card"><b>FP32 → FP64</b>{float(baselines["fp32_to_fp64"]["median_ms"]):.6f} ms median</div><div class="card"><b>Raw FP64</b>{raw_median:.6f} ms median</div></div>
<div class="figure"><img src="dyadic-normal32-dispersion.svg" alt="DyadicNormal32 DOT time versus normalized segment dispersion"></div>
<div class="finding">The directly timed genuine <code>N(0,1)</code> source has expected <code>X={genuine_expected_x:.3f}</code>, measured <code>X={genuine_actual_x:.3f}</code>, and takes {genuine_time:.6f} ms: {genuine_time / float(baselines["fp32_to_fp64"]["median_ms"]):.3f}× FP32→FP64 and {genuine_time / raw_median:.3f}× raw FP64. Across the artificial hot-plus-uniform sweep, the slowest median is {sensitivity:.3f}× the fastest. {html.escape(crossing_text)}</div>
<h2>Baselines</h2><table><thead><tr><th>Storage and arithmetic</th><th>Median ms</th><th>Q1 ms</th><th>Q3 ms</th><th>/ raw FP64</th></tr></thead><tbody>{baseline_rows}</tbody></table>
<h2>Measured points</h2><table><thead><tr><th>Target X</th><th>Actual X</th><th>Unique h / warp</th><th>Median ms</th><th>Q1 ms</th><th>Q3 ms</th><th>/ raw FP64</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Exact format convention</h2><p class="method">Bit 31 is the sign. Bits 30 through 0 are the magnitude rank <code>r</code>. Counting leading one bits in <code>r</code> gives segment <code>h</code>. For <code>h=0…30</code>, the zero delimiter is followed by <code>30-h</code> payload bits <code>l</code>. Segment boundaries split the half-normal code-point density with σ=√3 into tail probabilities <code>2^-h</code>. Each segment uses midpoint-spaced linear reconstruction <code>fma(double(l), B[h], A[h])</code>. The all-ones magnitude rank is <code>h=31,l=0</code> and maps to the finite <code>2^-32</code> tail boundary, approximately 10.978 for source σ=1. The 32 coefficient pairs occupy exactly 512 bytes and are staged in shared memory once per timed first-stage block.</p>
<p class="method">For 32 segment entries, <code>X=(E[unique h per warp]-1)/(E[unique h under uniform draws]-1)</code>. Sweep codes use a deterministic mixture of hot segment zero and uniform segment draws. The solver chooses mixture probability <code>q</code> for every target X. The genuine point samples the actual segment probabilities induced by an <code>N(0,1)</code> source. It is timed directly because equal X values can still map to different shared-memory bank conflicts. Payload bits and signs are independently randomized; left and right arrays use independent seeds. Each point is split between an ascending and descending pass. Every baseline is likewise split before and after the sweep, with the after order reversed to limit drift bias.</p>
<div class="artifacts"><b>Artifacts.</b> <a href="{html.escape(samples_href)}">Raw timing samples</a> · <a href="{html.escape(metrics_href)}">Measured segment dispersion</a> · <a href="{html.escape(correctness_href)}">CPU/GPU decoder checks</a> · <a href="{html.escape(coefficients_href)}">Exact coefficient table</a> · <a href="timing_summary.csv">Timing summary</a> · <a href="baseline_summary.csv">Baseline summary</a> · <a href="../run_manifest.txt">Run manifest</a></div>
</main></body></html>"""


def maybe_make_png(svg_path: Path, png_path: Path) -> bool:
    convert = shutil.which("convert") or shutil.which("magick")
    if convert is None:
        return False
    command = [convert]
    if Path(convert).name == "magick":
        command.append("convert")
    command.extend([str(svg_path), str(png_path)])
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        png_path.unlink(missing_ok=True)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--coefficients", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()

    rows = read_rows(arguments.samples)
    metrics = read_rows(arguments.metrics)
    correctness = arguments.correctness.read_text()
    validate_contract(rows, metrics, correctness)
    summary, baselines, genuine = build_summary(rows)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output_dir / "timing_summary.csv", [*summary, genuine])
    write_csv(
        arguments.output_dir / "baseline_summary.csv",
        [
            {
                "baseline": variant,
                "label": baseline["label"],
                **{key: value for key, value in baseline.items()
                   if key not in {"variant", "label"}},
            }
            for variant, baseline in baselines.items()
        ],
    )
    svg_path = arguments.output_dir / "dyadic-normal32-dispersion.svg"
    svg_path.write_text(make_svg(summary, baselines, genuine))
    made_png = maybe_make_png(
        svg_path, arguments.output_dir / "dyadic-normal32-dispersion.png"
    )
    samples_href = os.path.relpath(arguments.samples, arguments.output_dir)
    metrics_href = os.path.relpath(arguments.metrics, arguments.output_dir)
    correctness_href = os.path.relpath(arguments.correctness, arguments.output_dir)
    coefficients_href = os.path.relpath(arguments.coefficients, arguments.output_dir)
    (arguments.output_dir / "report.html").write_text(
        make_report(
            summary,
            baselines,
            genuine,
            samples_href,
            metrics_href,
            correctness_href,
            coefficients_href,
        )
    )
    print(f"wrote {svg_path} and {arguments.output_dir / 'report.html'}")
    print(f"PNG generated: {'yes' if made_png else 'no converter available'}")
    for baseline in baselines.values():
        print(
            f"{baseline['label']} median: "
            f"{float(baseline['median_ms']):.9f} ms"
        )
    print(
        "DyadicNormal32 endpoint medians: "
        f"X=0 {float(summary[0]['median_ms']):.9f} ms, "
        f"X=1 {float(summary[-1]['median_ms']):.9f} ms"
    )
    print(
        "genuine N(0,1): "
        f"expected X {float(genuine['target_x']):.9f}, "
        f"actual X {float(genuine['actual_x']):.9f}, "
        f"median {float(genuine['median_ms']):.9f} ms"
    )


if __name__ == "__main__":
    main()
