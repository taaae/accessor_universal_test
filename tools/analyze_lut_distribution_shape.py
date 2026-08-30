#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path


FORMATS = ("t16", "posit16_es1", "lns16_r11")
LABELS = {
    "t16": "T16",
    "posit16_es1": "posit<16,1>",
    "lns16_r11": "lns<16,11>",
}
COLORS = {
    "t16": "#e45756",
    "posit16_es1": "#4c78a8",
    "lns16_r11": "#54a24b",
}
MARKERS = {"t16": "circle", "posit16_es1": "square", "lns16_r11": "diamond"}
EXPECTED_SANITIZED = {"t16": 0, "posit16_es1": 1, "lns16_r11": 1}
CANONICAL_N = 1 << 26
CANONICAL_WARMUP = 10
CANONICAL_SAMPLES = 50
CANONICAL_LEFT_SEED = 0x243F6A8885A308D3
CANONICAL_RIGHT_SEED = 0x13198A2E03707344
RAW_LEFT_SEED = 0xA4093822299F31D0
RAW_RIGHT_SEED = 0x082EFA98EC4E6C89
RAW_N2P26_SANITY_ESTIMATE_MS = 0.25982871422400845
UNIFORM_UNIQUE = 8192 * (1 - (1 - 1 / 8192) ** 32)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"no timing rows in {path}")
    return rows


def validate_contract(
    rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    expected_samples: int,
) -> None:
    if expected_samples != CANONICAL_SAMPLES:
        raise ValueError(
            f"full reports require exactly {CANONICAL_SAMPLES} samples, got {expected_samples}"
        )
    required = {
        "gpu",
        "mode",
        "kernel",
        "N",
        "storage_bits",
        "arithmetic_type",
        "access_method",
        "packet_values",
        "lut_entries",
        "lut_bytes",
        "q",
        "q_eighths",
        "mean_unique_left",
        "mean_unique_right",
        "mean_unique_both",
        "normalized_sector_dispersion",
        "format",
        "sanitized_lut_entries",
        "warmup",
        "sample",
        "execution_order",
        "kernel_ms",
        "result",
    }
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"timing CSV is missing columns: {sorted(missing_columns)}")
    expected_constant = {
        "mode": "full",
        "kernel": "dot",
        "N": str(CANONICAL_N),
        "storage_bits": "16",
        "arithmetic_type": "fp32",
        "access_method": "scalar",
        "packet_values": "1",
        "lut_entries": "65536",
        "lut_bytes": "262144",
        "warmup": str(CANONICAL_WARMUP),
    }
    if len({row["gpu"] for row in rows}) != 1:
        raise ValueError("LUT timing rows contain more than one GPU model")
    for row_index, row in enumerate(rows, start=2):
        for field, expected in expected_constant.items():
            if row[field] != expected:
                raise ValueError(
                    f"timing row {row_index}: {field}={row[field]!r}, expected {expected!r}"
                )
        fmt = row["format"]
        if fmt not in FORMATS:
            raise ValueError(f"timing row {row_index}: unexpected format {fmt!r}")
        if int(row["sanitized_lut_entries"]) != EXPECTED_SANITIZED[fmt]:
            raise ValueError(f"timing row {row_index}: wrong sanitized count for {fmt}")
        q_eighths = int(row["q_eighths"])
        if not 0 <= q_eighths <= 8 or not math.isclose(
            float(row["q"]), q_eighths / 8, rel_tol=0, abs_tol=1e-15
        ):
            raise ValueError(f"timing row {row_index}: inconsistent q")
        left = float(row["mean_unique_left"])
        right = float(row["mean_unique_right"])
        both = float(row["mean_unique_both"])
        x = float(row["normalized_sector_dispersion"])
        if not math.isclose(both, 0.5 * (left + right), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"timing row {row_index}: inconsistent mean-sector fields")
        expected_x = (both - 1) / (UNIFORM_UNIQUE - 1)
        if not math.isclose(x, expected_x, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"timing row {row_index}: X does not match sector means")
        milliseconds = float(row["kernel_ms"])
        result = float(row["result"])
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise ValueError(f"timing row {row_index}: invalid kernel time")
        if not math.isfinite(result):
            raise ValueError(f"timing row {row_index}: nonfinite DOT result")

    by_q_sample: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    by_q_format: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        q_eighths = int(row["q_eighths"])
        sample = int(row["sample"])
        if not 0 <= sample < expected_samples:
            raise ValueError(f"sample index outside [0,{expected_samples}): {sample}")
        by_q_sample[(q_eighths, sample)].append(row)
        by_q_format[(q_eighths, row["format"])].append(row)
    for q_eighths in range(9):
        for sample in range(expected_samples):
            group = by_q_sample[(q_eighths, sample)]
            if {row["format"] for row in group} != set(FORMATS):
                raise ValueError(f"q={q_eighths}/8 sample={sample}: missing/duplicate format")
            if {int(row["execution_order"]) for row in group} != {0, 1, 2}:
                raise ValueError(f"q={q_eighths}/8 sample={sample}: invalid execution order")
        for fmt in FORMATS:
            group = by_q_format[(q_eighths, fmt)]
            if len(group) != expected_samples:
                raise ValueError(
                    f"q={q_eighths}/8 {fmt}: expected {expected_samples} rows, got {len(group)}"
                )
            if {int(row["sample"]) for row in group} != set(range(expected_samples)):
                raise ValueError(f"q={q_eighths}/8 {fmt}: duplicate/missing samples")

    metric_required = {
        "q",
        "q_eighths",
        "N",
        "left_seed",
        "right_seed",
        "hot_sector_base",
        "mean_unique_left",
        "mean_unique_right",
        "mean_unique_both",
        "uniform_mean_unique",
        "normalized_sector_dispersion",
    }
    if not metric_rows or metric_required - set(metric_rows[0]):
        raise ValueError("access-metrics CSV is empty or missing required columns")
    if len(metric_rows) != 9 or {int(row["q_eighths"]) for row in metric_rows} != set(range(9)):
        raise ValueError("access-metrics CSV must contain exactly q=0/8 through 8/8")
    for metric in metric_rows:
        q_eighths = int(metric["q_eighths"])
        if (
            metric["N"] != str(CANONICAL_N)
            or metric["hot_sector_base"] != "0"
            or int(metric["left_seed"]) != CANONICAL_LEFT_SEED
            or int(metric["right_seed"]) != CANONICAL_RIGHT_SEED
        ):
            raise ValueError(f"q={q_eighths}/8: wrong metric N, seed, or hot sector")
        if not math.isclose(float(metric["q"]), q_eighths / 8, rel_tol=0, abs_tol=1e-15):
            raise ValueError(f"q={q_eighths}/8: inconsistent metric q")
        if not math.isclose(float(metric["uniform_mean_unique"]), UNIFORM_UNIQUE, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"q={q_eighths}/8: wrong uniform-sector reference")
        for fmt in FORMATS:
            for timing in by_q_format[(q_eighths, fmt)]:
                for field in (
                    "mean_unique_left",
                    "mean_unique_right",
                    "mean_unique_both",
                    "normalized_sector_dispersion",
                ):
                    if not math.isclose(
                        float(metric[field]),
                        float(timing[field]),
                        rel_tol=0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"q={q_eighths}/8 {fmt}: metrics/timing mismatch in {field}"
                        )
    x_zero = float(by_q_format[(0, FORMATS[0])][0]["normalized_sector_dispersion"])
    x_uniform = float(by_q_format[(8, FORMATS[0])][0]["normalized_sector_dispersion"])
    if abs(x_zero) > 1e-15 or not 0.995 <= x_uniform <= 1.005:
        raise ValueError(f"endpoint X values are implausible: q=0 {x_zero}, q=1 {x_uniform}")


def summarize(rows: list[dict[str, str]], expected_samples: int | None) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["q_eighths"]), row["format"])].append(row)
    expected_groups = {(q, fmt) for q in range(9) for fmt in FORMATS}
    if set(groups) != expected_groups:
        missing = sorted(expected_groups - set(groups))
        extra = sorted(set(groups) - expected_groups)
        raise ValueError(f"unexpected groups; missing={missing}, extra={extra}")

    summary: list[dict[str, object]] = []
    x_by_q: dict[int, set[float]] = defaultdict(set)
    for (q_eighths, fmt), group in sorted(groups.items()):
        if expected_samples is not None and len(group) != expected_samples:
            raise ValueError(
                f"q={q_eighths}/8 {fmt}: expected {expected_samples} samples, got {len(group)}"
            )
        x_values = {float(row["normalized_sector_dispersion"]) for row in group}
        if len(x_values) != 1:
            raise ValueError(f"q={q_eighths}/8 {fmt}: X changed within the group")
        x = next(iter(x_values))
        x_by_q[q_eighths].add(x)
        times = [float(row["kernel_ms"]) for row in group]
        summary.append(
            {
                "q_eighths": q_eighths,
                "q": q_eighths / 8.0,
                "x": x,
                "format": fmt,
                "label": LABELS[fmt],
                "samples": len(times),
                "median_ms": statistics.median(times),
                "q1_ms": percentile(times, 0.25),
                "q3_ms": percentile(times, 0.75),
                "minimum_ms": min(times),
                "maximum_ms": max(times),
            }
        )
    if any(len(values) != 1 for values in x_by_q.values()):
        raise ValueError("formats at a given q did not use identical input/X")
    ordered_x = [next(iter(x_by_q[q])) for q in range(9)]
    if any(right <= left for left, right in zip(ordered_x, ordered_x[1:])):
        raise ValueError("X is not strictly increasing")
    return summary


def validate_raw_fp32(rows: list[dict[str, str]]) -> None:
    required = {
        "gpu", "mode", "kernel", "N", "blocks", "threads", "storage_bits",
        "arithmetic_type", "storage_layout", "access_method", "packet_values",
        "lut_entries", "lut_bytes", "format", "x_semantics", "left_seed",
        "right_seed", "warmup", "sample", "kernel_ms", "result",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"raw-FP32 CSV is missing columns: {sorted(missing)}")
    expected = {
        "mode": "full",
        "kernel": "dot",
        "N": str(CANONICAL_N),
        "blocks": "512",
        "threads": "256",
        "storage_bits": "32",
        "arithmetic_type": "fp32",
        "storage_layout": "natural",
        "access_method": "scalar",
        "packet_values": "1",
        "lut_entries": "0",
        "lut_bytes": "0",
        "format": "raw_fp32",
        "x_semantics": "undefined_no_lut",
        "left_seed": str(RAW_LEFT_SEED),
        "right_seed": str(RAW_RIGHT_SEED),
        "warmup": str(CANONICAL_WARMUP),
    }
    if len(rows) != CANONICAL_SAMPLES:
        raise ValueError(
            f"raw FP32 requires {CANONICAL_SAMPLES} samples, got {len(rows)}"
        )
    if len({row["gpu"] for row in rows}) != 1:
        raise ValueError("raw-FP32 rows contain more than one GPU model")
    for row_index, row in enumerate(rows, start=2):
        for field, value in expected.items():
            if row[field] != value:
                raise ValueError(
                    f"raw-FP32 row {row_index}: {field}={row[field]!r}, expected {value!r}"
                )
        milliseconds = float(row["kernel_ms"])
        result = float(row["result"])
        if not math.isfinite(milliseconds) or milliseconds <= 0:
            raise ValueError(f"raw-FP32 row {row_index}: invalid kernel time")
        if not math.isfinite(result):
            raise ValueError(f"raw-FP32 row {row_index}: nonfinite DOT result")
    if {int(row["sample"]) for row in rows} != set(range(CANONICAL_SAMPLES)):
        raise ValueError("raw FP32 contains duplicate or missing sample indices")


def summarize_raw_fp32(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    values = [float(row["kernel_ms"]) for row in rows]
    return {
        "format": "raw_fp32",
        "label": "Raw FP32",
        "samples": len(values),
        "median_ms": statistics.median(values),
        "q1_ms": percentile(values, 0.25),
        "q3_ms": percentile(values, 0.75),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


def write_raw_summary(path: Path, row: dict[str, float | int | str]) -> None:
    fields = (
        "format", "label", "samples", "median_ms", "q1_ms", "q3_ms",
        "minimum_ms", "maximum_ms",
    )
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "q_eighths",
        "q",
        "x",
        "format",
        "label",
        "samples",
        "median_ms",
        "q1_ms",
        "q3_ms",
        "minimum_ms",
        "maximum_ms",
    )
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def marker_svg(kind: str, x: float, y: float, color: str) -> str:
    if kind == "square":
        return f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" rx="1" fill="{color}"/>'
    if kind == "diamond":
        return f'<path d="M {x:.2f} {y - 5:.2f} L {x + 5:.2f} {y:.2f} L {x:.2f} {y + 5:.2f} L {x - 5:.2f} {y:.2f} Z" fill="{color}"/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.4" fill="{color}"/>'


def make_svg(
    rows: list[dict[str, object]],
    raw_fp32: dict[str, float | int | str] | None = None,
) -> str:
    width, height = 1120, 660
    left, right, top, bottom = 92, 210, 64, 88
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_low = [float(row["q1_ms"]) for row in rows]
    all_high = [float(row["q3_ms"]) for row in rows]
    if raw_fp32 is not None:
        all_low.append(float(raw_fp32["q1_ms"]))
        all_high.append(float(raw_fp32["q3_ms"]))
    y_min, y_max = min(all_low), max(all_high)
    padding = max((y_max - y_min) * 0.14, y_max * 0.015)
    y_min -= padding
    y_max += padding

    def px(value: float) -> float:
        return left + value / 1.02 * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">16-bit global LUT DOT performance versus access dispersion</title>',
        '<desc id="desc">Median kernel time and interquartile range for T16, posit 16 es1, and LNS 16 r11, with a raw FP32 horizontal reference.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;fill:#263238}.title{font-size:24px;font-weight:700}.subtitle{font-size:14px;fill:#607078}.axis{stroke:#263238;stroke-width:1.5}.grid{stroke:#dfe5e8;stroke-width:1}.tick{font-size:12px;fill:#607078}.axis-label{font-size:15px;font-weight:600}.curve{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.iqr{stroke-width:1.5;opacity:.7}.direct{font-size:14px;font-weight:700}.note{font-size:12px;fill:#607078}</style>',
        f'<text class="title" x="{left}" y="31">LUT dispersion versus raw FP32</text>' if raw_fp32 is not None else f'<text class="title" x="{left}" y="31">Same LUT kernel, different table contents</text>',
        f'<text class="subtitle" x="{left}" y="52">DOT, N=2²⁶, scalar x1, FP32 arithmetic; LUT series use uint16 storage</text>' if raw_fp32 is not None else f'<text class="subtitle" x="{left}" y="52">DOT, N=2²⁶, uint16 storage, 65,536-entry global FP32 LUT, scalar x1</text>',
    ]
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = py(value)
        pieces.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        pieces.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.3f}</text>')
    for index in range(6):
        value = index * 0.2
        x = px(value)
        pieces.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        pieces.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle">{value:.1f}</text>')
    pieces.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text class="axis-label" x="{left + plot_width / 2}" y="{height - 28}" text-anchor="middle">Normalized LUT-sector dispersion</text>',
            f'<text class="axis-label" transform="translate(25 {top + plot_height / 2}) rotate(-90)" text-anchor="middle">DOT kernel time (ms)</text>',
        ]
    )

    if raw_fp32 is not None:
        raw_q1 = py(float(raw_fp32["q1_ms"]))
        raw_q3 = py(float(raw_fp32["q3_ms"]))
        raw_y = py(float(raw_fp32["median_ms"]))
        band_y = min(raw_q1, raw_q3)
        band_height = abs(raw_q3 - raw_q1)
        pieces.extend(
            [
                f'<rect x="{left}" y="{band_y:.2f}" width="{plot_width}" height="{max(band_height, 1.0):.2f}" fill="#263238" opacity="0.08"/>',
                f'<line x1="{left}" y1="{raw_y:.2f}" x2="{left + plot_width}" y2="{raw_y:.2f}" stroke="#111820" stroke-width="2.2" stroke-dasharray="8 6"/>',
                f'<line x1="{left + plot_width}" y1="{raw_y:.2f}" x2="{left + plot_width + 30}" y2="{raw_y:.2f}" stroke="#111820" stroke-width="1.5" stroke-dasharray="4 4"/>',
                f'<text class="direct" x="{left + plot_width + 42}" y="{raw_y + 5:.2f}" fill="#111820">Raw FP32</text>',
            ]
        )

    by_format: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_format[str(row["format"])].append(row)
    endpoints: list[tuple[str, float, float]] = []
    for fmt in FORMATS:
        series = sorted(by_format[fmt], key=lambda row: float(row["x"]))
        color = COLORS[fmt]
        points = " ".join(
            f'{px(float(row["x"])):.2f},{py(float(row["median_ms"])):.2f}'
            for row in series
        )
        pieces.append(f'<polyline class="curve" stroke="{color}" points="{points}"/>')
        for row in series:
            x = px(float(row["x"]))
            q1 = py(float(row["q1_ms"]))
            q3 = py(float(row["q3_ms"]))
            median_y = py(float(row["median_ms"]))
            pieces.extend(
                [
                    f'<line class="iqr" stroke="{color}" x1="{x:.2f}" y1="{q1:.2f}" x2="{x:.2f}" y2="{q3:.2f}"/>',
                    f'<line class="iqr" stroke="{color}" x1="{x - 4:.2f}" y1="{q1:.2f}" x2="{x + 4:.2f}" y2="{q1:.2f}"/>',
                    f'<line class="iqr" stroke="{color}" x1="{x - 4:.2f}" y1="{q3:.2f}" x2="{x + 4:.2f}" y2="{q3:.2f}"/>',
                    marker_svg(MARKERS[fmt], x, median_y, color),
                ]
            )
        last = series[-1]
        endpoints.append((fmt, px(float(last["x"])), py(float(last["median_ms"]))))

    # Spread direct labels if overlapping curves finish at almost the same time.
    labels = sorted(endpoints, key=lambda item: item[2])
    minimum_gap = 29.0
    adjusted: list[tuple[str, float, float, float]] = []
    previous = top - minimum_gap
    for fmt, x, y in labels:
        label_y = max(y, previous + minimum_gap)
        adjusted.append((fmt, x, y, label_y))
        previous = label_y
    overflow = max((item[3] for item in adjusted), default=top) - (top + plot_height - 12)
    if overflow > 0:
        adjusted = [(fmt, x, y, label_y - overflow) for fmt, x, y, label_y in adjusted]
    label_x = left + plot_width + 42
    for fmt, endpoint_x, endpoint_y, label_y in adjusted:
        color = COLORS[fmt]
        pieces.append(
            f'<path d="M {endpoint_x + 7:.2f} {endpoint_y:.2f} L {label_x - 12:.2f} {label_y:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="4 4" fill="none"/>'
        )
        pieces.append(
            f'<text class="direct" x="{label_x}" y="{label_y + 5:.2f}" fill="{color}">{html.escape(LABELS[fmt])}</text>'
        )
    note = "Points: median of 50 launches; bars: interquartile range. Raw FP32 has no LUT, so X is undefined and its dotted line is only a reference." if raw_fp32 is not None else "Points: median of 50 launches; bars: interquartile range. X is calculated from the actual left/right code arrays."
    pieces.append(f'<text class="note" x="{left}" y="{height - 8}">{note}</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def spread_statistics(rows: list[dict[str, object]]) -> tuple[float, float]:
    by_q: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_q[int(row["q_eighths"])].append(float(row["median_ms"]))
    relative = [(max(values) - min(values)) / statistics.mean(values) for values in by_q.values()]
    return statistics.median(relative), max(relative)


def make_report(
    summary: list[dict[str, object]],
    svg_name: str,
    lut_raw_href: str,
    summary_name: str,
    raw_fp32: dict[str, float | int | str] | None = None,
    raw_fp32_href: str | None = None,
) -> str:
    median_spread, maximum_spread = spread_statistics(summary)
    conclusion = (
        "The curves overlap closely: LUT contents did not materially change runtime."
        if maximum_spread < 0.05
        else "The curves do not fully overlap; inspect the measured spread before treating LUT contents as performance-neutral."
    )
    raw_result = ""
    raw_artifact = ""
    if raw_fp32 is not None:
        raw_median = float(raw_fp32["median_ms"])
        estimate_delta = (
            raw_median / RAW_N2P26_SANITY_ESTIMATE_MS - 1.0
        ) * 100.0
        raw_result = (
            f" Raw FP32 measured {raw_median:.6f} ms median "
            f"[{float(raw_fp32['q1_ms']):.6f}, {float(raw_fp32['q3_ms']):.6f}] IQR. "
            f"That is {estimate_delta:+.2f}% versus the pre-run half-N sanity estimate "
            f"of {RAW_N2P26_SANITY_ESTIMATE_MS:.6f} ms (obtained by halving the "
            "0.519657 ms N=2²⁷ bare-H200 result). Its X is undefined because it performs no LUT lookup; "
            "the horizontal dotted line is a reference, not a sampled X series."
        )
        raw_artifact = f' · <a href="{html.escape(raw_fp32_href or "")}">raw FP32 samples</a> · <a href="raw_fp32_summary.csv">raw FP32 summary</a>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>16-bit LUT distribution experiment</title>
<style>
:root{{--ink:#1d272c;--muted:#607078;--line:#dfe5e8;--accent:#325f83}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}main{{max-width:1220px;margin:0 auto;padding:54px 28px 72px}}h1{{font-size:38px;line-height:1.1;margin:0 0 14px}}.lead{{font-size:18px;line-height:1.55;color:var(--muted);max-width:900px;margin:0 0 30px}}.card{{background:white;border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 30px rgba(36,58,70,.08);overflow:hidden}}.card img{{display:block;width:100%;height:auto}}.result{{margin-top:24px;padding:24px 28px;background:#e8f1f6;border-left:5px solid var(--accent);border-radius:8px;font-size:17px;line-height:1.55}}.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-top:24px}}.meta div{{background:white;border:1px solid var(--line);border-radius:10px;padding:16px}}.meta b{{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px}}a{{color:#245d87}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em}}</style></head>
<body><main><h1>Does LUT content affect the dispersion curve?</h1>
<p class="lead">Three 16-bit formats use the same uint16 → global FP32 LUT → FP32 DOT kernel. The code arrays are byte-for-byte identical across formats at every dispersion point; only the 256 KiB table contents change.</p>
<section class="card"><img src="{html.escape(svg_name)}" alt="Kernel time versus normalized LUT-sector dispersion for T16, posit 16 es1, and LNS 16 r11"></section>
<section class="result"><strong>Result.</strong> {html.escape(conclusion)} The median relative spread between formats is {median_spread * 100:.2f}% across X points; the maximum is {maximum_spread * 100:.2f}%.{html.escape(raw_result)}</section>
<section class="meta"><div><b>Input</b><code>N=2^26</code>, independent left/right arrays</div><div><b>Timing</b>10 warmups, 50 measured launches per case</div><div><b>Kernel</b>512 × 256 first stage, same final reduction and event boundary</div><div><b>Artifacts</b><a href="{html.escape(lut_raw_href)}">LUT raw samples</a> · <a href="{html.escape(summary_name)}">LUT summary</a>{raw_artifact}</div></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--raw-fp32-samples", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()

    rows = read_rows(arguments.samples)
    metric_rows = read_rows(arguments.metrics)
    validate_contract(rows, metric_rows, CANONICAL_SAMPLES)
    summary = summarize(rows, CANONICAL_SAMPLES)
    raw_summary = None
    if arguments.raw_fp32_samples is not None:
        raw_rows = read_rows(arguments.raw_fp32_samples)
        validate_raw_fp32(raw_rows)
        lut_gpu = {row["gpu"] for row in rows}
        raw_gpu = {row["gpu"] for row in raw_rows}
        if lut_gpu != raw_gpu:
            raise ValueError(
                f"LUT/raw GPU mismatch: LUT={sorted(lut_gpu)}, raw={sorted(raw_gpu)}"
            )
        raw_summary = summarize_raw_fp32(raw_rows)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = arguments.output_dir / "timing_summary.csv"
    svg_path = arguments.output_dir / "lut-distribution-shape.svg"
    report_path = arguments.output_dir / "report.html"
    write_summary(summary_path, summary)
    if raw_summary is not None:
        write_raw_summary(arguments.output_dir / "raw_fp32_summary.csv", raw_summary)
    svg_path.write_text(make_svg(summary, raw_summary))
    lut_raw_href = os.path.relpath(arguments.samples, arguments.output_dir)
    raw_fp32_href = (
        os.path.relpath(arguments.raw_fp32_samples, arguments.output_dir)
        if arguments.raw_fp32_samples is not None
        else None
    )
    report_path.write_text(
        make_report(
            summary,
            svg_path.name,
            lut_raw_href,
            summary_path.name,
            raw_summary,
            raw_fp32_href,
        )
    )
    median_spread, maximum_spread = spread_statistics(summary)
    print(f"wrote {summary_path}")
    print(f"wrote {svg_path}")
    print(f"wrote {report_path}")
    print(f"median relative format spread: {median_spread * 100:.3f}%")
    print(f"maximum relative format spread: {maximum_spread * 100:.3f}%")
    if raw_summary is not None:
        print(
            "raw FP32 median [IQR]: "
            f"{float(raw_summary['median_ms']):.9f} "
            f"[{float(raw_summary['q1_ms']):.9f}, {float(raw_summary['q3_ms']):.9f}] ms"
        )


if __name__ == "__main__":
    main()
