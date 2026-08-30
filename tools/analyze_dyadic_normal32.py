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
        "label": "Raw FP64", "storage_bits": "64", "arithmetic_type": "fp64",
        "before_order": 0, "after_order": 85, "color": "#111820", "dash": "9 6",
    },
    "raw_fp32": {
        "label": "Raw FP32", "storage_bits": "32", "arithmetic_type": "fp32",
        "before_order": 1, "after_order": 84, "color": "#18794e", "dash": "5 5",
    },
    "fp32_to_fp64": {
        "label": "FP32 to FP64", "storage_bits": "32", "arithmetic_type": "fp64",
        "before_order": 2, "after_order": 83, "color": "#315f9b", "dash": "7 5",
    },
}
DYADIC_VARIANTS = {
    "dyadic_normal32": {
        "label": "Current DyadicNormal32", "table_location": "shared", "color": "#b5532b",
    },
    "dyadic_sign_fused": {
        "label": "Sign-fused decoder", "table_location": "shared", "color": "#9b3f80",
    },
    "dyadic_bitcast_shared": {
        "label": "BitCast, shared coefficients", "table_location": "shared", "color": "#d18b16",
    },
    "dyadic_bitcast_constant": {
        "label": "BitCast, constant coefficients", "table_location": "constant", "color": "#007f86",
    },
}
STRATEGY_ORDER = tuple(DYADIC_VARIANTS)
GRAPH_DYADIC_VARIANTS = tuple(
    variant for variant in DYADIC_VARIANTS
    if variant != "dyadic_bitcast_constant"
)


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


def validate_coefficients(path: Path) -> None:
    rows = read_rows(path)
    required = {
        "segment", "lower_boundary", "upper_boundary", "payload_bits", "levels",
        "linear_start", "linear_step", "bitcast_offset", "bitcast_span",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"coefficient CSV is missing columns: {sorted(missing)}")
    if len(rows) != 32:
        raise ValueError("coefficient CSV must contain 32 segments")
    previous_upper = 0.0
    for segment, row in enumerate(rows):
        if int(row["segment"]) != segment:
            raise ValueError("coefficient segments are not consecutive")
        numeric = {
            field: float(row[field]) for field in (
                "lower_boundary", "upper_boundary", "linear_start", "linear_step",
                "bitcast_offset", "bitcast_span",
            )
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError(f"segment {segment}: nonfinite coefficient")
        if not math.isclose(numeric["lower_boundary"], previous_upper,
                            rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"segment {segment}: discontinuous boundaries")
        if numeric["upper_boundary"] <= numeric["lower_boundary"]:
            raise ValueError(f"segment {segment}: boundaries do not increase")
        expected_payload_bits = 30 - segment if segment < 30 else 0
        expected_levels = 1 << expected_payload_bits if segment < 31 else 1
        if int(row["payload_bits"]) != expected_payload_bits:
            raise ValueError(f"segment {segment}: wrong payload bit count")
        if int(row["levels"]) != expected_levels:
            raise ValueError(f"segment {segment}: wrong level count")
        if segment < 31:
            expected_step = (
                numeric["upper_boundary"] - numeric["lower_boundary"]
            ) / expected_levels
            expected_start = numeric["lower_boundary"] + 0.5 * expected_step
            if not math.isclose(numeric["linear_step"], expected_step,
                                rel_tol=2e-15, abs_tol=1e-300):
                raise ValueError(f"segment {segment}: wrong linear step")
            if not math.isclose(numeric["linear_start"], expected_start,
                                rel_tol=2e-15, abs_tol=1e-300):
                raise ValueError(f"segment {segment}: wrong linear start")
            expected_span = numeric["linear_step"] * expected_levels
            expected_offset = numeric["linear_start"] - expected_span
            if not math.isclose(numeric["bitcast_span"], expected_span,
                                rel_tol=2e-15, abs_tol=1e-300):
                raise ValueError(f"segment {segment}: wrong BitCast span")
            if not math.isclose(numeric["bitcast_offset"], expected_offset,
                                rel_tol=2e-15, abs_tol=1e-300):
                raise ValueError(f"segment {segment}: wrong BitCast offset")
        else:
            if numeric["linear_step"] != 0 or numeric["bitcast_span"] != 0:
                raise ValueError("terminal segment must have zero step")
            if numeric["linear_start"] != numeric["upper_boundary"]:
                raise ValueError("terminal linear value must equal upper boundary")
            if numeric["bitcast_offset"] != numeric["linear_start"]:
                raise ValueError("terminal BitCast value must equal linear value")
        previous_upper = numeric["upper_boundary"]


def finite_positive(row: dict[str, str], field: str, context: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{context}: invalid {field}={row[field]!r}")
    return value


def require_fields(rows: list[dict[str, str]]) -> None:
    required = {
        "gpu", "mode", "variant", "distribution", "phase", "kernel", "N",
        "blocks", "threads", "storage_bits", "arithmetic_type", "table_location",
        "segments", "coefficient_bytes", "target_x", "q", "mean_unique_segments",
        "actual_x", "genuine_n01_expected_x", "warmup", "warmups_this_batch",
        "sample", "execution_order", "kernel_ms", "result",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"timing CSV is missing columns: {sorted(missing)}")


def validate_baselines(rows: list[dict[str, str]]) -> None:
    for variant, specification in BASELINES.items():
        group = [row for row in rows if row["variant"] == variant]
        if len(group) != CANONICAL_SAMPLES:
            raise ValueError(f"{variant} baseline has the wrong sample count")
        for index, row in enumerate(group, start=1):
            expected = {
                "mode": "full", "kernel": "dot", "N": str(CANONICAL_N),
                "blocks": "512", "threads": "256",
                "storage_bits": str(specification["storage_bits"]),
                "arithmetic_type": str(specification["arithmetic_type"]),
                "table_location": "none", "segments": "0", "coefficient_bytes": "0",
                "warmup": str(CANONICAL_WARMUP), "distribution": "raw",
            }
            for field, value in expected.items():
                if row[field] != value:
                    raise ValueError(f"{variant} row {index}: {field} mismatch")
            finite_positive(row, "kernel_ms", f"{variant} row {index}")
            if not math.isfinite(float(row["result"])):
                raise ValueError(f"{variant} row {index}: nonfinite result")
        if {int(row["sample"]) for row in group} != set(range(CANONICAL_SAMPLES)):
            raise ValueError(f"{variant} has missing or duplicate samples")
        phases = {phase: [row for row in group if row["phase"] == phase]
                  for phase in ("before", "after")}
        if any(len(batch) != CANONICAL_SAMPLES // 2 for batch in phases.values()):
            raise ValueError(f"{variant} was not split before and after")
        if {int(row["sample"]) for row in phases["before"]} != set(range(25)) \
                or {int(row["sample"]) for row in phases["after"]} != set(range(25, 50)):
            raise ValueError(f"{variant} baseline sample halves are wrong")
        if any({row["warmups_this_batch"] for row in batch} != {"5"}
               for batch in phases.values()):
            raise ValueError(f"{variant} batches do not have five warmups")
        for phase, order_key in (("before", "before_order"), ("after", "after_order")):
            if {int(row["execution_order"]) for row in phases[phase]} != {
                int(specification[order_key])
            }:
                raise ValueError(f"{variant} {phase} execution order is wrong")


def expected_dyadic_order(
    variant: str, distribution: str, phase: str, target_index: int | None
) -> int:
    strategy_index = STRATEGY_ORDER.index(variant)
    if distribution == "hot_uniform" and phase == "forward":
        assert target_index is not None
        return 3 + target_index * 4 + strategy_index
    if distribution == "genuine_n01" and phase == "forward":
        return 39 + strategy_index
    reverse_position = len(STRATEGY_ORDER) - 1 - strategy_index
    if distribution == "genuine_n01" and phase == "reverse":
        return 43 + reverse_position
    if distribution == "hot_uniform" and phase == "reverse":
        assert target_index is not None
        return 47 + (len(TARGETS) - 1 - target_index) * 4 + reverse_position
    raise ValueError("invalid Dyadic phase")


def validate_dyadic(rows: list[dict[str, str]]) -> float:
    genuine_x_values: set[str] = set()
    for variant, specification in DYADIC_VARIANTS.items():
        measured = [row for row in rows if row["variant"] == variant]
        if len(measured) != (len(TARGETS) + 1) * CANONICAL_SAMPLES:
            raise ValueError(f"{variant} has the wrong total sample count")
        groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
        for index, row in enumerate(measured, start=1):
            expected = {
                "mode": "full", "kernel": "dot", "N": str(CANONICAL_N),
                "blocks": "512", "threads": "256", "storage_bits": "32",
                "arithmetic_type": "fp64",
                "table_location": str(specification["table_location"]),
                "segments": "32", "coefficient_bytes": "512",
                "warmup": str(CANONICAL_WARMUP),
            }
            for field, value in expected.items():
                if row[field] != value:
                    raise ValueError(f"{variant} row {index}: {field} mismatch")
            finite_positive(row, "kernel_ms", f"{variant} row {index}")
            if not math.isfinite(float(row["result"])):
                raise ValueError(f"{variant} row {index}: nonfinite result")
            genuine_x_values.add(row["genuine_n01_expected_x"])
            distribution = row["distribution"]
            if distribution == "hot_uniform":
                target = float(row["target_x"])
                target_index = next(
                    (i for i, value in enumerate(TARGETS)
                     if math.isclose(target, value, abs_tol=1e-15)), None
                )
                if target_index is None or abs(float(row["actual_x"]) - target) > 0.01:
                    raise ValueError(f"{variant} row {index}: invalid target X")
                if not 0 <= float(row["q"]) <= 1:
                    raise ValueError(f"{variant} row {index}: q outside [0,1]")
            elif distribution == "genuine_n01":
                target = float(row["target_x"])
                target_index = None
                if row["q"] != "nan":
                    raise ValueError("genuine rows must not have mixture q")
            else:
                raise ValueError(f"{variant} row {index}: unknown distribution")
            groups[(distribution, target)].append(row)
            if int(row["execution_order"]) != expected_dyadic_order(
                variant, distribution, row["phase"], target_index
            ):
                raise ValueError(f"{variant} row {index}: execution order is wrong")
        if len(groups) != len(TARGETS) + 1:
            raise ValueError(f"{variant} timing groups are incomplete")
        for (distribution, target), group in groups.items():
            if len(group) != CANONICAL_SAMPLES:
                raise ValueError(f"{variant}/{distribution}/{target}: wrong samples")
            if {int(row["sample"]) for row in group} != set(range(CANONICAL_SAMPLES)):
                raise ValueError(f"{variant}/{distribution}/{target}: bad sample IDs")
            if len({row["actual_x"] for row in group}) != 1:
                raise ValueError(f"{variant}/{distribution}/{target}: X changed")
            phases = {name: [row for row in group if row["phase"] == name]
                      for name in ("forward", "reverse")}
            if any(len(batch) != CANONICAL_SAMPLES // 2 for batch in phases.values()):
                raise ValueError(f"{variant}/{distribution}/{target}: bad split")
            if {int(row["sample"]) for row in phases["forward"]} != set(range(25)) \
                    or {int(row["sample"]) for row in phases["reverse"]} != set(range(25, 50)):
                raise ValueError(
                    f"{variant}/{distribution}/{target}: wrong sample halves"
                )
            if any({row["warmups_this_batch"] for row in batch} != {"5"}
                   for batch in phases.values()):
                raise ValueError(f"{variant}/{distribution}/{target}: bad warmups")
    if len(genuine_x_values) != 1:
        raise ValueError("genuine N(0,1) expected X is inconsistent")
    genuine_x = float(next(iter(genuine_x_values)))
    if not 0 < genuine_x < 1:
        raise ValueError("genuine N(0,1) expected X is outside (0,1)")
    return genuine_x


def validate_metrics(
    rows: list[dict[str, str]], metrics: list[dict[str, str]], genuine_x: float
) -> None:
    required = {
        "distribution", "target_x", "q", "N", "left_unique_segments",
        "right_unique_segments", "mean_unique_segments", "actual_x",
    }
    missing = required - set(metrics[0])
    if missing:
        raise ValueError(f"metrics CSV is missing columns: {sorted(missing)}")
    if len(metrics) != len(TARGETS) + 1:
        raise ValueError("metrics CSV has the wrong row count")
    current_rows = [row for row in rows if row["variant"] == "dyadic_normal32"]
    for metric in metrics:
        distribution = metric["distribution"]
        target = float(metric["target_x"])
        actual = float(metric["actual_x"])
        if metric["N"] != str(CANONICAL_N):
            raise ValueError("metrics CSV has the wrong N")
        if distribution == "hot_uniform":
            if target not in TARGETS or abs(actual - target) > 0.01:
                raise ValueError("metrics CSV has an invalid X point")
        elif distribution == "genuine_n01":
            if abs(target - genuine_x) > 1e-12:
                raise ValueError("genuine metrics row has the wrong expected X")
        else:
            raise ValueError("metrics CSV has an unknown distribution")
        timing = next(
            row for row in current_rows
            if row["distribution"] == distribution
            and math.isclose(float(row["target_x"]), target, abs_tol=1e-14)
        )
        if not math.isclose(actual, float(timing["actual_x"]), abs_tol=1e-14):
            raise ValueError("timing and metrics X disagree")


def validate_contract(
    rows: list[dict[str, str]], metrics: list[dict[str, str]], correctness: str
) -> None:
    require_fields(rows)
    required_checks = (
        "current_cpu_gpu_bit_mismatches=0",
        "bitcast_shared_cpu_gpu_bit_mismatches=0",
        "bitcast_constant_cpu_gpu_bit_mismatches=0",
        "segments_covered=32", "dot_current_passed=1", "dot_sign_fused_passed=1",
        "dot_bitcast_shared_passed=1", "dot_bitcast_constant_passed=1",
        "timed_current_sign_fused_bit_mismatches=0",
        "timed_bitcast_shared_constant_bit_mismatches=0",
        "timed_result_validation_passed=1",
    )
    for check in required_checks:
        if check not in correctness:
            raise ValueError(f"correctness record failed: {check}")
    if "timed_result_checks=500" not in correctness:
        raise ValueError("full run did not compare all 500 timed result groups")
    known = set(BASELINES) | set(DYADIC_VARIANTS)
    unknown = {row["variant"] for row in rows} - known
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    orders = [int(row["execution_order"]) for row in rows]
    if orders != sorted(orders):
        raise ValueError("timing CSV is not in physical execution order")
    if set(orders) != set(range(86)):
        raise ValueError("timing CSV does not contain every execution batch")
    for order in range(86):
        samples = [int(row["sample"]) for row in rows
                   if int(row["execution_order"]) == order]
        if samples != sorted(samples):
            raise ValueError(f"execution batch {order} is not in sample order")
    validate_baselines(rows)
    genuine_x = validate_dyadic(rows)
    validate_metrics(rows, metrics, genuine_x)


def summarize(group: list[dict[str, str]]) -> dict[str, float | int]:
    values = [float(row["kernel_ms"]) for row in group]
    return {
        "samples": len(values), "median_ms": statistics.median(values),
        "q1_ms": percentile(values, 0.25), "q3_ms": percentile(values, 0.75),
        "minimum_ms": min(values), "maximum_ms": max(values),
    }


def build_summary(rows: list[dict[str, str]]) -> tuple[
    list[dict[str, object]], dict[str, dict[str, object]], list[dict[str, object]]
]:
    baselines: dict[str, dict[str, object]] = {}
    for variant, specification in BASELINES.items():
        group = [row for row in rows if row["variant"] == variant]
        baselines[variant] = {
            "variant": variant, "label": specification["label"], **summarize(group)
        }
    sweep: list[dict[str, object]] = []
    genuine: list[dict[str, object]] = []
    for variant, specification in DYADIC_VARIANTS.items():
        variant_rows = [row for row in rows if row["variant"] == variant]
        for target in TARGETS:
            group = [
                row for row in variant_rows
                if row["distribution"] == "hot_uniform"
                and math.isclose(float(row["target_x"]), target, abs_tol=1e-15)
            ]
            sweep.append({
                "variant": variant, "label": specification["label"], "target_x": target,
                "actual_x": float(group[0]["actual_x"]),
                "mean_unique_segments": float(group[0]["mean_unique_segments"]),
                **summarize(group),
            })
        group = [row for row in variant_rows if row["distribution"] == "genuine_n01"]
        genuine.append({
            "variant": variant, "label": specification["label"],
            "target_x": float(group[0]["target_x"]),
            "actual_x": float(group[0]["actual_x"]),
            "mean_unique_segments": float(group[0]["mean_unique_segments"]),
            **summarize(group),
        })
    return sweep, baselines, genuine


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_svg(
    sweep: list[dict[str, object]], baselines: dict[str, dict[str, object]],
    genuine: list[dict[str, object]],
) -> str:
    width, height = 1420, 760
    left, right, top, bottom = 98, 390, 92, 105
    plot_width, plot_height = width - left - right, height - top - bottom
    plotted_sweep = [row for row in sweep
                     if row["variant"] in GRAPH_DYADIC_VARIANTS]
    plotted_genuine = [row for row in genuine
                       if row["variant"] in GRAPH_DYADIC_VARIANTS]
    all_rows = [*plotted_sweep, *plotted_genuine, *baselines.values()]
    y_min = min(float(row["q1_ms"]) for row in all_rows)
    y_max = max(float(row["q3_ms"]) for row in all_rows)
    padding = max((y_max - y_min) * 0.14, y_max * 0.01)
    y_min, y_max = y_min - padding, y_max + padding

    def px(value: float) -> float:
        return left + value * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">DyadicNormal32 decoder strategies versus segment dispersion</title>',
        '<desc id="desc">Median DOT kernel time with interquartile ranges for three DyadicNormal32 decoders and three baselines. The outlying constant-memory BitCast decoder is retained in the tables but omitted from this plot.</desc>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;fill:#243238}.title{font-size:25px;font-weight:700}.subtitle{font-size:14px;fill:#60727b}.axis{stroke:#243238;stroke-width:1.5}.grid{stroke:#dfe6e9;stroke-width:1}.tick{font-size:12px;fill:#60727b}.axis-label{font-size:15px;font-weight:600}.direct{font-size:13px;font-weight:700}.note{font-size:12px;fill:#60727b}</style>',
        f'<text class="title" x="{left}" y="36">DyadicNormal32 decoder strategies versus segment dispersion</text>',
        f'<text class="subtitle" x="{left}" y="61">DOT, N=2²⁶, scalar x1; three decoder curves and three drift-controlled baselines</text>',
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
    pieces.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        f'<text class="axis-label" x="{left + plot_width / 2}" y="{height - 38}" text-anchor="middle">Normalized segment-table dispersion X</text>',
        f'<text class="axis-label" transform="translate(27 {top + plot_height / 2}) rotate(-90)" text-anchor="middle">DOT kernel time (ms)</text>',
    ])
    genuine_x = float(genuine[0]["target_x"])
    pieces.append(f'<line x1="{px(genuine_x):.2f}" y1="{top}" x2="{px(genuine_x):.2f}" y2="{top + plot_height}" stroke="#87979e" stroke-width="1.4" stroke-dasharray="4 5"/>')
    label_sources: list[tuple[str, str, float, str]] = []
    for variant in GRAPH_DYADIC_VARIANTS:
        specification = DYADIC_VARIANTS[variant]
        rows = sorted((row for row in sweep if row["variant"] == variant),
                      key=lambda row: float(row["actual_x"]))
        color = str(specification["color"])
        points = " ".join(
            f'{px(float(row["actual_x"])):.2f},{py(float(row["median_ms"])):.2f}'
            for row in rows
        )
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        for row in rows:
            x, y = px(float(row["actual_x"])), py(float(row["median_ms"]))
            q1, q3 = py(float(row["q1_ms"])), py(float(row["q3_ms"]))
            pieces.extend([
                f'<line x1="{x:.2f}" y1="{q1:.2f}" x2="{x:.2f}" y2="{q3:.2f}" stroke="{color}" stroke-width="1.4"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="{color}"/>',
            ])
        genuine_row = next(row for row in genuine if row["variant"] == variant)
        gx, gy = px(float(genuine_row["actual_x"])), py(float(genuine_row["median_ms"]))
        pieces.append(f'<path d="M {gx:.2f} {gy - 5:.2f} L {gx + 5:.2f} {gy:.2f} L {gx:.2f} {gy + 5:.2f} L {gx - 5:.2f} {gy:.2f} Z" fill="{color}"/>')
        label_sources.append((str(specification["label"]), color,
                              py(float(rows[-1]["median_ms"])), "3 4"))
    for variant, baseline in baselines.items():
        specification = BASELINES[variant]
        color, dash = str(specification["color"]), str(specification["dash"])
        y = py(float(baseline["median_ms"]))
        pieces.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
        label_sources.append((str(baseline["label"]), color, y, dash))
    labels = sorted(label_sources, key=lambda item: item[2])
    positioned: list[tuple[str, str, float, float, str]] = []
    for label, color, source_y, dash in labels:
        label_y = source_y if not positioned else max(source_y, positioned[-1][2] + 27)
        positioned.append((label, color, label_y, source_y, dash))
    overflow = positioned[-1][2] - (top + plot_height - 8)
    if overflow > 0:
        positioned = [(label, color, label_y - overflow, source_y, dash)
                      for label, color, label_y, source_y, dash in positioned]
    label_x = left + plot_width + 50
    for label, color, label_y, source_y, dash in positioned:
        pieces.append(f'<path d="M {left + plot_width:.2f} {source_y:.2f} L {label_x - 12:.2f} {label_y:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}" fill="none"/>')
        pieces.append(f'<text class="direct" x="{label_x}" y="{label_y + 5:.2f}" fill="{color}">{html.escape(label)}</text>')
    pieces.append(f'<text class="note" x="{left}" y="{height - 10}">Each point and line is the median of 50 launches; bars show IQR. Diamonds mark genuine N(0,1) at expected X={genuine_x:.3f}. BitCast/constant remains in the tables.</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def make_report(
    sweep: list[dict[str, object]], baselines: dict[str, dict[str, object]],
    genuine: list[dict[str, object]], samples_href: str, metrics_href: str,
    correctness_href: str, coefficients_href: str,
) -> str:
    raw_fp64 = float(baselines["raw_fp64"]["median_ms"])
    current = next(row for row in genuine if row["variant"] == "dyadic_normal32")
    fastest = min(genuine, key=lambda row: float(row["median_ms"]))
    fastest_time, current_time = float(fastest["median_ms"]), float(current["median_ms"])
    strategy_rows = "".join(
        f'<tr><td>{html.escape(str(row["label"]))}</td><td>{float(row["median_ms"]):.6f}</td>'
        f'<td>{float(row["q1_ms"]):.6f}</td><td>{float(row["q3_ms"]):.6f}</td>'
        f'<td>{float(row["median_ms"]) / current_time:.3f}×</td>'
        f'<td>{float(row["median_ms"]) / raw_fp64:.3f}×</td></tr>' for row in genuine
    )
    baseline_rows = "".join(
        f'<tr><td>{html.escape(str(row["label"]))}</td><td>{float(row["median_ms"]):.6f}</td>'
        f'<td>{float(row["q1_ms"]):.6f}</td><td>{float(row["q3_ms"]):.6f}</td>'
        f'<td>{float(row["median_ms"]) / raw_fp64:.3f}×</td></tr>'
        for row in baselines.values()
    )
    sweep_rows = "".join(
        f'<tr><td>{html.escape(str(row["label"]))}</td><td>{float(row["target_x"]):.3f}</td>'
        f'<td>{float(row["actual_x"]):.6f}</td><td>{float(row["median_ms"]):.6f}</td>'
        f'<td>{float(row["q1_ms"]):.6f}</td><td>{float(row["q3_ms"]):.6f}</td>'
        f'<td>{float(row["median_ms"]) / raw_fp64:.3f}×</td></tr>' for row in sweep
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DyadicNormal32 decoder strategies</title>
<style>:root{{--ink:#1d292f;--muted:#60727b;--line:#dbe3e7;--surface:#fff;--accent:#007f86;--soft:#e8f3f3}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}main{{max-width:1450px;margin:0 auto;padding:54px 30px 80px}}h1{{font-size:40px;line-height:1.08;margin:0 0 14px}}h2{{font-size:25px;margin:40px 0 14px}}.lead{{max-width:1050px;font-size:18px;line-height:1.55;color:var(--muted);margin:0 0 30px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:13px;margin:0 0 34px}}.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px}}.card b{{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px}}.figure{{background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 8px 26px rgba(36,58,70,.07)}}.figure img{{display:block;width:100%;height:auto}}.finding{{margin-top:18px;padding:18px 20px;background:var(--soft);border-left:5px solid var(--accent);border-radius:8px;font-size:17px;line-height:1.55}}.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);font-variant-numeric:tabular-nums}}th,td{{padding:11px 12px;text-align:right;border-bottom:1px solid var(--line)}}th:first-child,td:first-child{{text-align:left}}th{{background:#eaf0f2;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}}.method{{line-height:1.65;color:#33454e}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.93em}}.artifacts{{margin-top:40px;padding:20px 22px;background:#e7f0f5;border-left:5px solid #315f80;border-radius:8px;line-height:1.7}}a{{color:#245d87}}</style></head>
<body><main><h1>DyadicNormal32 decoder strategies</h1><p class="lead">One H200 allocation, one executable, identical 32-bit code arrays, and the same scalar x1 DOT geometry. The run compares the current integer-to-FP64 decoder with sign fusion and two decoders that construct the interpolation coordinate as FP64 bits.</p>
<div class="cards"><div class="card"><b>Input</b><code>N=2^26</code>, scalar x1 DOT</div><div class="card"><b>Strategies</b>4 Dyadic decoders</div><div class="card"><b>Fastest genuine N(0,1)</b>{html.escape(str(fastest["label"]))}<br>{fastest_time:.6f} ms</div><div class="card"><b>Current decoder</b>{current_time:.6f} ms</div><div class="card"><b>Raw FP64</b>{raw_fp64:.6f} ms</div></div>
<div class="figure"><img src="dyadic-normal32-strategies.svg" alt="Four DyadicNormal32 decoder strategies and three baselines"></div>
<div class="finding">On genuine N(0,1), {html.escape(str(fastest["label"]))} is {current_time / fastest_time:.3f}× as fast as the current decoder and takes {fastest_time / raw_fp64:.3f}× raw FP64 time. The artificial sweep shows whether that result survives changes in segment-table dispersion.</div>
<h2>Genuine N(0,1)</h2><div class="table-wrap"><table><thead><tr><th>Decoder</th><th>Median ms</th><th>Q1 ms</th><th>Q3 ms</th><th>/ current</th><th>/ raw FP64</th></tr></thead><tbody>{strategy_rows}</tbody></table></div>
<h2>Baselines</h2><div class="table-wrap"><table><thead><tr><th>Storage and arithmetic</th><th>Median ms</th><th>Q1 ms</th><th>Q3 ms</th><th>/ raw FP64</th></tr></thead><tbody>{baseline_rows}</tbody></table></div>
<h2>Dispersion sweep</h2><div class="table-wrap"><table><thead><tr><th>Decoder</th><th>Target X</th><th>Actual X</th><th>Median ms</th><th>Q1 ms</th><th>Q3 ms</th><th>/ raw FP64</th></tr></thead><tbody>{sweep_rows}</tbody></table></div>
<h2>What changed</h2><p class="method"><code>Current</code> decodes both signs separately and converts each integer payload to FP64. <code>Sign-fused</code> decodes positive magnitudes and applies the XOR of the two signs once before the multiply-add. The BitCast variants replace each integer-to-FP64 payload conversion with an exact FP64 coordinate assembled from the code bits. One stages transformed coefficient pairs in shared memory; the other reads them from CUDA constant memory. All variants preserve the same 32-bit format and reconstruction points to normal FP64 rounding tolerance.</p>
<p class="method">Forward batches use current, sign-fused, BitCast/shared, then BitCast/constant. Reverse batches invert that order. The baselines bracket the decoder sweep in reverse order. Each curve point and genuine-source point combines 25 samples from each direction.</p>
<div class="artifacts"><b>Artifacts.</b> <a href="{html.escape(samples_href)}">Raw timing samples</a> · <a href="{html.escape(metrics_href)}">Measured segment dispersion</a> · <a href="{html.escape(correctness_href)}">CPU/GPU checks</a> · <a href="{html.escape(coefficients_href)}">Coefficient table</a> · <a href="timing_summary.csv">Sweep summary</a> · <a href="genuine_summary.csv">Genuine N(0,1) summary</a> · <a href="baseline_summary.csv">Baseline summary</a> · <a href="../run_manifest.txt">Run manifest</a></div>
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
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
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
    validate_coefficients(arguments.coefficients)
    validate_contract(rows, metrics, correctness)
    sweep, baselines, genuine = build_summary(rows)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output_dir / "timing_summary.csv", sweep)
    write_csv(arguments.output_dir / "genuine_summary.csv", genuine)
    write_csv(arguments.output_dir / "baseline_summary.csv", list(baselines.values()))
    svg_path = arguments.output_dir / "dyadic-normal32-strategies.svg"
    svg_path.write_text(make_svg(sweep, baselines, genuine))
    made_png = maybe_make_png(svg_path, arguments.output_dir / "dyadic-normal32-strategies.png")
    samples_href = os.path.relpath(arguments.samples, arguments.output_dir)
    metrics_href = os.path.relpath(arguments.metrics, arguments.output_dir)
    correctness_href = os.path.relpath(arguments.correctness, arguments.output_dir)
    coefficients_href = os.path.relpath(arguments.coefficients, arguments.output_dir)
    (arguments.output_dir / "report.html").write_text(
        make_report(sweep, baselines, genuine, samples_href, metrics_href,
                    correctness_href, coefficients_href)
    )
    print(f"wrote {svg_path} and {arguments.output_dir / 'report.html'}")
    print(f"PNG generated: {'yes' if made_png else 'no converter available'}")
    for row in genuine:
        print(f"{row['label']} genuine N(0,1): {float(row['median_ms']):.9f} ms")
    for row in baselines.values():
        print(f"{row['label']} median: {float(row['median_ms']):.9f} ms")


if __name__ == "__main__":
    main()
