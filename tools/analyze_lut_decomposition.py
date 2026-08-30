#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


CANONICAL_N = 1 << 26
CANONICAL_WARMUP = 10
CANONICAL_SAMPLES = 50
TARGETS = tuple(index / 8 for index in range(9))


@dataclass(frozen=True)
class Variant:
    graph: str
    identifier: str
    label: str
    storage_bits: int
    arithmetic: str
    location: str
    component_bits: int
    components: int
    entries: int
    color: str


VARIANTS = (
    Variant("g1_u16_global_fp64", "u16_global_fp64", "16-bit full global LUT to FP64", 16, "fp64", "global", 16, 1, 65536, "#d95f4f"),
    Variant("g2_u8_shared_fp32", "u8_shared_fp32", "8-bit shared LUT to FP32", 8, "fp32", "shared", 8, 1, 256, "#3478a8"),
    Variant("g3_u8_shared_fp64", "u8_shared_fp64", "8-bit shared LUT to FP64", 8, "fp64", "shared", 8, 1, 256, "#5b8f3d"),
    Variant("g4_u16_split_fp32", "u16_2x8_shared_fp32", "2 x 8-bit shared LUTs", 16, "fp32", "shared", 8, 2, 256, "#7654a6"),
    Variant("g4_u16_split_fp32", "u16_4x4_shared_fp32", "4 x 4-bit shared LUTs", 16, "fp32", "shared", 4, 4, 16, "#d8902f"),
    Variant("g5_u32_split_fp64", "u32_2x16_global_fp64", "2 x 16-bit global LUTs", 32, "fp64", "global", 16, 2, 65536, "#d95f4f"),
    Variant("g5_u32_split_fp64", "u32_4x8_shared_fp64", "4 x 8-bit shared LUTs", 32, "fp64", "shared", 8, 4, 256, "#3478a8"),
    Variant("g5_u32_split_fp64", "u32_8x4_shared_fp64", "8 x 4-bit shared LUTs", 32, "fp64", "shared", 4, 8, 16, "#5b8f3d"),
)
VARIANT_BY_ID = {variant.identifier: variant for variant in VARIANTS}

GRAPH_TITLES = {
    "g1_u16_global_fp64": "16-bit full global LUT to FP64",
    "g2_u8_shared_fp32": "8-bit shared LUT to FP32",
    "g3_u8_shared_fp64": "8-bit shared LUT to FP64",
    "g4_u16_split_fp32": "16-bit split-table decoding to FP32",
    "g5_u32_split_fp64": "32-bit split-table decoding to FP64",
}
GRAPH_SUBTITLES = {
    "g1_u16_global_fp64": "One 65,536-entry global FP64 table",
    "g2_u8_shared_fp32": "One 256-entry shared FP32 table",
    "g3_u8_shared_fp64": "One 256-entry shared FP64 table",
    "g4_u16_split_fp32": "Two 8-bit lookups versus four 4-bit lookups; component results are summed",
    "g5_u32_split_fp64": "Global 16-bit tables versus shared 8-bit and 4-bit tables; component results are summed",
}
GRAPH_BASELINE = {
    "g1_u16_global_fp64": "raw_fp64",
    "g2_u8_shared_fp32": "raw_fp32",
    "g3_u8_shared_fp64": "raw_fp64",
    "g4_u16_split_fp32": "raw_fp32",
    "g5_u32_split_fp64": "raw_fp64",
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


def validate_raw_fp32(rows: list[dict[str, str]]) -> None:
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
        "warmup": str(CANONICAL_WARMUP),
    }
    if len(rows) != CANONICAL_SAMPLES:
        raise ValueError(f"raw FP32 needs {CANONICAL_SAMPLES} rows")
    for index, row in enumerate(rows, start=2):
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"raw FP32 row {index}: {field} mismatch")
        finite_positive(row, "kernel_ms", f"raw FP32 row {index}")
        if not math.isfinite(float(row["result"])):
            raise ValueError(f"raw FP32 row {index}: nonfinite result")
    if {int(row["sample"]) for row in rows} != set(range(CANONICAL_SAMPLES)):
        raise ValueError("raw FP32 has missing or duplicate samples")


def validate_contract(
    rows: list[dict[str, str]], metrics: list[dict[str, str]], raw_fp32: list[dict[str, str]]
) -> None:
    required = {
        "gpu", "mode", "graph_id", "variant", "kernel", "N", "blocks",
        "threads", "storage_bits", "arithmetic_type", "table_location",
        "component_bits", "components", "lut_entries_per_component",
        "total_lut_entries", "total_lut_bytes", "dynamic_shared_bytes",
        "target_x", "q", "mean_unique_indices", "actual_x",
        "mean_unique_sectors", "mean_shared_wavefronts", "warmup", "sample",
        "execution_order", "kernel_ms", "result",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"timing CSV missing columns: {sorted(missing)}")
    gpus = {row["gpu"] for row in rows}
    if len(gpus) != 1 or not next(iter(gpus)).startswith("NVIDIA H200"):
        raise ValueError(f"unexpected timing GPUs: {sorted(gpus)}")
    if {row["gpu"] for row in raw_fp32} != gpus:
        raise ValueError("raw FP32 and decomposition runs used different GPU models")

    raw64 = [row for row in rows if row["variant"] == "raw_fp64"]
    if len(raw64) != CANONICAL_SAMPLES:
        raise ValueError("raw FP64 baseline has the wrong sample count")
    for index, row in enumerate(raw64, start=1):
        expected = {
            "mode": "full", "graph_id": "raw_fp64", "kernel": "dot",
            "N": str(CANONICAL_N), "blocks": "512", "threads": "256",
            "storage_bits": "64", "arithmetic_type": "fp64",
            "table_location": "none", "warmup": str(CANONICAL_WARMUP),
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ValueError(f"raw FP64 row {index}: {field} mismatch")
        finite_positive(row, "kernel_ms", f"raw FP64 row {index}")
        if not math.isfinite(float(row["result"])):
            raise ValueError(f"raw FP64 row {index}: nonfinite result")
    if {int(row["sample"]) for row in raw64} != set(range(CANONICAL_SAMPLES)):
        raise ValueError("raw FP64 has missing or duplicate samples")

    measured = [row for row in rows if row["variant"] != "raw_fp64"]
    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row_index, row in enumerate(measured, start=2):
        variant = VARIANT_BY_ID.get(row["variant"])
        if variant is None:
            raise ValueError(f"timing row {row_index}: unknown variant")
        expected = {
            "mode": "full", "graph_id": variant.graph, "kernel": "dot",
            "N": str(CANONICAL_N), "blocks": "512", "threads": "256",
            "storage_bits": str(variant.storage_bits),
            "arithmetic_type": variant.arithmetic,
            "table_location": variant.location,
            "component_bits": str(variant.component_bits),
            "components": str(variant.components),
            "lut_entries_per_component": str(variant.entries),
            "warmup": str(CANONICAL_WARMUP),
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ValueError(
                    f"timing row {row_index}: {field}={row[field]!r}, expected {value!r}"
                )
        target = float(row["target_x"])
        if not any(math.isclose(target, expected_target, abs_tol=1e-15) for expected_target in TARGETS):
            raise ValueError(f"timing row {row_index}: unexpected target X")
        actual = float(row["actual_x"])
        if abs(actual - target) > 0.01:
            raise ValueError(f"timing row {row_index}: actual X misses target")
        if not 0 <= float(row["q"]) <= 1:
            raise ValueError(f"timing row {row_index}: invalid q")
        finite_positive(row, "kernel_ms", f"timing row {row_index}")
        if not math.isfinite(float(row["result"])):
            raise ValueError(f"timing row {row_index}: nonfinite result")
        if variant.location == "global":
            if not math.isfinite(float(row["mean_unique_sectors"])) or row["mean_shared_wavefronts"] != "nan":
                raise ValueError(f"timing row {row_index}: wrong global diagnostics")
        else:
            if not math.isfinite(float(row["mean_shared_wavefronts"])) or row["mean_unique_sectors"] != "nan":
                raise ValueError(f"timing row {row_index}: wrong shared diagnostics")
        groups[(variant.identifier, target)].append(row)

    expected_groups = {(variant.identifier, target) for variant in VARIANTS for target in TARGETS}
    if set(groups) != expected_groups:
        raise ValueError("timing CSV is missing variant/target groups")
    for key, group in groups.items():
        if len(group) != CANONICAL_SAMPLES:
            raise ValueError(f"{key}: wrong sample count")
        if {int(row["sample"]) for row in group} != set(range(CANONICAL_SAMPLES)):
            raise ValueError(f"{key}: missing or duplicate samples")
        if len({row["actual_x"] for row in group}) != 1:
            raise ValueError(f"{key}: X changed between samples")

    metric_required = {
        "graph_id", "variant", "target_x", "q", "component",
        "component_bits", "table_location", "entry_bytes", "N",
        "left_unique_indices", "right_unique_indices", "mean_unique_indices",
        "actual_x", "left_unique_sectors", "right_unique_sectors",
        "mean_unique_sectors", "left_shared_wavefronts",
        "right_shared_wavefronts", "mean_shared_wavefronts",
    }
    if metric_required - set(metrics[0]):
        raise ValueError("metrics CSV is missing required columns")
    metric_groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        variant = VARIANT_BY_ID.get(row["variant"])
        if variant is None:
            raise ValueError("metrics CSV contains an unknown variant")
        target = float(row["target_x"])
        actual = float(row["actual_x"])
        if abs(actual - target) > 0.01:
            raise ValueError(f"{variant.identifier} component misses target X")
        metric_groups[(variant.identifier, target)].append(row)
    if set(metric_groups) != expected_groups:
        raise ValueError("metrics CSV is missing variant/target groups")
    for key, group in metric_groups.items():
        variant = VARIANT_BY_ID[key[0]]
        if len(group) != variant.components:
            raise ValueError(f"{key}: wrong component metric count")
        if {int(row["component"]) for row in group} != set(range(variant.components)):
            raise ValueError(f"{key}: component indices are incomplete")
        component_x = [float(row["actual_x"]) for row in group]
        if max(component_x) - min(component_x) > 0.005:
            raise ValueError(f"{key}: component lookup dispersions diverge")
        timing_x = float(groups[key][0]["actual_x"])
        if not math.isclose(statistics.mean(component_x), timing_x, abs_tol=1e-12):
            raise ValueError(f"{key}: timing and component X disagree")


def summarize_samples(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    times = [float(row["kernel_ms"]) for row in rows]
    return {
        "samples": len(times),
        "median_ms": statistics.median(times),
        "q1_ms": percentile(times, 0.25),
        "q3_ms": percentile(times, 0.75),
        "minimum_ms": min(times),
        "maximum_ms": max(times),
    }


def build_summary(
    rows: list[dict[str, str]], raw_fp32: list[dict[str, str]]
) -> tuple[list[dict[str, object]], dict[str, dict[str, float | int | str]]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["variant"] != "raw_fp64":
            grouped[(row["variant"], float(row["target_x"]))].append(row)
    summary: list[dict[str, object]] = []
    for variant in VARIANTS:
        for target in TARGETS:
            group = grouped[(variant.identifier, target)]
            values = summarize_samples(group)
            summary.append(
                {
                    "graph_id": variant.graph,
                    "variant": variant.identifier,
                    "label": variant.label,
                    "target_x": target,
                    "actual_x": float(group[0]["actual_x"]),
                    **values,
                }
            )
    raw64 = summarize_samples([row for row in rows if row["variant"] == "raw_fp64"])
    raw32 = summarize_samples(raw_fp32)
    baselines = {
        "raw_fp32": {"label": "Raw FP32", **raw32},
        "raw_fp64": {"label": "Raw FP64", **raw64},
    }
    return summary, baselines


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_svg(
    graph_id: str,
    summary: list[dict[str, object]],
    baseline: dict[str, float | int | str],
) -> str:
    graph_rows = [row for row in summary if row["graph_id"] == graph_id]
    variants = [variant for variant in VARIANTS if variant.graph == graph_id]
    width, height = 1220, 660
    left, right, top, bottom = 92, 330, 76, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    lows = [float(row["q1_ms"]) for row in graph_rows] + [float(baseline["q1_ms"])]
    highs = [float(row["q3_ms"]) for row in graph_rows] + [float(baseline["q3_ms"])]
    y_min, y_max = min(lows), max(highs)
    padding = max((y_max - y_min) * 0.14, y_max * 0.012)
    y_min -= padding
    y_max += padding

    def px(value: float) -> float:
        return left + value * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(GRAPH_TITLES[graph_id])}</title>',
        f'<desc id="desc">DOT kernel time versus normalized lookup dispersion, with {html.escape(str(baseline["label"]))} baseline.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;fill:#263238}.title{font-size:24px;font-weight:700}.subtitle{font-size:14px;fill:#607078}.axis{stroke:#263238;stroke-width:1.5}.grid{stroke:#dfe5e8;stroke-width:1}.tick{font-size:12px;fill:#607078}.axis-label{font-size:15px;font-weight:600}.curve{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.iqr{stroke-width:1.5;opacity:.75}.direct{font-size:13px;font-weight:700}.note{font-size:12px;fill:#607078}</style>',
        f'<text class="title" x="{left}" y="32">{html.escape(GRAPH_TITLES[graph_id])}</text>',
        f'<text class="subtitle" x="{left}" y="55">{html.escape(GRAPH_SUBTITLES[graph_id])}; DOT, N=2²⁶, scalar x1</text>',
    ]
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = py(value)
        pieces.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        pieces.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end">{value:.3f}</text>')
    for index in range(6):
        value = index / 5
        x = px(value)
        pieces.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        pieces.append(f'<text class="tick" x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle">{value:.1f}</text>')
    pieces.extend([
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        f'<text class="axis-label" x="{left + plot_width / 2}" y="{height - 30}" text-anchor="middle">Normalized lookup dispersion</text>',
        f'<text class="axis-label" transform="translate(25 {top + plot_height / 2}) rotate(-90)" text-anchor="middle">DOT kernel time (ms)</text>',
    ])

    baseline_y = py(float(baseline["median_ms"]))
    baseline_q1 = py(float(baseline["q1_ms"]))
    baseline_q3 = py(float(baseline["q3_ms"]))
    pieces.extend([
        f'<rect x="{left}" y="{min(baseline_q1, baseline_q3):.2f}" width="{plot_width}" height="{max(abs(baseline_q3 - baseline_q1), 1.0):.2f}" fill="#111820" opacity="0.08"/>',
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{left + plot_width}" y2="{baseline_y:.2f}" stroke="#111820" stroke-width="2.2" stroke-dasharray="8 6"/>',
    ])

    endpoints: list[tuple[str, str, float, str]] = [
        (str(baseline["label"]), "#111820", baseline_y, "baseline")
    ]
    for variant in variants:
        series = sorted(
            [row for row in graph_rows if row["variant"] == variant.identifier],
            key=lambda row: float(row["actual_x"]),
        )
        points = " ".join(
            f'{px(float(row["actual_x"])):.2f},{py(float(row["median_ms"])):.2f}'
            for row in series
        )
        pieces.append(f'<polyline class="curve" stroke="{variant.color}" points="{points}"/>')
        for row in series:
            x = px(float(row["actual_x"]))
            y = py(float(row["median_ms"]))
            q1 = py(float(row["q1_ms"]))
            q3 = py(float(row["q3_ms"]))
            pieces.extend([
                f'<line class="iqr" stroke="{variant.color}" x1="{x:.2f}" y1="{q1:.2f}" x2="{x:.2f}" y2="{q3:.2f}"/>',
                f'<line class="iqr" stroke="{variant.color}" x1="{x - 4:.2f}" y1="{q1:.2f}" x2="{x + 4:.2f}" y2="{q1:.2f}"/>',
                f'<line class="iqr" stroke="{variant.color}" x1="{x - 4:.2f}" y1="{q3:.2f}" x2="{x + 4:.2f}" y2="{q3:.2f}"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{variant.color}"/>',
            ])
        endpoints.append((variant.label, variant.color, py(float(series[-1]["median_ms"])), variant.identifier))

    ordered = sorted(endpoints, key=lambda item: item[2])
    gap = 32.0
    labels: list[tuple[str, str, float, float, str]] = []
    previous = top - gap
    for label, color, endpoint_y, identifier in ordered:
        label_y = max(endpoint_y, previous + gap)
        labels.append((label, color, endpoint_y, label_y, identifier))
        previous = label_y
    overflow = max(label_y for _, _, _, label_y, _ in labels) - (top + plot_height - 10)
    if overflow > 0:
        labels = [(label, color, endpoint, label_y - overflow, identifier) for label, color, endpoint, label_y, identifier in labels]
    label_x = left + plot_width + 44
    for label, color, endpoint_y, label_y, identifier in labels:
        dash = "5 5" if identifier == "baseline" else "3 4"
        pieces.append(
            f'<path d="M {left + plot_width:.2f} {endpoint_y:.2f} L {label_x - 12:.2f} {label_y:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}" fill="none"/>'
        )
        pieces.append(f'<text class="direct" x="{label_x}" y="{label_y + 5:.2f}" fill="{color}">{html.escape(label)}</text>')
    pieces.append(f'<text class="note" x="{left}" y="{height - 8}">Curves show medians of 50 launches; bars show IQR. The black dotted baseline has no lookup dispersion.</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def ratio_at_endpoints(summary: list[dict[str, object]], variant: str) -> float:
    rows = sorted(
        [row for row in summary if row["variant"] == variant],
        key=lambda row: float(row["actual_x"]),
    )
    return float(rows[-1]["median_ms"]) / float(rows[0]["median_ms"])


def make_report(
    summary: list[dict[str, object]],
    baselines: dict[str, dict[str, float | int | str]],
    sample_href: str,
    metric_href: str,
    raw_fp32_href: str,
) -> str:
    sections = []
    for index, graph_id in enumerate(GRAPH_TITLES, start=1):
        filename = f"graph-{index}-{graph_id}.svg"
        variants = [variant for variant in VARIANTS if variant.graph == graph_id]
        ratios = ", ".join(
            f"{html.escape(variant.label)}: {ratio_at_endpoints(summary, variant.identifier):.2f}×"
            for variant in variants
        )
        sections.append(
            f'<section><h2>{index}. {html.escape(GRAPH_TITLES[graph_id])}</h2>'
            f'<div class="figure"><img src="{filename}" alt="{html.escape(GRAPH_TITLES[graph_id])}"></div>'
            f'<p class="shape"><b>X=1 / X=0 median time:</b> {ratios}</p></section>'
        )
    fp32 = baselines["raw_fp32"]
    fp64 = baselines["raw_fp64"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LUT decomposition performance</title>
<style>
:root{{--ink:#1c262b;--muted:#607078;--line:#dce3e6;--surface:#fff;--accent:#315f80}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}main{{max-width:1280px;margin:0 auto;padding:54px 30px 80px}}h1{{font-size:40px;line-height:1.08;margin:0 0 14px}}.lead{{max-width:980px;font-size:18px;line-height:1.55;color:var(--muted);margin:0 0 30px}}.method{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:13px;margin:0 0 40px}}.method div{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px}}.method b{{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px}}section{{margin-top:42px}}h2{{font-size:25px;margin:0 0 14px}}.figure{{background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 8px 26px rgba(36,58,70,.07)}}.figure img{{display:block;width:100%;height:auto}}.shape{{margin:12px 4px 0;color:var(--muted)}}.artifacts{{margin-top:44px;padding:20px 22px;background:#e7f0f5;border-left:5px solid var(--accent);border-radius:8px;line-height:1.7}}a{{color:#245d87}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.93em}}</style></head>
<body><main><h1>Lookup-table decomposition performance</h1>
<p class="lead">Five DOT experiments isolate table width, table placement, arithmetic type, and the number of component lookups. Every component table receives the same target normalized lookup dispersion. Shared tables are staged once per block inside the timed first-stage kernel.</p>
<div class="method"><div><b>Input</b><code>N=2^26</code>, scalar x1, independent left/right fields</div><div><b>Geometry</b>512 × 256 first stage plus the same final reduction</div><div><b>Timing</b>10 warmups and 50 measured launches per point</div><div><b>Baselines</b>Raw FP32 {float(fp32['median_ms']):.6f} ms; raw FP64 {float(fp64['median_ms']):.6f} ms</div></div>
{''.join(sections)}
<div class="artifacts"><b>Artifacts.</b> <a href="{html.escape(sample_href)}">Raw timing samples</a> · <a href="{html.escape(metric_href)}">Access diagnostics</a> · <a href="{html.escape(raw_fp32_href)}">Raw FP32 source samples</a> · <a href="timing_summary.csv">Timing summary</a> · <a href="baseline_summary.csv">Baseline summary</a></div>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--raw-fp32-samples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()

    rows = read_rows(arguments.samples)
    metrics = read_rows(arguments.metrics)
    raw_fp32 = read_rows(arguments.raw_fp32_samples)
    validate_raw_fp32(raw_fp32)
    validate_contract(rows, metrics, raw_fp32)
    summary, baselines = build_summary(rows, raw_fp32)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(arguments.output_dir / "timing_summary.csv", summary)
    baseline_rows = [
        {"baseline": identifier, **values} for identifier, values in baselines.items()
    ]
    write_csv(arguments.output_dir / "baseline_summary.csv", baseline_rows)
    for index, graph_id in enumerate(GRAPH_TITLES, start=1):
        baseline = baselines[GRAPH_BASELINE[graph_id]]
        (arguments.output_dir / f"graph-{index}-{graph_id}.svg").write_text(
            make_svg(graph_id, summary, baseline)
        )
    sample_href = os.path.relpath(arguments.samples, arguments.output_dir)
    metric_href = os.path.relpath(arguments.metrics, arguments.output_dir)
    raw_fp32_href = os.path.relpath(arguments.raw_fp32_samples, arguments.output_dir)
    (arguments.output_dir / "report.html").write_text(
        make_report(summary, baselines, sample_href, metric_href, raw_fp32_href)
    )
    print(f"wrote five graphs and {arguments.output_dir / 'report.html'}")
    print(
        "raw baselines: "
        f"FP32 {float(baselines['raw_fp32']['median_ms']):.9f} ms, "
        f"FP64 {float(baselines['raw_fp64']['median_ms']):.9f} ms"
    )


if __name__ == "__main__":
    main()
