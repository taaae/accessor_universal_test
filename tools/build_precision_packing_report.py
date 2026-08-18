#!/usr/bin/env python3
"""Add the precision-packing bottleneck experiment to the unified report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import build_storage_performance_report as base
from typing import Iterable


PAGE_NAME = "packing-bottlenecks.html"
PAGE_LABEL = "Packing bottlenecks"

STORAGE_LABELS = {
    "fp4_e2m1": "E2M1 aka FP4",
    "fp8_e4m3": "E4M3 aka FP8",
    "fp8_e5m2": "E5M2 aka FP8",
    "fp16_e5m10": "E5M10 aka FP16",
    "bf16_e8m7": "E8M7 aka BF16",
    "fp32_e8m23": "E8M23 aka FP32",
    "fp64_e11m52": "E11M52 aka FP64",
}

STORAGE_ORDER = tuple(STORAGE_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="experiment 018 run; default: newest complete run",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/018_precision_packing_bottlenecks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/report"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def newest_complete_run(root: Path) -> Path:
    required = (
        "timing_summary.csv",
        "roof_metrics.csv",
        "component_summary.csv",
        "profile_operations.csv",
    )
    candidates = sorted(
        path
        for path in root.glob("run_*")
        if all((path / filename).is_file() for filename in required)
    )
    if not candidates:
        raise SystemExit(f"no complete precision-packing run below {root}")
    return candidates[-1]


def geometric_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not finite:
        return math.nan
    return math.exp(statistics.fmean(math.log(value) for value in finite))


def grouped_geometric_rows(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
    numeric_fields: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, current in sorted(groups.items()):
        result: dict[str, object] = dict(zip(keys, key))
        for field in numeric_fields:
            result[field] = geometric_mean(float(row[field]) for row in current)
        output.append(result)
    return output


def aggregate_run(run_dir: Path) -> dict[str, object]:
    timing_rows = read_csv(run_dir / "timing_summary.csv")
    roof_rows = read_csv(run_dir / "roof_metrics.csv")
    component_rows = read_csv(run_dir / "component_summary.csv")
    profile_rows = read_csv(run_dir / "profile_operations.csv")

    expected_storage = set(STORAGE_LABELS)
    if {row["storage"] for row in timing_rows} != expected_storage:
        raise SystemExit("precision-packing timing data has unexpected storage coverage")
    if {row["kernel"] for row in timing_rows} != {"dot", "gemv"}:
        raise SystemExit("precision-packing timing data is missing DOT or GEMV")

    maximum_n = {
        kernel: max(int(row["n"]) for row in timing_rows if row["kernel"] == kernel)
        for kernel in ("dot", "gemv")
    }
    timing_large = [
        row for row in timing_rows if int(row["n"]) == maximum_n[row["kernel"]]
    ]
    timing = grouped_geometric_rows(
        timing_large,
        ("kernel", "storage", "arithmetic", "family", "lanes"),
        ("median_time_ms", "useful_flops", "logical_storage_bytes"),
    )
    baseline = {
        (str(row["kernel"]), str(row["storage"]), str(row["arithmetic"])): float(
            row["median_time_ms"]
        )
        for row in timing
        if row["family"] == "scalar_single" and row["lanes"] == "1"
    }
    for row in timing:
        key = (str(row["kernel"]), str(row["storage"]), str(row["arithmetic"]))
        time_ms = float(row["median_time_ms"])
        row["lanes"] = int(str(row["lanes"]))
        row["storage_bits"] = int(
            next(
                source["storage_bits"]
                for source in timing_large
                if source["storage"] == row["storage"]
            )
        )
        row["speedup"] = baseline[key] / time_ms
        row["useful_gflops"] = float(row["useful_flops"]) / (time_ms * 1.0e6)
        row["arithmetic_intensity"] = float(row["useful_flops"]) / float(
            row["logical_storage_bytes"]
        )

    roof_large = [
        row for row in roof_rows if int(row["n"]) == maximum_n[row["kernel"]]
    ]
    roof = grouped_geometric_rows(
        roof_large,
        ("kernel", "storage", "arithmetic", "family", "lanes"),
        (
            "sustainable_hbm_gb_per_s",
            "empirical_arithmetic_peak_gflop_per_s",
            "roof_efficiency",
        ),
    )
    roof_by_key = {
        (
            str(row["kernel"]),
            str(row["storage"]),
            str(row["arithmetic"]),
            str(row["family"]),
            int(str(row["lanes"])),
        ): row
        for row in roof
    }
    for row in timing:
        key = (
            str(row["kernel"]),
            str(row["storage"]),
            str(row["arithmetic"]),
            str(row["family"]),
            int(row["lanes"]),
        )
        current = roof_by_key.get(key)
        if current:
            row["sustainable_hbm_gb_per_s"] = current["sustainable_hbm_gb_per_s"]
            row["arithmetic_peak_gflop_per_s"] = current[
                "empirical_arithmetic_peak_gflop_per_s"
            ]
            row["roof_efficiency"] = current["roof_efficiency"]

    components = grouped_geometric_rows(
        component_rows,
        ("component", "storage", "arithmetic", "family", "lanes"),
        (
            "logical_values",
            "median_time_ms",
            "logical_storage_bytes",
            "modeled_conversions",
        ),
    )
    for row in components:
        row["lanes"] = int(str(row["lanes"]))
        seconds = float(row["median_time_ms"]) * 1.0e-3
        row["gvalues_per_s"] = float(row["logical_values"]) / seconds / 1.0e9
        row["effective_gb_per_s"] = (
            float(row["logical_storage_bytes"]) / seconds / 1.0e9
        )
        conversions = float(row["modeled_conversions"])
        row["gconversions_per_s"] = conversions / seconds / 1.0e9

    profile = []
    profile_numeric = (
        "instructions_per_logical_value",
        "dram_percent_peak",
        "sm_percent_peak",
        "registers_per_thread",
        "long_scoreboard_stall_per_issue",
        "math_pipe_throttle_stall_per_issue",
        "mio_throttle_stall_per_issue",
    )
    for source in profile_rows:
        row: dict[str, object] = {
            key: source[key]
            for key in ("kernel", "storage", "arithmetic", "family")
        }
        row["lanes"] = int(source["lanes"])
        for field in profile_numeric:
            row[field] = float(source[field])
        profile.append(row)

    return {
        "run": run_dir.name,
        "maximum_n": maximum_n,
        "storage_order": STORAGE_ORDER,
        "storage_labels": STORAGE_LABELS,
        "timing": timing,
        "components": components,
        "profile": profile,
    }


PAGE_STYLE = r"""
.packing-controls { align-items: end; display: flex; flex-wrap: wrap; gap: 14px; margin: 12px 0 26px; }
.packing-controls label { color: var(--muted); display: grid; gap: 4px; font-size: 0.88rem; }
.packing-controls select { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--fg); font: inherit; min-width: 220px; padding: 7px 30px 7px 9px; }
.packing-chart-grid { display: grid; gap: 18px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 14px; }
.packing-chart-grid.resources { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.packing-chart { min-width: 0; }
.packing-chart svg { background: var(--surface); border: 1px solid var(--border); display: block; height: auto; width: 100%; }
.packing-legend { align-items: center; display: flex; flex-wrap: wrap; gap: 7px 16px; margin: 8px 0 2px; min-height: 24px; }
.packing-legend-item { align-items: center; display: inline-flex; gap: 6px; white-space: nowrap; }
.packing-legend-line { border-top: 3px solid var(--legend-color); display: inline-block; width: 22px; }
.packing-legend-marker { background: var(--surface); border: 2px solid var(--legend-color); display: inline-block; height: 10px; transform: rotate(var(--marker-rotation, 0deg)); width: 10px; }
.packing-legend-marker.circle { border-radius: 50%; }
@media (max-width: 900px) { .packing-chart-grid.resources { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 680px) { .packing-chart-grid, .packing-chart-grid.resources { grid-template-columns: 1fr; } }
"""


PAGE_BODY = r"""
<section class="text-section">
<div class="packing-controls"><label for="packing-storage">Storage format<select id="packing-storage"></select></label></div>
</section>

<section class="graph-section">
<h2>Complete-kernel speedup</h2>
<div id="speed-legend" class="packing-legend" aria-label="Speed chart legend"></div>
<div id="speed-grid" class="packing-chart-grid"></div>
</section>

<section class="graph-section">
<h2>Roofline trajectory</h2>
<div id="roof-grid" class="packing-chart-grid"></div>
</section>

<section class="graph-section">
<h2>Bottleneck transition</h2>
<div id="resource-legend" class="packing-legend" aria-label="Resource chart legend"></div>
<div id="resource-grid" class="packing-chart-grid resources"></div>
</section>

<section class="graph-section">
<h2>Attribution ratios</h2>
<div id="attribution-legend" class="packing-legend" aria-label="Attribution chart legend"></div>
<div id="attribution-grid" class="packing-chart-grid"></div>
</section>

<section class="graph-section">
<h2>Isolated load and conversion throughput</h2>
<div id="mechanism-legend" class="packing-legend" aria-label="Microbenchmark chart legend"></div>
<div id="mechanism-grid" class="packing-chart-grid"></div>
</section>
"""


PAGE_SCRIPT = r"""
const packingData = JSON.parse(document.getElementById('packing-data').textContent);
const storageSelect = document.getElementById('packing-storage');
const NS = 'http://www.w3.org/2000/svg';
const familyColors = { scalar_unrolled: '#6c757d', vector_packet: '#0072B2', packed_arithmetic: '#D55E00' };
const familyLabels = { scalar_unrolled: 'Scalar unrolled', vector_packet: 'Vector packet', packed_arithmetic: 'Native packed arithmetic' };
const arithmeticColors = { fp16: '#0072B2', bf16: '#009E73', fp32: '#E69F00', fp64: '#D55E00' };
const arithmeticLabels = { fp16: 'FP16', bf16: 'BF16', fp32: 'FP32', fp64: 'FP64' };
const arithmeticMarkers = { fp16: 'circle', bf16: 'triangle', fp32: 'square', fp64: 'diamond' };
const widths = [1, 2, 4, 8];

for (const storage of packingData.storage_order) {
  const option = document.createElement('option');
  option.value = storage;
  option.textContent = packingData.storage_labels[storage];
  option.selected = storage === 'fp16_e5m10';
  storageSelect.append(option);
}

function svgElement(name, attributes = {}, text = '') {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text) node.textContent = text;
  return node;
}

function markerPath(kind, x, y, size) {
  if (kind === 'circle') return svgElement('circle', { cx: x, cy: y, r: size });
  if (kind === 'triangle') return svgElement('path', { d: `M ${x} ${y-size} L ${x+size} ${y+size} L ${x-size} ${y+size} Z` });
  if (kind === 'diamond') return svgElement('path', { d: `M ${x} ${y-size} L ${x+size} ${y} L ${x} ${y+size} L ${x-size} ${y} Z` });
  return svgElement('rect', { x: x-size, y: y-size, width: 2*size, height: 2*size });
}

function niceLinearTicks(minimum, maximum, count = 5) {
  if (!(maximum > minimum)) return [minimum];
  const rough = (maximum - minimum) / Math.max(1, count - 1);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const residual = rough / magnitude;
  const step = (residual >= 5 ? 5 : residual >= 2 ? 2 : 1) * magnitude;
  const start = Math.ceil(minimum / step) * step;
  const ticks = [];
  for (let value = start; value <= maximum + step * 0.25; value += step) ticks.push(value);
  return ticks;
}

function logTicks(minimum, maximum) {
  const ticks = [];
  const low = Math.floor(Math.log10(minimum));
  const high = Math.ceil(Math.log10(maximum));
  for (let exponent = low; exponent <= high; exponent += 1) {
    for (const multiplier of [1, 2, 5]) {
      const value = multiplier * 10 ** exponent;
      if (value >= minimum && value <= maximum) ticks.push(value);
    }
  }
  return ticks;
}

function formatTick(value) {
  if (Math.abs(value) >= 10000) return `${(value / 1000).toFixed(value >= 100000 ? 0 : 1)}k`;
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (Math.abs(value) >= 10) return value.toFixed(0);
  if (Math.abs(value) >= 1) return value.toFixed(1).replace(/\.0$/, '');
  return value.toPrecision(2);
}

function drawChart(target, config) {
  target.replaceChildren();
  const width = 540;
  const height = 345;
  const margin = { top: 40, right: 32, bottom: 54, left: 68 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const values = config.series.flatMap(series => series.points).filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
  const xValues = values.map(point => point.x);
  const yValues = values.map(point => point.y);
  let xMin = config.xDomain?.[0] ?? Math.min(...xValues);
  let xMax = config.xDomain?.[1] ?? Math.max(...xValues);
  let yMin = config.yDomain?.[0] ?? Math.min(0, ...yValues);
  let yMax = config.yDomain?.[1] ?? Math.max(...yValues) * 1.08;
  if (config.reference !== undefined) {
    yMin = Math.min(yMin, config.reference);
    yMax = Math.max(yMax, config.reference);
  }
  if (config.yScale === 'log') {
    yMin = config.yDomain?.[0] ?? Math.min(...yValues.filter(value => value > 0)) * 0.8;
    yMax = config.yDomain?.[1] ?? Math.max(...yValues) * 1.25;
  }
  if (xMin === xMax) { xMin *= 0.8; xMax *= 1.2; }
  if (yMin === yMax) { yMin *= 0.8; yMax *= 1.2; }
  const mapX = value => margin.left + (config.xScale === 'log'
    ? (Math.log10(value) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))
    : (value - xMin) / (xMax - xMin)) * innerWidth;
  const mapY = value => margin.top + (1 - (config.yScale === 'log'
    ? (Math.log10(value) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))
    : (value - yMin) / (yMax - yMin))) * innerHeight;
  const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': config.accessible });
  svg.append(svgElement('title', {}, config.title));
  svg.append(svgElement('desc', {}, config.accessible));
  svg.append(svgElement('text', { x: width / 2, y: 24, 'text-anchor': 'middle', fill: 'var(--fg)', 'font-size': 15, 'font-weight': 500 }, config.title));
  const plot = svgElement('g');
  const xTicks = config.xTicks ?? (config.xScale === 'log' ? logTicks(xMin, xMax) : niceLinearTicks(xMin, xMax));
  const yTicks = config.yTicks ?? (config.yScale === 'log' ? logTicks(yMin, yMax) : niceLinearTicks(yMin, yMax));
  for (const tick of yTicks) {
    const y = mapY(tick);
    plot.append(svgElement('line', { x1: margin.left, x2: width-margin.right, y1: y, y2: y, stroke: 'var(--border)', 'stroke-width': 1 }));
    plot.append(svgElement('text', { x: margin.left-8, y: y+4, 'text-anchor': 'end', fill: 'var(--muted)', 'font-size': 11 }, config.yFormat ? config.yFormat(tick) : formatTick(tick)));
  }
  for (const tick of xTicks) {
    if (tick < xMin || tick > xMax) continue;
    const x = mapX(tick);
    plot.append(svgElement('line', { x1: x, x2: x, y1: margin.top, y2: height-margin.bottom, stroke: 'var(--border)', 'stroke-width': 1 }));
    plot.append(svgElement('text', { x, y: height-margin.bottom+19, 'text-anchor': 'middle', fill: 'var(--muted)', 'font-size': 11 }, config.xFormat ? config.xFormat(tick) : formatTick(tick)));
  }
  plot.append(svgElement('line', { x1: margin.left, x2: width-margin.right, y1: height-margin.bottom, y2: height-margin.bottom, stroke: 'var(--muted)' }));
  plot.append(svgElement('line', { x1: margin.left, x2: margin.left, y1: margin.top, y2: height-margin.bottom, stroke: 'var(--muted)' }));
  if (config.reference !== undefined) {
    const y = mapY(config.reference);
    plot.append(svgElement('line', { x1: margin.left, x2: width-margin.right, y1: y, y2: y, stroke: '#6c757d', 'stroke-width': 1.4, 'stroke-dasharray': '6 5' }));
  }
  for (const series of config.series) {
    const points = [...series.points].filter(point => Number.isFinite(point.x) && Number.isFinite(point.y)).sort((a, b) => a.x-b.x);
    if (!points.length) continue;
    if (points.length > 1) {
      plot.append(svgElement('path', {
        d: points.map((point, index) => `${index ? 'L' : 'M'} ${mapX(point.x)} ${mapY(point.y)}`).join(' '),
        fill: 'none', stroke: series.color, 'stroke-width': series.width ?? 2.2,
        'stroke-dasharray': series.dash ?? '', opacity: series.opacity ?? 1,
      }));
    }
    if (series.showMarkers !== false) {
      for (const point of points) {
        const marker = markerPath(series.marker ?? 'circle', mapX(point.x), mapY(point.y), 4.4);
        marker.setAttribute('fill', 'var(--surface)');
        marker.setAttribute('stroke', series.color);
        marker.setAttribute('stroke-width', '2.2');
        marker.append(svgElement('title', {}, point.tooltip ?? `${series.label}: ${point.y}`));
        plot.append(marker);
        if (point.label) {
          plot.append(svgElement('text', {
            x: mapX(point.x) + (point.labelDx ?? 8),
            y: mapY(point.y) + (point.labelDy ?? -7),
            'text-anchor': point.labelAnchor ?? 'start',
            fill: 'var(--fg)', 'font-size': 11, 'font-weight': 500,
          }, point.label));
        }
      }
    }
    if (series.lineLabel && points.length) {
      const point = points[points.length - 1];
      plot.append(svgElement('text', {
        x: mapX(point.x) - 6, y: mapY(point.y) + (series.lineLabelDy ?? -7), 'text-anchor': 'end',
        fill: 'var(--fg)', 'font-size': 11, 'font-weight': 500,
      }, series.lineLabel));
    }
  }
  svg.append(plot);
  svg.append(svgElement('text', { x: margin.left+innerWidth/2, y: height-10, 'text-anchor': 'middle', fill: 'var(--fg)', 'font-size': 12 }, config.xLabel));
  const yLabel = svgElement('text', { x: 16, y: margin.top+innerHeight/2, 'text-anchor': 'middle', fill: 'var(--fg)', 'font-size': 12, transform: `rotate(-90 16 ${margin.top+innerHeight/2})` }, config.yLabel);
  svg.append(yLabel);
  target.append(svg);
}

function setLegend(targetId, entries) {
  const target = document.getElementById(targetId);
  target.replaceChildren();
  const seen = new Set();
  for (const entry of entries) {
    const key = `${entry.label}|${entry.color}|${entry.marker ?? ''}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const item = document.createElement('span');
    item.className = 'packing-legend-item';
    const sample = document.createElement('span');
    if (entry.marker) {
      sample.className = `packing-legend-marker ${entry.marker}`;
      sample.style.setProperty('--legend-color', entry.color);
      if (entry.marker === 'diamond') sample.style.setProperty('--marker-rotation', '45deg');
    } else {
      sample.className = 'packing-legend-line';
      sample.style.setProperty('--legend-color', entry.color);
    }
    const label = document.createElement('span');
    label.textContent = entry.label;
    item.append(sample, label);
    target.append(item);
  }
}

function panelTarget(grid, id) {
  const target = document.createElement('div');
  target.className = 'packing-chart';
  target.id = id;
  grid.append(target);
  return target;
}

function arithmeticModes(storage) {
  return [...new Set(packingData.timing.filter(row => row.storage === storage).map(row => row.arithmetic))]
    .sort((a, b) => ['fp16', 'bf16', 'fp32', 'fp64'].indexOf(a) - ['fp16', 'bf16', 'fp32', 'fp64'].indexOf(b));
}

function baselineRow(storage, arithmetic, kernel) {
  return packingData.timing.find(row => row.storage === storage && row.arithmetic === arithmetic && row.kernel === kernel && row.family === 'scalar_single');
}

function familySeries(storage, arithmetic, kernel, family) {
  const base = baselineRow(storage, arithmetic, kernel);
  const points = base ? [{ ...base, lanes: 1 }] : [];
  points.push(...packingData.timing.filter(row => row.storage === storage && row.arithmetic === arithmetic && row.kernel === kernel && row.family === family));
  return points.sort((a, b) => a.lanes-b.lanes);
}

function renderSpeed(storage) {
  const grid = document.getElementById('speed-grid');
  grid.replaceChildren();
  const modes = arithmeticModes(storage);
  const legend = [];
  for (const kernel of ['dot', 'gemv']) {
    for (const arithmetic of modes) {
      const series = [];
      for (const family of ['scalar_unrolled', 'vector_packet', 'packed_arithmetic']) {
        const rows = familySeries(storage, arithmetic, kernel, family);
        if (rows.length < 2) continue;
        series.push({
          label: familyLabels[family], color: familyColors[family], marker: 'circle',
          points: rows.map(row => ({ x: row.lanes, y: row.speedup, tooltip: `${packingData.storage_labels[storage]}→${arithmeticLabels[arithmetic]} · ${familyLabels[family]} ×${row.lanes}: ${row.speedup.toFixed(2)}×, ${row.median_time_ms.toFixed(4)} ms` })),
        });
        legend.push({ label: familyLabels[family], color: familyColors[family] });
      }
      if (!series.length) continue;
      const pairLabel = `${packingData.storage_labels[storage]}→${arithmeticLabels[arithmetic]}`;
      drawChart(panelTarget(grid, `speed-${kernel}-${arithmetic}`), {
        title: `${kernel.toUpperCase()} · ${pairLabel}`,
        accessible: `${kernel.toUpperCase()} ${pairLabel} complete-kernel speedup by width and implementation family`,
        series, xDomain: [1, 8], xTicks: widths, xFormat: value => `×${value}`,
        yDomain: [0, Math.max(1.1, ...series.flatMap(item => item.points.map(point => point.y))) * 1.08],
        xLabel: 'Values per thread', yLabel: 'Speedup over x1', reference: 1,
      });
    }
  }
  setLegend('speed-legend', legend);
}

function renderRoofline(storage) {
  const grid = document.getElementById('roof-grid');
  grid.replaceChildren();
  const modes = arithmeticModes(storage);
  for (const kernel of ['dot', 'gemv']) {
    for (const arithmetic of modes) {
      const rows = familySeries(storage, arithmetic, kernel, 'vector_packet').filter(row => row.sustainable_hbm_gb_per_s);
      if (!rows.length) continue;
      const bandwidth = rows[0].sustainable_hbm_gb_per_s;
      const peak = rows[0].arithmetic_peak_gflop_per_s;
      const series = [
        {
          label: 'HBM roof', lineLabel: 'HBM roof', lineLabelDy: 24, color: '#6c757d', dash: '7 5', width: 1.5,
          showMarkers: false, points: [{x: 0.08, y: bandwidth*0.08}, {x: 5, y: bandwidth*5}],
        },
        {
          label: `${arithmeticLabels[arithmetic]} roof`, lineLabel: `${arithmeticLabels[arithmetic]} roof`,
          color: arithmeticColors[arithmetic], dash: '3 5', width: 1.2, opacity: 0.75,
          showMarkers: false, points: [{x: 0.08, y: peak}, {x: 5, y: peak}],
        },
        {
          label: 'Measured', color: arithmeticColors[arithmetic], marker: 'circle',
          points: rows.map(row => {
            const offsets = {
              1: { labelDx: 8, labelDy: 12, labelAnchor: 'start' },
              2: { labelDx: -8, labelDy: 4, labelAnchor: 'end' },
              4: { labelDx: 8, labelDy: 4, labelAnchor: 'start' },
              8: { labelDx: 8, labelDy: -8, labelAnchor: 'start' },
            };
            return {
              x: row.arithmetic_intensity, y: row.useful_gflops, label: `×${row.lanes}`,
              ...offsets[row.lanes],
              tooltip: `${packingData.storage_labels[storage]}→${arithmeticLabels[arithmetic]} ×${row.lanes}: ${row.useful_gflops.toFixed(0)} GFLOP/s, ${(100*row.roof_efficiency).toFixed(1)}% of roof`,
            };
          }),
        },
      ];
      const pairLabel = `${packingData.storage_labels[storage]}→${arithmeticLabels[arithmetic]}`;
      drawChart(panelTarget(grid, `roof-${kernel}-${arithmetic}`), {
        title: `${kernel.toUpperCase()} · ${pairLabel}`,
        accessible: `${kernel.toUpperCase()} ${pairLabel} roofline trajectory from scalar x1 to vector packet x8`, series,
        xScale: 'log', yScale: 'log', xDomain: [0.08, 5], yDomain: [100, 100000],
        xLabel: 'Useful FLOP / encoded byte', yLabel: 'Useful GFLOP/s',
      });
    }
  }
}

function renderResources(storage) {
  const grid = document.getElementById('resource-grid');
  grid.replaceChildren();
  const modes = arithmeticModes(storage);
  const metrics = [
    ['instructions_per_logical_value', 'Instructions / value', null],
    ['dram_percent_peak', 'DRAM utilization', [0, 100]],
    ['sm_percent_peak', 'SM utilization', [0, 100]],
  ];
  for (const kernel of ['dot', 'gemv']) {
    for (const [field, title, domain] of metrics) {
      const series = modes.map(arithmetic => {
        const rows = packingData.profile.filter(row => row.storage === storage && row.arithmetic === arithmetic && row.kernel === kernel && (row.family === 'scalar_single' || row.family === 'vector_packet')).sort((a, b) => a.lanes-b.lanes);
        return {
          label: arithmeticLabels[arithmetic], color: arithmeticColors[arithmetic], marker: arithmeticMarkers[arithmetic],
          points: rows.map(row => ({ x: row.lanes, y: row[field], tooltip: `${arithmeticLabels[arithmetic]} ×${row.lanes}: ${row[field].toFixed(2)}${field.includes('percent') ? '%' : ''}; ${row.registers_per_thread.toFixed(0)} registers/thread; MIO ${row.mio_throttle_stall_per_issue.toFixed(2)}; long scoreboard ${row.long_scoreboard_stall_per_issue.toFixed(2)}` })),
        };
      }).filter(item => item.points.length);
      drawChart(panelTarget(grid, `resource-${kernel}-${field}`), {
        title: `${kernel.toUpperCase()} · ${title}`, accessible: `${kernel.toUpperCase()} ${title.toLowerCase()} by packet width`, series,
        xDomain: [1, 8], xTicks: widths, xFormat: value => `×${value}`,
        yDomain: domain ?? undefined, xLabel: 'Packet width', yLabel: field.includes('percent') ? 'Percent of sustained peak' : 'Instructions per logical value',
        yFormat: field.includes('percent') ? value => `${value.toFixed(0)}%` : undefined,
      });
    }
  }
  setLegend('resource-legend', modes.map(arithmetic => ({ label: arithmeticLabels[arithmetic], color: arithmeticColors[arithmetic], marker: arithmeticMarkers[arithmetic] })));
}

function renderAttribution(storage) {
  const grid = document.getElementById('attribution-grid');
  grid.replaceChildren();
  const modes = arithmeticModes(storage);
  const narrow = modes[0];
  for (const kernel of ['dot', 'gemv']) {
    const packetSeries = modes.map(arithmetic => {
      const points = [2, 4, 8].map(lanes => {
        const unrolled = packingData.timing.find(row => row.storage === storage && row.arithmetic === arithmetic && row.kernel === kernel && row.family === 'scalar_unrolled' && row.lanes === lanes);
        const packet = packingData.timing.find(row => row.storage === storage && row.arithmetic === arithmetic && row.kernel === kernel && row.family === 'vector_packet' && row.lanes === lanes);
        if (!unrolled || !packet) return null;
        const ratio = unrolled.median_time_ms / packet.median_time_ms;
        return { x: lanes, y: ratio, tooltip: `${arithmeticLabels[arithmetic]} ×${lanes}: packet is ${ratio.toFixed(2)}× the equally coarsened scalar throughput` };
      }).filter(Boolean);
      return { label: arithmeticLabels[arithmetic], color: arithmeticColors[arithmetic], marker: arithmeticMarkers[arithmetic], points };
    });
    drawChart(panelTarget(grid, `packet-ratio-${kernel}`), {
      title: `${kernel.toUpperCase()} · packet-only benefit`, accessible: `${kernel.toUpperCase()} vector packet benefit over scalar unrolling`, series: packetSeries,
      xDomain: [2, 8], xTicks: [2, 4, 8], xFormat: value => `×${value}`, yDomain: [0.7, Math.max(1.15, ...packetSeries.flatMap(item => item.points.map(point => point.y))) * 1.08],
      xLabel: 'Values per thread', yLabel: 'Unrolled time / packet time', reference: 1,
    });

    const fp64Available = modes.includes('fp64') && narrow !== 'fp64';
    const mixedPoints = fp64Available ? widths.map(lanes => {
      const family = lanes === 1 ? 'scalar_single' : 'vector_packet';
      const nativeRow = packingData.timing.find(row => row.storage === storage && row.arithmetic === narrow && row.kernel === kernel && row.family === family && row.lanes === lanes);
      const mixedRow = packingData.timing.find(row => row.storage === storage && row.arithmetic === 'fp64' && row.kernel === kernel && row.family === family && row.lanes === lanes);
      if (!nativeRow || !mixedRow) return null;
      const ratio = mixedRow.median_time_ms / nativeRow.median_time_ms;
      return { x: lanes, y: ratio, tooltip: `FP64 / ${arithmeticLabels[narrow]} at ×${lanes}: ${ratio.toFixed(2)}× time` };
    }).filter(Boolean) : [];
    const mixedSeries = [{ label: fp64Available ? `FP64 / ${arithmeticLabels[narrow]}` : 'No wider mode', color: '#D55E00', marker: 'diamond', points: mixedPoints }];
    drawChart(panelTarget(grid, `mixed-ratio-${kernel}`), {
      title: `${kernel.toUpperCase()} · mixed-arithmetic penalty`, accessible: `${kernel.toUpperCase()} FP64 arithmetic slowdown relative to the narrowest arithmetic`, series: mixedSeries,
      xDomain: [1, 8], xTicks: widths, xFormat: value => `×${value}`, yDomain: [0.85, Math.max(1.1, ...mixedPoints.map(point => point.y)) * 1.08],
      xLabel: 'Packet width', yLabel: 'FP64 time / narrow time', reference: 1,
    });
  }
  setLegend('attribution-legend', modes.map(arithmetic => ({ label: arithmeticLabels[arithmetic], color: arithmeticColors[arithmetic], marker: arithmeticMarkers[arithmetic] })));
}

function renderMechanisms(storage) {
  const grid = document.getElementById('mechanism-grid');
  grid.replaceChildren();
  const modes = arithmeticModes(storage);
  const loadSeries = ['scalar_unrolled', 'vector_packet'].map(family => {
    const base = packingData.components.find(row => row.component === 'stream_load' && row.storage === storage && row.family === 'scalar_single');
    const rows = base ? [{ ...base, lanes: 1 }, ...packingData.components.filter(row => row.component === 'stream_load' && row.storage === storage && row.family === family)] : [];
    return {
      label: familyLabels[family], color: familyColors[family], marker: family === 'vector_packet' ? 'circle' : 'square',
      points: rows.sort((a, b) => a.lanes-b.lanes).map(row => ({ x: row.lanes, y: row.gvalues_per_s, tooltip: `${familyLabels[family]} ×${row.lanes}: ${row.gvalues_per_s.toFixed(0)} Gvalues/s, ${row.effective_gb_per_s.toFixed(0)} GB/s` })),
    };
  });
  drawChart(panelTarget(grid, 'stream-load-throughput'), {
    title: 'Stream-load issue throughput', accessible: 'Isolated stream-load throughput for scalar unrolling and vector packets', series: loadSeries,
    xDomain: [1, 8], xTicks: widths, xFormat: value => `×${value}`, yScale: 'log',
    xLabel: 'Values per thread', yLabel: 'Gvalues/s',
  });
  const decodeSeries = modes.map(arithmetic => {
    const rows = packingData.components.filter(row => row.component === 'register_decode' && row.storage === storage && row.arithmetic === arithmetic).sort((a, b) => a.lanes-b.lanes);
    return {
      label: arithmeticLabels[arithmetic], color: arithmeticColors[arithmetic], marker: arithmeticMarkers[arithmetic],
      points: rows.map(row => ({ x: row.lanes, y: row.gconversions_per_s, tooltip: `${arithmeticLabels[arithmetic]} ×${row.lanes}: ${row.gconversions_per_s.toFixed(0)} Gconversions/s` })),
    };
  });
  drawChart(panelTarget(grid, 'register-decode-throughput'), {
    title: 'Register-resident conversion throughput', accessible: 'Register-resident conversion throughput by arithmetic type and independent lane count', series: decodeSeries,
    xDomain: [1, 8], xTicks: widths, xFormat: value => `×${value}`, yScale: 'log',
    xLabel: 'Independent decoded lanes', yLabel: 'Gconversions/s',
  });
  setLegend('mechanism-legend', [
    ...loadSeries.map(series => ({ label: series.label, color: series.color })),
    ...modes.map(arithmetic => ({ label: arithmeticLabels[arithmetic], color: arithmeticColors[arithmetic], marker: arithmeticMarkers[arithmetic] })),
  ]);
}

function renderAll() {
  const storage = storageSelect.value;
  renderSpeed(storage);
  renderRoofline(storage);
  renderResources(storage);
  renderAttribution(storage);
  renderMechanisms(storage);
}

storageSelect.addEventListener('change', renderAll);
renderAll();
"""


def navigation_html() -> str:
    """Delegate to the shared page list.

    This module used to keep its own copy, which silently went stale: it still
    named the conversion sections by their old labels and never gained the LNS
    or summary pages.
    """
    return base.navigation(PAGE_NAME)


def report_document(data: dict[str, object]) -> str:
    def json_safe(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    encoded = json.dumps(json_safe(data), separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )
    run_name = html.escape(str(data["run"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Packing and mixed-arithmetic bottlenecks — H200 storage-format report</title>
<link rel="stylesheet" href="report.css">
<style>{PAGE_STYLE}</style>
</head>
<body>
<header><div class="shell"><a class="brand" href="index.html">H200 storage-format report</a><nav aria-label="Report sections">{navigation_html()}</nav></div></header>
<main class="shell">
<h1>Packing and mixed-arithmetic bottlenecks</h1>
{PAGE_BODY}
</main>
<footer><div class="shell">Experiment: <code>{run_name}</code></div></footer>
<script type="application/json" id="packing-data">{encoded}</script>
<script>{PAGE_SCRIPT}</script>
</body>
</html>
"""


def patch_navigation(output_dir: Path) -> int:
    changed = 0
    for path in output_dir.glob("*.html"):
        if path.name == PAGE_NAME:
            continue
        text = path.read_text(encoding="utf-8")
        if f'href="{PAGE_NAME}"' in text:
            continue
        pattern = re.compile(
            r'(<nav aria-label="Report sections">.*?<a href="general-info\.html"[^>]*>General info</a>)(</nav>)',
            re.DOTALL,
        )
        replacement = rf'\1<a href="{PAGE_NAME}">{PAGE_LABEL}</a>\2'
        updated, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise SystemExit(f"could not add packing navigation to {path}")
        path.write_text(updated, encoding="utf-8")
        changed += 1
    return changed


def patch_index(output_dir: Path) -> None:
    path = output_dir / "index.html"
    text = path.read_text(encoding="utf-8")
    if f'href="{PAGE_NAME}"' in text.split("</nav>", 1)[-1]:
        return
    card = (
        f'<a class="report-link" href="{PAGE_NAME}"><strong>{PAGE_LABEL}</strong>'
        '<span>Packet speedup, roofline motion, resource transitions, and mixed-arithmetic cost.</span></a>'
    )
    marker = re.compile(
        r'(<a class="report-link" href="general-info\.html">.*?</a>)', re.DOTALL
    )
    updated, count = marker.subn(rf"\1{card}", text, count=1)
    if count != 1:
        raise SystemExit("could not add packing card to report index")
    path.write_text(updated, encoding="utf-8")


def patch_manifest(output_dir: Path, run_name: str) -> None:
    path = output_dir / "report_manifest.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if not line.startswith("packing_bottleneck_run=")]
    lines.append(f"packing_bottleneck_run={run_name}")
    for index, line in enumerate(lines):
        if line.startswith("pages=") and PAGE_NAME not in line.split("=", 1)[1].split(","):
            lines[index] = line + f",{PAGE_NAME}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else newest_complete_run(results_root)
    output_dir = args.output_dir.resolve()
    if not (output_dir / "index.html").is_file():
        raise SystemExit(f"base report is missing below {output_dir}")
    data = aggregate_run(run_dir)
    (output_dir / PAGE_NAME).write_text(report_document(data), encoding="utf-8")
    patched = patch_navigation(output_dir)
    patch_index(output_dir)
    patch_manifest(output_dir, run_dir.name)
    print(f"Wrote {output_dir / PAGE_NAME} and updated {patched} report navigation bars")


if __name__ == "__main__":
    main()
