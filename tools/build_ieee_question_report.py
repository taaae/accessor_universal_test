#!/usr/bin/env python3
"""Build question-led IEEE conversion pages from the unified H200 run.

The first rollout deliberately covers only the six-bit formats.  The helpers
are width-agnostic so the same page structure can be enabled for the remaining
formats after the presentation and comparison semantics have been reviewed.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_storage_performance_report as base


SIX_BIT_FORMATS = ("e0m5", "e1m4", "e2m3", "e3m2", "e4m1", "e5m0")
COMPUTES = ("fp32", "fp64")
KERNELS = ("dot", "gemv")
LAYOUTS = ("dense", "padded")
SCOPES = ("x1", "best")
DISTRIBUTIONS = ("uniform_0_1", "normal_0_1")

DECODER_LABELS = {
    "raw": "Raw arithmetic baseline",
    "generic": "Generic decoder",
    "direct_branchy": "Direct bit construction",
    "fixed_integer": "Fixed-point integer path",
    "e1_integer": "E1 integer path",
    "full_lut_global": "Full lookup · cached global",
    "full_lut_shared": "Full lookup · shared memory",
}

DECODER_EXPLANATIONS = {
    "generic": (
        "Runs the general IEEE-like decoder after extracting each code, then "
        "returns the requested arithmetic type."
    ),
    "direct_branchy": (
        "Extracts sign, exponent, and mantissa fields and constructs the target "
        "floating-point value directly, with separate zero/subnormal/normal paths."
    ),
    "fixed_integer": (
        "Treats the exponent-free payload as a signed fixed-point integer and "
        "applies its compile-time power-of-two scale."
    ),
    "e1_integer": (
        "Uses the E1 layout's integer-like representation and an exact "
        "power-of-two scale instead of a general floating-point decoder."
    ),
    "full_lut_global": (
        "Uses the six-bit code as an index into a 64-entry table containing the "
        "final arithmetic value. The table is read through the cached global path."
    ),
    "full_lut_shared": (
        "Stages a 64-entry table in shared memory, then uses each six-bit code "
        "as the lookup index for the final arithmetic value."
    ),
}

DECODER_COLORS = {
    "raw": "#222222",
    "generic": "#6c757d",
    "direct_branchy": "#0072B2",
    "fixed_integer": "#009E73",
    "e1_integer": "#56B4E9",
    "full_lut_global": "#D55E00",
    "full_lut_shared": "#CC79A7",
}


@dataclass(frozen=True)
class Selection:
    score_ms: float
    strategy_id: str
    row: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unified-run-dir",
        type=Path,
        help="experiment 021 run directory; default: newest complete run",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/021_unified_strategy_performance"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/report")
    )
    return parser.parse_args()


def newest_unified_run(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.glob("run_*")
        if (path / "unified_core/full/timing_summary.csv").is_file()
        and (path / "unified_core/full/validation.txt").is_file()
    )
    if not candidates:
        raise SystemExit(f"no complete unified strategy run below {root}")
    return candidates[-1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def geometric_mean(values: Iterable[float]) -> float:
    positive = list(values)
    if not positive or any(value <= 0.0 for value in positive):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in positive))


def strategy_score(rows: Sequence[dict[str, str]], kernel: str) -> float:
    sizes = sorted({int(row["N"]) for row in rows})
    if len(sizes) < 2:
        raise ValueError("strategy does not contain two N values")
    large_sizes = set(sizes[-2:])
    values = [
        float(row["median_ms"])
        for row in rows
        if int(row["N"]) in large_sizes
        and row["distribution"] in DISTRIBUTIONS
    ]
    expected = len(large_sizes) * len(DISTRIBUTIONS)
    if len(values) != expected:
        raise ValueError(
            f"incomplete {kernel} large-N aggregate: {len(values)} of {expected}"
        )
    return geometric_mean(values)


def format_rows(
    rows: Sequence[dict[str, str]], compute: str, format_name: str, kernel: str
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["arithmetic_type"] == compute
        and row["format"] == format_name
        and row["kernel"] == kernel
        and row.get("all_valid", "1") == "1"
    ]


def eligible_rows(
    rows: Sequence[dict[str, str]], layout: str, scope: str
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["storage_layout"] == layout]
    if scope == "x1":
        selected = [
            row
            for row in selected
            if row["access_method"] == "scalar"
            and int(row["packet_values"]) == 1
        ]
    elif scope != "best":
        raise ValueError(f"unknown scope {scope}")
    return selected


def best_selection(
    rows: Sequence[dict[str, str]],
    compute: str,
    format_name: str,
    kernel: str,
    layout: str,
    scope: str,
    *,
    access_method: str | None = None,
) -> Selection | None:
    candidates = eligible_rows(
        format_rows(rows, compute, format_name, kernel), layout, scope
    )
    if access_method is not None:
        candidates = [
            row for row in candidates if row["access_method"] == access_method
        ]
    by_strategy: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_strategy[row["strategy_id"]].append(row)
    if not by_strategy:
        return None
    ranked = [
        Selection(strategy_score(strategy_rows, kernel), strategy_id, strategy_rows[0])
        for strategy_id, strategy_rows in by_strategy.items()
    ]
    return min(ranked, key=lambda item: (item.score_ms, item.strategy_id))


def baseline_score(
    rows: Sequence[dict[str, str]], compute: str, kernel: str
) -> float:
    raw_format = f"raw_{compute}"
    raw = [
        row
        for row in rows
        if row["arithmetic_type"] == compute
        and row["format"] == raw_format
        and row["kernel"] == kernel
        and row["access_method"] == "scalar"
        and int(row["packet_values"]) == 1
        and row.get("all_valid", "1") == "1"
    ]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in raw:
        grouped[(row["distribution"], int(row["N"]))].append(
            float(row["median_ms"])
        )
    collapsed = [
        {
            "distribution": distribution,
            "N": str(size),
            "median_ms": str(statistics.median(values)),
        }
        for (distribution, size), values in grouped.items()
    ]
    return strategy_score(collapsed, kernel)


def format_bits(rows: Sequence[dict[str, str]], compute: str) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for row in rows:
        if row["arithmetic_type"] != compute or row["format"].startswith("raw_"):
            continue
        result.setdefault(
            row["format"],
            (int(row["exponent_bits"]), int(row["mantissa_bits"])),
        )
    return result


def faster_precise_peer(
    rows: Sequence[dict[str, str]],
    compute: str,
    format_name: str,
    kernel: str,
    layout: str,
    scope: str,
) -> tuple[str, Selection] | None:
    inventory = format_bits(rows, compute)
    exponent_bits, mantissa_bits = inventory[format_name]
    peers = [
        candidate
        for candidate, (candidate_e, candidate_m) in inventory.items()
        if candidate != format_name
        and candidate_e >= exponent_bits
        and candidate_m >= mantissa_bits
        and (candidate_e > exponent_bits or candidate_m > mantissa_bits)
    ]
    selections = []
    for candidate in peers:
        selection = best_selection(
            rows, compute, candidate, kernel, layout, scope
        )
        if selection is not None:
            selections.append((candidate, selection))
    return min(selections, key=lambda item: item[1].score_ms) if selections else None


def row_label(kernel: str, layout: str, scope: str) -> str:
    scope_label = "x1" if scope == "x1" else "Best"
    return f"{kernel.upper()} {layout.title()} {scope_label}"


def ratio_cell(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return '<td class="ratio-neutral">—</td>'
    css = "ratio-good" if value > 1.0 else "ratio-bad"
    return f'<td class="{css}">{value:.3f}×</td>'


def usefulness_table(
    rows: Sequence[dict[str, str]], compute: str, format_name: str
) -> str:
    body = []
    for kernel in KERNELS:
        baseline = baseline_score(rows, compute, kernel)
        for scope in SCOPES:
            for layout in LAYOUTS:
                selection = best_selection(
                    rows, compute, format_name, kernel, layout, scope
                )
                speedup = baseline / selection.score_ms if selection else None
                body.append(
                    f"<tr><th>{row_label(kernel, layout, scope)}</th>"
                    f"{ratio_cell(speedup)}</tr>"
                )
    return question_table(
        "Is this type even useful?",
        ("Case", f"Speedup over raw {compute.upper()}"),
        body,
    )


def precise_peer_table(
    rows: Sequence[dict[str, str]], compute: str, format_name: str
) -> str:
    body = []
    for kernel in KERNELS:
        for scope in SCOPES:
            for layout in LAYOUTS:
                current = best_selection(
                    rows, compute, format_name, kernel, layout, scope
                )
                peer = faster_precise_peer(
                    rows, compute, format_name, kernel, layout, scope
                )
                if current is None or peer is None:
                    peer_cell = "<td>—</td>"
                    speedup = None
                else:
                    peer_name, peer_selection = peer
                    peer_file = base.compute_conversion_format_filename(
                        compute, peer_name
                    )
                    peer_cell = (
                        f'<td><a href="{html.escape(peer_file)}">'
                        f"{html.escape(base.label(peer_name))}</a></td>"
                    )
                    speedup = peer_selection.score_ms / current.score_ms
                body.append(
                    f"<tr><th>{row_label(kernel, layout, scope)}</th>"
                    f"{peer_cell}{ratio_cell(speedup)}</tr>"
                )
    return question_table(
        "Is there a more precise type that is faster?",
        ("Case", "Fastest more-precise type", "Current type speedup"),
        body,
    )


def padded_table(
    rows: Sequence[dict[str, str]], compute: str, format_name: str
) -> str:
    body = []
    for kernel in KERNELS:
        for scope in SCOPES:
            dense = best_selection(rows, compute, format_name, kernel, "dense", scope)
            padded = best_selection(
                rows, compute, format_name, kernel, "padded", scope
            )
            speedup = (
                dense.score_ms / padded.score_ms if dense and padded else None
            )
            body.append(
                f"<tr><th>{kernel.upper()} {'x1' if scope == 'x1' else 'Best'}</th>"
                f"{ratio_cell(speedup)}</tr>"
            )
    return question_table(
        "Is the padded strategy faster?",
        ("Case", "Padded speedup over dense"),
        body,
    )


def same_bit_tables(
    rows: Sequence[dict[str, str]], compute: str, format_name: str
) -> str:
    tables = []
    for kernel in KERNELS:
        for scope in SCOPES:
            for layout in LAYOUTS:
                if format_name == "e0m5":
                    type_cells = []
                    timing_cells = []
                    for candidate in SIX_BIT_FORMATS:
                        selection = best_selection(
                            rows, compute, candidate, kernel, layout, scope
                        )
                        current = (
                            ' class="current-format-column"'
                            if candidate == format_name
                            else ""
                        )
                        filename = base.compute_conversion_format_filename(
                            compute, candidate
                        )
                        type_cells.append(
                            f"<th{current}><a href=\"{html.escape(filename)}\">"
                            f"{html.escape(base.label(candidate))}</a></th>"
                        )
                        timing_cells.append(
                            f"<td{current}>"
                            + (
                                f"{selection.score_ms:.6g} ms"
                                if selection is not None
                                else "—"
                            )
                            + "</td>"
                        )
                    tables.append(
                        '<div class="comparison-mini-table comparison-mini-table-wide">'
                        f"<h3>{kernel.upper()} · {layout.title()} · "
                        f"{'x1' if scope == 'x1' else 'Best'}</h3>"
                        '<div class="table-wrap"><table class="strategy-table '
                        'same-bit-transposed"><tbody><tr><th>6-bit type</th>'
                        f"{''.join(type_cells)}</tr><tr><th>Best time</th>"
                        f"{''.join(timing_cells)}</tr></tbody></table></div></div>"
                    )
                    continue

                baseline = baseline_score(rows, compute, kernel)
                body = []
                for candidate in SIX_BIT_FORMATS:
                    selection = best_selection(
                        rows, compute, candidate, kernel, layout, scope
                    )
                    speedup = baseline / selection.score_ms if selection else None
                    current = " class=\"current-format\"" if candidate == format_name else ""
                    filename = base.compute_conversion_format_filename(
                        compute, candidate
                    )
                    body.append(
                        f"<tr{current}><th><a href=\"{html.escape(filename)}\">"
                        f"{html.escape(base.label(candidate))}</a></th>"
                        f"{ratio_cell(speedup)}</tr>"
                    )
                tables.append(
                    '<div class="comparison-mini-table">'
                    f"<h3>{kernel.upper()} · {layout.title()} · "
                    f"{'x1' if scope == 'x1' else 'Best'}</h3>"
                    '<div class="table-wrap"><table class="strategy-table">'
                    f"<thead><tr><th>6-bit type</th><th>vs raw {compute.upper()}</th>"
                    f"</tr></thead><tbody>{''.join(body)}</tbody></table></div></div>"
                )
    return (
        '<section class="text-section question-section"><h2>'
        "Comparison of same-bit types</h2>"
        f'<div class="question-table-grid">{"".join(tables)}</div></section>'
    )


def packing_table(
    rows: Sequence[dict[str, str]], compute: str, format_name: str
) -> str:
    body = []
    for kernel in KERNELS:
        for layout in LAYOUTS:
            scalar = best_selection(
                rows, compute, format_name, kernel, layout, "x1"
            )
            packet = best_selection(
                rows,
                compute,
                format_name,
                kernel,
                layout,
                "best",
                access_method="thread_packet",
            )
            speedup = scalar.score_ms / packet.score_ms if scalar and packet else None
            body.append(
                f"<tr><th>{kernel.upper()} {layout.title()}</th>"
                f"{ratio_cell(speedup)}</tr>"
            )
    return question_table(
        "How much does packing improve performance?",
        ("Case", "Best packet speedup over x1"),
        body,
    )


def access_explanation(row: dict[str, str]) -> tuple[str, str]:
    layout = row["storage_layout"]
    access = row["access_method"]
    packet = int(row["packet_values"])
    decoder = row["decoder"]
    if layout == "dense":
        storage = "Reads the compact six-bit bitstream without byte padding."
        storage_pipeline = "dense 6-bit stream"
    else:
        storage = "Reads one byte per value; two padding bits are unused."
        storage_pipeline = "padded 8-bit slots"
    if access == "scalar":
        access_text = "Each thread loads and decodes one value at a time."
        access_pipeline = "scalar x1"
    elif access == "thread_packet":
        access_text = (
            f"Each thread loads one packet containing {packet} adjacent values "
            "and processes all decoded lanes."
        )
        access_pipeline = f"thread packet x{packet}"
    else:
        access_text = (
            f"A cooperative thread group loads a dense {packet}-value chunk, "
            "redistributes its words with warp shuffles, and divides decoded "
            "values among consumers."
        )
        access_pipeline = f"shuffle chunk x{packet}"
    decoder_text = DECODER_EXPLANATIONS.get(
        decoder, f"Uses the {decoder.replace('_', ' ')} decoder."
    )
    pipeline = (
        f"{storage_pipeline} → {access_pipeline} → "
        f"{DECODER_LABELS.get(decoder, decoder)} → {row['arithmetic_type'].upper()} FMA"
    )
    return f"{storage} {access_text} {decoder_text}", pipeline


def explanations_section(
    rows: Sequence[dict[str, str]], compute: str, format_name: str
) -> str:
    blocks = []
    for kernel in KERNELS:
        for scope in SCOPES:
            for layout in LAYOUTS:
                selection = best_selection(
                    rows, compute, format_name, kernel, layout, scope
                )
                if selection is None:
                    continue
                explanation, pipeline = access_explanation(selection.row)
                blocks.append(
                    '<article class="strategy-explanation">'
                    f"<h3>{row_label(kernel, layout, scope)}</h3>"
                    f"<p><code>{html.escape(selection.strategy_id)}</code></p>"
                    f"<p>{html.escape(explanation)}</p>"
                    f'<p class="strategy-pipeline">{html.escape(pipeline)}</p>'
                    "</article>"
                )
    return (
        '<section class="text-section question-section"><h2>'
        "Best strategies explained</h2>"
        f'<div class="strategy-explanation-grid">{"".join(blocks)}</div></section>'
    )


def question_table(
    title: str, headings: Sequence[str], body: Sequence[str]
) -> str:
    header = "".join(f"<th>{html.escape(item)}</th>" for item in headings)
    return f"""<section class="text-section question-section">
<h2>{html.escape(title)}</h2>
<div class="table-wrap"><table class="strategy-table question-table">
<thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
</section>"""


def strategy_label(row: dict[str, str]) -> str:
    if row["decoder"] == "raw":
        return f"Raw {row['strategy_id'].removeprefix('raw_').upper()} baseline"
    layout = row["storage_layout"].title()
    if row["access_method"] == "scalar":
        access = "x1"
    elif row["access_method"] == "thread_packet":
        access = f"packet x{row['packet_values']}"
    else:
        access = f"shuffle x{row['packet_values']}"
    decoder = DECODER_LABELS.get(row["decoder"], row["decoder"].replace("_", " "))
    return f"{layout} · {access} · {decoder}"


def plot_strategy_component(
    rows: Sequence[dict[str, str]],
    compute: str,
    format_name: str,
    kernel: str,
    path: Path,
) -> tuple[Path, list[dict[str, str]]]:
    selected = format_rows(rows, compute, format_name, kernel)
    raw_by_case: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["arithmetic_type"] == compute
            and row["format"] == f"raw_{compute}"
            and row["kernel"] == kernel
            and row["access_method"] == "scalar"
            and int(row["packet_values"]) == 1
        ):
            raw_by_case[(row["distribution"], int(row["N"]))].append(
                float(row["median_ms"])
            )
    selected.extend(
        {
            "distribution": distribution,
            "N": str(size),
            "median_ms": str(statistics.median(values)),
            "strategy_id": f"raw_{compute}",
            "storage_layout": "baseline",
            "access_method": "baseline",
            "packet_values": "1",
            "decoder": "raw",
        }
        for (distribution, size), values in raw_by_case.items()
    )
    metadata: dict[str, dict[str, str]] = {}
    for row in selected:
        metadata.setdefault(row["strategy_id"], row)
    strategies = sorted(
        metadata,
        key=lambda strategy_id: (
            metadata[strategy_id]["storage_layout"],
            metadata[strategy_id]["access_method"],
            int(metadata[strategy_id]["packet_values"]),
            metadata[strategy_id]["decoder"],
        ),
    )
    figure, axis = plt.subplots(figsize=(13.8, 7.2))
    series_metadata: list[dict[str, str]] = []
    for distribution in DISTRIBUTIONS:
        for strategy_id in strategies:
            current = sorted(
                (
                    row
                    for row in selected
                    if row["strategy_id"] == strategy_id
                    and row["distribution"] == distribution
                ),
                key=lambda row: int(row["N"]),
            )
            if not current:
                continue
            row = metadata[strategy_id]
            access = row["access_method"]
            packet = int(row["packet_values"])
            is_baseline = row["decoder"] == "raw"
            linestyle = (
                "-"
                if is_baseline
                else ":"
                if access == "scalar"
                else "--"
                if packet == 2
                else "-."
                if packet == 4
                else "-"
            )
            line = axis.plot(
                [int(item["N"]) for item in current],
                [float(item["median_ms"]) for item in current],
                color=DECODER_COLORS.get(row["decoder"], "#333333"),
                linestyle=linestyle,
                linewidth=2.5 if is_baseline else 1.65,
                marker=(
                    "D"
                    if is_baseline
                    else "o"
                    if row["storage_layout"] == "dense"
                    else "s"
                ),
                markerfacecolor=(
                    DECODER_COLORS.get(row["decoder"], "#333333")
                    if row["storage_layout"] in {"dense", "baseline"}
                    else "white"
                ),
                markersize=4.0,
                alpha=0.86,
            )[0]
            index = len(series_metadata)
            line.set_gid(f"question-series-{index}")
            series_metadata.append(
                {
                    "distribution": distribution,
                    "strategy_id": strategy_id,
                    "layout": row["storage_layout"],
                    "access": f"{access}-x{packet}",
                    "decoder": row["decoder"],
                    "label": strategy_label(row),
                }
            )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Reduction length N")
    axis.set_ylabel("Complete kernel time (ms)")
    axis.set_title(
        f"{base.label(format_name)} → {compute.upper()} · {kernel.upper()}",
        fontsize=15,
    )
    base.format_axis_labels(axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return path, series_metadata


def filter_option(
    kind: str, value: str, text: str, *, radio: bool = False, checked: bool = True
) -> str:
    input_type = "radio" if radio else "checkbox"
    name = ' name="distribution"' if radio else ""
    checked_attribute = " checked" if checked else ""
    data = "" if radio else f' data-filter="{html.escape(kind)}"'
    return (
        '<label class="filter-option">'
        f'<input type="{input_type}"{name}{data} value="{html.escape(value)}"'
        f"{checked_attribute}>"
        f"<span>{html.escape(text)}</span></label>"
    )


def strategy_chart_document(
    svg_path: Path,
    metadata: Sequence[dict[str, str]],
    compute: str,
    format_name: str,
    kernel: str,
) -> str:
    svg = svg_path.read_text(encoding="utf-8")
    svg = svg[svg.index("<svg ") :]
    for index, item in enumerate(metadata):
        opening = f'<g id="question-series-{index}">'
        replacement = (
            f'<g id="question-series-{index}" data-series="true" '
            f'data-distribution="{item["distribution"]}" '
            f'data-layout="{item["layout"]}" data-access="{item["access"]}" '
            f'data-decoder="{item["decoder"]}" '
            f'data-strategy="{html.escape(item["strategy_id"])}">'
            f'<title>{html.escape(item["label"])}</title>'
        )
        if opening not in svg:
            raise ValueError(f"missing SVG series {index} in {svg_path}")
        svg = svg.replace(opening, replacement, 1)

    strategy_rows: dict[str, dict[str, str]] = {}
    for item in metadata:
        strategy_rows.setdefault(item["strategy_id"], item)
    distribution_controls = "".join(
        (
            filter_option(
                "distribution",
                distribution,
                "U(0,1)" if distribution == "uniform_0_1" else "N(0,1)",
                radio=True,
                checked=distribution == "normal_0_1",
            )
            for distribution in DISTRIBUTIONS
        )
    )
    layout_controls = "".join(
        filter_option("layout", layout, layout.title()) for layout in LAYOUTS
    )
    accesses = sorted(
        {item["access"] for item in metadata if item["access"] != "baseline-x1"}
    )
    access_controls = "".join(
        filter_option(
            "access",
            access,
            access.replace("thread_packet", "Packet").replace(
                "cooperative_shuffle", "Shuffle"
            ).replace("scalar", "Scalar").replace("-x", " x"),
        )
        for access in accesses
    )
    decoders = sorted({item["decoder"] for item in metadata})
    decoder_controls = "".join(
        filter_option(
            "decoder", decoder, DECODER_LABELS.get(decoder, decoder)
        )
        for decoder in decoders
    )
    strategy_controls = "".join(
        filter_option(
            "strategy",
            strategy_id,
            item["label"],
        )
        for strategy_id, item in strategy_rows.items()
    )
    root_id = f"question-chart-{compute}-{format_name}-{kernel}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(base.label(format_name))} {kernel.upper()} strategies</title>
<style>
:root {{ color-scheme: light; --fg:#1f252b; --muted:#5d6872; --focus:#075f9a; --border:#d8dee4; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#fff; color:var(--fg); font:14px/1.4 system-ui,sans-serif; }}
.root {{ padding:12px 14px 2px; }} .filters {{ display:grid; gap:8px; margin-bottom:6px; }}
fieldset {{ border:0; margin:0; padding:0; }} legend {{ color:var(--muted); font-size:.84rem; margin-bottom:3px; }}
.options {{ display:flex; flex-wrap:wrap; gap:4px 14px; }} .filter-option {{ align-items:center; cursor:pointer; display:inline-flex; gap:5px; min-height:26px; }}
.filter-option input {{ accent-color:var(--focus); height:15px; margin:0; width:15px; }} details {{ border-top:1px solid var(--border); padding-top:6px; }}
summary {{ color:var(--muted); cursor:pointer; }} .status {{ color:var(--muted); margin:4px 0; }}
svg {{ display:block; height:auto; width:100%; }} [data-series].hidden {{ display:none; }}
</style></head><body><main id="{root_id}" class="root">
<section class="filters" aria-label="Chart filters">
<fieldset><legend>Distribution</legend><div class="options">{distribution_controls}</div></fieldset>
<fieldset><legend>Storage layout</legend><div class="options">{layout_controls}</div></fieldset>
<fieldset><legend>Access</legend><div class="options">{access_controls}</div></fieldset>
<fieldset><legend>Decoder</legend><div class="options">{decoder_controls}</div></fieldset>
<details><summary>Individual strategies</summary><div class="options">{strategy_controls}</div></details>
</section><p class="status" aria-live="polite"></p>{svg}</main>
<script>
const root=document.getElementById('{root_id}');
const controls=[...root.querySelectorAll('input')]; const series=[...root.querySelectorAll('[data-series]')];
function enabled(kind,value){{const input=controls.find(item=>item.dataset.filter===kind&&item.value===value);return input?input.checked:true;}}
function reportHeight(){{if(window.parent!==window)window.parent.postMessage({{type:'performance-chart-height',height:Math.ceil(root.getBoundingClientRect().bottom+2)}},'*');}}
function update(){{const distribution=root.querySelector('input[name="distribution"]:checked').value;let visible=0;for(const line of series){{const show=line.dataset.distribution===distribution&&enabled('layout',line.dataset.layout)&&enabled('access',line.dataset.access)&&enabled('decoder',line.dataset.decoder)&&enabled('strategy',line.dataset.strategy);line.classList.toggle('hidden',!show);line.setAttribute('aria-hidden',String(!show));if(show)visible++;}}root.querySelector('.status').textContent=`${{visible}} strategies visible · lower is faster`;requestAnimationFrame(reportHeight);}}
root.addEventListener('change',update);window.addEventListener('load',update);window.addEventListener('message',event=>{{if(event.data?.type==='request-performance-chart-height')reportHeight();}});if('ResizeObserver'in window)new ResizeObserver(reportHeight).observe(root);
</script></body></html>"""


def chart_sections(compute: str, format_name: str) -> str:
    blocks = []
    for kernel in KERNELS:
        source = f"interactive/question-{compute}-{format_name}-{kernel}.html"
        blocks.append(
            f"<h3>{kernel.upper()}</h3>"
            '<figure class="interactive-figure">'
            f'<iframe class="interactive-chart-frame question-chart-frame" '
            f'data-performance-chart src="{source}" '
            f'title="Interactive {html.escape(base.label(format_name))} '
            f'{kernel.upper()} strategy timing"></iframe>'
            f'<figcaption><a href="{source}">Open the chart full size</a>.</figcaption>'
            "</figure>"
        )
    return (
        '<section class="graph-section question-section"><h2>'
        "How do all strategies scale with N?</h2>" + "".join(blocks) + "</section>"
    )


def page_body(
    rows: Sequence[dict[str, str]],
    run_dir: Path,
    output_dir: Path,
    compute: str,
    format_name: str,
) -> str:
    filename = base.compute_conversion_format_filename(compute, format_name)
    raw_prefix = Path(os.path.relpath(run_dir / "unified_core/full", output_dir)).as_posix()
    return (
        base.conversion_navigation(filename, compute)
        + '<section class="experiment-status completed-section">'
        '<span class="status-badge">H200 measured</span>'
        f"<strong>Experiment 021 · {html.escape(run_dir.name)}</strong></section>"
        + (
            f'<h2 class="question-format-heading">{html.escape(base.label(format_name))}</h2>'
            if format_name == "e0m5"
            else ""
        )
        + usefulness_table(rows, compute, format_name)
        + precise_peer_table(rows, compute, format_name)
        + padded_table(rows, compute, format_name)
        + same_bit_tables(rows, compute, format_name)
        + packing_table(rows, compute, format_name)
        + explanations_section(rows, compute, format_name)
        + chart_sections(compute, format_name)
        + f'<section class="text-section"><h2>Raw data</h2><ul><li><a href="'
        f'{raw_prefix}/timing_summary.csv">Unified timing summary</a></li></ul></section>'
    )


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def update_overview_cards(output_dir: Path, compute: str) -> None:
    overview = output_dir / base.compute_conversion_overview_filename(compute)
    document = overview.read_text(encoding="utf-8")
    for format_name in SIX_BIT_FORMATS:
        filename = base.compute_conversion_format_filename(compute, format_name)
        pattern = re.compile(
            rf'<a class="report-link pending-report-link" href="{re.escape(filename)}">'
            r'(?P<label><strong>.*?</strong>)<span>.*?</span></a>'
        )
        replacement = (
            f'<a class="report-link" href="{filename}">'
            r'\g<label><span>Unified H200 question report</span></a>'
        )
        document, count = pattern.subn(replacement, document, count=1)
        if count == 0:
            existing_pattern = re.compile(
                rf'<a class="report-link" href="{re.escape(filename)}">'
                r'(?P<label><strong>.*?</strong>)<span>.*?</span></a>'
            )
            document, count = existing_pattern.subn(
                replacement, document, count=1
            )
        if count != 1:
            raise ValueError(f"could not update overview card for {compute}/{format_name}")
    document = document.replace(
        "<span class=\"status-badge\">Pending H200 data</span>",
        "<span class=\"status-badge\">6-bit H200 data available</span>",
        1,
    )
    overview.write_text(document, encoding="utf-8")


EXTRA_CSS = """
/* Question-led unified IEEE pages. */
.completed-section { border-style: solid; }
.question-section { max-width: 1220px; }
.question-table { width: min(100%, 680px); }
.question-table th:first-child { width: auto; }
.question-table td:last-child { font-variant-numeric: tabular-nums; font-weight: 650; text-align: right; white-space: nowrap; }
.ratio-good { background: color-mix(in srgb, #1b8a4b 14%, transparent); color: #146b3a; }
.ratio-bad { background: color-mix(in srgb, #c43838 14%, transparent); color: #a12626; }
.ratio-neutral { color: var(--muted); }
.question-table-grid { display: grid; gap: 24px 30px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.comparison-mini-table h3 { margin: 0 0 7px; }
.comparison-mini-table .strategy-table { width: 100%; }
.comparison-mini-table-wide { grid-column: 1 / -1; }
.current-format { background: var(--active); box-shadow: inset 4px 0 0 var(--link); }
.question-format-heading { font-size: clamp(1.55rem, 3vw, 2rem); margin: 0 0 24px; }
.same-bit-transposed { min-width: 760px; table-layout: fixed; }
.same-bit-transposed th, .same-bit-transposed td { text-align: center; width: auto; }
.same-bit-transposed th:first-child { text-align: left; width: 112px; }
.same-bit-transposed td { font-variant-numeric: tabular-nums; white-space: nowrap; }
.current-format-column { background: var(--active); box-shadow: inset 0 4px 0 var(--link); }
.strategy-explanation-grid { display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.strategy-explanation { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }
.strategy-explanation h3 { margin: 0 0 8px; }
.strategy-explanation p { margin: 7px 0; }
.strategy-pipeline { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .86rem; }
.question-chart-frame { height: 760px; }
@media (prefers-color-scheme: dark) {
  .ratio-good { color: #82d8a5; }
  .ratio-bad { color: #ff9b9b; }
}
@media (max-width: 760px) {
  .question-table-grid, .strategy-explanation-grid { grid-template-columns: 1fr; }
}
"""


def validate(rows: Sequence[dict[str, str]]) -> None:
    for compute in COMPUTES:
        available = {
            row["format"]
            for row in rows
            if row["arithmetic_type"] == compute and row["bits"] == "6"
        }
        if available != set(SIX_BIT_FORMATS):
            raise SystemExit(
                f"{compute} six-bit inventory mismatch: {sorted(available)}"
            )
        for format_name in SIX_BIT_FORMATS:
            for kernel in KERNELS:
                for layout in LAYOUTS:
                    for scope in SCOPES:
                        if best_selection(
                            rows, compute, format_name, kernel, layout, scope
                        ) is None:
                            raise SystemExit(
                                f"missing {compute}/{format_name}/{kernel}/{layout}/{scope}"
                            )


def main() -> None:
    args = parse_args()
    run_dir = args.unified_run_dir or newest_unified_run(args.results_root)
    output_dir = args.output_dir
    if not (output_dir / "report.css").is_file():
        raise SystemExit(
            f"{output_dir} is not a built report; run the base report builder first"
        )
    timing_path = run_dir / "unified_core/full/timing_summary.csv"
    rows = read_rows(timing_path)
    validate(rows)
    manifest = read_manifest(output_dir / "report_manifest.txt")

    assets = output_dir / "assets"
    interactive = output_dir / "interactive"
    assets.mkdir(parents=True, exist_ok=True)
    interactive.mkdir(parents=True, exist_ok=True)
    for compute in COMPUTES:
        for format_name in SIX_BIT_FORMATS:
            for kernel in KERNELS:
                svg_path, chart_metadata = plot_strategy_component(
                    rows,
                    compute,
                    format_name,
                    kernel,
                    assets / f"question-{compute}-{format_name}-{kernel}.svg",
                )
                (interactive / f"question-{compute}-{format_name}-{kernel}.html").write_text(
                    strategy_chart_document(
                        svg_path,
                        chart_metadata,
                        compute,
                        format_name,
                        kernel,
                    ),
                    encoding="utf-8",
                )
            filename = base.compute_conversion_format_filename(compute, format_name)
            document = base.page_document(
                filename=filename,
                title=f"{base.label(format_name)} → {compute.upper()} arithmetic",
                intro=(
                    "Question-led H200 comparison of dense and padded storage, "
                    "scalar and packed access, and every qualified decoder."
                ),
                body=page_body(rows, run_dir, output_dir, compute, format_name),
                performance_run_name=manifest.get("performance_run", "unknown"),
                accuracy_run_name=manifest.get("accuracy_run", "unknown"),
                strategy_run_name=manifest.get("strategy_run", "unknown"),
                all_strategy_run_name=manifest.get("all_strategy_run", "unknown"),
                expanded_strategy_run_name=(
                    manifest.get("expanded_strategy_run", "unknown")
                    + f" + {run_dir.name}"
                ),
            )
            (output_dir / filename).write_text(document, encoding="utf-8")
        update_overview_cards(output_dir, compute)

    css_path = output_dir / "report.css"
    css = css_path.read_text(encoding="utf-8")
    if "/* Question-led unified IEEE pages. */" not in css:
        css_path.write_text(css.rstrip() + "\n" + EXTRA_CSS, encoding="utf-8")

    manifest_path = output_dir / "report_manifest.txt"
    current_manifest = "\n".join(
        line
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("unified_strategy_run=", "unified_six_bit_rows="))
    ).rstrip()
    manifest_path.write_text(
        current_manifest
        + f"\nunified_strategy_run={run_dir.name}"
        + f"\nunified_six_bit_rows={sum(row['bits'] == '6' for row in rows)}\n",
        encoding="utf-8",
    )
    print(f"Updated six-bit IEEE question pages in {output_dir}")


if __name__ == "__main__":
    main()
