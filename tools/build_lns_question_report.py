#!/usr/bin/env python3
"""Build question-led LNS conversion pages from the experiment 022 H200 run.

An ``LNS<B,R>`` value is one sign bit plus a signed ``B-1`` bit fixed-point
logarithm carrying ``R`` fractional bits, so its integer part ``I = B-1-R``
sets dynamic range and ``R`` sets relative precision.  That makes ``(I, R)``
the exact counterpart of an IEEE format's ``(exponent, mantissa)`` pair, which
is what lets these pages reuse the IEEE precision ordering and name a matched
IEEE type for every LNS format.
"""

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
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_storage_performance_report as base


COMPUTES = ("fp32", "fp64")
KERNELS = ("dot", "gemv")
SCOPES = ("x1", "best")
LAYOUTS = ("dense", "padded")
MULTIPLIES = ("ordinary", "fused")
DISTRIBUTIONS = ("uniform_0_1", "normal_0_1")

DECODER_LABELS = {
    "raw": "Raw arithmetic baseline",
    "reference_exp2": "Reference exp2",
    "ex2_approx": "Hardware ex2.approx",
    "full_lut_global": "Full lookup · cached global",
    "full_lut_shared": "Full lookup · shared memory",
    "fraction_lut_global": "Fraction lookup · cached global",
    "fraction_lut_shared": "Fraction lookup · shared memory",
    "fraction_lut_warp": "Fraction lookup · warp registers",
    "split_linear": "Split lookup + linear term",
    "split_quadratic": "Split lookup + quadratic term",
    "split_cubic": "Split lookup + cubic term",
    "pair_lut_global": "Pair-product lookup · cached global",
    "pair_lut_shared": "Pair-product lookup · shared memory",
}

DECODER_COLORS = {
    "raw": "#222222",
    "reference_exp2": "#6c757d",
    "ex2_approx": "#0072B2",
    "full_lut_global": "#D55E00",
    "full_lut_shared": "#CC79A7",
    "fraction_lut_global": "#009E73",
    "fraction_lut_shared": "#56B4E9",
    "fraction_lut_warp": "#7E57C2",
    "split_linear": "#E69F00",
    "split_quadratic": "#B8860B",
    "split_cubic": "#C17C00",
    "pair_lut_global": "#8C564B",
    "pair_lut_shared": "#3B8ED0",
}

MULTIPLY_LABELS = {"ordinary": "ordinary", "fused": "fused"}


@dataclass(frozen=True)
class Selection:
    score_ms: float
    strategy_id: str
    row: dict[str, str]


@dataclass(frozen=True)
class Layout:
    """Bit allocation of one LNS format."""

    bits: int
    integer_bits: int
    fraction_bits: int

    @property
    def ieee_label(self) -> str:
        return f"E{self.integer_bits}M{self.fraction_bits}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lns-run-dir",
        type=Path,
        help="experiment 022 run directory; default: newest complete run",
    )
    parser.add_argument(
        "--lns-results-root",
        type=Path,
        default=Path("results/022_lns_strategy_performance"),
    )
    parser.add_argument(
        "--ieee-results-root",
        type=Path,
        default=Path("results/021_unified_strategy_performance"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/report"))
    parser.add_argument(
        "--formats",
        default="",
        help="comma-separated LNS formats to build; default: every format in the run",
    )
    return parser.parse_args()


def newest_run(root: Path, relative: str) -> Path:
    candidates = sorted(
        path for path in root.glob("run_*") if (path / relative).is_file()
    )
    if not candidates:
        raise SystemExit(f"no complete run below {root}")
    return candidates[-1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def geometric_mean(values: Iterable[float]) -> float:
    positive = list(values)
    if not positive or any(value <= 0.0 for value in positive):
        raise ValueError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in positive))


def strategy_score(rows: Sequence[dict[str, str]], field: str = "median_ms") -> float:
    """Geometric mean over both distributions at the two largest N."""
    sizes = sorted({int(row["N"]) for row in rows})
    if len(sizes) < 2:
        raise ValueError("strategy does not span two N values")
    large = set(sizes[-2:])
    values = [
        float(row[field])
        for row in rows
        if int(row["N"]) in large and row["distribution"] in DISTRIBUTIONS
    ]
    expected = len(large) * len(DISTRIBUTIONS)
    if len(values) != expected:
        raise ValueError(f"incomplete aggregate: {len(values)} of {expected}")
    return geometric_mean(values)


class LnsIndex:
    """Groups the LNS run once and memoises every derived selection."""

    def __init__(self, rows: Sequence[dict[str, str]], raw_rows: Sequence[dict[str, str]]):
        self.rows = rows
        self.by_case: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        self.layouts: dict[str, Layout] = {}
        self.computes: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.get("all_valid", "1") != "1":
                continue
            self.by_case[
                (row["arithmetic_type"], row["format"], row["kernel"])
            ].append(row)
            self.computes[row["format"]].add(row["arithmetic_type"])
            self.layouts.setdefault(
                row["format"],
                Layout(
                    int(row["bits"]),
                    int(row["log_integer_bits"]),
                    int(row["log_fraction_bits"]),
                ),
            )
        self.raw_by_case: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in raw_rows:
            if row.get("valid", "1") != "1":
                continue
            self.raw_by_case[(row["arithmetic_type"], row["kernel"])].append(row)
        self.selections: dict[tuple, Selection | None] = {}
        self.baselines: dict[tuple[str, str], float] = {}

    def formats(self, compute: str) -> tuple[str, ...]:
        """Formats measured with this arithmetic, narrow to wide then by range."""
        names = [f for f, c in self.computes.items() if compute in c]
        return tuple(
            sorted(
                names,
                key=lambda f: (self.layouts[f].bits, self.layouts[f].integer_bits),
            )
        )

    def width_group(self, compute: str, format_name: str) -> tuple[str, ...]:
        bits = self.layouts[format_name].bits
        return tuple(
            f for f in self.formats(compute) if self.layouts[f].bits == bits
        )

    def has_padded(self, format_name: str) -> bool:
        """True when a padded variant wastes bits, i.e. the width is unaligned."""
        return {8, 16, 32}.isdisjoint({self.layouts[format_name].bits})


def eligible(
    rows: Sequence[dict[str, str]],
    scope: str,
    layout: str | None,
    multiply: str | None,
    access: str | None,
) -> list[dict[str, str]]:
    selected = list(rows)
    if layout is not None:
        selected = [r for r in selected if r["storage_layout"] == layout]
    if multiply is not None:
        selected = [r for r in selected if r["multiply_method"] == multiply]
    if access is not None:
        selected = [r for r in selected if r["access_method"] == access]
    if scope == "x1":
        selected = [
            r
            for r in selected
            if r["access_method"] == "scalar" and int(r["packet_values"]) == 1
        ]
    elif scope != "best":
        raise ValueError(f"unknown scope {scope}")
    return selected


def best_selection(
    index: LnsIndex,
    compute: str,
    format_name: str,
    kernel: str,
    scope: str,
    *,
    layout: str | None = None,
    multiply: str | None = None,
    access: str | None = None,
) -> Selection | None:
    key = (compute, format_name, kernel, scope, layout, multiply, access)
    if key in index.selections:
        return index.selections[key]
    candidates = eligible(
        index.by_case.get((compute, format_name, kernel), ()),
        scope,
        layout,
        multiply,
        access,
    )
    by_strategy: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_strategy[row["strategy_id"]].append(row)
    ranked = [
        Selection(strategy_score(rows), strategy_id, rows[0])
        for strategy_id, rows in by_strategy.items()
    ]
    result = min(ranked, key=lambda s: (s.score_ms, s.strategy_id)) if ranked else None
    index.selections[key] = result
    return result


def baseline_score(index: LnsIndex, compute: str, kernel: str) -> float:
    """Raw FP32/FP64 anchor remeasured inside the same allocation."""
    if (compute, kernel) in index.baselines:
        return index.baselines[(compute, kernel)]
    raw = [
        row
        for row in index.raw_by_case.get((compute, kernel), ())
        if row["format"] == f"raw_{compute}"
        and row["access_method"] == "scalar"
        and int(row["packet_values"]) == 1
    ]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in raw:
        grouped[(row["distribution"], int(row["N"]))].append(float(row["mean_ms"]))
    collapsed = [
        {
            "distribution": distribution,
            "N": str(size),
            "median_ms": str(statistics.median(values)),
        }
        for (distribution, size), values in grouped.items()
    ]
    index.baselines[(compute, kernel)] = score = strategy_score(collapsed)
    return score


# --------------------------------------------------------------------------
# IEEE counterpart, experiment 021
# --------------------------------------------------------------------------


class IeeeIndex:
    """The matched IEEE measurements, keyed by (exponent, mantissa)."""

    def __init__(self, rows: Sequence[dict[str, str]]):
        self.by_case: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        self.by_allocation: dict[tuple[str, int, int], str] = {}
        for row in rows:
            if row.get("all_valid", "1") != "1" or row["format"].startswith("raw_"):
                continue
            self.by_case[
                (row["arithmetic_type"], row["format"], row["kernel"])
            ].append(row)
            self.by_allocation.setdefault(
                (
                    row["arithmetic_type"],
                    int(row["exponent_bits"]),
                    int(row["mantissa_bits"]),
                ),
                row["format"],
            )
        self.selections: dict[tuple, Selection | None] = {}

    def match(self, compute: str, layout: Layout) -> str | None:
        return self.by_allocation.get(
            (compute, layout.integer_bits, layout.fraction_bits)
        )

    def best(
        self, compute: str, format_name: str, kernel: str, scope: str
    ) -> Selection | None:
        key = (compute, format_name, kernel, scope)
        if key in self.selections:
            return self.selections[key]
        rows = self.by_case.get((compute, format_name, kernel), ())
        if scope == "x1":
            rows = [
                r
                for r in rows
                if r["access_method"] == "scalar" and int(r["packet_values"]) == 1
            ]
        by_strategy: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_strategy[row["strategy_id"]].append(row)
        ranked = [
            Selection(strategy_score(rs), sid, rs[0])
            for sid, rs in by_strategy.items()
        ]
        result = (
            min(ranked, key=lambda s: (s.score_ms, s.strategy_id)) if ranked else None
        )
        self.selections[key] = result
        return result


# --------------------------------------------------------------------------
# Presentation helpers
# --------------------------------------------------------------------------


def case_label(kernel: str, scope: str) -> str:
    return f"{kernel.upper()} {'x1' if scope == 'x1' else 'Best'}"


def ratio_cell(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return '<td class="ratio-neutral">—</td>'
    css = "ratio-good" if value > 1.0 else "ratio-bad"
    return f'<td class="{css}">{value:.3f}×</td>'


def question_table(title: str, headings: Sequence[str], body: Sequence[str]) -> str:
    header = "".join(f"<th>{html.escape(item)}</th>" for item in headings)
    return f"""<section class="text-section question-section">
<h2>{html.escape(title)}</h2>
<div class="table-wrap"><table class="strategy-table question-table">
<thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
</section>"""


def lns_label(index: LnsIndex, format_name: str) -> str:
    layout = index.layouts[format_name]
    return f"LNS&lt;{layout.bits},{layout.fraction_bits}&gt;"


def lns_plain_label(index: LnsIndex, format_name: str) -> str:
    layout = index.layouts[format_name]
    return f"LNS<{layout.bits},{layout.fraction_bits}>"


def lns_page_filename(compute: str, format_name: str) -> str:
    return f"lns-{compute}-{format_name}.html"


def lns_overview_filename(compute: str) -> str:
    return f"lns-{compute}.html"


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def usefulness_table(index: LnsIndex, compute: str, format_name: str) -> str:
    body = []
    for kernel in KERNELS:
        baseline = baseline_score(index, compute, kernel)
        for scope in SCOPES:
            best = best_selection(index, compute, format_name, kernel, scope)
            speedup = baseline / best.score_ms if best else None
            body.append(
                f"<tr><th>{case_label(kernel, scope)}</th>{ratio_cell(speedup)}</tr>"
            )
    return question_table(
        "Is this type even useful?",
        ("Case", f"Speedup over raw {compute.upper()}"),
        body,
    )


def precise_peer_table(index: LnsIndex, compute: str, format_name: str) -> str:
    current_layout = index.layouts[format_name]
    peers = [
        candidate
        for candidate in index.formats(compute)
        if candidate != format_name
        and index.layouts[candidate].integer_bits >= current_layout.integer_bits
        and index.layouts[candidate].fraction_bits >= current_layout.fraction_bits
        and (
            index.layouts[candidate].integer_bits > current_layout.integer_bits
            or index.layouts[candidate].fraction_bits > current_layout.fraction_bits
        )
    ]
    body = []
    for kernel in KERNELS:
        for scope in SCOPES:
            current = best_selection(index, compute, format_name, kernel, scope)
            ranked = []
            for candidate in peers:
                selection = best_selection(index, compute, candidate, kernel, scope)
                if selection is not None:
                    ranked.append((selection.score_ms, candidate, selection))
            if current is None or not ranked:
                peer_cell = "<td>—</td>"
                speedup = None
            else:
                _, peer_name, peer_selection = min(ranked)
                link = lns_page_filename(compute, peer_name)
                peer_cell = (
                    f'<td><a href="{html.escape(link)}">'
                    f"{lns_label(index, peer_name)}</a></td>"
                )
                speedup = peer_selection.score_ms / current.score_ms
            body.append(
                f"<tr><th>{case_label(kernel, scope)}</th>"
                f"{peer_cell}{ratio_cell(speedup)}</tr>"
            )
    return question_table(
        "Is there a more precise type that is faster?",
        ("Case", "Fastest more-precise type", "Current type speedup"),
        body,
    )


def ieee_comparison_table(
    index: LnsIndex, ieee: IeeeIndex, compute: str, format_name: str
) -> str:
    layout = index.layouts[format_name]
    match = ieee.match(compute, layout)
    if match is None:
        note = (
            f'<p class="lns-match-note">{lns_label(index, format_name)} allocates its '
            f"logarithm as {layout.ieee_label}, but no IEEE format with that "
            f"allocation was measured with {compute.upper()} arithmetic.</p>"
        )
        return (
            '<section class="text-section question-section">'
            "<h2>Is it better than similar IEEE?</h2>" + note + "</section>"
        )
    ieee_href = base.compute_conversion_format_filename(compute, match)
    note = (
        f'<p class="lns-match-note">{lns_label(index, format_name)} spends its '
        f"{layout.bits} bits as one sign bit, {layout.integer_bits} integer and "
        f"{layout.fraction_bits} fractional logarithm bits, so its range and "
        f"resolution match "
        f'<a href="{html.escape(ieee_href)}">{html.escape(base.label(match))}</a>.</p>'
    )
    body = []
    for kernel in KERNELS:
        for scope in SCOPES:
            current = best_selection(index, compute, format_name, kernel, scope)
            counterpart = ieee.best(compute, match, kernel, scope)
            speedup = (
                counterpart.score_ms / current.score_ms
                if current and counterpart
                else None
            )
            body.append(
                f"<tr><th>{case_label(kernel, scope)}</th>{ratio_cell(speedup)}</tr>"
            )
    header = "".join(
        f"<th>{item}</th>"
        for item in (
            "Case",
            f"Speedup over {html.escape(base.label(match))} → {compute.upper()}",
        )
    )
    return f"""<section class="text-section question-section">
<h2>Is it better than similar IEEE?</h2>
{note}
<div class="table-wrap"><table class="strategy-table question-table">
<thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
</section>"""


def fusion_table(index: LnsIndex, compute: str, format_name: str) -> str:
    body = []
    for kernel in KERNELS:
        for scope in SCOPES:
            ordinary = best_selection(
                index, compute, format_name, kernel, scope, multiply="ordinary"
            )
            fused = best_selection(
                index, compute, format_name, kernel, scope, multiply="fused"
            )
            speedup = (
                ordinary.score_ms / fused.score_ms if ordinary and fused else None
            )
            body.append(
                f"<tr><th>{case_label(kernel, scope)}</th>{ratio_cell(speedup)}</tr>"
            )
    return question_table(
        "Is LNS fusion faster?",
        ("Case", "Fusion speedup over normal"),
        body,
    )


def padded_table(index: LnsIndex, compute: str, format_name: str) -> str:
    if not index.has_padded(format_name):
        return ""
    body = []
    for kernel in KERNELS:
        for scope in SCOPES:
            dense = best_selection(
                index, compute, format_name, kernel, scope, layout="dense"
            )
            padded = best_selection(
                index, compute, format_name, kernel, scope, layout="padded"
            )
            speedup = dense.score_ms / padded.score_ms if dense and padded else None
            body.append(
                f"<tr><th>{case_label(kernel, scope)}</th>{ratio_cell(speedup)}</tr>"
            )
    return question_table(
        "Is the padded strategy faster?",
        ("Case", "Padded speedup over dense"),
        body,
    )


def packing_table(index: LnsIndex, compute: str, format_name: str) -> str:
    body = []
    for kernel in KERNELS:
        scalar = best_selection(index, compute, format_name, kernel, "x1")
        packet = best_selection(
            index, compute, format_name, kernel, "best", access="thread_packet"
        )
        speedup = scalar.score_ms / packet.score_ms if scalar and packet else None
        body.append(f"<tr><th>{kernel.upper()}</th>{ratio_cell(speedup)}</tr>")
    return question_table(
        "How much does packing improve performance?",
        ("Case", "Best packet speedup over x1"),
        body,
    )


def same_bit_tables(index: LnsIndex, compute: str, format_name: str) -> str:
    peers = index.width_group(compute, format_name)
    if len(peers) < 2:
        return ""
    bits = index.layouts[format_name].bits
    min_width = 112 + 128 * len(peers)
    tables = []
    for kernel in KERNELS:
        for scope in SCOPES:
            type_cells = []
            timing_cells = []
            for candidate in peers:
                selection = best_selection(index, compute, candidate, kernel, scope)
                current = (
                    ' class="current-format-column"'
                    if candidate == format_name
                    else ""
                )
                link = lns_page_filename(compute, candidate)
                type_cells.append(
                    f'<th{current}><a href="{html.escape(link)}">'
                    f"{lns_label(index, candidate)}</a></th>"
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
                f"<h3>{kernel.upper()} · {'x1' if scope == 'x1' else 'Best'}</h3>"
                '<div class="table-wrap"><table class="strategy-table '
                f'same-bit-transposed" style="min-width:{min_width}px">'
                f"<tbody><tr><th>{bits}-bit type</th>{''.join(type_cells)}</tr>"
                f"<tr><th>Best time</th>{''.join(timing_cells)}</tr>"
                "</tbody></table></div></div>"
            )
    return (
        '<section class="text-section question-section"><h2>'
        f"Comparison of same-bit types ({bits}-bit)</h2>"
        f'<div class="question-table-grid">{"".join(tables)}</div></section>'
    )


def padded_container_bits(bits: int) -> int:
    for container in (8, 16, 32):
        if bits <= container:
            return container
    raise ValueError(f"no padded container for {bits}-bit storage")


def strategy_explanation(index: LnsIndex, row: dict[str, str]) -> tuple[str, str]:
    layout_name = row["storage_layout"]
    access = row["access_method"]
    packet = int(row["packet_values"])
    decoder = row["decoder"]
    multiply = row["multiply_method"]
    bits = int(row["bits"])
    fraction_bits = int(row["log_fraction_bits"])

    container = padded_container_bits(bits)
    if layout_name == "padded" and container > bits:
        storage = (
            f"Reads one {container}-bit slot per value; {container - bits} padding "
            "bits are unused."
        )
        storage_pipeline = f"padded {container}-bit slots"
    else:
        storage = f"Reads the compact {bits}-bit bitstream without padding."
        storage_pipeline = f"dense {bits}-bit stream"

    if access == "scalar":
        access_text = "Each thread loads and decodes one value at a time."
        access_pipeline = "scalar x1"
    elif access == "thread_packet":
        access_text = (
            f"Each thread loads one packet of {packet} adjacent values and "
            "processes every decoded lane."
        )
        access_pipeline = f"thread packet x{packet}"
    else:
        loaders = int(row["loader_threads"])
        consumers = int(row["consumer_threads"])
        word_bits = int(row["load_word_bits"])
        access_text = (
            f"A cooperative group loads a dense {packet}-value chunk as {loaders} × "
            f"{word_bits}-bit words, redistributes them with warp shuffles, and "
            f"gives each of {consumers} consumer lanes its own values to decode."
        )
        access_pipeline = f"shuffle chunk x{packet}"

    if multiply == "fused":
        multiply_text = (
            "The two stored logarithms are added as widened integers, so the "
            "product is formed before any conversion and only the result is "
            "decoded — one decode per product instead of two."
        )
        multiply_pipeline = "fused log-add → 1 decode"
    else:
        multiply_text = (
            "Both operands are decoded to the arithmetic type first and then "
            "multiplied normally."
        )
        multiply_pipeline = "2 decodes → multiply"

    scale = 1 << fraction_bits
    if decoder == "reference_exp2":
        decoder_text = (
            f"Evaluates exp2 of the stored logarithm divided by {scale}, using the "
            "arithmetic type's own exp2."
        )
    elif decoder == "ex2_approx":
        decoder_text = (
            "Uses the hardware ex2.approx.f32 instruction instead of an exact exp2, "
            "trading accuracy for a single-instruction exponential."
        )
    elif decoder.startswith("full_lut"):
        decoder_text = (
            f"Uses the whole {bits}-bit code as an index into a {1 << bits:,}-entry "
            "table holding the final value, so no exponential is evaluated."
        )
    elif decoder.startswith("fraction_lut"):
        where = {
            "fraction_lut_global": "read through the cached global path",
            "fraction_lut_shared": "staged in shared memory",
            "fraction_lut_warp": "distributed one entry per warp lane and fetched "
            "with a shuffle",
        }[decoder]
        decoder_text = (
            f"Splits the logarithm into its integer and {fraction_bits}-bit "
            f"fractional parts, looks the fraction up in a {scale:,}-entry table "
            f"({where}), then applies the integer part as an exponent adjustment."
        )
    elif decoder.startswith("split_"):
        order = {"split_linear": "linear", "split_quadratic": "quadratic",
                 "split_cubic": "cubic"}[decoder]
        decoder_text = (
            "Looks up a coarse exponential from the leading fraction bits and "
            f"reconstructs the remainder with a {order} polynomial, which keeps the "
            "table small for wide fractions."
        )
    elif decoder.startswith("pair_lut"):
        where = "shared memory" if decoder.endswith("shared") else "the cached global path"
        decoder_text = (
            f"Indexes a {1 << (2 * bits):,}-entry table with both {bits}-bit codes at "
            f"once, returning the finished product directly from {where}."
        )
    else:
        decoder_text = f"Uses the {decoder.replace('_', ' ')} decoder."

    pipeline = (
        f"{storage_pipeline} → {access_pipeline} → "
        f"{DECODER_LABELS.get(decoder, decoder)} → {multiply_pipeline} → "
        f"{row['arithmetic_type'].upper()} accumulate"
    )
    return f"{storage} {access_text} {decoder_text} {multiply_text}", pipeline


def explanations_section(index: LnsIndex, compute: str, format_name: str) -> str:
    blocks = []
    for kernel in KERNELS:
        for scope in SCOPES:
            selection = best_selection(index, compute, format_name, kernel, scope)
            if selection is None:
                continue
            explanation, pipeline = strategy_explanation(index, selection.row)
            blocks.append(
                '<article class="strategy-explanation">'
                f"<h3>{case_label(kernel, scope)}</h3>"
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


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def strategy_label(row: dict[str, str]) -> str:
    if row["decoder"] == "raw":
        return f"Raw {row['arithmetic_type'].upper()} baseline"
    layout_name = row["storage_layout"].title()
    if row["access_method"] == "scalar":
        access = "x1"
    elif row["access_method"] == "thread_packet":
        access = f"packet x{row['packet_values']}"
    else:
        access = f"shuffle x{row['packet_values']}"
    decoder = DECODER_LABELS.get(row["decoder"], row["decoder"].replace("_", " "))
    return f"{row['multiply_method']} · {layout_name} · {access} · {decoder}"


def plot_strategies(
    index: LnsIndex, compute: str, format_name: str, kernel: str, path: Path
) -> tuple[Path, list[dict[str, str]]]:
    selected = list(index.by_case.get((compute, format_name, kernel), ()))
    raw_by_case: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in index.raw_by_case.get((compute, kernel), ()):
        if row["format"] == f"raw_{compute}" and row["access_method"] == "scalar" \
                and int(row["packet_values"]) == 1:
            raw_by_case[(row["distribution"], int(row["N"]))].append(
                float(row["mean_ms"])
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
            "multiply_method": "baseline",
            "arithmetic_type": compute,
        }
        for (distribution, size), values in raw_by_case.items()
    )
    metadata: dict[str, dict[str, str]] = {}
    for row in selected:
        metadata.setdefault(row["strategy_id"], row)
    strategies = sorted(
        metadata,
        key=lambda sid: (
            metadata[sid]["multiply_method"],
            metadata[sid]["storage_layout"],
            metadata[sid]["access_method"],
            int(metadata[sid]["packet_values"]),
            metadata[sid]["decoder"],
        ),
    )
    figure, axis = plt.subplots(figsize=(13.8, 7.2))
    series: list[dict[str, str]] = []
    for distribution in DISTRIBUTIONS:
        for strategy_id in strategies:
            points = sorted(
                (
                    row
                    for row in selected
                    if row["strategy_id"] == strategy_id
                    and row["distribution"] == distribution
                ),
                key=lambda row: int(row["N"]),
            )
            if not points:
                continue
            row = metadata[strategy_id]
            packet = int(row["packet_values"])
            is_baseline = row["decoder"] == "raw"
            linestyle = (
                "-"
                if is_baseline
                else ":"
                if row["access_method"] == "scalar"
                else "--"
                if packet == 2
                else "-."
                if packet == 4
                else "-"
            )
            line = axis.plot(
                [int(p["N"]) for p in points],
                [float(p["median_ms"]) for p in points],
                color=DECODER_COLORS.get(row["decoder"], "#333333"),
                linestyle=linestyle,
                linewidth=2.5 if is_baseline else 1.65,
                marker=(
                    "D"
                    if is_baseline
                    else "o"
                    if row["multiply_method"] == "ordinary"
                    else "^"
                ),
                markerfacecolor=(
                    DECODER_COLORS.get(row["decoder"], "#333333")
                    if row["storage_layout"] in {"dense", "baseline"}
                    else "white"
                ),
                markersize=4.0,
                alpha=0.86,
            )[0]
            line.set_gid(f"lns-series-{len(series)}")
            series.append(
                {
                    "distribution": distribution,
                    "strategy_id": strategy_id,
                    "layout": row["storage_layout"],
                    "multiply": row["multiply_method"],
                    "access": f"{row['access_method']}-x{packet}",
                    "decoder": row["decoder"],
                    "label": strategy_label(row),
                }
            )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Reduction length N")
    axis.set_ylabel("Complete kernel time (ms)")
    axis.set_title(
        f"{lns_plain_label(index, format_name)} → {compute.upper()} · {kernel.upper()}",
        fontsize=15,
    )
    base.format_axis_labels(axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return path, series


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


def chart_document(
    svg_path: Path,
    metadata: Sequence[dict[str, str]],
    index: LnsIndex,
    compute: str,
    format_name: str,
    kernel: str,
) -> str:
    svg = svg_path.read_text(encoding="utf-8")
    svg = svg[svg.index("<svg ") :]
    for position, item in enumerate(metadata):
        opening = f'<g id="lns-series-{position}">'
        replacement = (
            f'<g id="lns-series-{position}" data-series="true" '
            f'data-distribution="{item["distribution"]}" '
            f'data-layout="{item["layout"]}" data-multiply="{item["multiply"]}" '
            f'data-access="{item["access"]}" data-decoder="{item["decoder"]}" '
            f'data-strategy="{html.escape(item["strategy_id"])}">'
            f'<title>{html.escape(item["label"])}</title>'
        )
        if opening not in svg:
            raise ValueError(f"missing SVG series {position} in {svg_path}")
        svg = svg.replace(opening, replacement, 1)

    strategy_rows: dict[str, dict[str, str]] = {}
    for item in metadata:
        strategy_rows.setdefault(item["strategy_id"], item)
    distribution_controls = "".join(
        filter_option(
            "distribution",
            distribution,
            "U(0,1)" if distribution == "uniform_0_1" else "N(0,1)",
            radio=True,
            checked=distribution == "normal_0_1",
        )
        for distribution in DISTRIBUTIONS
    )
    layouts = sorted({item["layout"] for item in metadata if item["layout"] != "baseline"})
    layout_controls = "".join(
        filter_option("layout", layout, layout.title()) for layout in layouts
    )
    multiplies = sorted(
        {item["multiply"] for item in metadata if item["multiply"] != "baseline"}
    )
    multiply_controls = "".join(
        filter_option("multiply", multiply, multiply.title()) for multiply in multiplies
    )
    accesses = sorted(
        {item["access"] for item in metadata if item["access"] != "baseline-x1"}
    )
    access_controls = "".join(
        filter_option(
            "access",
            access,
            access.replace("thread_packet", "Packet")
            .replace("cooperative_shuffle", "Shuffle")
            .replace("scalar", "Scalar")
            .replace("-x", " x"),
        )
        for access in accesses
    )
    decoders = sorted({item["decoder"] for item in metadata})
    decoder_controls = "".join(
        filter_option("decoder", decoder, DECODER_LABELS.get(decoder, decoder))
        for decoder in decoders
    )
    strategy_controls = "".join(
        filter_option("strategy", strategy_id, item["label"])
        for strategy_id, item in strategy_rows.items()
    )
    root_id = f"lns-chart-{compute}-{format_name}-{kernel}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{lns_plain_label(index, format_name)} {kernel.upper()} strategies</title>
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
<fieldset><legend>Multiplication</legend><div class="options">{multiply_controls}</div></fieldset>
<fieldset><legend>Access</legend><div class="options">{access_controls}</div></fieldset>
<fieldset><legend>Decoder</legend><div class="options">{decoder_controls}</div></fieldset>
<details><summary>Individual strategies</summary><div class="options">{strategy_controls}</div></details>
</section><p class="status" aria-live="polite"></p>{svg}</main>
<script>
const root=document.getElementById('{root_id}');
const controls=[...root.querySelectorAll('input')]; const series=[...root.querySelectorAll('[data-series]')];
function enabled(kind,value){{const input=controls.find(item=>item.dataset.filter===kind&&item.value===value);return input?input.checked:true;}}
function reportHeight(){{if(window.parent!==window)window.parent.postMessage({{type:'performance-chart-height',height:Math.ceil(root.getBoundingClientRect().bottom+2)}},'*');}}
function update(){{const distribution=root.querySelector('input[name="distribution"]:checked').value;let visible=0;for(const line of series){{const show=line.dataset.distribution===distribution&&enabled('layout',line.dataset.layout)&&enabled('multiply',line.dataset.multiply)&&enabled('access',line.dataset.access)&&enabled('decoder',line.dataset.decoder)&&enabled('strategy',line.dataset.strategy);line.classList.toggle('hidden',!show);line.setAttribute('aria-hidden',String(!show));if(show)visible++;}}root.querySelector('.status').textContent=`${{visible}} strategies visible · lower is faster`;requestAnimationFrame(reportHeight);}}
root.addEventListener('change',update);window.addEventListener('load',update);window.addEventListener('message',event=>{{if(event.data?.type==='request-performance-chart-height')reportHeight();}});if('ResizeObserver'in window)new ResizeObserver(reportHeight).observe(root);
</script></body></html>"""


def chart_sections(index: LnsIndex, compute: str, format_name: str) -> str:
    blocks = []
    for kernel in KERNELS:
        source = f"interactive/lns-{compute}-{format_name}-{kernel}.html"
        blocks.append(
            f"<h3>{kernel.upper()}</h3>"
            '<figure class="interactive-figure">'
            f'<iframe class="interactive-chart-frame question-chart-frame" '
            f'data-performance-chart src="{source}" '
            f'title="Interactive {lns_plain_label(index, format_name)} '
            f'{kernel.upper()} strategy timing"></iframe>'
            f'<figcaption><a href="{source}">Open the chart full size</a>.</figcaption>'
            "</figure>"
        )
    return (
        '<section class="graph-section question-section"><h2>'
        "How do all strategies scale with N?</h2>" + "".join(blocks) + "</section>"
    )


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def lns_navigation(index: LnsIndex, compute: str, current: str) -> str:
    """Mirrors the IEEE conversion subnav so both families share one layout.

    The shared stylesheet pins the overview link across a fixed number of grid
    rows, so the group count travels as a custom property instead.
    """
    overview = lns_overview_filename(compute)
    overview_active = (
        ' aria-current="page" class="conversion-overview-link active"'
        if current == overview
        else ' class="conversion-overview-link"'
    )
    groups: dict[int, list[str]] = defaultdict(list)
    for format_name in index.formats(compute):
        groups[index.layouts[format_name].bits].append(format_name)
    rendered = []
    for bits in sorted(groups):
        links = []
        for format_name in groups[bits]:
            filename = lns_page_filename(compute, format_name)
            active = (
                ' aria-current="page" class="active"' if filename == current else ""
            )
            links.append(
                f'<a href="{html.escape(filename)}"{active}>'
                f"{lns_label(index, format_name)}</a>"
            )
        group_id = f"lns-{compute}-formats-{bits}-bit"
        rendered.append(
            '<div class="conversion-format-group" role="group" '
            f'aria-labelledby="{group_id}">'
            f'<span class="conversion-format-group-label" id="{group_id}">'
            f"{bits}-bit</span>"
            f'<span class="conversion-format-links">{"".join(links)}</span></div>'
        )
    return (
        '<nav class="subnav conversion-subnav lns-subnav" '
        f'style="--lns-format-groups: {len(groups)}" '
        f'aria-label="LNS to {compute.upper()} arithmetic formats">'
        f'<a href="{html.escape(overview)}"{overview_active}>Overview</a>'
        f'{"".join(rendered)}</nav>'
    )


def page_body(
    index: LnsIndex,
    ieee: IeeeIndex,
    run_dir: Path,
    output_dir: Path,
    compute: str,
    format_name: str,
) -> str:
    filename = lns_page_filename(compute, format_name)
    raw_prefix = Path(
        os.path.relpath(run_dir / "full", output_dir)
    ).as_posix()
    return (
        lns_navigation(index, compute, filename)
        + '<section class="experiment-status completed-section">'
        '<span class="status-badge">H200 measured</span>'
        f"<strong>Experiment 022 · {html.escape(run_dir.name)}</strong></section>"
        + f'<h2 class="question-format-heading">{lns_label(index, format_name)}</h2>'
        + usefulness_table(index, compute, format_name)
        + precise_peer_table(index, compute, format_name)
        + ieee_comparison_table(index, ieee, compute, format_name)
        + fusion_table(index, compute, format_name)
        + padded_table(index, compute, format_name)
        + packing_table(index, compute, format_name)
        + same_bit_tables(index, compute, format_name)
        + explanations_section(index, compute, format_name)
        + chart_sections(index, compute, format_name)
        + '<section class="text-section"><h2>Raw data</h2><ul><li><a href="'
        f'{raw_prefix}/timing_summary.csv">Unified LNS timing summary</a></li>'
        f'<li><a href="{raw_prefix}/raw_anchor_samples.csv">Raw arithmetic anchors'
        "</a></li></ul></section>"
    )


def overview_body(
    index: LnsIndex, run_dir: Path, compute: str, built: Sequence[str]
) -> str:
    groups: dict[int, list[str]] = defaultdict(list)
    for format_name in index.formats(compute):
        groups[index.layouts[format_name].bits].append(format_name)
    sections = []
    for bits in sorted(groups):
        cards = []
        for format_name in groups[bits]:
            filename = lns_page_filename(compute, format_name)
            layout = index.layouts[format_name]
            if format_name in built:
                cards.append(
                    f'<a class="report-link" href="{html.escape(filename)}">'
                    f"<strong>{lns_label(index, format_name)}</strong>"
                    f"<span>{layout.ieee_label} allocation · question report</span></a>"
                )
            else:
                cards.append(
                    '<a class="report-link pending-report-link" '
                    f'href="{html.escape(filename)}">'
                    f"<strong>{lns_label(index, format_name)}</strong>"
                    f"<span>{layout.ieee_label} allocation · page pending</span></a>"
                )
        sections.append(
            '<section class="text-section"><h2>'
            f"{bits}-bit formats</h2>"
            f'<div class="report-link-grid">{"".join(cards)}</div></section>'
        )
    return (
        lns_navigation(index, compute, lns_overview_filename(compute))
        + '<section class="experiment-status completed-section">'
        '<span class="status-badge">H200 measured</span>'
        f"<strong>Experiment 022 · {html.escape(run_dir.name)}</strong></section>"
        + '<section class="text-section"><h2>What these formats are</h2>'
        "<p>An <code>LNS&lt;B,R&gt;</code> value stores one sign bit and a signed "
        "<code>B-1</code> bit fixed-point logarithm with <code>R</code> fractional "
        "bits. The integer part <code>I = B-1-R</code> sets dynamic range and "
        "<code>R</code> sets relative precision, so every format has an exact IEEE "
        "counterpart <code>E&lt;I&gt;M&lt;R&gt;</code> named on its page.</p>"
        "<p>Fused multiplication adds the two stored logarithms as widened integers "
        "and decodes only the product, replacing two decodes with one.</p></section>"
        + "".join(sections)
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


EXTRA_CSS = """
/* Question-led LNS pages. */
.lns-match-note { color: var(--muted); margin: 0 0 14px; max-width: 70ch; }
/* The IEEE subnav pins its overview link across 17 width groups; LNS has
   fewer, so the count arrives as a custom property on the nav element. */
.lns-subnav .conversion-overview-link { grid-row: 1 / span var(--lns-format-groups, 17); }
.lns-subnav .conversion-format-group-label { flex-basis: 54px; }
@media (max-width: 760px) {
  .lns-subnav .conversion-overview-link { grid-row: auto; justify-self: start; }
  .lns-subnav .conversion-format-group-label { flex-basis: auto; }
}
"""


def main() -> None:
    args = parse_args()
    run_dir = args.lns_run_dir or newest_run(
        args.lns_results_root, "full/timing_summary.csv"
    )
    ieee_run = newest_run(
        args.ieee_results_root, "unified_core/full/timing_summary.csv"
    )
    output_dir = args.output_dir
    if not (output_dir / "report.css").is_file():
        raise SystemExit(
            f"{output_dir} is not a built report; run the base report builder first"
        )

    index = LnsIndex(
        read_rows(run_dir / "full/timing_summary.csv"),
        read_rows(run_dir / "full/raw_anchor_samples.csv"),
    )
    ieee = IeeeIndex(read_rows(ieee_run / "unified_core/full/timing_summary.csv"))
    manifest = read_manifest(output_dir / "report_manifest.txt")

    requested = tuple(f for f in args.formats.split(",") if f)
    for format_name in requested:
        if format_name not in index.layouts:
            raise SystemExit(f"{format_name} is not present in {run_dir}")

    assets = output_dir / "assets"
    interactive = output_dir / "interactive"
    assets.mkdir(parents=True, exist_ok=True)
    interactive.mkdir(parents=True, exist_ok=True)

    pages = 0
    for compute in COMPUTES:
        available = index.formats(compute)
        build = tuple(f for f in available if not requested or f in requested)
        for position, format_name in enumerate(build, start=1):
            print(
                f"[{compute} {position}/{len(build)}] "
                f"{lns_plain_label(index, format_name)}",
                flush=True,
            )
            for kernel in KERNELS:
                svg_path, chart_metadata = plot_strategies(
                    index,
                    compute,
                    format_name,
                    kernel,
                    assets / f"lns-{compute}-{format_name}-{kernel}.svg",
                )
                (
                    interactive / f"lns-{compute}-{format_name}-{kernel}.html"
                ).write_text(
                    chart_document(
                        svg_path, chart_metadata, index, compute, format_name, kernel
                    ),
                    encoding="utf-8",
                )
            filename = lns_page_filename(compute, format_name)
            document = base.page_document(
                filename=filename,
                title=(
                    f"{lns_plain_label(index, format_name)} → "
                    f"{compute.upper()} arithmetic"
                ),
                intro=(
                    "Question-led H200 comparison of ordinary and fused logarithmic "
                    "multiplication, dense and padded storage, scalar and packed "
                    "access, and every qualified decoder."
                ),
                body=page_body(index, ieee, run_dir, output_dir, compute, format_name),
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
            pages += 1

        overview = lns_overview_filename(compute)
        (output_dir / overview).write_text(
            base.page_document(
                filename=overview,
                title=f"LNS → {compute.upper()}",
                intro=(
                    "Logarithmic storage measured against raw arithmetic, against "
                    "more precise logarithmic formats, and against the IEEE format "
                    "with the same range and resolution."
                ),
                body=overview_body(index, run_dir, compute, build),
                performance_run_name=manifest.get("performance_run", "unknown"),
                accuracy_run_name=manifest.get("accuracy_run", "unknown"),
                strategy_run_name=manifest.get("strategy_run", "unknown"),
                all_strategy_run_name=manifest.get("all_strategy_run", "unknown"),
                expanded_strategy_run_name=(
                    manifest.get("expanded_strategy_run", "unknown")
                    + f" + {run_dir.name}"
                ),
            ),
            encoding="utf-8",
        )

    css_path = output_dir / "report.css"
    css = css_path.read_text(encoding="utf-8")
    marker = "/* Question-led LNS pages. */"
    if marker in css:
        css = css[: css.index(marker)]
    css_path.write_text(css.rstrip() + "\n" + EXTRA_CSS, encoding="utf-8")

    manifest_path = output_dir / "report_manifest.txt"
    current = "\n".join(
        line
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("lns_strategy_run=", "lns_question_pages="))
    ).rstrip()
    manifest_path.write_text(
        current
        + f"\nlns_strategy_run={run_dir.name}"
        + f"\nlns_question_pages={pages}\n",
        encoding="utf-8",
    )
    print(f"Updated {pages} LNS question pages in {output_dir}")


if __name__ == "__main__":
    main()
