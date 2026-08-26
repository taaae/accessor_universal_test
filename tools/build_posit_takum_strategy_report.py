#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import statistics
from collections import Counter
from pathlib import Path


MAIN_DISTRIBUTIONS = (
    "field_balanced_finite",
    "paired_log_uniform_finite",
)
KERNELS = ("dot", "gemv")
FAMILIES = ("posit", "takum", "takum_log")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def short_format(name: str) -> str:
    names = {
        "posit8_es0": "posit<8,0>",
        "posit14_es1": "posit<14,1>",
        "posit16_es1": "posit<16,1>",
        "posit32_es2": "posit<32,2>",
        "takum8": "takum<8>",
        "takum14": "takum<14>",
        "takum16": "takum<16>",
        "takum32": "takum<32>",
        "takum_log8": "takum_log<8>",
        "takum_log14": "takum_log<14>",
        "takum_log16": "takum_log<16>",
        "takum_log32": "takum_log<32>",
    }
    return names.get(name, name)


def short_strategy(name: str) -> str:
    names = {
        "direct": "direct",
        "full_lut_shared": "shared LUT",
        "full_lut_global": "global LUT",
        "native_scalar": "native",
        "direct_branchy": "direct branchy",
        "direct_masked": "direct masked",
        "subnormal_lut_global": "subnormal LUT",
        "prefix_lut_global": "prefix LUT",
    }
    return names.get(name, name)


def short_distribution(name: str) -> str:
    return {
        "field_balanced_finite": "field-balanced",
        "paired_log_uniform_finite": "paired log-uniform",
        "lut_scattered_control": "scattered",
        "lut_concentrated_control": "concentrated",
    }[name]


def ratio_cell(row: dict[str, str]) -> str:
    ratio = float(row["ratio"])
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    label = {
        "equivalent": "equivalent",
        "numerator_faster": "faster",
        "numerator_slower": "slower",
        "inconclusive": "inconclusive",
    }[row["classification"]]
    return f"{ratio:.3f} [{lower:.3f}, {upper:.3f}], {label}"


def winner_cell(
    winner: dict[str, str], comparison: dict[str, str]
) -> str:
    return (
        f"{short_strategy(winner['winning_strategy'])}, "
        f"{float(winner['median_ms']):.3f} ms; {ratio_cell(comparison)} vs direct"
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def ratio_stats(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    values = [float(row["ratio"]) for row in rows]
    return min(values), statistics.median(values), max(values)


def interval_chart(
    title: str,
    groups: list[tuple[str, float, float, float]],
    *,
    lower: float | None = None,
    upper: float | None = None,
    reference: float | None = 1.0,
    band: tuple[float, float] | None = None,
) -> str:
    values = [value for _, minimum, median, maximum in groups for value in (minimum, median, maximum)]
    if reference is not None:
        values.append(reference)
    if band is not None:
        values.extend(band)
    data_min = min(values) if lower is None else lower
    data_max = max(values) if upper is None else upper
    span = max(data_max - data_min, 1e-9)
    if lower is None:
        data_min -= span * 0.06
    if upper is None:
        data_max += span * 0.06

    width = 820
    left = 190
    right = 32
    top = 42
    row_height = 52
    bottom = 48
    plot_width = width - left - right
    height = top + row_height * len(groups) + bottom

    def x(value: float) -> float:
        return left + (value - data_min) / (data_max - data_min) * plot_width

    pieces = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect class="pt-chart-frame" x="{left}" y="{top - 18}" width="{plot_width}" height="{row_height * len(groups) + 14}"/>',
    ]
    if band is not None:
        band_x = x(band[0])
        pieces.append(
            f'<rect class="pt-equivalence-band" x="{band_x:.2f}" y="{top - 18}" '
            f'width="{x(band[1]) - band_x:.2f}" height="{row_height * len(groups) + 14}"/>'
        )
    ticks = [data_min + (data_max - data_min) * index / 5 for index in range(6)]
    for tick in ticks:
        tick_x = x(tick)
        pieces.extend(
            [
                f'<line class="pt-grid-line" x1="{tick_x:.2f}" y1="{top - 18}" x2="{tick_x:.2f}" y2="{top + row_height * len(groups) - 4}"/>',
                f'<text class="pt-axis-label" x="{tick_x:.2f}" y="{height - 16}" text-anchor="middle">{tick:.2f}</text>',
            ]
        )
    if reference is not None:
        pieces.append(
            f'<line class="pt-reference-line" x1="{x(reference):.2f}" y1="{top - 18}" '
            f'x2="{x(reference):.2f}" y2="{top + row_height * len(groups) - 4}"/>'
        )
    for index, (label, minimum, median, maximum) in enumerate(groups):
        y = top + index * row_height + 8
        pieces.extend(
            [
                f'<text class="pt-row-label" x="{left - 12}" y="{y + 4}" text-anchor="end">{html.escape(label)}</text>',
                f'<line class="pt-range-line" x1="{x(minimum):.2f}" y1="{y}" x2="{x(maximum):.2f}" y2="{y}"/>',
                f'<line class="pt-range-cap" x1="{x(minimum):.2f}" y1="{y - 6}" x2="{x(minimum):.2f}" y2="{y + 6}"/>',
                f'<line class="pt-range-cap" x1="{x(maximum):.2f}" y1="{y - 6}" x2="{x(maximum):.2f}" y2="{y + 6}"/>',
                f'<circle class="pt-median-point" cx="{x(median):.2f}" cy="{y}" r="5"/>',
                f'<text class="pt-value-label" x="{x(maximum) + 8:.2f}" y="{y + 4}">{median:.3f}</text>',
            ]
        )
    pieces.append("</svg>")
    return "".join(pieces)


def outcome_chart(counts: Counter[str]) -> str:
    categories = [
        ("equivalent", "Equivalent", "pt-outcome-equivalent"),
        ("inconclusive", "Inconclusive", "pt-outcome-inconclusive"),
        ("numerator_slower", "Slower", "pt-outcome-slower"),
    ]
    total = sum(counts[key] for key, _, _ in categories)
    width = 820
    left = 26
    bar_y = 52
    bar_width = width - 2 * left
    x = left
    pieces = [
        f'<svg viewBox="0 0 {width} 150" role="img" aria-label="Outcome counts across {total} alternative versus IEEE cases">',
        "<title>Alternative versus IEEE outcome counts</title>",
    ]
    for key, label, css_class in categories:
        count = counts[key]
        segment = bar_width * count / total
        pieces.append(
            f'<rect class="{css_class}" x="{x:.2f}" y="{bar_y}" width="{segment:.2f}" height="34"/>'
        )
        if segment >= 70:
            pieces.append(
                f'<text class="pt-outcome-label" x="{x + segment / 2:.2f}" y="{bar_y + 22}" text-anchor="middle">{count}</text>'
            )
        x += segment
    legend_x = left
    for key, label, css_class in categories:
        count = counts[key]
        pieces.extend(
            [
                f'<rect class="{css_class}" x="{legend_x}" y="108" width="12" height="12"/>',
                f'<text class="pt-axis-label" x="{legend_x + 18}" y="119">{html.escape(label)} {count}</text>',
            ]
        )
        legend_x += 180
    pieces.append("</svg>")
    return "".join(pieces)


def html_table(headers: list[str], rows: list[list[str]], *, css_class: str = "") -> str:
    class_name = f"strategy-table pt-table {css_class}".strip()
    heading = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table class="{class_name}"><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'


def classification_badge(classification: str) -> str:
    labels = {
        "equivalent": "Equivalent",
        "inconclusive": "Inconclusive",
        "numerator_faster": "Faster",
        "numerator_slower": "Slower",
    }
    return f'<span class="pt-result pt-{classification}">{labels[classification]}</span>'


def build_html_report(analysis_dir: Path, output: Path) -> None:
    comparisons = read_csv(analysis_dir / "case_comparisons.csv")
    timing = read_csv(analysis_dir / "timing_summary.csv")
    q1 = [row for row in comparisons if row["question"] == "1"]
    q2 = [row for row in comparisons if row["question"] == "2"]
    q3 = [row for row in comparisons if row["question"] == "3"]
    q4 = [row for row in comparisons if row["question"] == "4"]
    q4_counts = Counter(row["classification"] for row in q4)

    q1_groups: list[tuple[str, float, float, float]] = []
    for bits in ("8", "14"):
        for strategy in ("full_lut_shared", "full_lut_global"):
            rows = [
                row for row in q1
                if row["bits"] == bits and row["numerator_strategy"] == strategy
            ]
            q1_groups.append((f"{bits}-bit {short_strategy(strategy)}", *ratio_stats(rows)))

    q2_groups: list[tuple[str, float, float, float]] = []
    for bits in ("8", "14"):
        for strategy in ("full_lut_shared", "full_lut_global"):
            rows = [
                row for row in q2
                if row["bits"] == bits and row["numerator_strategy"] == strategy
            ]
            q2_groups.append((f"{bits}-bit {short_strategy(strategy)}", *ratio_stats(rows)))

    q3_groups: list[tuple[str, float, float, float]] = []
    for fmt in ("posit16_es1", "takum16", "takum_log16"):
        rows = [row for row in q3 if row["numerator_format"] == fmt]
        q3_groups.append((short_format(fmt), *ratio_stats(rows)))

    width_rows: list[list[str]] = []
    for bits in ("8", "14", "16", "32"):
        rows = [row for row in q4 if row["bits"] == bits]
        counts = Counter(row["classification"] for row in rows)
        minimum, _, maximum = ratio_stats(rows)
        width_rows.append(
            [
                bits,
                str(len(rows)),
                str(counts["equivalent"]),
                str(counts["inconclusive"]),
                str(counts["numerator_slower"]),
                f"{minimum:.3f} to {maximum:.3f}",
            ]
        )

    direct32_rows: list[list[str]] = []
    for fmt in ("posit32_es2", "takum32", "takum_log32"):
        for arithmetic in ("fp32", "fp64"):
            rows = [
                row for row in timing
                if row["format"] == fmt
                and row["arithmetic"] == arithmetic
                and row["strategy"] == "direct"
                and row["distribution"] in MAIN_DISTRIBUTIONS
            ]
            values = [float(row["median_ms"]) for row in rows]
            direct32_rows.append(
                [short_format(fmt), arithmetic.upper(), f"{min(values):.3f} to {max(values):.3f} ms"]
            )

    detail_rows: list[list[str]] = []
    for row in sorted(
        q4,
        key=lambda item: (
            int(item["bits"]), item["numerator_format"], item["arithmetic"],
            item["distribution"], item["kernel"],
        ),
    ):
        reference = (
            f"{html.escape(row['denominator_format'])} "
            f"<code>{html.escape(short_strategy(row['denominator_strategy']))}</code>"
        )
        detail_rows.append(
            [
                html.escape(short_format(row["numerator_format"])),
                html.escape(row["arithmetic"].upper()),
                html.escape(short_distribution(row["distribution"])),
                html.escape(row["kernel"].upper()),
                f"{float(row['ratio']):.3f}",
                f"[{float(row['ci_lower']):.3f}, {float(row['ci_upper']):.3f}]",
                classification_badge(row["classification"]),
                reference,
            ]
        )

    q1_chart = interval_chart(
        "Full LUT time divided by direct decoder time",
        q1_groups,
        lower=0.10,
        upper=1.95,
    )
    q2_chart = interval_chart(
        "Alternative LUT divided by IEEE LUT",
        q2_groups,
        lower=0.97,
        upper=1.03,
        band=(0.97, 1.03),
    )
    q3_chart = interval_chart(
        "16-bit global LUT divided by direct decoder time",
        q3_groups,
        lower=0.15,
        upper=0.70,
    )
    q4_chart = outcome_chart(q4_counts)

    styles = """
.pt-hero { margin-bottom: 34px; }
.pt-kicker { color: var(--link); font-size: .78rem; font-weight: 650; letter-spacing: .05em; margin: 0 0 7px; text-transform: uppercase; }
.pt-verdict { background: var(--surface); border: 1px solid var(--border); border-left: 5px solid var(--link); border-radius: 8px; margin: 24px 0 30px; max-width: 1050px; padding: 18px 20px; }
.pt-verdict strong { display: block; font-size: 1.1rem; margin-bottom: 5px; }
.pt-verdict p { margin: 0; max-width: 90ch; }
.pt-summary { display: grid; gap: 12px; grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 20px 0 42px; }
.pt-summary > div { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 15px 16px; }
.pt-summary span { color: var(--muted); display: block; font-size: .82rem; }
.pt-summary strong { display: block; font-size: 1.45rem; margin: 3px 0; }
.pt-summary small { color: var(--muted); }
.pt-question { border-top: 1px solid var(--border); margin-top: 46px; padding-top: 32px; }
.pt-question-number { color: var(--link); font-size: .8rem; font-weight: 650; letter-spacing: .04em; text-transform: uppercase; }
.pt-answer { font-size: 1.08rem; max-width: 88ch; }
.pt-chart-grid { display: grid; gap: 18px; grid-template-columns: minmax(0, 1.35fr) minmax(270px, .65fr); margin: 22px 0; }
.pt-chart-grid > * { min-width: 0; }
.pt-chart { min-width: 0; }
.pt-chart svg { background: var(--surface); border: 1px solid var(--border); display: block; height: auto; width: 100%; }
.pt-chart figcaption { max-width: 82ch; }
.pt-chart-frame { fill: transparent; stroke: var(--border); }
.pt-grid-line { stroke: var(--border); stroke-width: 1; }
.pt-reference-line { stroke: var(--fg); stroke-dasharray: 5 5; stroke-width: 1.5; }
.pt-equivalence-band { fill: color-mix(in srgb, #1b8a4b 12%, transparent); }
.pt-range-line, .pt-range-cap { stroke: var(--link); stroke-width: 3; }
.pt-range-cap { stroke-width: 1.5; }
.pt-median-point { fill: var(--surface); stroke: var(--link); stroke-width: 3; }
.pt-row-label, .pt-axis-label, .pt-value-label { fill: var(--fg); font: 13px system-ui, sans-serif; }
.pt-axis-label { fill: var(--muted); font-size: 12px; }
.pt-value-label { fill: var(--muted); font-size: 12px; }
.pt-outcome-equivalent { fill: #2f8f5b; }
.pt-outcome-inconclusive { fill: #c58b18; }
.pt-outcome-slower { fill: #bd4a4a; }
.pt-outcome-label { fill: white; font: 500 13px system-ui, sans-serif; }
.pt-fact-list { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; margin: 0; padding: 16px 20px 16px 38px; }
.pt-fact-list li { margin: 7px 0; }
.pt-table { width: 100%; }
.pt-table th { width: auto; }
.pt-table td { font-variant-numeric: tabular-nums; }
.pt-result { border-radius: 999px; display: inline-block; font-size: .75rem; font-weight: 650; padding: 2px 8px; white-space: nowrap; }
.pt-equivalent { background: color-mix(in srgb, #1b8a4b 16%, transparent); color: #146b3a; }
.pt-inconclusive { background: color-mix(in srgb, #c58b18 18%, transparent); color: #805809; }
.pt-numerator_slower { background: color-mix(in srgb, #c43838 15%, transparent); color: #a12626; }
.pt-numerator_faster { background: color-mix(in srgb, #1b8a4b 16%, transparent); color: #146b3a; }
.pt-files { display: grid; gap: 10px; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.pt-file { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; color: var(--fg); padding: 12px 14px; text-decoration: none; }
.pt-file:hover, .pt-file:focus-visible { border-color: var(--link); }
.pt-file span { color: var(--muted); display: block; font-size: .82rem; }
.pt-details { margin-top: 20px; }
.pt-details summary { color: var(--link); cursor: pointer; font-weight: 500; margin-bottom: 12px; }
.pt-method-grid { display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.pt-method-grid > div { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
.pt-method-grid h3 { margin-top: 0; }
@media (prefers-color-scheme: dark) {
  .pt-equivalent, .pt-numerator_faster { color: #82d8a5; }
  .pt-inconclusive { color: #f4c66e; }
  .pt-numerator_slower { color: #ff9b9b; }
}
@media (max-width: 900px) {
  .pt-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pt-chart-grid { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 620px) {
  .pt-summary, .pt-method-grid, .pt-files { grid-template-columns: 1fr; }
}
"""

    navigation = "".join(
        [
            '<a href="index.html">Overview</a>',
            '<a href="conversion-strategies.html">IEEE conversion</a>',
            '<a href="lns-fp64.html">LNS conversion</a>',
            '<a href="ieee-lns-summary.html">IEEE, LNS summary</a>',
            '<a href="posit-takum.html" aria-current="page" class="active">Posit, takum</a>',
        ]
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Posit and takum conversion on H200</title>
<link rel="stylesheet" href="report.css">
<style>{styles}</style>
</head>
<body>
<header><div class="shell"><a class="brand" href="index.html">H200 storage-format report</a><nav aria-label="Report sections">{navigation}</nav></div></header>
<main class="shell">
<section class="pt-hero">
<p class="pt-kicker">Experiment 024 · run_20260826T183425Z</p>
<h1>Posit and takum conversion on H200</h1>
<p class="lead">Scalar storage conversion for posit, linear takum, and logarithmic takum. The benchmark covers 8, 14, 16, and 32 bits with FP32 and FP64 arithmetic in DOT and GEMV.</p>
<div class="pt-verdict"><strong>Fast LUT conversion does not give these formats a speed advantage over IEEE.</strong><p>Full tables are the right decoder through 16 bits. At 8 bits they tie the fastest same-width IEEE paths. At 14 bits and above, the alternatives lose. The 32-bit direct decoders are several times slower.</p></div>
<div class="pt-summary">
<div><span>Low-bit winner</span><strong>Full LUT</strong><small>Shared at 8 bits, global at 14</small></div>
<div><span>LUT content controls</span><strong>96 / 96</strong><small>Equivalent inside the 3% band</small></div>
<div><span>16-bit strategy</span><strong>24 / 24</strong><small>Global LUT beats direct</small></div>
<div><span>Faster than IEEE</span><strong>0 / 96</strong><small>16 equivalent, 2 inconclusive</small></div>
</div>
</section>

<section class="pt-question">
<span class="pt-question-number">Question 1</span>
<h2>Is a full LUT best at low bit counts?</h2>
<p class="pt-answer"><strong>Yes.</strong> Shared LUT won all 24 main 8-bit cases. Global LUT won all 24 main 14-bit cases. The oversized 14-bit shared table sometimes lost to direct decoding, but that never changed the selected strategy.</p>
<div class="pt-chart-grid"><figure class="pt-chart">{q1_chart}<figcaption>Alternative LUT time divided by direct-decoder time. Values below 1 are faster. The point is the median; the line spans the observed cases.</figcaption></figure><ul class="pt-fact-list"><li>87 of 96 individual LUT comparisons beat direct.</li><li>The nine losses were 14-bit FP64 shared-LUT cases.</li><li>Staging a 128 KiB table per block caused those losses.</li><li>Global LUT remained faster than direct in every 14-bit main case.</li></ul></div>
</section>

<section class="pt-question">
<span class="pt-question-number">Question 2</span>
<h2>Does LUT content change conversion speed?</h2>
<p class="pt-answer"><strong>No measurable effect.</strong> The controls reuse the same index stream and compiled table-only kernel. Changing entries between IEEE, posit, and takum values left every paired result inside the predeclared 3% equivalence band.</p>
<div class="pt-chart-grid"><figure class="pt-chart">{q2_chart}<figcaption>Alternative-table time divided by IEEE-table time. The shaded region is the 0.97 to 1.03 equivalence band.</figcaption></figure><ul class="pt-fact-list"><li>All 96 controls were equivalent.</li><li>The observed ratios ran from 0.989 to 1.009.</li><li>The result held for both kernels, arithmetic types, widths, traces, and placements.</li><li>With indices fixed, table contents do not affect throughput.</li></ul></div>
</section>

<section class="pt-question">
<span class="pt-question-number">Question 3</span>
<h2>What wins after a shared full LUT becomes too large?</h2>
<p class="pt-answer">At 16 bits, use the global full LUT. It beat direct conversion in all 24 cases. At 32 bits, a complete table would occupy 16 GiB for FP32 or 32 GiB for FP64, so the experiment measured direct conversion only.</p>
<div class="pt-chart-grid"><figure class="pt-chart">{q3_chart}<figcaption>16-bit global-LUT time divided by direct-decoder time. Every measured case favors the global table.</figcaption></figure><div><h3>32-bit direct baseline</h3>{html_table(["Format", "Arithmetic", "Range across cases"], direct32_rows)}</div></div>
<p>This 32-bit result is a baseline, not proof that direct decoding beats every segmented or approximate decoder.</p>
</section>

<section class="pt-question">
<span class="pt-question-number">Question 4</span>
<h2>How do the best alternatives compare with same-width IEEE?</h2>
<p class="pt-answer"><strong>No alternative case was faster.</strong> The only ties occur at 8 bits. Past 8 bits, conversion cost removes the storage-width advantage before the kernel finishes.</p>
<div class="pt-chart-grid"><figure class="pt-chart">{q4_chart}<figcaption>Classification of the 96 best-alternative versus fastest-retained-IEEE comparisons.</figcaption></figure><div><h3>Outcome by width</h3>{html_table(["Bits", "Cases", "Equivalent", "Inconclusive", "Slower", "Ratio range"], width_rows)}</div></div>
<details class="pt-details"><summary>Show all 96 comparisons</summary>{html_table(["Alternative", "Arithmetic", "Distribution", "Kernel", "Ratio", "95% CI", "Result", "IEEE reference"], detail_rows, css_class="pt-detail-table")}</details>
<p>The main comparison uses each format's own safe input interval. Question 2 is the controlled conversion-only result because it holds the index stream and kernel constant.</p>
</section>

<section class="pt-question">
<span class="pt-question-number">Method</span>
<h2>What the benchmark measured</h2>
<div class="pt-method-grid">
<div><h3>Workload</h3><ul><li>One NVIDIA H200 NVL GPU</li><li>Scalar x1 storage access</li><li>DOT with N = 2<sup>27</sup></li><li>GEMV with M = 1024 and N = 65,536</li><li>FP32 and FP64 arithmetic</li><li>Field-balanced and paired log-uniform finite inputs</li></ul></div>
<div><h3>Timing and statistics</h3><ul><li>10 warmups and 30 samples per case</li><li>Paired round bootstrap confidence intervals</li><li>Alternative time divided by reference time</li><li>Equivalent only when the full interval lies inside 0.97 to 1.03</li><li>Encoding and input preparation outside timed kernels</li></ul></div>
</div>
</section>

<section class="pt-question">
<span class="pt-question-number">Validation</span>
<h2>Run integrity</h2>
<div class="pt-summary">
<div><span>Timed cases</span><strong>604</strong><small>18,120 successful samples</small></div>
<div><span>Decoder checks</span><strong>119 / 119</strong><small>All passed</small></div>
<div><span>Histogram rows</span><strong>23,668</strong><small>Zero infeasible cases</small></div>
<div><span>Independent review</span><strong>No findings</strong><small>Context-free CUDA review</small></div>
</div>
<p>The benchmark tests storage conversion followed by ordinary FP32 or FP64 arithmetic. It does not test quires, native posit or takum arithmetic, segmented 32-bit decoders, or an accuracy-matched application.</p>
</section>

<section class="pt-question">
<span class="pt-question-number">Files</span>
<h2>Reproduce or inspect the run</h2>
<div class="pt-files">
<a class="pt-file" href="../024_posit_takum_strategy_performance/run_20260826T183425Z/full/timing_samples.csv"><strong>Raw timings</strong><span>18,120 samples</span></a>
<a class="pt-file" href="../024_posit_takum_strategy_performance/run_20260826T183425Z/analysis/case_comparisons.csv"><strong>All comparisons</strong><span>Ratios and confidence intervals</span></a>
<a class="pt-file" href="../024_posit_takum_strategy_performance/run_20260826T183425Z/analysis/strategy_winners.csv"><strong>Strategy winners</strong><span>Best decoder per case</span></a>
<a class="pt-file" href="../024_posit_takum_strategy_performance/run_20260826T183425Z/full/decoder_validation.csv"><strong>Decoder validation</strong><span>119 checks</span></a>
<a class="pt-file" href="../024_posit_takum_strategy_performance/run_20260826T183425Z/full/histograms.csv"><strong>Input histograms</strong><span>Field and trace coverage</span></a>
<a class="pt-file" href="../024_posit_takum_strategy_performance/run_20260826T183425Z/run_manifest.txt"><strong>Run manifest</strong><span>Exact experiment settings</span></a>
<a class="pt-file" href="../024_posit_takum_strategy_performance/code_review.md"><strong>Independent review</strong><span>CUDA audit and resolution</span></a>
<a class="pt-file" href="../../docs/posit_takum_strategy_benchmark.md"><strong>Experiment specification</strong><span>Decisions fixed before implementation</span></a>
<a class="pt-file" href="../024_posit_takum_strategy_performance/summary.md"><strong>Markdown tables</strong><span>Machine-friendly companion report</span></a>
</div>
</section>
</main>
<footer><div class="shell">Experiment 024 · <code>run_20260826T183425Z</code> · one NVIDIA H200 NVL</div></footer>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def build_report(analysis_dir: Path, output: Path) -> None:
    winners = read_csv(analysis_dir / "strategy_winners.csv")
    comparisons = read_csv(analysis_dir / "case_comparisons.csv")
    timing = read_csv(analysis_dir / "timing_summary.csv")

    winner_by_case = {
        (
            row["format"],
            row["arithmetic"],
            row["distribution"],
            row["kernel"],
        ): row
        for row in winners
    }
    comparison_by_case = {
        (
            row["question"],
            row["numerator_format"],
            row["arithmetic"],
            row["distribution"],
            row["kernel"],
            row["numerator_strategy"],
        ): row
        for row in comparisons
    }
    timing_by_case = {
        (
            row["format"],
            row["arithmetic"],
            row["distribution"],
            row["kernel"],
            row["strategy"],
        ): row
        for row in timing
    }

    lines = [
        "# Posit and takum GPU storage-conversion results",
        "",
        "This report answers the four questions fixed before implementation. The run used one NVIDIA H200 NVL GPU, scalar x1 access, 30 timing samples after 10 warm-ups, DOT with N = 2^27, and GEMV with M = 1024 and N = 65,536. Ratios below are alternative time divided by reference time. Lower is faster. The brackets give the 95% bootstrap confidence interval. `Equivalent` means the full interval lies inside [0.97, 1.03].",
        "",
        "## Case tables",
        "",
        "### Question 1: low-bit strategy winner",
        "",
        "Every cell reports the case winner, its median kernel time, and its paired ratio to the direct decoder.",
        "",
    ]

    q1_rows: list[list[str]] = []
    low_formats = [
        "posit8_es0",
        "takum8",
        "takum_log8",
        "posit14_es1",
        "takum14",
        "takum_log14",
    ]
    for fmt in low_formats:
        for arithmetic in ("fp32", "fp64"):
            row = [short_format(fmt), arithmetic.upper()]
            for distribution in MAIN_DISTRIBUTIONS:
                for kernel in KERNELS:
                    winner = winner_by_case[(fmt, arithmetic, distribution, kernel)]
                    comparison = comparison_by_case[
                        (
                            "1",
                            fmt,
                            arithmetic,
                            distribution,
                            kernel,
                            winner["winning_strategy"],
                        )
                    ]
                    row.append(winner_cell(winner, comparison))
            q1_rows.append(row)
    lines.extend(
        markdown_table(
            [
                "Format",
                "Arithmetic",
                "Field DOT",
                "Field GEMV",
                "Log-uniform DOT",
                "Log-uniform GEMV",
            ],
            q1_rows,
        )
    )

    lines += [
        "",
        "### Question 2: alternative LUT divided by IEEE LUT",
        "",
        "These controls reuse the same raw index stream and compiled table-only kernel. Only table contents change. Each row covers one alternative family, width, arithmetic type, trace, and table placement.",
        "",
    ]
    q2 = [row for row in comparisons if row["question"] == "2"]
    q2_key = {
        (
            row["bits"],
            row["arithmetic"],
            row["distribution"],
            row["numerator_strategy"],
            row["numerator_format"],
            row["kernel"],
        ): row
        for row in q2
    }
    q2_rows: list[list[str]] = []
    format_by_family = {
        ("8", "posit"): "posit8_es0",
        ("8", "takum"): "takum8",
        ("8", "takum_log"): "takum_log8",
        ("14", "posit"): "posit14_es1",
        ("14", "takum"): "takum14",
        ("14", "takum_log"): "takum_log14",
    }
    for bits in ("8", "14"):
        for arithmetic in ("fp32", "fp64"):
            for trace in ("lut_scattered_control", "lut_concentrated_control"):
                for strategy in ("full_lut_shared", "full_lut_global"):
                    for family in FAMILIES:
                        fmt = format_by_family[(bits, family)]
                        dot = q2_key[(bits, arithmetic, trace, strategy, fmt, "dot")]
                        gemv = q2_key[(bits, arithmetic, trace, strategy, fmt, "gemv")]
                        q2_rows.append(
                            [
                                bits,
                                arithmetic.upper(),
                                short_distribution(trace),
                                short_strategy(strategy),
                                short_format(fmt),
                                ratio_cell(dot),
                                ratio_cell(gemv),
                            ]
                        )
    lines.extend(
        markdown_table(
            ["Bits", "Arithmetic", "Trace", "Placement", "Family", "DOT", "GEMV"],
            q2_rows,
        )
    )

    lines += [
        "",
        "### Question 3: strategy after the shared LUT limit",
        "",
        "At 16 bits, the table reports global-LUT time divided by direct-decoder time. At 32 bits, a complete LUT would require 16 GiB for FP32 or 32 GiB for FP64, so direct was the only full decoder tested. The 32-bit cells report its median time.",
        "",
    ]
    q3_rows: list[list[str]] = []
    for fmt in ("posit16_es1", "takum16", "takum_log16"):
        for arithmetic in ("fp32", "fp64"):
            row = [short_format(fmt), arithmetic.upper()]
            for distribution in MAIN_DISTRIBUTIONS:
                for kernel in KERNELS:
                    comparison = comparison_by_case[
                        ("3", fmt, arithmetic, distribution, kernel, "full_lut_global")
                    ]
                    row.append(ratio_cell(comparison))
            q3_rows.append(row)
    lines.extend(
        markdown_table(
            [
                "16-bit format",
                "Arithmetic",
                "Field DOT",
                "Field GEMV",
                "Log-uniform DOT",
                "Log-uniform GEMV",
            ],
            q3_rows,
        )
    )
    lines.append("")
    q3_32_rows: list[list[str]] = []
    for fmt in ("posit32_es2", "takum32", "takum_log32"):
        for arithmetic in ("fp32", "fp64"):
            row = [short_format(fmt), arithmetic.upper()]
            for distribution in MAIN_DISTRIBUTIONS:
                for kernel in KERNELS:
                    result = timing_by_case[(fmt, arithmetic, distribution, kernel, "direct")]
                    row.append(f"{float(result['median_ms']):.3f} ms")
            q3_32_rows.append(row)
    lines.extend(
        markdown_table(
            [
                "32-bit format",
                "Arithmetic",
                "Field DOT",
                "Field GEMV",
                "Log-uniform DOT",
                "Log-uniform GEMV",
            ],
            q3_32_rows,
        )
    )

    lines += [
        "",
        "### Question 4: best alternative divided by fastest retained same-width IEEE",
        "",
        "The alternative winner and IEEE reference are selected independently inside each format, arithmetic, distribution, and kernel case. The IEEE reference shown in each cell is the fastest retained format and scalar strategy for that case.",
        "",
    ]
    q4 = [row for row in comparisons if row["question"] == "4"]
    q4_key = {
        (
            row["numerator_format"],
            row["arithmetic"],
            row["distribution"],
            row["kernel"],
        ): row
        for row in q4
    }
    q4_rows: list[list[str]] = []
    all_alt_formats = [
        "posit8_es0", "takum8", "takum_log8",
        "posit14_es1", "takum14", "takum_log14",
        "posit16_es1", "takum16", "takum_log16",
        "posit32_es2", "takum32", "takum_log32",
    ]
    for fmt in all_alt_formats:
        for arithmetic in ("fp32", "fp64"):
            for distribution in MAIN_DISTRIBUTIONS:
                row = [short_format(fmt), arithmetic.upper(), short_distribution(distribution)]
                for kernel in KERNELS:
                    comparison = q4_key[(fmt, arithmetic, distribution, kernel)]
                    reference = (
                        f"{comparison['denominator_format']} "
                        f"{short_strategy(comparison['denominator_strategy'])}"
                    )
                    row.append(f"{ratio_cell(comparison)} vs {reference}")
                q4_rows.append(row)
    lines.extend(
        markdown_table(
            ["Format", "Arithmetic", "Distribution", "DOT", "GEMV"],
            q4_rows,
        )
    )

    q1 = [row for row in comparisons if row["question"] == "1"]
    q3 = [row for row in comparisons if row["question"] == "3"]
    q4_classes = Counter(row["classification"] for row in q4)

    def ratio_range(rows: list[dict[str, str]]) -> tuple[float, float]:
        values = [float(row["ratio"]) for row in rows]
        return min(values), max(values)

    q4_ranges = {
        bits: ratio_range([row for row in q4 if row["bits"] == bits])
        for bits in ("8", "14", "16", "32")
    }

    lines += [
        "",
        "## Answers",
        "",
        "### 1. Is a full LUT best at low bit counts?",
        "",
        "Yes for every measured 8-bit and 14-bit main case. Shared LUT won all 24 8-bit cases. Global LUT won all 24 14-bit cases. Across the 96 individual LUT-versus-direct comparisons, 87 LUT cases were faster. The nine slower cases were 14-bit FP64 shared-LUT cases, where each block stages a 128 KiB table. That staging cost does not change the winner because the global LUT remained faster than direct.",
        "",
        "### 2. Does LUT speed depend on whether the entries contain IEEE, posit, or takum values?",
        "",
        f"No measurable dependence appeared. All {len(q2)} paired controls were equivalent under the predeclared 3% band. This held for both widths, arithmetic types, traces, placements, and kernels. The observed alternative-to-IEEE ratios ranged from {min(float(row['ratio']) for row in q2):.3f} to {max(float(row['ratio']) for row in q2):.3f}. Once the raw index stream and kernel are fixed, the LUT contents do not affect conversion throughput.",
        "",
        "### 3. What wins after a shared full LUT becomes too large?",
        "",
        f"At 16 bits, the global full LUT beat direct decoding in all {len(q3)} cases. Its paired ratio to direct ranged from {min(float(row['ratio']) for row in q3):.3f} to {max(float(row['ratio']) for row in q3):.3f}. At 32 bits, direct decoding is the only complete strategy measured because a full table is impractical. The experiment therefore establishes a 32-bit direct baseline, not that direct is better than every possible segmented or approximate decoder.",
        "",
        "### 4. How do the best alternatives compare with retained IEEE formats of the same width?",
        "",
        f"No alternative case was faster. Of 96 cases, {q4_classes['equivalent']} were equivalent, {q4_classes['inconclusive']} were inconclusive, and {q4_classes['numerator_slower']} were slower. All equivalent and inconclusive results occurred at 8 bits. The ratio ranges were {q4_ranges['8'][0]:.3f} to {q4_ranges['8'][1]:.3f} at 8 bits, {q4_ranges['14'][0]:.3f} to {q4_ranges['14'][1]:.3f} at 14 bits, {q4_ranges['16'][0]:.3f} to {q4_ranges['16'][1]:.3f} at 16 bits, and {q4_ranges['32'][0]:.3f} to {q4_ranges['32'][1]:.3f} at 32 bits.",
        "",
        "The main-format comparisons do not reuse identical values because each format has its own safe interval and field-balanced generator. Question 2 is the clean conversion-only comparison. It shows equal LUT throughput. Question 4 measures the specified end-to-end kernels with each format's own inputs, so cache locality and code distribution can also affect the result.",
        "",
        "## Validation and limits",
        "",
        "The full run contains 604 timed cases and 18,120 successful samples. All 119 decoder validation rows passed. The run recorded 23,668 histogram rows and no infeasible case. The build gate compiled all 44 targets, the smoke run covered the same 604-case matrix, and the full job completed in 421 seconds. A separate context-free CUDA review ended with no remaining finding.",
        "",
        "The benchmark measures storage conversion followed by ordinary FP32 or FP64 arithmetic. It does not test posit quires, native posit or takum arithmetic, segmented 32-bit decoders, or an accuracy-matched application. These timings alone cannot show that any family has a better accuracy-performance trade-off.",
        "",
        "## Reproducibility files",
        "",
        "- [Raw timing samples](run_20260826T183425Z/full/timing_samples.csv)",
        "- [Decoder validation](run_20260826T183425Z/full/decoder_validation.csv)",
        "- [Input histograms](run_20260826T183425Z/full/histograms.csv)",
        "- [All confidence intervals](run_20260826T183425Z/analysis/case_comparisons.csv)",
        "- [All strategy winners](run_20260826T183425Z/analysis/strategy_winners.csv)",
        "- [Timing medians](run_20260826T183425Z/analysis/timing_summary.csv)",
        "- [Run manifest](run_20260826T183425Z/run_manifest.txt)",
        "- [GPU and compiler environment](run_20260826T183425Z/environment.txt)",
        "- [Independent CUDA review](code_review.md)",
        "- [Predeclared experiment specification](../../docs/posit_takum_strategy_benchmark.md)",
        "",
    ]
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-output", type=Path)
    args = parser.parse_args()
    build_report(args.analysis_dir, args.output)
    if args.html_output is not None:
        build_html_report(args.analysis_dir, args.html_output)


if __name__ == "__main__":
    main()
