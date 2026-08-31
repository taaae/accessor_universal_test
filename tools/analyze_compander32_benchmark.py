#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from collections import defaultdict
from pathlib import Path


FULL_NS = (1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28)
EXPECTED_VARIANTS = (
    "raw_fp32",
    "fp32_to_fp64",
    "raw_fp64",
    "int32",
    "quadratic32",
    "blended_quadratic32",
    "blended_cubic32",
    "pwl2_compand32",
    "pwl4_compand32",
    "e11m20",
    "e10m21",
    "dyadic_normal32",
    "posit32_es2",
    "takum32",
    "lns32_r23",
)
COLORS = {
    "raw_fp32": "#111820",
    "fp32_to_fp64": "#52606d",
    "raw_fp64": "#89939e",
    "int32": "#00876c",
    "quadratic32": "#24a148",
    "blended_quadratic32": "#76a713",
    "blended_cubic32": "#b18c00",
    "pwl2_compand32": "#008ccf",
    "pwl4_compand32": "#5865d8",
    "e11m20": "#8b5cf6",
    "e10m21": "#a8559d",
    "dyadic_normal32": "#c4562d",
    "posit32_es2": "#d83a52",
    "takum32": "#a72f62",
    "lns32_r23": "#725141",
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
        raise ValueError(f"no timing rows in {path}")
    return rows


def validate(rows: list[dict[str, str]]) -> tuple[str, tuple[int, ...], int]:
    required = {
        "gpu", "mode", "distribution", "kernel", "format", "label", "group",
        "storage_bits", "arithmetic_type", "storage_layout", "access_method",
        "packet_values", "decoder", "N", "physical_input_bytes", "blocks",
        "threads", "warmups", "sample", "execution_order", "kernel_ms",
        "scaled_result", "valid",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"timing CSV misses columns: {sorted(missing)}")
    modes = {row["mode"] for row in rows}
    if len(modes) != 1:
        raise ValueError("timing CSV mixes benchmark modes")
    mode = next(iter(modes))
    expected_ns = FULL_NS if mode == "full" else (1 << 20,)
    expected_samples = 50 if mode == "full" else 3
    expected_warmups = 10 if mode == "full" else 1
    if {row["format"] for row in rows} != set(EXPECTED_VARIANTS):
        raise ValueError("timing CSV has missing or unexpected variants")
    for row in rows:
        fixed = {
            "distribution": "normal_clipped",
            "kernel": "dot",
            "storage_layout": "natural",
            "access_method": "scalar",
            "packet_values": "1",
            "blocks": "512",
            "threads": "256",
            "warmups": str(expected_warmups),
            "valid": "1",
        }
        for field, value in fixed.items():
            if row[field] != value:
                raise ValueError(f"{row['format']}: {field}={row[field]!r}")
        n = int(row["N"])
        if n not in expected_ns:
            raise ValueError(f"unexpected N={n}")
        kernel_ms = float(row["kernel_ms"])
        result = float(row["scaled_result"])
        if not math.isfinite(kernel_ms) or kernel_ms <= 0:
            raise ValueError("kernel time must be finite and positive")
        if not math.isfinite(result):
            raise ValueError("DOT result must be finite")
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["format"], int(row["N"]))].append(row)
    for variant in EXPECTED_VARIANTS:
        for n in expected_ns:
            group = groups[(variant, n)]
            if len(group) != expected_samples:
                raise ValueError(f"{variant}/N={n}: wrong sample count")
            if {int(row["sample"]) for row in group} != set(range(expected_samples)):
                raise ValueError(f"{variant}/N={n}: bad sample indices")
    for n in expected_ns:
        for sample in range(expected_samples):
            round_rows = [
                row for row in rows
                if int(row["N"]) == n and int(row["sample"]) == sample
            ]
            if {int(row["execution_order"]) for row in round_rows} != set(range(15)):
                raise ValueError(f"N={n}/sample={sample}: incomplete rotated order")
    return mode, expected_ns, expected_samples


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["format"], int(row["N"]))].append(row)
    output: list[dict[str, object]] = []
    for variant in EXPECTED_VARIANTS:
        ns = sorted(n for key_variant, n in groups if key_variant == variant)
        for n in ns:
            group = groups[(variant, n)]
            values = [float(row["kernel_ms"]) for row in group]
            first = group[0]
            median = statistics.median(values)
            output.append({
                "format": variant,
                "label": first["label"],
                "group": first["group"],
                "decoder": first["decoder"],
                "storage_bits": int(first["storage_bits"]),
                "arithmetic_type": first["arithmetic_type"],
                "N": n,
                "median_ms": median,
                "q1_ms": percentile(values, 0.25),
                "q3_ms": percentile(values, 0.75),
                "time_per_element_ns": median * 1.0e6 / n,
                "effective_input_gb_s": float(first["physical_input_bytes"]) /
                                           (median * 1.0e6),
            })
    anchors = {
        int(row["N"]): float(row["median_ms"])
        for row in output if row["format"] == "fp32_to_fp64"
    }
    for row in output:
        row["ratio_to_fp32_to_fp64"] = (
            float(row["median_ms"]) / anchors[int(row["N"])]
        )
    return output


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def line_graph(rows: list[dict[str, object]]) -> str:
    width, height = 1420, 850
    left, right, top, bottom = 105, 1040, 72, 725
    by_variant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["format"])].append(row)
    values = [float(row["median_ms"]) for row in rows]
    lower = min(values) * 0.88
    upper = max(values) * 1.15
    log_lower, log_upper = math.log10(lower), math.log10(upper)

    def x(n: int) -> float:
        return left + (math.log2(n) - 20) / 8 * (right - left)

    def y(value: float) -> float:
        return bottom - (math.log10(value) - log_lower) / (log_upper - log_lower) * (bottom - top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Median DOT time versus N for all tested 32-bit storage formats">',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif;fill:#26343d}'
        '.grid{stroke:#dbe3e8;stroke-width:1}.axis{stroke:#53636e;stroke-width:1.4}'
        '.tick{font-size:13px}.title{font-size:23px;font-weight:750}.sub{font-size:14px;fill:#61717d}'
        '.legend{font-size:13px}.line{fill:none;stroke-width:2.5}.point{stroke:#fff;stroke-width:1.2}</style>',
        f'<rect width="{width}" height="{height}" rx="18" fill="#fff"/>',
        '<text x="105" y="35" class="title">DOT kernel time versus prefix length</text>',
        '<text x="105" y="57" class="sub">512 blocks × 256 threads, scalar x1, medians of 50 launches, logarithmic time axis</text>',
    ]
    for exponent in (20, 22, 24, 26, 28):
        xpos = x(1 << exponent)
        parts.append(f'<line class="grid" x1="{xpos:.1f}" y1="{top}" x2="{xpos:.1f}" y2="{bottom}"/>')
        parts.append(f'<text class="tick" x="{xpos:.1f}" y="{bottom + 28}" text-anchor="middle">2^{exponent}</text>')
    tick_count = 7
    for index in range(tick_count):
        power = log_lower + index / (tick_count - 1) * (log_upper - log_lower)
        value = 10 ** power
        ypos = y(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{ypos:.1f}" x2="{right}" y2="{ypos:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 13}" y="{ypos + 4:.1f}" text-anchor="end">{value:.3f}</text>')
    parts.extend([
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<text class="tick" x="{(left + right) / 2:.1f}" y="{bottom + 61}" text-anchor="middle">N</text>',
        f'<text class="tick" transform="translate(28 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">Median kernel time (ms)</text>',
    ])
    for variant in EXPECTED_VARIANTS:
        points = sorted(by_variant[variant], key=lambda row: int(row["N"]))
        coordinates = " ".join(f'{x(int(row["N"])):.1f},{y(float(row["median_ms"])):.1f}' for row in points)
        color = COLORS[variant]
        dash = ' stroke-dasharray="7 5"' if str(points[0]["group"]) == "baseline" else ""
        parts.append(f'<polyline class="line" points="{coordinates}" stroke="{color}"{dash}><title>{escape(points[0]["label"])}</title></polyline>')
        for row in points:
            parts.append(f'<circle class="point" cx="{x(int(row["N"])):.1f}" cy="{y(float(row["median_ms"])):.1f}" r="3.5" fill="{color}"><title>{escape(row["label"])}: {float(row["median_ms"]):.6f} ms</title></circle>')
    legend_x, legend_y = 1080, 92
    for index, variant in enumerate(EXPECTED_VARIANTS):
        row = by_variant[variant][0]
        ypos = legend_y + index * 43
        color = COLORS[variant]
        dash = ' stroke-dasharray="7 5"' if str(row["group"]) == "baseline" else ""
        parts.append(f'<line x1="{legend_x}" y1="{ypos}" x2="{legend_x + 30}" y2="{ypos}" stroke="{color}" stroke-width="3"{dash}/>')
        parts.append(f'<text class="legend" x="{legend_x + 39}" y="{ypos + 4}">{escape(row["label"])}</text>')
    parts.append('</svg>')
    return "".join(parts)


def ranking_graph(rows: list[dict[str, object]], target_n: int) -> str:
    selected = sorted(
        (row for row in rows if int(row["N"]) == target_n),
        key=lambda row: float(row["median_ms"]),
    )
    width, height = 1500, 1020
    left, solid_end, label_x = 110, 1040, 1190
    top, bottom = 90, 900
    minimum = min(float(row["median_ms"]) for row in selected) * 0.92
    maximum = max(float(row["median_ms"]) for row in selected) * 1.05

    def actual_y(value: float) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    label_positions: list[float] = []
    minimum_gap = 38.0
    for row in selected:
        candidate = actual_y(float(row["median_ms"]))
        label_positions.append(candidate)
    for index in range(1, len(label_positions)):
        if label_positions[index] > label_positions[index - 1] - minimum_gap:
            label_positions[index] = label_positions[index - 1] - minimum_gap
    if label_positions[-1] < top:
        shift = top - label_positions[-1]
        label_positions = [value + shift for value in label_positions]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="N equals {target_n} DOT kernel timing ranking">',
        '<style>text{font-family:Inter,ui-sans-serif,system-ui,sans-serif;fill:#26343d}'
        '.grid{stroke:#dbe3e8;stroke-width:1}.axis{stroke:#53636e;stroke-width:1.4}'
        '.title{font-size:23px;font-weight:750}.sub{font-size:14px;fill:#61717d}.tick{font-size:13px}.label{font-size:14px;font-weight:650}</style>',
        f'<rect width="{width}" height="{height}" rx="18" fill="#fff"/>',
        f'<text x="110" y="36" class="title">32-bit storage decoder ranking at N = 2^{int(math.log2(target_n))}</text>',
        '<text x="110" y="59" class="sub">Line height is the median DOT time. Dotted extensions connect crowded lines to direct labels.</text>',
    ]
    for index in range(7):
        value = minimum + index / 6 * (maximum - minimum)
        ypos = actual_y(value)
        parts.append(f'<line class="grid" x1="{left}" y1="{ypos:.1f}" x2="{solid_end}" y2="{ypos:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 13}" y="{ypos + 4:.1f}" text-anchor="end">{value:.3f}</text>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    parts.append(f'<text class="tick" transform="translate(28 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">Median kernel time (ms)</text>')
    for row, label_y in zip(selected, label_positions):
        value = float(row["median_ms"])
        ypos = actual_y(value)
        color = COLORS[str(row["format"])]
        solid_dash = ' stroke-dasharray="8 5"' if str(row["group"]) == "baseline" else ""
        parts.append(f'<line x1="{left + 15}" y1="{ypos:.1f}" x2="{solid_end}" y2="{ypos:.1f}" stroke="{color}" stroke-width="5" stroke-linecap="round"{solid_dash}/>')
        parts.append(f'<path d="M {solid_end} {ypos:.1f} L {label_x - 17} {label_y:.1f}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="4 6"/>')
        parts.append(f'<text class="label" x="{label_x}" y="{label_y + 5:.1f}">{escape(row["label"])} · {value:.6f} ms · {float(row["ratio_to_fp32_to_fp64"]):.2f}×</text>')
    parts.append('</svg>')
    return "".join(parts)


STRATEGY_ROWS = (
    ("Raw FP32", "FP32", "Native FP32 load and FP32 FMA"),
    ("FP32 to FP64", "FP64", "Native FP32 load, conversion to FP64"),
    ("Raw FP64", "FP64", "Native FP64 load"),
    ("Int32", "FP64", "Signed integer to FP64"),
    ("Quadratic32", "FP64", "Integer to FP64, abs, multiply"),
    ("BlendedQuadratic32", "FP64", "Integer to FP64, abs, FMA, multiply; α=0.65"),
    ("BlendedCubic32", "FP64", "Integer to FP64, square, FMA, multiply; α=0.65"),
    ("PWL2Compand32", "FP64", "Sign | 1 zone bit | 30 payload bits; branchless select and FMA"),
    ("PWL4Compand32", "FP64", "Sign | 2 zone bits | 29 payload bits; branchless select and FMA"),
    ("E11M20", "FP64", "direct_masked"),
    ("E10M21", "FP64", "direct_branchy"),
    ("DyadicNormal32", "FP64", "Existing sign-fused decoder; genuine N(0,1) codes"),
    ("Posit<32,2>", "FP64", "Existing direct decoder"),
    ("Takum32", "FP64", "Existing linear-takum direct decoder"),
    ("LNS<32,23>", "FP64", "ordinary scalar x1 ex2_approx"),
)


def read_optional(path: Path | None) -> str:
    if path is None or not path.exists():
        return "Not supplied."
    return path.read_text(errors="replace")


def build_report(
    rows: list[dict[str, object]],
    line_svg: str,
    ranking_svg: str,
    correctness: str,
    compiler: str,
    environment: str,
) -> str:
    nmax_rows = sorted(
        (row for row in rows if int(row["N"]) == 1 << 28),
        key=lambda row: float(row["median_ms"]),
    )
    dyadic = next(row for row in nmax_rows if row["format"] == "dyadic_normal32")
    fp32_to_fp64 = next(row for row in nmax_rows if row["format"] == "fp32_to_fp64")
    raw_fp64 = next(row for row in nmax_rows if row["format"] == "raw_fp64")

    def classification(row: dict[str, object]) -> str:
        value = float(row["median_ms"])
        anchor = float(fp32_to_fp64["median_ms"])
        if value <= 1.05 * anchor:
            return "promising"
        if value < float(dyadic["median_ms"]):
            return "marginal"
        if value >= float(raw_fp64["median_ms"]):
            return "poor"
        return "unattractive"

    strategy_table = "".join(
        f"<tr><td>{escape(name)}</td><td>{escape(arithmetic)}</td><td>{escape(decoder)}</td></tr>"
        for name, arithmetic, decoder in STRATEGY_ROWS
    )
    timing_table = "".join(
        "<tr>"
        f"<td>{escape(row['label'])}</td>"
        f"<td>{float(row['median_ms']):.6f}</td>"
        f"<td>{float(row['q1_ms']):.6f}–{float(row['q3_ms']):.6f}</td>"
        f"<td>{float(row['ratio_to_fp32_to_fp64']):.3f}×</td>"
        f"<td>{float(row['time_per_element_ns']):.4f}</td>"
        f"<td><span class=\"tag {classification(row)}\">{classification(row)}</span></td>"
        "</tr>"
        for row in nmax_rows
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>32-bit compander conversion cost on H200</title>
<style>
:root{{--ink:#1f2d35;--muted:#61717d;--line:#d9e2e7;--paper:#fff;--back:#eef3f5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--back);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{max-width:1510px;margin:0 auto;padding:38px 28px 70px}}h1{{font-size:36px;letter-spacing:-.035em;margin:0 0 7px}}h2{{font-size:24px;margin:38px 0 13px}}p{{max-width:1030px;color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:25px 0}}.card,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 26px #21313b0d}}.card{{padding:16px}}.card small{{display:block;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);font-weight:700}}.card strong{{font-size:18px}}
.panel{{padding:14px;margin:14px 0;overflow:auto}}.panel svg{{display:block;width:100%;height:auto}}table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden}}th,td{{text-align:left;padding:11px 13px;border-bottom:1px solid #e6ecef;vertical-align:top}}th{{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);background:#f7fafb}}code{{font:13px ui-monospace,SFMono-Regular,Menlo,monospace}}pre{{white-space:pre-wrap;background:#18232a;color:#dbe7ed;padding:16px;border-radius:12px;max-height:360px;overflow:auto;font-size:12px}}.tag{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700}}.promising{{background:#dff5e9;color:#17623c}}.marginal{{background:#fff2c7;color:#725700}}.unattractive{{background:#fee5d7;color:#8b3d16}}.poor{{background:#fde1e7;color:#8e2940}}.note{{border-left:4px solid #3483a8;padding:8px 14px;background:#edf7fb;color:#385664}}@media(max-width:900px){{.cards{{grid-template-columns:1fr 1fr}}main{{padding:24px 12px}}}}
</style></head><body><main>
<h1>32-bit compander conversion cost on H200</h1>
<p>This run asks a narrow question: which 32-bit decoders stay near native FP32-to-FP64 conversion cost inside a scalar DOT kernel? It measures performance only. The chosen compander parameters are representative code paths, not accuracy winners.</p>
<div class="cards">
<div class="card"><small>Input</small><strong>one clipped N(0,1) allocation</strong><br>prefixes from 2^20 through 2^28</div>
<div class="card"><small>Geometry</small><strong>512 × 256</strong><br>fixed for every N and strategy</div>
<div class="card"><small>Timing</small><strong>10 + 50 launches</strong><br>rotated order, CUDA events</div>
<div class="card"><small>Access</small><strong>scalar x1</strong><br>two-stage DOT, encoding excluded</div>
</div>
<h2>Scaling with N</h2><div class="panel">{line_svg}</div>
<p class="note">The time axis is logarithmic so the slow reference decoders do not flatten the low-cost group. Every N uses a prefix of the same maximum-size encoded arrays.</p>
<h2>Ranking at N = 2^28</h2><div class="panel">{ranking_svg}</div>
<p>The labels use the FP32-to-FP64 median at the same N as 1.00×. “Promising” means within 5%. “Marginal” ends at the DyadicNormal32 time. Results at or above DyadicNormal32 are unattractive for this use, and results at or above raw FP64 are poor.</p>
<h2>N = 2^28 timing table</h2>
<table><thead><tr><th>Format</th><th>Median ms</th><th>IQR ms</th><th>vs FP32→FP64</th><th>ns/element</th><th>Screen</th></tr></thead><tbody>{timing_table}</tbody></table>
<h2>Exact strategy set</h2>
<table><thead><tr><th>Format</th><th>Arithmetic</th><th>Timed decode</th></tr></thead><tbody>{strategy_table}</tbody></table>
<h2>What the timer includes</h2>
<p>Each sample covers one 512-block first-stage DOT launch and one 256-thread final reduction. It excludes allocation, random generation, encoding, transfers, and scale restoration. Int32 and the polynomial companders restore their value scale once after reduction for correctness checks. PWL2 uses the boundary 1.168. PWL4 uses 0.552, 1.168, and 1.992. Both cover ±8 and give every positive zone the same number of codes.</p>
<p>Effective input bandwidth is present in the summary CSV as bytes read from the two storage arrays divided by elapsed time. It is not a fair single ranking across raw FP64 and 32-bit formats because raw FP64 moves twice the input bytes. The figures therefore use time and time per element.</p>
<h2>Correctness</h2><pre>{escape(correctness)}</pre>
<h2>Compiler evidence</h2><pre>{escape(compiler)}</pre>
<h2>Environment</h2><pre>{escape(environment)}</pre>
<h2>Limits</h2>
<p>This run fixes one GPU, one DOT geometry, scalar x1 access, one clipped normal input, and one alpha. It does not compare numerical error, choose alpha by application loss, or claim that a decoder with a fast path is the best representation for a workload. The input distribution can still affect data-dependent decoders such as DyadicNormal32 and branchy IEEE conversion.</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--correctness", type=Path)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--environment", type=Path)
    args = parser.parse_args()

    raw = read_rows(args.samples)
    mode, ns, samples = validate(raw)
    summary = summarize(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir / "timing_summary.csv", summary)
    graph = line_graph(summary)
    target_n = max(int(row["N"]) for row in summary)
    ranking = ranking_graph(summary, target_n)
    (args.output_dir / "time_vs_n.svg").write_text(graph)
    ranking_name = f"ranking_n2p{int(math.log2(target_n))}.svg"
    (args.output_dir / ranking_name).write_text(ranking)
    if mode == "full":
        report = build_report(
            summary,
            graph,
            ranking,
            read_optional(args.correctness),
            read_optional(args.compiler),
            read_optional(args.environment),
        )
        (args.output_dir / "report.html").write_text(report)
    print(
        f"validated {len(EXPECTED_VARIANTS)} variants × {len(ns)} N values × "
        f"{samples} samples; wrote {args.output_dir}"
    )


if __name__ == "__main__":
    main()
