#!/usr/bin/env python3
"""Build the IEEE and LNS summary page.

The page answers a practical question the per-format pages cannot: which
storage types are worth implementing in the memory accessor, and which decode
strategy each one should use.  Findings are added here as the analysis
establishes them, so ``SECTIONS`` starts effectively empty and grows.
"""

from __future__ import annotations

import argparse
import collections
import csv
import html
import math
import statistics
from pathlib import Path
from typing import Sequence

import build_ieee_question_report as ieee
import build_storage_performance_report as base


# The two decoders compete on very different terms, so the tables below rank
# every decoder and report the runner-up rather than a fixed rival.
STRATEGY_ROOT = Path("results/021_unified_strategy_performance")
# The screen stage measured exactly one size per kernel; using that same size
# for the full stage keeps screened-out variants comparable with survivors.
COMMON_N = {"dot": 16_777_216, "gemv": 16_384}
DISTRIBUTIONS = ("uniform_0_1", "normal_0_1")
# Above this width `bitwidth_strategy_bench.cu` builds no table decoders.
MAX_LUT_BITS = 14


FILENAME = "ieee-lns-summary.html"

INTRO = ""

def code(text: str) -> str:
    """A source block; the base stylesheet already styles bare ``pre``."""
    return f"<pre><code>{html.escape(text)}</code></pre>"


# Each entry is (heading, [paragraphs...]).  Filled in as findings are settled.
SECTIONS: Sequence[tuple[str, Sequence[str]]] = (
    (
        "full_lut_shared",
        (
            "<code>a[i]</code> semantic does not work as it is.",
            "H200 allows 227 KB shared memory per block, so the table fits up "
            "to 15 bits (128 KB) &mdash; same for FP32 and FP64, one 32-bit "
            "word per code.",
            "FP64 tables store 32-bit entries holding the double&rsquo;s high "
            "word; the low word is a literal zero.",
            code(
                """__global__ void kernel(...) {
  extern __shared__ uint32_t table[];

  for (i = threadIdx.x; i < table_size; i += blockDim.x)
    table[i] = global_table[i];

  __syncthreads();  // block-collective, so it cannot live inside operator[]

  out[i] = table[a[i]] * x[i];
}"""
            ),
            "Solution:",
            code(
                """template <typename Acc>
__global__ void kernel(Acc acc, const double *x, double *out, std::size_t n) {
  auto a = acc.stage();                     // barrier inside; all threads
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * x[i];
}"""
            ),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/report"))
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def newest_strategy_run(root: Path) -> Path | None:
    """Newest experiment 021 run that finished its screen and full stages."""
    candidates = [
        directory
        for directory in sorted(root.glob("run_*"))
        if (directory / "unified_core/full/timing_summary.csv").is_file()
        and (directory / "unified_core/screen/timing_summary.csv").is_file()
    ]
    return candidates[-1] if candidates else None


def read_stage(path: Path) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    """Valid rows at the one N the screen stage shares with the full stage."""
    index: dict[tuple[str, str, str, str], list[dict[str, str]]] = (
        collections.defaultdict(list)
    )
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("all_valid", "1") != "1":
                continue
            if int(row["N"]) != COMMON_N[row["kernel"]]:
                continue
            key = (
                row["arithmetic_type"],
                row["format"],
                row["kernel"],
                row["strategy_id"],
            )
            index[key].append(row)
    return index


def variant_score(rows: Sequence[dict[str, str]]) -> float | None:
    values = [
        float(row["median_ms"])
        for row in rows
        if row["distribution"] in DISTRIBUTIONS
    ]
    if len(values) != len(DISTRIBUTIONS):
        return None
    return math.exp(statistics.fmean(math.log(value) for value in values))


def ranked_decoders(
    full: dict, screen: dict, compute: str, format_name: str, kernel: str, scope: str
) -> list[tuple[str, float]]:
    """Fastest variant per decoder, fastest decoder first.

    The full stage only ran the screen's finalists, so a decoder missing there
    lost the screen rather than going unimplemented; fall back to its screen
    measurement instead of dropping it.
    """
    best: dict[str, float] = {}
    prefix = (compute, format_name, kernel)
    for key in set(full) | set(screen):
        if key[:3] != prefix:
            continue
        _, access, lanes, decoder = key[3].split("/")
        if scope == "x1" and not (access == "scalar" and lanes == "x1"):
            continue
        score = variant_score(full[key] if key in full else screen[key])
        if score is None:
            continue
        if decoder not in best or score < best[decoder]:
            best[decoder] = score
    return sorted(best.items(), key=lambda item: item[1])


def e0_formats(full: dict, compute: str) -> list[str]:
    names = {key[1] for key in full if key[0] == compute and key[1].startswith("e0m")}
    return sorted(names, key=lambda name: int(name[3:]) + 1)


def e0_table(full: dict, screen: dict, compute: str, kernel: str, scope: str) -> str:
    headings = (
        "type", "bits", "winner", "ms", "runner-up", "ms", "gap", "note",
    )
    body_rows = []
    for name in e0_formats(full, compute):
        bits = int(name[3:]) + 1
        ranking = ranked_decoders(full, screen, compute, name, kernel, scope)
        note = "lut strategies excluded" if bits > MAX_LUT_BITS else ""
        cells = [base.label(name), str(bits)]
        if len(ranking) >= 2:
            (winner, winner_ms), (runner, runner_ms) = ranking[0], ranking[1]
            cells += [
                f"<code>{winner}</code>", f"{winner_ms:.4f}",
                f"<code>{runner}</code>", f"{runner_ms:.4f}",
                f"{runner_ms / winner_ms:.3f}x",
            ]
        elif ranking:
            cells += [f"<code>{ranking[0][0]}</code>", f"{ranking[0][1]:.4f}",
                      "-", "-", "-"]
        else:
            cells += ["-", "-", "-", "-", "-"]
        cells.append(note)
        body_rows.append(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"
        )
    header = "".join(f"<th>{heading}</th>" for heading in headings)
    title = f"{kernel.upper()} &rarr; {compute.upper()}"
    return (
        f"<h4>{title}</h4>"
        '<div class="table-wrap"><table class="strategy-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def e0_section(run_dir: Path) -> str:
    full = read_stage(run_dir / "unified_core/full/timing_summary.csv")
    screen = read_stage(run_dir / "unified_core/screen/timing_summary.csv")
    blocks = []
    for scope in ("x1", "best"):
        blocks.append(f"<h3>{scope} scope</h3>")
        for compute in ("fp32", "fp64"):
            for kernel in ("dot", "gemv"):
                blocks.append(e0_table(full, screen, compute, kernel, scope))
    return (
        '<section class="text-section"><h2>E0</h2>'
        "<p><code>fixed_integer</code> and <code>full_lut_shared</code> are "
        "dominant strategies.</p>"
        "<details><summary>Best strategy vs runner-up &mdash; 8 tables</summary>"
        "<p>Times at a single size (DOT 2<sup>24</sup>, GEMV 2<sup>14</sup>), "
        "geometric mean over both distributions.</p>"
        f"{''.join(blocks)}</details></section>"
    )


def e1_section() -> str:
    """E1 is an integer format, but both test distributions sit inside its
    subnormal range, so its measured margins say more about the data than the
    format."""
    shares = ieee.subnormal_shares("e1m1")
    threshold = ieee.smallest_normal(1)
    return (
        '<section class="text-section"><h2>E1 &mdash; unreliable</h2>'
        "<p>The test data sits almost entirely in E1&rsquo;s subnormal range.</p>"
        '<div class="table-wrap"><table class="strategy-table">'
        "<thead><tr><th></th><th>uniform(0,1)</th><th>normal(0,1)</th></tr></thead>"
        f"<tbody><tr><td>share below smallest normal ({threshold:g})</td>"
        f"<td>{shares[0]:.0%}</td><td>{shares[1]:.1%}</td></tr></tbody>"
        "</table></div>"
        "<details><summary>What this breaks</summary>"
        "<p>E1 has one exponent bit, so the bias is 0: subnormals span "
        "[0,&nbsp;2) and the single normal binade spans [2,&nbsp;4), both in "
        "steps of 2<sup>1&minus;M</sup>.  The code space is one uniform ramp, "
        "which is what lets <code>e1_integer</code> decode it branchlessly as "
        "<code>magnitude &times; 2<sup>1&minus;M</sup></code> &mdash; the "
        "exponent bit is just the high bit of the magnitude field.  E1 is an "
        "integer format in the same sense E0 is.</p>"
        "<p><code>e1_integer</code> measures 1.20&ndash;1.64x faster than "
        "<code>direct_branchy</code>, but that margin is bought by the table "
        "above.  <code>direct_branchy</code> renormalizes subnormals &mdash; "
        "leading-bit search, variable shift, exponent reassembly &mdash; and "
        "with 95&ndash;100% of values below the smallest normal it takes that "
        "path on essentially every value.  On data inside [2,&nbsp;4) it would "
        "take the cheap path instead and the gap would shrink.  Neither "
        "distribution puts meaningful mass there, so this run cannot say by "
        "how much.</p>"
        "<p><code>e1_integer</code>&rsquo;s own times are identical across the "
        "two distributions to four digits &mdash; it is genuinely "
        "data-independent.  The sensitivity lives entirely in its competitor.</p>"
        "<p>E1M0 is separately degenerate: 1 sign bit, 1 exponent bit, no "
        "mantissa, so it encodes only &plusmn;0 and &plusmn;inf and no finite "
        "nonzero values.  <code>e1_integer</code> does not apply to it.</p>"
        "</details></section>"
    )


# Decoders that compile to the exponent-width fast path.  E8 formats keep the
# general `storage::decode`, so `generic` is only a shift for E11.
SHIFT_DECODERS = {
    8: {"direct_branchy", "direct_masked"},
    11: {"direct_branchy", "direct_masked", "generic"},
}
NATIVE_FORMATS = {
    "fp4_e2m1", "fp8_e4m3", "fp8_e5m2", "fp16_e5m10", "bf16_e8m7",
    "fp32_e8m23", "raw_fp32", "raw_fp64",
}


def shift_table(
    full: dict, screen: dict, exponent: int, compute: str, kernel: str, scope: str,
    widths: dict,
) -> str:
    shift_kinds = SHIFT_DECODERS[exponent]
    names = sorted(
        (
            name
            for (arith, name, _, _) in full
            if arith == compute
            and widths.get(name, (None,))[0] == exponent
            and name not in NATIVE_FORMATS
        ),
        key=lambda name: widths[name][1],
    )
    body_rows = []
    for name in dict.fromkeys(names):
        ranking = ranked_decoders(full, screen, compute, name, kernel, scope)
        scores = dict(ranking)
        shift = min((v for k, v in scores.items() if k in shift_kinds), default=None)
        other = [(k, v) for k, v in ranking if k not in shift_kinds]
        if shift is None or not other:
            continue
        rival, rival_ms = other[0]
        cells = [
            base.label(name), str(widths[name][1]), f"{shift:.4f}",
            f"<code>{rival}</code>", f"{rival_ms:.4f}", f"{rival_ms / shift:.3f}x",
            "shift loses" if rival_ms < shift else "",
        ]
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    headings = ("type", "bits", "shift", "best non-shift", "ms", "gap", "note")
    header = "".join(f"<th>{h}</th>" for h in headings)
    return (
        f"<h4>{kernel.upper()} {scope}</h4>"
        '<div class="table-wrap"><table class="strategy-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def shift_section(run_dir: Path) -> str:
    full = read_stage(run_dir / "unified_core/full/timing_summary.csv")
    screen = read_stage(run_dir / "unified_core/screen/timing_summary.csv")
    widths: dict[str, tuple[int, int]] = {}
    with (run_dir / "unified_core/full/timing_summary.csv").open(encoding="utf-8") as h:
        for row in csv.DictReader(h):
            widths[row["format"]] = (int(row["exponent_bits"]), int(row["bits"]))
    layout = (
        "FP32   S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM\n"
        "FP64   S EEEEEEEEEEE " + "M" * 52
    )
    blocks = []
    for exponent, compute in ((8, "fp32"), (11, "fp64")):
        blocks.append(f"<h3>E{exponent} &rarr; {compute.upper()}</h3>")
        for scope in ("x1", "best"):
            for kernel in ("dot", "gemv"):
                blocks.append(
                    shift_table(full, screen, exponent, compute, kernel, scope, widths)
                )
    return (
        '<section class="text-section">'
        "<h2>Shifters: E8 &rarr; FP32, E11 &rarr; FP64</h2>"
        f"{code(layout)}"
        "<p>Matching the target&rsquo;s exponent width matches its bias too, so "
        "decoding is one left shift &mdash; the same number moved into a bigger "
        "container.</p>"
        "<details><summary>Shift vs the best decoder that is not a shift "
        "&mdash; 8 tables</summary>"
        "<p>Times at a single size (DOT 2<sup>24</sup>, GEMV 2<sup>14</sup>), "
        "geometric mean over both distributions.  Native formats are excluded.</p>"
        f"{''.join(blocks)}</details></section>"
    )


# Decoders that reach the hardware converter for a native format.  `generic`
# is short-circuited to `decode_native_scalar` for exactly these types.
NATIVE_DECODERS = {"native_scalar", "native_packed", "generic"}
# Native formats in the order they are reported, widest range last.
NATIVE_ORDER = (
    "fp4_e2m1", "fp8_e4m3", "fp8_e5m2", "fp16_e5m10", "bf16_e8m7", "fp32_e8m23",
)
# Gaps inside this band are reported as ties rather than as a winner: the
# sample counts here do not resolve differences that small.
TIE_BAND = 0.025


def native_table(
    full: dict, screen: dict, compute: str, kernel: str, scope: str, widths: dict
) -> str:
    body_rows = []
    for name in NATIVE_ORDER:
        ranking = ranked_decoders(full, screen, compute, name, kernel, scope)
        scores = dict(ranking)
        native = min(
            (v for k, v in scores.items() if k in NATIVE_DECODERS), default=None
        )
        other = [(k, v) for k, v in ranking if k not in NATIVE_DECODERS]
        if native is None or not other:
            continue
        rival, rival_ms = other[0]
        gap = rival_ms / native
        if abs(gap - 1) < TIE_BAND:
            note = "tie"
        elif gap < 1:
            note = "native loses"
        else:
            note = ""
        cells = [
            base.label(name), str(widths[name][1]), f"{native:.4f}",
            f"<code>{rival}</code>", f"{rival_ms:.4f}", f"{gap:.3f}x", note,
        ]
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    headings = ("type", "bits", "native", "best non-native", "ms", "gap", "note")
    header = "".join(f"<th>{h}</th>" for h in headings)
    return (
        f"<h4>{kernel.upper()} {scope} &rarr; {compute.upper()}</h4>"
        '<div class="table-wrap"><table class="strategy-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def native_section(run_dir: Path) -> str:
    full = read_stage(run_dir / "unified_core/full/timing_summary.csv")
    screen = read_stage(run_dir / "unified_core/screen/timing_summary.csv")
    widths: dict[str, tuple[int, int]] = {}
    with (run_dir / "unified_core/full/timing_summary.csv").open(encoding="utf-8") as h:
        for row in csv.DictReader(h):
            widths[row["format"]] = (int(row["exponent_bits"]), int(row["bits"]))
    blocks = [
        native_table(full, screen, compute, kernel, scope, widths)
        for scope in ("x1", "best")
        for compute in ("fp32", "fp64")
        for kernel in ("dot", "gemv")
    ]
    return (
        '<section class="text-section">'
        "<h2>Native: FP4, FP8 (E4M3, E5M2), FP16, BF16, FP32</h2>"
        "<p>Native conversion wins overall, but not for FP4 &mdash; there "
        "<code>full_lut_shared</code> is the best.  For FP8 "
        "<code>full_lut_shared</code> also wins occasionally.  For BF16 and "
        "FP32 in FP32 cases the bit shift often shows the same performance.</p>"
        "<details><summary>Native vs the best decoder that is not native "
        "&mdash; 8 tables</summary>"
        "<p>Times at a single size (DOT 2<sup>24</sup>, GEMV 2<sup>14</sup>), "
        "geometric mean over both distributions.  <em>native</em> is the best "
        "of <code>native_scalar</code>, <code>native_packed</code> and "
        "<code>generic</code>, which all dispatch to the hardware converter "
        "for these formats.  Gaps within 2.5% are marked as ties.</p>"
        f"{''.join(blocks)}</details></section>"
    )


FULL_LUTS = ("full_lut_shared", "full_lut_global")


def narrow_cells(full: dict, screen: dict, widths: dict) -> list[tuple]:
    """Every cell left once the special-cased families are removed.

    E0/E1/E2 have their own sections, the shifters decode with one instruction,
    and native formats have a hardware converter -- none of them tests whether
    a table is the right choice.  What remains is the general case.
    """
    cells = []
    for compute in ("fp32", "fp64"):
        names = sorted(
            {name for (arith, name, _, _) in full if arith == compute},
            key=lambda name: widths[name][1],
        )
        for name in dict.fromkeys(names):
            exponent, bits = widths[name]
            if name in NATIVE_FORMATS or exponent < 3 or bits > MAX_LUT_BITS:
                continue
            if exponent == 8 and compute == "fp32":
                continue
            if exponent == 11 and compute == "fp64":
                continue
            for scope in ("x1", "best"):
                for kernel in ("dot", "gemv"):
                    ranking = ranked_decoders(
                        full, screen, compute, name, kernel, scope
                    )
                    if ranking:
                        cells.append((name, bits, compute, kernel, scope, ranking))
    return cells


def narrow_section(run_dir: Path) -> str:
    full = read_stage(run_dir / "unified_core/full/timing_summary.csv")
    screen = read_stage(run_dir / "unified_core/screen/timing_summary.csv")
    widths: dict[str, tuple[int, int]] = {}
    with (run_dir / "unified_core/full/timing_summary.csv").open(encoding="utf-8") as h:
        for row in csv.DictReader(h):
            widths[row["format"]] = (int(row["exponent_bits"]), int(row["bits"]))
    cells = narrow_cells(full, screen, widths)
    total = len(cells)
    wins = collections.Counter(
        ranking[0][0] for (*_, ranking) in cells if ranking[0][0] in FULL_LUTS
    )
    lost = [cell for cell in cells if cell[5][0][0] not in FULL_LUTS]

    rate_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{count}</td>"
        f"<td>{count / total:.0%}</td></tr>"
        for name, count in wins.most_common()
    ) + (
        f"<tr><td>neither</td><td>{len(lost)}</td>"
        f"<td>{len(lost) / total:.0%}</td></tr>"
    )

    by_width: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    formats: dict[int, set[str]] = collections.defaultdict(set)
    for name, bits, _, _, _, ranking in cells:
        winner = ranking[0][0]
        by_width[bits][winner if winner in FULL_LUTS else "other"] += 1
        formats[bits].add(base.label(name))
    width_rows = "".join(
        f"<tr><td>{bits}</td><td>{by_width[bits]['full_lut_shared']}</td>"
        f"<td>{by_width[bits]['full_lut_global']}</td>"
        f"<td>{by_width[bits]['other']}</td>"
        f"<td>{', '.join(sorted(formats[bits]))}</td></tr>"
        for bits in sorted(by_width)
    )

    detail = []
    for name, bits, compute, kernel, scope, ranking in lost:
        best = ranking[0][1]
        rows = "".join(
            f"<tr><td>{rank}</td><td><code>{decoder}</code></td>"
            f"<td>{score:.4f}</td><td>{score / best:.3f}x</td></tr>"
            for rank, (decoder, score) in enumerate(ranking, 1)
        )
        detail.append(
            f"<h4>{base.label(name)} &mdash; {kernel.upper()} {scope} "
            f"&rarr; {compute.upper()}</h4>"
            '<div class="table-wrap"><table class="strategy-table">'
            "<thead><tr><th>rank</th><th>decoder</th><th>ms</th>"
            "<th>vs best</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    return (
        '<section class="text-section"><h2>&le; 14 bit width</h2>'
        "<p><code>full_lut_shared</code> and <code>full_lut_global</code> "
        "dominate.  Only at the widest 14 bits (E5M8) does another strategy "
        "win.</p>"
        "<h3>Win rate</h3>"
        '<div class="table-wrap"><table class="strategy-table">'
        "<thead><tr><th>decoder</th><th>cells</th><th>share</th></tr></thead>"
        f"<tbody>{rate_rows}</tbody></table></div>"
        f"<p>A full LUT wins {sum(wins.values())} of {total} &mdash; "
        f"{100 * sum(wins.values()) / total:.1f}%.</p>"
        "<h3>By width</h3>"
        '<div class="table-wrap"><table class="strategy-table">'
        "<thead><tr><th>bits</th><th>shared</th><th>global</th><th>other</th>"
        "<th>formats</th></tr></thead>"
        f"<tbody>{width_rows}</tbody></table></div>"
        f"<details><summary>The {len(lost)} cells a full LUT lost</summary>"
        f"{''.join(detail)}</details></section>"
    )


# Dense beats padded by more than this, or the cell is not worth listing.
LAYOUT_TIE_BAND = 0.025


def layout_best(
    full: dict, screen: dict, compute: str, name: str, kernel: str, scope: str
) -> dict[str, tuple[float, str]]:
    """Fastest variant per storage layout, with the variant that achieved it."""
    out: dict[str, tuple[float, str]] = {}
    for key in set(full) | set(screen):
        if key[:3] != (compute, name, kernel):
            continue
        layout, access, lanes, decoder = key[3].split("/")
        if scope == "x1" and not (access == "scalar" and lanes == "x1"):
            continue
        score = variant_score(full[key] if key in full else screen[key])
        if score is None:
            continue
        if layout not in out or score < out[layout][0]:
            out[layout] = (score, f"{access}/{lanes}/{decoder}")
    return out


def dense_wins(full: dict, screen: dict, widths: dict) -> list[tuple]:
    """Cells where the dense layout beat padded by more than the tie band.

    Byte-aligned widths are skipped: their "padded" runs are the same bytes in
    memory under another label, so the comparison measures the loader alone.
    """
    found = []
    for compute in ("fp32", "fp64"):
        names = sorted(
            {name for (arith, name, _, _) in full if arith == compute},
            key=lambda name: widths[name][1],
        )
        known = set(ieee.compute_formats(compute))
        for name in dict.fromkeys(names):
            # `raw_fp32`/`raw_fp64` are anchor series, not storage formats.
            if name not in known or not ieee.padded_is_distinct(compute, name):
                continue
            for scope in ("x1", "best"):
                for kernel in ("dot", "gemv"):
                    best = layout_best(full, screen, compute, name, kernel, scope)
                    dense, padded = best.get("dense"), best.get("padded")
                    if not dense or not padded:
                        continue
                    gap = padded[0] / dense[0]
                    if gap - 1 >= LAYOUT_TIE_BAND:
                        found.append(
                            (name, widths[name][1], compute, kernel, scope,
                             dense[0], padded[0], gap, dense[1])
                        )
    return sorted(found, key=lambda row: (row[1], row[2], row[4], -row[7]))


def layout_section(run_dir: Path) -> str:
    full = read_stage(run_dir / "unified_core/full/timing_summary.csv")
    screen = read_stage(run_dir / "unified_core/screen/timing_summary.csv")
    widths: dict[str, tuple[int, int]] = {}
    with (run_dir / "unified_core/full/timing_summary.csv").open(encoding="utf-8") as h:
        for row in csv.DictReader(h):
            widths[row["format"]] = (int(row["exponent_bits"]), int(row["bits"]))
    wins = dense_wins(full, screen, widths)
    win_rows = "".join(
        "<tr>"
        f"<td>{base.label(name)}</td><td>{bits}</td><td>{compute.upper()}</td>"
        f"<td>{kernel.upper()}</td><td>{scope}</td>"
        f"<td>{dense:.4f}</td><td>{padded:.4f}</td><td>{gap:.3f}x</td>"
        f"<td><code>{via}</code></td></tr>"
        for name, bits, compute, kernel, scope, dense, padded, gap, via in wins
    )
    container_rows = "".join(
        f"<tr><td>{label}</td><td><code>{container}</code></td>"
        f"<td>{size}</td><td>{waste}</td></tr>"
        for label, container, size, waste in (
            ("2&ndash;8", "uint8_t", "1", "0&ndash;75%"),
            ("9&ndash;16", "uint16_t", "2", "0&ndash;44%"),
            ("17&ndash;32", "uint32_t", "4", "0&ndash;47%"),
        )
    )
    return (
        '<section class="text-section"><h2>Dense vs padded</h2>'
        "<p>Dense packs values back to back and reads them out of a 32-bit "
        "word pair:</p>"
        + code(
            """// one value, any width
bit   = index * Bits;
pair  = words[bit / 32] | (words[bit / 32 + 1] << 32);   // two loads
value = (pair >> (bit % 32)) & mask;                     // shift + mask"""
        )
        + "<p>Padded gives every value its own container and reads it "
        "directly:</p>"
        + code(
            """// one value, container is uint8_t / uint16_t / uint32_t
value = data[index] & mask;                              // one load"""
        )
        + "<p>The container is the smallest of 1, 2 or 4 bytes that holds the "
        "format:</p>"
        '<div class="table-wrap"><table class="strategy-table">'
        "<thead><tr><th>format bits</th><th>container</th>"
        "<th>bytes/value</th><th>wasted</th></tr></thead>"
        f"<tbody>{container_rows}</tbody></table></div>"
        "<p>At small widths the kernel is limited by per-value memory "
        "instructions, not by bytes.  Spending extra bandwidth to halve the "
        "load count is a good trade.</p>"
        "<p>For GEMV padded always wins.  For DOT dense wins only where the "
        "container would waste a lot: 9 and 10 bits unpacked, marginally at "
        "12, and 17&ndash;28 bits once packing is available &mdash; and every "
        "one of those wide wins is a shift decoder.</p>"
        f"<details><summary>The {len(wins)} cells where dense won</summary>"
        '<div class="table-wrap"><table class="strategy-table">'
        "<thead><tr><th>type</th><th>bits</th><th>cmp</th><th>kernel</th>"
        "<th>scope</th><th>dense</th><th>padded</th><th>gap</th>"
        "<th>dense via</th></tr></thead>"
        f"<tbody>{win_rows}</tbody></table></div></details></section>"
    )


def simple_table(headings: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "".join(f"<th>{h}</th>" for h in headings)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        '<div class="table-wrap"><table class="strategy-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{body_rows}</tbody>"
        "</table></div>"
    )


def packing_section() -> str:
    """Packing and shuffle.

    Hardcoded rather than generated: the result is a clean sweep, so there is
    nothing here that a re-run would refine, and the prose carries as many
    figures as the tables do.
    """
    return (
        '<section class="text-section"><h2>Packing and shuffle</h2>'
        "<p>Packing always wins, x8 usually better.</p>"
        "<details><summary>Every winner is a packet</summary>"
        "<p>Across 284 best-scope cells (every format &times; compute &times; "
        "kernel):</p>"
        + simple_table(
            ("winning access method", "cells", "share"),
            (
                ("<code>thread_packet</code>", "284", "100%"),
                ("<code>scalar</code> (x1)", "0", "0%"),
                ("<code>cooperative_shuffle</code>", "0", "0%"),
            ),
        )
        + simple_table(
            ("winning packet width", "cells", "share"),
            (("x8", "254", "89%"), ("x4", "30", "11%"),
             ("x2", "0", "0%"), ("x1", "0", "0%")),
        )
        + "<h3>x1 never wins &mdash; not once, not even close</h3>"
        "<p>Packing beats <code>scalar/x1</code> in every cell, median 1.536x, "
        "range 1.056x&ndash;2.884x.  Zero cells where x1 is within 2.5% of the "
        "best.</p>"
        "<p>The narrowest margins are all 14&ndash;16 bit FP64 GEMV &mdash; "
        "E7M8 1.056x, E2M13 1.059x, E6M9 1.078x.  That is the regime where the "
        "container is already 2&ndash;4 bytes so a packet load is not gathering "
        "much more per instruction.</p>"
        "<h3>Shuffle never wins, and is not close</h3>"
        "<p><code>cooperative_shuffle</code> exists in 258 cells and loses all "
        "of them:</p>"
        + simple_table(
            ("", "vs best"),
            (("best case", "1.220x behind"), ("median", "2.027x behind"),
             ("worst", "5.013x behind")),
        )
        + "<p>Its closest approach anywhere is E2M11 at 14 bits, FP64 GEMV, "
        "still 1.22x off.  Median means it is typically half the speed of the "
        "winner.</p>"
        "<h3>x4 beats x8 only marginally, and almost only at 8 bits</h3>"
        "<p>24 of the 30 x4 wins are 8-bit formats, and the margins are "
        "tiny:</p>"
        + simple_table(
            ("gap over x8", "count"),
            (("&lt; 1.02x", "17"), ("1.02x &ndash; 1.05x", "12"),
             ("&gt; 1.05x", "1 (E8M8, 1.063x)")),
        )
        + "<p>Half of them are inside 1.5%, i.e. ties.  The largest is E8M8 at "
        "17 bits FP32 DOT (1.063x), and there are four FP64 GEMV 8-bit cases "
        "around 1.04&ndash;1.05x.  Everything else is noise-level.</p>"
        "<h3>Practical reading</h3>"
        "<p>Always pack, always x8.  The rule costs at most 1.063x in the "
        "single worst cell and typically nothing, while not packing costs 1.5x "
        "on median.  x2 and x1 are never right, and "
        "<code>cooperative_shuffle</code> is never right by a wide "
        "margin.</p>"
        "</details></section>"
    )


def useful_types_section() -> str:
    """Which FP32 types are worth using, at x1.

    Hardcoded: the grouping carries judgement a generator would have to encode
    as special cases anyway -- FP4 is pulled out of the native group, and
    E0M15/E1M14 are pulled out of their structural group because they beat raw
    and nothing wider beats them.
    """
    ratio = ("type", "bits", "DOT", "GEMV")
    timed = ("type", "bits", "ms", "peer", "bits", "ms", "speedup")
    return (
        '<section class="text-section"><h2>Useful IEEE types</h2>'
        "<details><summary>FP32, x1 case</summary>"
        "<p>Peer is the fastest format carrying at least as much exponent and mantissa, with plain FP32 in the pool.  Ratios are peer time over format time, so above 1 means the format wins.  Anything from 0.97x to 1.03x is a tie and counts as useless.</p>"
        "<p>E8 shares FP32&rsquo;s exponent width and bias, so its decode is a single left shift.  <code>direct_branchy</code>, <code>exponent_only</code>, <code>native_scalar</code> and <code>full_lut_global</code> all land within 1.2% of each other on those formats, so every E8 strategy is labelled <code>shift</code>.</p>"
        "<h3>Useful in both kernels, 3 formats</h3>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>peer DOT</th><th>DOT strategy</th><th>DOT</th><th>peer GEMV</th><th>GEMV strategy</th><th>GEMV</th></tr></thead><tbody><tr><td>E5M10 aka FP16</td><td>16</td><td>E8M11</td><td><code>native</code></td><td>1.06x</td><td>raw FP32</td><td><code>native</code></td><td>1.84x</td></tr><tr><td>E8M7 aka BF16</td><td>16</td><td>E8M11</td><td><code>shift</code></td><td>1.05x</td><td>E8M8</td><td><code>shift</code></td><td>1.82x</td></tr><tr><td>E1M14</td><td>16</td><td>raw FP32</td><td><code>e1_integer</code></td><td>1.03x</td><td>raw FP32</td><td><code>e1_integer</code></td><td>1.07x</td></tr></tbody></table></div>'
        "<h3>Useful in DOT only, 15 formats</h3>"
        "<h4>Native, 2</h4>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>peer DOT</th><th>DOT strategy</th><th>DOT</th><th>peer GEMV</th><th>GEMV strategy</th><th>GEMV</th></tr></thead><tbody><tr><td>E4M3 aka FP8</td><td>8</td><td>E4M4</td><td><code>native</code></td><td>1.33x</td><td>FP16</td><td><code>native</code></td><td>0.91x</td></tr><tr><td>E5M2 aka FP8</td><td>8</td><td>E5M4</td><td><code>native</code></td><td>1.56x</td><td>FP16</td><td><code>native</code></td><td>0.90x</td></tr></tbody></table></div>'
        "<h4>Integer, 3</h4>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>peer DOT</th><th>DOT strategy</th><th>DOT</th><th>peer GEMV</th><th>GEMV strategy</th><th>GEMV</th></tr></thead><tbody><tr><td>E0M7</td><td>8</td><td>E0M8</td><td><code>fixed_integer</code></td><td>1.21x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.83x</td></tr><tr><td>E1M6</td><td>8</td><td>E2M7</td><td><code>e1_integer</code></td><td>1.55x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.85x</td></tr><tr><td>E0M8</td><td>9</td><td>FP16</td><td><code>fixed_integer</code></td><td>1.70x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr></tbody></table></div>'
        "<h4>Full LUT, 7</h4>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>peer DOT</th><th>DOT strategy</th><th>DOT</th><th>peer GEMV</th><th>GEMV strategy</th><th>GEMV</th></tr></thead><tbody><tr><td>E2M5</td><td>8</td><td>E2M7</td><td><code>full_lut_shared</code></td><td>1.49x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.85x</td></tr><tr><td>E3M4</td><td>8</td><td>E4M4</td><td><code>full_lut_shared</code></td><td>1.25x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.85x</td></tr><tr><td>E6M1</td><td>8</td><td>E8M1</td><td><code>full_lut_shared</code></td><td>1.30x</td><td>E8M5</td><td><code>full_lut_shared</code></td><td>0.88x</td></tr><tr><td>E7M0</td><td>8</td><td>E8M0</td><td><code>full_lut_shared</code></td><td>1.08x</td><td>E8M0</td><td><code>full_lut_shared</code></td><td>0.88x</td></tr><tr><td>E4M4</td><td>9</td><td>E5M4</td><td><code>full_lut_shared</code></td><td>1.17x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.60x</td></tr><tr><td>E2M7</td><td>10</td><td>FP16</td><td><code>full_lut_shared</code></td><td>1.33x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.70x</td></tr><tr><td>E5M4</td><td>10</td><td>FP16</td><td><code>full_lut_shared</code></td><td>1.37x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.68x</td></tr></tbody></table></div>'
        "<h4>Shift, 3</h4>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>peer DOT</th><th>DOT strategy</th><th>DOT</th><th>peer GEMV</th><th>GEMV strategy</th><th>GEMV</th></tr></thead><tbody><tr><td>E8M0</td><td>9</td><td>E8M1</td><td><code>shift</code></td><td>1.21x</td><td>E8M5</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M1</td><td>10</td><td>E8M3</td><td><code>shift</code></td><td>1.47x</td><td>E8M5</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M3</td><td>12</td><td>BF16</td><td><code>shift</code></td><td>1.05x</td><td>E8M5</td><td><code>shift</code></td><td>1.00x</td></tr></tbody></table></div>'
        "<p>FP16 is the GEMV peer for 8 of these 15 and beats them by 1.10x to 1.67x.  That single format is why the group is DOT only.  The three shift entries tie GEMV rather than losing it, which puts them a tier above the rest.</p>"
        "<h3>Useful in GEMV only, 2 formats</h3>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>peer DOT</th><th>DOT strategy</th><th>DOT</th><th>peer GEMV</th><th>GEMV strategy</th><th>GEMV</th></tr></thead><tbody><tr><td>E0M11</td><td>12</td><td>E1M14</td><td><code>fixed_integer</code></td><td>1.02x</td><td>E0M15</td><td><code>full_lut_shared</code></td><td>1.11x</td></tr><tr><td>E0M15</td><td>16</td><td>raw FP32</td><td><code>fixed_integer</code></td><td>1.02x</td><td>raw FP32</td><td><code>fixed_integer</code></td><td>1.07x</td></tr></tbody></table></div>'
        "<h3>E8, the shift family, 10 formats</h3>"
        "<p>A cross-cutting view.  These rows also appear in the tables above and below.</p>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>peer DOT</th><th>DOT strategy</th><th>DOT</th><th>peer GEMV</th><th>GEMV strategy</th><th>GEMV</th></tr></thead><tbody><tr><td>E8M0</td><td>9</td><td>E8M1</td><td><code>shift</code></td><td>1.21x</td><td>E8M5</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M1</td><td>10</td><td>E8M3</td><td><code>shift</code></td><td>1.47x</td><td>E8M5</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M3</td><td>12</td><td>BF16</td><td><code>shift</code></td><td>1.05x</td><td>E8M5</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M5</td><td>14</td><td>BF16</td><td><code>shift</code></td><td>1.00x</td><td>BF16</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M7 aka BF16</td><td>16</td><td>E8M11</td><td><code>shift</code></td><td>1.05x</td><td>E8M8</td><td><code>shift</code></td><td>1.82x</td></tr><tr><td>E8M8</td><td>17</td><td>E8M11</td><td><code>shift</code></td><td>0.99x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M11</td><td>20</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M15</td><td>24</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M19</td><td>28</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M23 aka FP32</td><td>32</td><td>raw FP32</td><td><code>shift</code></td><td>0.99x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr></tbody></table></div>'
        "<p>GEMV is a step function: 0.0187 ms from 9 through 16 bits, 0.0340 from 17 up.  That is the container boundary, and with decode free the time is purely bytes moved.  Every format from E8M0 to E8M5 is interchangeable in GEMV, so take E8M5 and get five more mantissa bits for nothing, or BF16 for seven.</p>"
        "<h3>Useless, 43 formats</h3>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>peer DOT</th><th>DOT strategy</th><th>DOT</th><th>peer GEMV</th><th>GEMV strategy</th><th>GEMV</th></tr></thead><tbody><tr><td>E8M11</td><td>20</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M5</td><td>14</td><td>BF16</td><td><code>shift</code></td><td>1.00x</td><td>BF16</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M19</td><td>28</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M15</td><td>24</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M23 aka FP32</td><td>32</td><td>raw FP32</td><td><code>shift</code></td><td>0.99x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E8M8</td><td>17</td><td>E8M11</td><td><code>shift</code></td><td>0.99x</td><td>raw FP32</td><td><code>shift</code></td><td>1.00x</td></tr><tr><td>E0M23</td><td>24</td><td>raw FP32</td><td><code>fixed_integer</code></td><td>0.99x</td><td>raw FP32</td><td><code>fixed_integer</code></td><td>0.95x</td></tr><tr><td>E7M8</td><td>16</td><td>E8M11</td><td><code>direct_branchy</code></td><td>0.94x</td><td>E8M8</td><td><code>prefix_lut_shared</code></td><td>0.87x</td></tr><tr><td>E6M9</td><td>16</td><td>E8M11</td><td><code>direct_branchy</code></td><td>0.93x</td><td>raw FP32</td><td><code>prefix_lut_shared</code></td><td>0.89x</td></tr><tr><td>E5M11</td><td>17</td><td>E8M11</td><td><code>direct_branchy</code></td><td>0.90x</td><td>raw FP32</td><td><code>prefix_lut_shared</code></td><td>0.78x</td></tr><tr><td>E5M14</td><td>20</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.89x</td><td>raw FP32</td><td><code>prefix_lut_shared</code></td><td>0.78x</td></tr><tr><td>E5M18</td><td>24</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.89x</td><td>raw FP32</td><td><code>prefix_lut_shared</code></td><td>0.78x</td></tr><tr><td>E5M22</td><td>28</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.89x</td><td>raw FP32</td><td><code>prefix_lut_shared</code></td><td>0.78x</td></tr><tr><td>E0M6</td><td>7</td><td>E1M6</td><td><code>fixed_integer</code></td><td>1.01x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.75x</td></tr><tr><td>E3M2</td><td>6</td><td>FP8 E5M2</td><td><code>full_lut_shared</code></td><td>0.92x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.74x</td></tr><tr><td>E3M3</td><td>7</td><td>FP8 E4M3</td><td><code>full_lut_shared</code></td><td>0.92x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E2M3</td><td>6</td><td>FP8 E4M3</td><td><code>full_lut_shared</code></td><td>0.92x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E1M4</td><td>6</td><td>E1M6</td><td><code>e1_integer</code></td><td>1.00x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E4M1</td><td>6</td><td>FP8 E5M2</td><td><code>full_lut_shared</code></td><td>0.92x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E5M0</td><td>6</td><td>FP8 E5M2</td><td><code>full_lut_shared</code></td><td>0.92x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E5M1</td><td>7</td><td>FP8 E5M2</td><td><code>full_lut_shared</code></td><td>0.91x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E0M5</td><td>6</td><td>E0M6</td><td><code>fixed_integer</code></td><td>1.00x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E1M1</td><td>3</td><td>FP8 E5M2</td><td><code>e1_integer</code></td><td>0.98x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E2M1 aka FP4</td><td>4</td><td>FP8 E5M2</td><td><code>full_lut_shared</code></td><td>0.92x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E0M1</td><td>2</td><td>FP8 E5M2</td><td><code>fixed_integer</code></td><td>0.97x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E1M0</td><td>2</td><td>FP8 E5M2</td><td><code>direct_branchy</code></td><td>0.96x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E2M2</td><td>5</td><td>FP8 E5M2</td><td><code>full_lut_shared</code></td><td>0.93x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E2M0</td><td>3</td><td>FP8 E5M2</td><td><code>full_lut_shared</code></td><td>0.92x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E0M2</td><td>3</td><td>FP8 E5M2</td><td><code>fixed_integer</code></td><td>0.95x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E0M4</td><td>5</td><td>E0M6</td><td><code>fixed_integer</code></td><td>0.99x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E1M2</td><td>4</td><td>FP8 E5M2</td><td><code>e1_integer</code></td><td>0.97x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E4M0</td><td>5</td><td>FP8 E5M2</td><td><code>full_lut_shared</code></td><td>0.92x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E0M3</td><td>4</td><td>FP8 E4M3</td><td><code>fixed_integer</code></td><td>0.97x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E3M0</td><td>4</td><td>FP8 E5M2</td><td><code>full_lut_shared</code></td><td>0.92x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.73x</td></tr><tr><td>E4M11</td><td>16</td><td>E8M11</td><td><code>direct_branchy</code></td><td>0.88x</td><td>raw FP32</td><td><code>prefix_lut_shared</code></td><td>0.70x</td></tr><tr><td>E2M11</td><td>14</td><td>E8M11</td><td><code>full_lut_global</code></td><td>0.88x</td><td>raw FP32</td><td><code>subnormal_lut_shared</code></td><td>0.70x</td></tr><tr><td>E4M23</td><td>28</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.84x</td><td>raw FP32</td><td><code>prefix_lut_shared</code></td><td>0.65x</td></tr><tr><td>E2M13</td><td>16</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.83x</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.61x</td></tr><tr><td>E5M6</td><td>12</td><td>FP16</td><td><code>full_lut_global</code></td><td>0.93x</td><td>FP16</td><td><code>full_lut_shared</code></td><td>0.61x</td></tr><tr><td>E2M14</td><td>17</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.80x</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.56x</td></tr><tr><td>E2M17</td><td>20</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.80x</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.55x</td></tr><tr><td>E3M12</td><td>16</td><td>raw FP32</td><td><code>direct_branchy</code></td><td>0.78x</td><td>raw FP32</td><td><code>subnormal_lut_global</code></td><td>0.55x</td></tr><tr><td>E5M8</td><td>14</td><td>FP16</td><td><code>full_lut_global</code></td><td>0.86x</td><td>FP16</td><td><code>prefix_lut_shared</code></td><td>0.46x</td></tr></tbody></table></div>'
        "<p>Three bands.  The first six rows are E8 formats tying raw FP32 or BF16, useless because something more precise costs the same, not because they are slow.  Rows 14 to 34 are the sub-8-bit range, flat at roughly 0.92x DOT and 0.73x GEMV because they share the 1-byte container and the same table decoder.  The last nine are the real failures, 12 to 28 bits with no cheap decoder, ending at E5M8 losing GEMV to FP16 by 0.46x.</p>"
        "<p>3 + 15 + 2 + 43 = 63, with the E8 table repeating 10 of those rows.</p>"
        "</details>"
        "<details><summary>FP32, best case</summary>"
        "<p>The <code>raw</code> anchor is unusable here: its x8 variant runs 1.85x slower than its x4.  E8M23 stores the same 4-byte floats and reaches 0.0377 ms in DOT and 0.0204 ms in GEMV through <code>padded/x8/direct_branchy</code>, so that is the plain-FP32 reference.  DOT and GEMV columns compare against it; under 1 is faster.</p>"
        "<p>The peer columns name the fastest format that carries at least as much exponent and mantissa, and the ratio is peer time over format time, so above 1 means the format wins.  Every one of the 62 formats has such a peer, because FP16, BF16 and FP32 itself are more precise than almost everything.</p>"
        "<h3>Beats every more-precise peer in both kernels, 4 formats</h3>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>DOT</th><th>DOT peer</th><th>ratio</th><th>GEMV</th><th>GEMV peer</th><th>ratio</th></tr></thead><tbody><tr><td>E8M7 aka BF16</td><td>16</td><td>0.61x</td><td>E8M11</td><td>1.24x</td><td>0.53x</td><td>E8M8</td><td>1.88x</td></tr><tr><td>E5M10 aka FP16</td><td>16</td><td>0.62x</td><td>E8M11</td><td>1.23x</td><td>0.49x</td><td>E8M11</td><td>2.05x</td></tr><tr><td>E0M15</td><td>16</td><td>0.71x</td><td>E8M15</td><td>1.22x</td><td>0.94x</td><td>E8M15</td><td>1.06x</td></tr><tr><td>E1M14</td><td>16</td><td>0.71x</td><td>E8M15</td><td>1.21x</td><td>0.94x</td><td>E8M15</td><td>1.06x</td></tr></tbody></table></div>'
        "<p>All four are 16-bit.  FP16 and BF16 beat their nearest more-precise alternative by close to 2x in GEMV, the widest margin in the study.</p>"
        "<h3>Beats peers in DOT only, 15 formats</h3>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>DOT</th><th>DOT peer</th><th>ratio</th><th>GEMV</th><th>GEMV peer</th><th>ratio</th></tr></thead><tbody><tr><td>E6M1</td><td>8</td><td>0.49x</td><td>E8M7 aka BF16</td><td>1.26x</td><td>0.75x</td><td>E8M7 aka BF16</td><td>0.70x</td></tr><tr><td>E5M2 aka FP8</td><td>8</td><td>0.49x</td><td>E8M7 aka BF16</td><td>1.25x</td><td>0.76x</td><td>E5M10 aka FP16</td><td>0.64x</td></tr><tr><td>E4M3 aka FP8</td><td>8</td><td>0.49x</td><td>E8M7 aka BF16</td><td>1.25x</td><td>0.76x</td><td>E5M10 aka FP16</td><td>0.64x</td></tr><tr><td>E7M0</td><td>8</td><td>0.49x</td><td>E8M0</td><td>1.25x</td><td>0.76x</td><td>E8M7 aka BF16</td><td>0.70x</td></tr><tr><td>E2M5</td><td>8</td><td>0.53x</td><td>E8M7 aka BF16</td><td>1.16x</td><td>0.84x</td><td>E5M10 aka FP16</td><td>0.58x</td></tr><tr><td>E1M6</td><td>8</td><td>0.53x</td><td>E8M7 aka BF16</td><td>1.15x</td><td>0.84x</td><td>E5M10 aka FP16</td><td>0.58x</td></tr><tr><td>E3M4</td><td>8</td><td>0.53x</td><td>E8M7 aka BF16</td><td>1.15x</td><td>0.85x</td><td>E5M10 aka FP16</td><td>0.57x</td></tr><tr><td>E8M11</td><td>20</td><td>0.76x</td><td>E8M15</td><td>1.13x</td><td>1.00x</td><td>E8M15</td><td>1.00x</td></tr><tr><td>E8M15</td><td>24</td><td>0.86x</td><td>E8M19</td><td>1.12x</td><td>1.00x</td><td>E8M23 aka FP32</td><td>1.00x</td></tr><tr><td>E0M7</td><td>8</td><td>0.57x</td><td>E8M7 aka BF16</td><td>1.08x</td><td>0.92x</td><td>E5M10 aka FP16</td><td>0.53x</td></tr><tr><td>E1M4</td><td>6</td><td>0.50x</td><td>E2M5</td><td>1.05x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E8M19</td><td>28</td><td>0.96x</td><td>E8M23 aka FP32</td><td>1.04x</td><td>1.00x</td><td>E8M23 aka FP32</td><td>1.00x</td></tr><tr><td>E0M23</td><td>24</td><td>0.96x</td><td>E8M23 aka FP32</td><td>1.04x</td><td>1.08x</td><td>E8M23 aka FP32</td><td>0.93x</td></tr><tr><td>E0M6</td><td>7</td><td>0.51x</td><td>E1M6</td><td>1.03x</td><td>0.84x</td><td>E5M10 aka FP16</td><td>0.58x</td></tr><tr><td>E0M5</td><td>6</td><td>0.50x</td><td>E0M6</td><td>1.03x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr></tbody></table></div>'
        "<p>E8M11, E8M15 and E8M19 are a distinct case inside this group: their GEMV ratio is exactly 1.00x, a tie rather than a loss, so they are DOT wins with GEMV parity.  The rest lose GEMV to FP16 or BF16 by 0.53x to 0.93x.</p>"
        "<h3>Beaten by a more-precise peer, 43 formats</h3>"
        '<div class="table-wrap"><table class="strategy-table"><thead><tr><th>type</th><th>bits</th><th>DOT</th><th>DOT peer</th><th>ratio</th><th>GEMV</th><th>GEMV peer</th><th>ratio</th></tr></thead><tbody><tr><td>E8M0</td><td>9</td><td>0.61x</td><td>E8M7 aka BF16</td><td>1.00x</td><td>0.58x</td><td>E8M7 aka BF16</td><td>0.91x</td></tr><tr><td>E8M3</td><td>12</td><td>0.61x</td><td>E8M7 aka BF16</td><td>1.00x</td><td>0.58x</td><td>E8M7 aka BF16</td><td>0.91x</td></tr><tr><td>E0M4</td><td>5</td><td>0.50x</td><td>E0M5</td><td>1.00x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.61x</td></tr><tr><td>E8M1</td><td>10</td><td>0.62x</td><td>E8M7 aka BF16</td><td>0.98x</td><td>0.58x</td><td>E8M7 aka BF16</td><td>0.91x</td></tr><tr><td>E8M5</td><td>14</td><td>0.63x</td><td>E8M7 aka BF16</td><td>0.97x</td><td>0.58x</td><td>E8M7 aka BF16</td><td>0.91x</td></tr><tr><td>E2M2</td><td>5</td><td>0.50x</td><td>E5M2 aka FP8</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E5M1</td><td>7</td><td>0.50x</td><td>E6M1</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E0M2</td><td>3</td><td>0.50x</td><td>E5M2 aka FP8</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.61x</td></tr><tr><td>E2M1 aka FP4</td><td>4</td><td>0.50x</td><td>E6M1</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E3M3</td><td>7</td><td>0.50x</td><td>E4M3 aka FP8</td><td>0.97x</td><td>0.81x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E0M11</td><td>12</td><td>0.73x</td><td>E0M15</td><td>0.97x</td><td>0.97x</td><td>E1M14</td><td>0.97x</td></tr><tr><td>E0M3</td><td>4</td><td>0.50x</td><td>E4M3 aka FP8</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.61x</td></tr><tr><td>E1M1</td><td>3</td><td>0.50x</td><td>E6M1</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.61x</td></tr><tr><td>E1M2</td><td>4</td><td>0.50x</td><td>E5M2 aka FP8</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.61x</td></tr><tr><td>E1M0</td><td>2</td><td>0.50x</td><td>E6M1</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E0M1</td><td>2</td><td>0.50x</td><td>E6M1</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E2M3</td><td>6</td><td>0.50x</td><td>E4M3 aka FP8</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E3M2</td><td>6</td><td>0.50x</td><td>E5M2 aka FP8</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E2M0</td><td>3</td><td>0.50x</td><td>E6M1</td><td>0.97x</td><td>0.81x</td><td>E5M10 aka FP16</td><td>0.60x</td></tr><tr><td>E4M0</td><td>5</td><td>0.50x</td><td>E6M1</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.61x</td></tr><tr><td>E3M0</td><td>4</td><td>0.50x</td><td>E6M1</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.61x</td></tr><tr><td>E5M0</td><td>6</td><td>0.50x</td><td>E6M1</td><td>0.97x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.61x</td></tr><tr><td>E4M1</td><td>6</td><td>0.50x</td><td>E6M1</td><td>0.96x</td><td>0.80x</td><td>E5M10 aka FP16</td><td>0.61x</td></tr><tr><td>E5M22</td><td>28</td><td>1.08x</td><td>E8M23 aka FP32</td><td>0.93x</td><td>1.64x</td><td>E8M23 aka FP32</td><td>0.61x</td></tr><tr><td>E0M8</td><td>9</td><td>0.67x</td><td>E5M10 aka FP16</td><td>0.92x</td><td>0.82x</td><td>E5M10 aka FP16</td><td>0.59x</td></tr><tr><td>E5M18</td><td>24</td><td>1.07x</td><td>E8M19</td><td>0.90x</td><td>1.63x</td><td>E8M23 aka FP32</td><td>0.61x</td></tr><tr><td>E4M4</td><td>9</td><td>0.70x</td><td>E8M7 aka BF16</td><td>0.88x</td><td>0.95x</td><td>E5M10 aka FP16</td><td>0.51x</td></tr><tr><td>E5M4</td><td>10</td><td>0.71x</td><td>E8M7 aka BF16</td><td>0.86x</td><td>0.94x</td><td>E5M10 aka FP16</td><td>0.52x</td></tr><tr><td>E2M7</td><td>10</td><td>0.74x</td><td>E8M7 aka BF16</td><td>0.82x</td><td>1.06x</td><td>E5M10 aka FP16</td><td>0.46x</td></tr><tr><td>E5M14</td><td>20</td><td>1.06x</td><td>E8M15</td><td>0.81x</td><td>1.63x</td><td>E8M15</td><td>0.61x</td></tr><tr><td>E8M8</td><td>17</td><td>0.95x</td><td>E8M11</td><td>0.80x</td><td>0.99x</td><td>E8M11</td><td>1.01x</td></tr><tr><td>E4M23</td><td>28</td><td>1.31x</td><td>E8M23 aka FP32</td><td>0.76x</td><td>2.10x</td><td>E8M23 aka FP32</td><td>0.48x</td></tr><tr><td>E5M6</td><td>12</td><td>0.85x</td><td>E8M7 aka BF16</td><td>0.72x</td><td>1.21x</td><td>E5M10 aka FP16</td><td>0.40x</td></tr><tr><td>E5M11</td><td>17</td><td>1.08x</td><td>E8M11</td><td>0.70x</td><td>1.63x</td><td>E8M11</td><td>0.61x</td></tr><tr><td>E7M8</td><td>16</td><td>1.12x</td><td>E8M11</td><td>0.68x</td><td>1.63x</td><td>E8M8</td><td>0.61x</td></tr><tr><td>E6M9</td><td>16</td><td>1.12x</td><td>E8M11</td><td>0.68x</td><td>1.62x</td><td>E8M11</td><td>0.62x</td></tr><tr><td>E2M17</td><td>20</td><td>1.44x</td><td>E8M19</td><td>0.67x</td><td>2.33x</td><td>E8M23 aka FP32</td><td>0.43x</td></tr><tr><td>E2M14</td><td>17</td><td>1.44x</td><td>E8M15</td><td>0.60x</td><td>2.33x</td><td>E8M15</td><td>0.43x</td></tr><tr><td>E2M13</td><td>16</td><td>1.50x</td><td>E8M15</td><td>0.57x</td><td>2.34x</td><td>E8M15</td><td>0.42x</td></tr><tr><td>E4M11</td><td>16</td><td>1.35x</td><td>E8M11</td><td>0.56x</td><td>2.07x</td><td>E8M11</td><td>0.48x</td></tr><tr><td>E2M11</td><td>14</td><td>1.41x</td><td>E8M11</td><td>0.54x</td><td>2.04x</td><td>E8M11</td><td>0.49x</td></tr><tr><td>E5M8</td><td>14</td><td>1.16x</td><td>E5M10 aka FP16</td><td>0.53x</td><td>1.66x</td><td>E5M10 aka FP16</td><td>0.29x</td></tr><tr><td>E3M12</td><td>16</td><td>1.64x</td><td>E8M15</td><td>0.53x</td><td>2.52x</td><td>E8M15</td><td>0.40x</td></tr></tbody></table></div>'
        "<p>The top of this table is the near-miss band.  E8M0 and E8M3 tie their peer in DOT and lose GEMV by 0.91x; E0M4 and E0M11 lose by 3% or less.  The bottom is where the losses are structural: everything from E5M22 down also loses to plain FP32 outright, and E3M12 at 0.40x GEMV is 2.5x slower than the baseline it was meant to shrink.</p>"
        "<p>4 + 15 + 43 + 1 baseline = 63.</p>"
        "</details>"
        "<details><summary>FP64, x1 case</summary>"
        "<p>raw_fp64 costs 0.0825 ms in DOT and 0.0435 ms in GEMV, at 8 bytes "
        "per value.  Every ratio compares against that; under 1 is faster.  "
        "79 formats.</p>"
        "<p>Padded containers come in 1, 2 or 4 bytes, so every format moves 2 "
        "to 8 times less data than raw.  Formats sharing a container move "
        "exactly the same bytes, which leaves decode cost and precision as the "
        "only things separating them.</p>"
        "<p>One variable predicts almost every outcome: does the format have a "
        "cheap decoder?</p>"
        + simple_table(
            ("cheap decoder", "trigger", "formats", "lose to raw"),
            (
                ("native", "hardware converter exists", "6", "0"),
                ("E11 shift", "<code>exponent_bits == 11</code>", "8", "0"),
                ("integer", "<code>exponent_bits &le; 1</code>", "16", "0"),
                ("full LUT", "<code>bits &le; 14</code>", "19", "1"),
                ("none", "16 bits or wider, E&ge;2", "23", "19"),
            ),
        )
        + "<p>56 formats have one and lose once between them.  The 23 without "
        "lose 19 times.</p>"

        "<h3>Useful, 29 formats</h3>"
        "<h4>8 bits, 8 formats</h4>"
        "<p>All eight win.  Six decode through <code>full_lut_shared</code>, "
        "the two FP8s through hardware.  E0M7 and E1M6 win with the table, not "
        "the integer decoder: at 8 bits a 256-entry lookup beats the "
        "arithmetic.</p>"
        + simple_table(
            ("type", "DOT", "GEMV", "decoder"),
            (
                ("E6M1", "0.40x", "0.62x", "<code>full_lut_shared</code>"),
                ("E5M2 aka FP8", "0.40x", "0.58x", "<code>native_scalar</code>"),
                ("E4M3 aka FP8", "0.40x", "0.59x", "<code>native_scalar</code>"),
                ("E0M7", "0.40x", "0.54x", "<code>full_lut_shared</code>"),
                ("E1M6", "0.40x", "0.58x", "<code>full_lut_shared</code>"),
                ("E7M0", "0.40x", "0.62x", "<code>full_lut_shared</code>"),
                ("E3M4", "0.41x", "0.58x", "<code>full_lut_shared</code>"),
                ("E2M5", "0.41x", "0.58x", "<code>full_lut_shared</code>"),
            ),
        )
        + "<h4>9 and 10 bits, DOT only, 6 formats</h4>"
        "<p>Every one uses dense for DOT and padded for GEMV.  Padding puts "
        "them in the same 2-byte container as the 16-bit formats, so GEMV gives "
        "them no bandwidth advantage and FP16 or BF16 beats each of them.  They "
        "earn their place in DOT, where dense buys 0.50x to 0.59x against the "
        "16-bit formats&rsquo; 0.82x.</p>"
        + simple_table(
            ("type", "DOT", "GEMV", "GEMV loses to"),
            (
                ("E5M3", "0.50x", "0.73x", "FP16, 1.41x"),
                ("E2M6", "0.51x", "0.80x", "FP16, 1.55x"),
                ("E8M0", "0.51x", "0.77x", "BF16, 1.47x"),
                ("E5M4", "0.58x", "0.72x", "FP16, 1.40x"),
                ("E8M1", "0.58x", "0.73x", "BF16, 1.39x"),
                ("E2M7", "0.59x", "0.73x", "FP16, 1.41x"),
            ),
        )
        + "<h4>Integer at 16 and 32 bits, 4 formats</h4>"
        "<p>The only non-native formats above 14 bits whose timing ignores the "
        "data.  All of the test data falls in E1M14&rsquo;s subnormal range and "
        "its time does not move, because the integer decoders never check.</p>"
        + simple_table(
            ("type", "bits", "DOT", "GEMV", "decoder"),
            (
                ("E0M15", "16", "0.84x", "0.84x", "<code>fixed_integer</code>"),
                ("E1M14", "16", "0.85x", "0.81x", "<code>e1_integer</code>"),
                ("E0M31", "32", "0.90x", "0.90x", "<code>fixed_integer</code>"),
                ("E1M30", "32", "0.90x", "0.90x", "<code>e1_integer</code>"),
            ),
        )
        + "<h4>E11, one shift, 8 formats</h4>"
        "<p>FP64&rsquo;s bias is 1023, which is also E11&rsquo;s bias, so "
        "decoding is a single left shift.  GEMV holds flat inside each "
        "container.</p>"
        + simple_table(
            ("type", "bits", "DOT", "GEMV"),
            (("E11M0", "12", "0.79x", "0.60x"), ("E11M2", "14", "0.81x", "0.59x"),
             ("E11M4", "16", "0.80x", "0.60x"), ("E11M5", "17", "0.84x", "0.78x"),
             ("E11M8", "20", "0.84x", "0.79x"), ("E11M12", "24", "0.85x", "0.78x"),
             ("E11M16", "28", "0.85x", "0.78x"), ("E11M20", "32", "0.85x", "0.78x")),
        )
        + "<h4>Native at 16 and 32 bits, 3 formats</h4>"
        + simple_table(
            ("type", "bits", "DOT", "GEMV"),
            (("E5M10 aka FP16", "16", "0.82x", "0.52x"),
             ("E8M7 aka BF16", "16", "0.82x", "0.53x"),
             ("E8M23 aka FP32", "32", "0.86x", "0.79x")),
        )
        + "<p>The five groups above pass a robustness check by construction.  "
        "<code>full_lut</code>, <code>fixed_integer</code>, "
        "<code>e1_integer</code>, hardware conversion and the E11 shift all run "
        "branchless or table-driven, so subnormal share cannot move their "
        "timing.</p>"

        "<h3>Useless, 50 formats</h3>"
        "<h4>Wins that vanish under scrutiny, 4 formats</h4>"
        "<p>These beat raw only because no test value lands in their subnormal "
        "range, which lets <code>prefix_lut_shared</code> skip its fallback.  "
        "Take that one decoder away and all four lose.</p>"
        + simple_table(
            ("type", "sub%", "GEMV, all decoders", "no prefix or subnormal LUT",
             "no branchy path at all"),
            (("E7M8", "0%", "0.93x", "1.04x", "1.24x"),
             ("E6M9", "0%", "0.95x", "1.02x", "1.21x"),
             ("E9M6", "0%", "0.97x", "1.07x", "1.26x"),
             ("E10M5", "0%", "1.01x", "1.04x", "1.24x")),
        )
        + "<h4>Redundant, 26 formats</h4>"
        "<p>Another format fits the same container, carries at least as much "
        "precision, and runs faster.  23 of these sit below 8 bits and lose to "
        "an 8-bit format sharing the 1-byte container.  The E0 family loses to "
        "E0M7 rather than FP8, since FP8 carries fewer mantissa bits than E0M4 "
        "and E0M5.</p>"
        + simple_table(
            ("loses to", "n", "types"),
            (("E5M2 aka FP8", "11",
              "E1M1, E2M0, E1M2, FP4, E3M0, E2M2, E3M1, E4M0, E3M2, E4M1, E5M0"),
             ("E0M7", "5", "E0M1, E0M2, E0M3, E0M4, E0M5"),
             ("E4M3 aka FP8", "3", "E1M3, E2M3, E3M3"),
             ("E5M10 aka FP16", "3", "E2M9, E5M6, E5M8"),
             ("E6M1", "2", "E1M0, E6M0"),
             ("E3M4", "2", "E1M4, E2M4")),
        )
        + "<h4>Loses to raw in GEMV, 11 formats</h4>"
        "<p>Each one wins DOT between 0.96x and 1.00x, then loses GEMV.  E2M11 "
        "at 1.26x, E4M11 at 1.33x, E5M11 and E5M14 and E5M18 at 1.04x, E5M22 at "
        "1.12x, E5M26 at 1.07x, E6M25 and E7M24 at 1.06x, E9M22 at 1.09x, "
        "E10M21 at 1.10x.</p>"
        "<h4>Loses to raw in both kernels, 9 formats</h4>"
        + simple_table(
            ("type", "bits", "DOT", "GEMV"),
            (("E4M27", "32", "1.06x", "1.58x"), ("E2M13", "16", "1.13x", "1.68x"),
             ("E3M12", "16", "1.15x", "1.52x"), ("E2M17", "20", "1.18x", "1.98x"),
             ("E2M14", "17", "1.19x", "1.97x"), ("E2M29", "32", "1.19x", "1.99x"),
             ("E2M21", "24", "1.20x", "2.01x"), ("E2M25", "28", "1.20x", "2.02x"),
             ("E3M28", "32", "1.27x", "2.17x")),
        )
        + "<p>Every one is E2, E3 or E4 at 16 bits or wider: high subnormal "
        "share, no cheap decoder.  No format with equal or better precision "
        "beats them either, so you drop them because storing doubles is faster, "
        "not because a better small format exists.</p>"
        "<p>29 + 4 + 26 + 11 + 9 = 79.</p>"
        "</details>"
        "</section>"
    )


def body(run_dir: Path | None) -> str:
    blocks = []
    for heading, paragraphs in SECTIONS:
        rendered = "".join(
            block if block.startswith("<pre>") else f"<p>{block}</p>"
            for block in paragraphs
        )
        blocks.append(
            '<section class="text-section">'
            f"<h2>{html.escape(heading)}</h2>{rendered}</section>"
        )
    if run_dir is not None:
        blocks.append(e0_section(run_dir))
    blocks.append(e1_section())
    if run_dir is not None:
        blocks.append(shift_section(run_dir))
        blocks.append(native_section(run_dir))
        blocks.append(narrow_section(run_dir))
        blocks.append(layout_section(run_dir))
    blocks.append(packing_section())
    blocks.append(useful_types_section())
    return "".join(blocks)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not (output_dir / "report.css").is_file():
        raise SystemExit(
            f"{output_dir} is not a built report; run the base report builder first"
        )
    manifest = read_manifest(output_dir / "report_manifest.txt")
    run_dir = newest_strategy_run(STRATEGY_ROOT)
    if run_dir is None:
        raise SystemExit(f"no complete strategy run under {STRATEGY_ROOT}")
    document = base.page_document(
        filename=FILENAME,
        title="IEEE, LNS summary",
        intro=INTRO,
        body=body(run_dir),
        performance_run_name=manifest.get("performance_run", "unknown"),
        accuracy_run_name=manifest.get("accuracy_run", "unknown"),
        strategy_run_name=manifest.get("strategy_run", "unknown"),
        all_strategy_run_name=manifest.get("all_strategy_run", "unknown"),
        expanded_strategy_run_name=manifest.get("expanded_strategy_run", "unknown"),
    )
    (output_dir / FILENAME).write_text(document, encoding="utf-8")
    print(f"Wrote {FILENAME} with {body(newest_strategy_run(STRATEGY_ROOT)).count(chr(60) + 'section')} sections")


if __name__ == "__main__":
    main()
