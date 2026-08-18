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
    print(f"Wrote {FILENAME} with {len(SECTIONS) + 1} finding sections")


if __name__ == "__main__":
    main()
