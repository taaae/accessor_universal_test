#!/usr/bin/env python3
"""Build the IEEE and LNS summary page.

The page answers a practical question the per-format pages cannot: which
storage types are worth implementing in the memory accessor, and which decode
strategy each one should use.  Findings are added here as the analysis
establishes them, so ``SECTIONS`` starts effectively empty and grows.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Sequence

import build_storage_performance_report as base


FILENAME = "ieee-lns-summary.html"

INTRO = ""

# Each entry is (heading, [paragraphs...]).  Filled in as findings are settled.
SECTIONS: Sequence[tuple[str, Sequence[str]]] = ()


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


def body() -> str:
    blocks = []
    for heading, paragraphs in SECTIONS:
        rendered = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
        blocks.append(
            '<section class="text-section">'
            f"<h2>{html.escape(heading)}</h2>{rendered}</section>"
        )
    return "".join(blocks)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if not (output_dir / "report.css").is_file():
        raise SystemExit(
            f"{output_dir} is not a built report; run the base report builder first"
        )
    manifest = read_manifest(output_dir / "report_manifest.txt")
    document = base.page_document(
        filename=FILENAME,
        title="IEEE, LNS summary",
        intro=INTRO,
        body=body(),
        performance_run_name=manifest.get("performance_run", "unknown"),
        accuracy_run_name=manifest.get("accuracy_run", "unknown"),
        strategy_run_name=manifest.get("strategy_run", "unknown"),
        all_strategy_run_name=manifest.get("all_strategy_run", "unknown"),
        expanded_strategy_run_name=manifest.get("expanded_strategy_run", "unknown"),
    )
    (output_dir / FILENAME).write_text(document, encoding="utf-8")
    print(f"Wrote {FILENAME} with {len(SECTIONS)} finding sections")


if __name__ == "__main__":
    main()
