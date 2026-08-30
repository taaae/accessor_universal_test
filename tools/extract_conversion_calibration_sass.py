#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from conversion_calibration.sass import extract, merge_resources, validate_acceptance, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sass", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--warnings", type=Path)
    args = parser.parse_args()
    rows = extract(args.sass.read_text(encoding="utf-8", errors="replace"))
    merge_resources(rows, args.resources)
    warnings = validate_acceptance(rows)
    write_csv(rows, args.output)
    if args.warnings:
        args.warnings.write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")
    print(f"extracted {len(rows)} kernels; warnings={len(warnings)}")


if __name__ == "__main__": main()
