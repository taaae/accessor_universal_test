#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


MAIN_DISTRIBUTIONS = {"field_balanced_finite", "paired_log_uniform_finite"}
CONTROL_DISTRIBUTIONS = {"lut_scattered_control", "lut_concentrated_control"}


def read_all_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ratio_interval(
    numerator: list[float], denominator: list[float], seed: int
) -> tuple[float, float, float]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired timing groups must have the same nonzero size")
    logs = [math.log(left / right) for left, right in zip(numerator, denominator)]
    estimate = math.exp(statistics.median(logs))
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(10_000):
        sample = [logs[rng.randrange(len(logs))] for _ in logs]
        bootstrap.append(math.exp(statistics.median(sample)))
    bootstrap.sort()
    return estimate, bootstrap[249], bootstrap[9749]


def independent_ratio_interval(
    numerator: list[float], denominator: list[float], seed: int
) -> tuple[float, float, float]:
    if not numerator or not denominator:
        raise ValueError("timing groups must be nonempty")
    estimate = statistics.median(numerator) / statistics.median(denominator)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(10_000):
        left = [numerator[rng.randrange(len(numerator))] for _ in numerator]
        right = [denominator[rng.randrange(len(denominator))] for _ in denominator]
        bootstrap.append(statistics.median(left) / statistics.median(right))
    bootstrap.sort()
    return estimate, bootstrap[249], bootstrap[9749]


def stable_seed(*keys: tuple[str, ...]) -> int:
    digest = hashlib.sha256(repr(keys).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def classification(lower: float, upper: float) -> str:
    if lower >= 0.97 and upper <= 1.03:
        return "equivalent"
    if upper < 0.97:
        return "numerator_faster"
    if lower > 1.03:
        return "numerator_slower"
    return "inconclusive"


def analyze(samples_path: Path, output_dir: Path) -> None:
    all_rows = read_all_rows(samples_path)
    rows = [row for row in all_rows if row["status"] == "ok"]
    infeasible = {
        tuple(
            row[column]
            for column in (
                "format",
                "family",
                "bits",
                "arithmetic",
                "distribution",
                "kernel",
                "strategy",
                "status",
            )
        )
        for row in all_rows
        if row["status"] != "ok"
    }
    infeasible_rows = [
        dict(
            zip(
                (
                    "format",
                    "family",
                    "bits",
                    "arithmetic",
                    "distribution",
                    "kernel",
                    "strategy",
                    "status",
                ),
                key,
            )
        )
        for key in sorted(infeasible)
    ]
    write_rows(
        output_dir / "infeasible_cases.csv",
        [
            "format",
            "family",
            "bits",
            "arithmetic",
            "distribution",
            "kernel",
            "strategy",
            "status",
        ],
        infeasible_rows,
    )
    if not rows:
        raise ValueError("no successful timing rows")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(
            row[column]
            for column in (
                "format",
                "family",
                "bits",
                "arithmetic",
                "distribution",
                "kernel",
                "strategy",
            )
        )
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: int(row["round"]))

    summaries: list[dict[str, object]] = []
    medians: dict[tuple[str, ...], float] = {}
    for key, values in sorted(groups.items()):
        timing = [float(row["kernel_ms"]) for row in values]
        median = statistics.median(timing)
        medians[key] = median
        summaries.append(
            dict(
                zip(
                    (
                        "format",
                        "family",
                        "bits",
                        "arithmetic",
                        "distribution",
                        "kernel",
                        "strategy",
                    ),
                    key,
                )
            )
            | {
                "samples": len(timing),
                "median_ms": f"{median:.9g}",
                "minimum_ms": f"{min(timing):.9g}",
                "maximum_ms": f"{max(timing):.9g}",
            }
        )
    write_rows(
        output_dir / "timing_summary.csv",
        list(summaries[0]),
        summaries,
    )

    main_groups: dict[tuple[str, ...], list[tuple[tuple[str, ...], float]]] = defaultdict(list)
    for key, median in medians.items():
        if key[4] in MAIN_DISTRIBUTIONS:
            main_groups[key[:4] + key[4:6]].append((key, median))
    winners = []
    for case, candidates in sorted(main_groups.items()):
        key, median = min(candidates, key=lambda item: item[1])
        winners.append(
            {
                "format": key[0],
                "family": key[1],
                "bits": key[2],
                "arithmetic": key[3],
                "distribution": key[4],
                "kernel": key[5],
                "winning_strategy": key[6],
                "median_ms": f"{median:.9g}",
            }
        )
    write_rows(output_dir / "strategy_winners.csv", list(winners[0]), winners)

    comparisons: list[dict[str, object]] = []

    def add_comparison(
        question: int,
        numerator_key: tuple[str, ...],
        denominator_key: tuple[str, ...],
        note: str,
        paired: bool = True,
    ) -> None:
        numerator = [float(row["kernel_ms"]) for row in groups[numerator_key]]
        denominator = [float(row["kernel_ms"]) for row in groups[denominator_key]]
        seed = stable_seed(numerator_key, denominator_key)
        interval = ratio_interval if paired else independent_ratio_interval
        ratio, lower, upper = interval(numerator, denominator, seed)
        comparisons.append(
            {
                "question": question,
                "bits": numerator_key[2],
                "arithmetic": numerator_key[3],
                "distribution": numerator_key[4],
                "kernel": numerator_key[5],
                "numerator_format": numerator_key[0],
                "numerator_strategy": numerator_key[6],
                "denominator_format": denominator_key[0],
                "denominator_strategy": denominator_key[6],
                "ratio": f"{ratio:.9g}",
                "ci_lower": f"{lower:.9g}",
                "ci_upper": f"{upper:.9g}",
                "classification": classification(lower, upper),
                "interval_method": "paired_round_bootstrap" if paired else "independent_bootstrap",
                "note": note,
            }
        )

    # Questions 1 and 3 compare every retained alternative strategy with its
    # direct decoder in the same case.
    for key in sorted(groups):
        if key[1] not in {"posit", "takum", "takum_log"}:
            continue
        if key[4] not in MAIN_DISTRIBUTIONS or key[6] == "direct":
            continue
        direct = key[:-1] + ("direct",)
        add_comparison(1 if int(key[2]) <= 14 else 3, key, direct,
                       "alternative strategy divided by direct")

    # Question 2 uses one IEEE reference table at each width and arithmetic.
    ieee_reference = {8: "fp8_e4m3", 14: "e5m8"}
    for key in sorted(groups):
        if key[1] not in {"posit", "takum", "takum_log"}:
            continue
        if key[4] not in CONTROL_DISTRIBUTIONS:
            continue
        denominator = (
            ieee_reference[int(key[2])],
            "ieee",
            key[2],
            key[3],
            key[4],
            key[5],
            key[6],
        )
        add_comparison(2, key, denominator, "alternative LUT divided by IEEE LUT")

    # Question 4 compares each alternative winner with the fastest retained
    # same-width IEEE strategy. Formats remain separate on the numerator side.
    ieee_by_case: dict[tuple[str, ...], tuple[tuple[str, ...], float]] = {}
    for key, median in medians.items():
        if key[1] != "ieee" or key[4] not in MAIN_DISTRIBUTIONS:
            continue
        case = key[2:6]
        if case not in ieee_by_case or median < ieee_by_case[case][1]:
            ieee_by_case[case] = (key, median)
    for winner in winners:
        if winner["family"] not in {"posit", "takum", "takum_log"}:
            continue
        key = (
            winner["format"],
            winner["family"],
            winner["bits"],
            winner["arithmetic"],
            winner["distribution"],
            winner["kernel"],
            winner["winning_strategy"],
        )
        denominator = ieee_by_case[key[2:6]][0]
        add_comparison(4, key, denominator,
                       "best alternative strategy divided by fastest retained IEEE",
                       paired=False)

    write_rows(output_dir / "case_comparisons.csv", list(comparisons[0]), comparisons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyze(args.samples, args.output_dir)


if __name__ == "__main__":
    main()
