#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def load_analyzer(path: Path):
    specification = importlib.util.spec_from_file_location(
        "dyadic_normal32_analysis", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load DyadicNormal32 analyzer")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--coefficients", required=True, type=Path)
    parser.add_argument("--analyzer", required=True, type=Path)
    arguments = parser.parse_args()

    analyzer = load_analyzer(arguments.analyzer)
    analyzer.validate_coefficients(arguments.coefficients)
    rows = read_rows(arguments.samples)
    metrics = read_rows(arguments.metrics)
    if len(rows) != 57:
        raise ValueError("smoke timing CSV must contain 57 data rows")
    if len(metrics) != 4:
        raise ValueError("smoke metrics CSV must contain four data rows")
    known = set(analyzer.BASELINES) | set(analyzer.DYADIC_VARIANTS)
    if {row["variant"] for row in rows} != known:
        raise ValueError("smoke timing CSV does not contain all seven variants")
    if any(row["mode"] != "smoke" or row["N"] != "1048576"
           or row["kernel"] != "dot" or row["warmup"] != "1"
           for row in rows):
        raise ValueError("smoke timing rows have the wrong fixed settings")
    orders = [int(row["execution_order"]) for row in rows]
    if orders != sorted(orders) or set(orders) != set(range(38)):
        raise ValueError("smoke rows are not in complete physical execution order")
    for order in range(38):
        batch = [row for row in rows if int(row["execution_order"]) == order]
        if [int(row["sample"]) for row in batch] != sorted(
            int(row["sample"]) for row in batch
        ):
            raise ValueError(f"smoke batch {order} is not in sample order")
    smoke_baseline_orders = {
        "raw_fp64": (0, 37),
        "raw_fp32": (1, 36),
        "fp32_to_fp64": (2, 35),
    }
    for variant in analyzer.BASELINES:
        group = [row for row in rows if row["variant"] == variant]
        if len(group) != 3 or {row["phase"] for row in group} != {"before", "after"}:
            raise ValueError(f"{variant} smoke baseline split is wrong")
        if sorted(int(row["sample"]) for row in group) != [0, 1, 2]:
            raise ValueError(f"{variant} smoke samples are wrong")
        if {int(row["sample"]) for row in group if row["phase"] == "before"} \
                != {0, 1} or {int(row["sample"]) for row in group
                              if row["phase"] == "after"} != {2}:
            raise ValueError(f"{variant} smoke sample halves are wrong")
        before_order, after_order = smoke_baseline_orders[variant]
        if {int(row["execution_order"]) for row in group
                if row["phase"] == "before"} != {before_order} \
                or {int(row["execution_order"]) for row in group
                    if row["phase"] == "after"} != {after_order}:
            raise ValueError(f"{variant} smoke execution order is wrong")
    targets = {0.0, 0.5, 1.0}
    for variant in analyzer.DYADIC_VARIANTS:
        strategy_index = analyzer.STRATEGY_ORDER.index(variant)
        reverse_position = len(analyzer.STRATEGY_ORDER) - 1 - strategy_index
        group = [row for row in rows if row["variant"] == variant]
        if len(group) != 12:
            raise ValueError(f"{variant} smoke sample count is wrong")
        artificial = [row for row in group if row["distribution"] == "hot_uniform"]
        genuine = [row for row in group if row["distribution"] == "genuine_n01"]
        if len(artificial) != 9 or {float(row["target_x"]) for row in artificial} != targets:
            raise ValueError(f"{variant} smoke targets are incomplete")
        if len(genuine) != 3:
            raise ValueError(f"{variant} genuine smoke point is incomplete")
        for distribution, target in [
            *(('hot_uniform', value) for value in targets),
            ('genuine_n01', float(genuine[0]["target_x"])),
        ]:
            point = [row for row in group if row["distribution"] == distribution
                     and float(row["target_x"]) == target]
            if sorted(int(row["sample"]) for row in point) != [0, 1, 2]:
                raise ValueError(f"{variant}/{distribution}/{target}: bad samples")
            if [row["phase"] for row in point].count("forward") != 2 \
                    or [row["phase"] for row in point].count("reverse") != 1:
                raise ValueError(f"{variant}/{distribution}/{target}: bad split")
            if {int(row["sample"]) for row in point
                    if row["phase"] == "forward"} != {0, 1} \
                    or {int(row["sample"]) for row in point
                        if row["phase"] == "reverse"} != {2}:
                raise ValueError(
                    f"{variant}/{distribution}/{target}: wrong sample halves"
                )
            if distribution == "hot_uniform":
                target_index = (0.0, 0.5, 1.0).index(target)
                expected_forward = 3 + target_index * 4 + strategy_index
                expected_reverse = 23 + (2 - target_index) * 4 + reverse_position
            else:
                expected_forward = 15 + strategy_index
                expected_reverse = 19 + reverse_position
            if {int(row["execution_order"]) for row in point
                    if row["phase"] == "forward"} != {expected_forward} \
                    or {int(row["execution_order"]) for row in point
                        if row["phase"] == "reverse"} != {expected_reverse}:
                raise ValueError(
                    f"{variant}/{distribution}/{target}: wrong execution order"
                )
    if {row["distribution"] for row in metrics} != {"hot_uniform", "genuine_n01"}:
        raise ValueError("smoke metrics distributions are incomplete")
    if {float(row["target_x"]) for row in metrics
            if row["distribution"] == "hot_uniform"} != targets:
        raise ValueError("smoke metrics targets are incomplete")
    correctness = arguments.correctness.read_text()
    for check in (
        "current_cpu_gpu_bit_mismatches=0",
        "bitcast_shared_cpu_gpu_bit_mismatches=0",
        "bitcast_constant_cpu_gpu_bit_mismatches=0",
        "dot_current_passed=1", "dot_sign_fused_passed=1",
        "dot_bitcast_shared_passed=1", "dot_bitcast_constant_passed=1",
        "timed_current_sign_fused_bit_mismatches=0",
        "timed_bitcast_shared_constant_bit_mismatches=0",
        "timed_result_validation_passed=1",
        "timed_result_checks=12",
    ):
        if check not in correctness:
            raise ValueError(f"smoke correctness record failed: {check}")
    print("DyadicNormal32 smoke artifact contract passed")


if __name__ == "__main__":
    main()
