#!/usr/bin/env python3
"""Validate and summarize experiment 018 raw timing/component samples."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


FORMATS = {
    "fp16_e5m10": (16, ("fp16", "fp64")),
    "bf16_e8m7": (16, ("bf16", "fp64")),
    "fp32_e8m23": (32, ("fp32", "fp64")),
    "fp8_e4m3": (8, ("fp16", "fp32", "fp64")),
    "fp8_e5m2": (8, ("fp16", "fp32", "fp64")),
    "fp4_e2m1": (4, ("fp16", "fp32", "fp64")),
    "fp64_e11m52": (64, ("fp64",)),
}

NARROWEST_ARITHMETIC = {
    "fp16_e5m10": "fp16",
    "bf16_e8m7": "bf16",
    "fp32_e8m23": "fp32",
    "fp8_e4m3": "fp16",
    "fp8_e5m2": "fp16",
    "fp4_e2m1": "fp16",
    "fp64_e11m52": "fp64",
}

TIMING_KEYS = (
    "distribution",
    "kernel",
    "storage",
    "storage_bits",
    "arithmetic",
    "family",
    "lanes",
    "m",
    "n",
)

COMPONENT_KEYS = (
    "distribution",
    "component",
    "storage",
    "storage_bits",
    "arithmetic",
    "family",
    "lanes",
    "logical_values",
    "repeats",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError) as error:
        raise SystemExit(f"invalid {key!r} in row: {row}") from error


def as_int(row: dict[str, str], key: str) -> int:
    value = as_float(row, key)
    if not math.isfinite(value) or value != int(value):
        raise SystemExit(f"invalid integer {key!r} in row: {row}")
    return int(value)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_groups(
    rows: list[dict[str, str]], keys: tuple[str, ...], value_key: str
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        values = [as_float(row, value_key) for row in group]
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise SystemExit(f"non-positive/non-finite {value_key} for group {key}")
        median = statistics.median(values)
        deviations = [abs(value - median) for value in values]
        result: dict[str, object] = dict(zip(keys, key))
        result.update(
            {
                "samples": len(values),
                "median_time_ms": median,
                "q25_time_ms": percentile(values, 0.25),
                "q75_time_ms": percentile(values, 0.75),
                "mad_time_ms": statistics.median(deviations),
                "min_time_ms": min(values),
                "max_time_ms": max(values),
            }
        )
        for field in (
            "useful_flops",
            "logical_storage_bytes",
            "modeled_load_instructions",
            "modeled_conversions",
            "logical_values",
            "repeats",
        ):
            if field in group[0]:
                result[field] = as_float(group[0], field)
        summaries.append(result)
    return summaries


def expected_kernel_variants(storage: str) -> set[tuple[str, str, int]]:
    result: set[tuple[str, str, int]] = set()
    for arithmetic in FORMATS[storage][1]:
        result.add((arithmetic, "scalar_single", 1))
        for lanes in (2, 4, 8):
            result.add((arithmetic, "scalar_unrolled", lanes))
            result.add((arithmetic, "vector_packet", lanes))
    if storage in ("fp16_e5m10", "bf16_e8m7"):
        native = NARROWEST_ARITHMETIC[storage]
        for lanes in (2, 4, 8):
            result.add((native, "packed_arithmetic", lanes))
    return result


def expected_component_variants(storage: str) -> set[tuple[str, str, str, int]]:
    result: set[tuple[str, str, str, int]] = {
        ("stream_load", "none", "scalar_single", 1)
    }
    for lanes in (2, 4, 8):
        result.add(("stream_load", "none", "scalar_unrolled", lanes))
        result.add(("stream_load", "none", "vector_packet", lanes))
    for arithmetic in FORMATS[storage][1]:
        result.add(("stream_decode", arithmetic, "scalar_single", 1))
        for lanes in (2, 4, 8):
            result.add(("stream_decode", arithmetic, "scalar_unrolled", lanes))
            result.add(("stream_decode", arithmetic, "vector_packet", lanes))
        for lanes in (1, 2, 4, 8):
            result.add(("register_decode", arithmetic, "register_resident", lanes))
            result.add(("arithmetic_chain", arithmetic, "independent_chains", lanes))
    return result


def validate_samples(
    rows: list[dict[str, str]],
    *,
    components: bool,
    expected_rounds: int,
    expected_samples: int,
) -> None:
    expected_count = expected_rounds * expected_samples
    seen_formats = {row["storage"] for row in rows}
    if seen_formats != set(FORMATS):
        raise SystemExit(
            f"format mismatch: expected {sorted(FORMATS)}, got {sorted(seen_formats)}"
        )
    group_keys = COMPONENT_KEYS if components else TIMING_KEYS
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    for key, group in groups.items():
        if len(group) != expected_count:
            raise SystemExit(
                f"group {key} has {len(group)} samples, expected {expected_count}"
            )
        observations = {(as_int(row, "round"), as_int(row, "sample")) for row in group}
        expected_observations = {
            (round_index, sample_index)
            for round_index in range(expected_rounds)
            for sample_index in range(expected_samples)
        }
        if observations != expected_observations:
            raise SystemExit(f"round/sample coverage mismatch for group {key}")

    for storage in FORMATS:
        actual = set()
        for row in rows:
            if row["storage"] != storage:
                continue
            if components:
                actual.add(
                    (
                        row["component"],
                        row["arithmetic"],
                        row["family"],
                        as_int(row, "lanes"),
                    )
                )
            else:
                actual.add(
                    (row["arithmetic"], row["family"], as_int(row, "lanes"))
                )
        expected = (
            expected_component_variants(storage)
            if components
            else expected_kernel_variants(storage)
        )
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise SystemExit(
                f"{storage} inventory mismatch; missing={missing}, extra={extra}"
            )


def validate_correctness(rows: list[dict[str, str]]) -> None:
    tolerances = {"fp16": 0.15, "bf16": 0.25, "fp32": 2.0e-4, "fp64": 2.0e-10}
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    keys = ("distribution", "kernel", "storage", "arithmetic", "m", "n")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    for key, group in groups.items():
        baseline_rows = [row for row in group if row["family"] == "scalar_single"]
        if len(baseline_rows) != 1:
            raise SystemExit(f"validation group {key} lacks one x1 baseline")
        baseline = as_float(baseline_rows[0], "result_checksum")
        if not math.isfinite(baseline):
            raise SystemExit(f"non-finite validation baseline for {key}")
        tolerance = tolerances[key[3]]
        scale = max(1.0, abs(baseline))
        for row in group:
            result = as_float(row, "result_checksum")
            if not math.isfinite(result):
                raise SystemExit(f"non-finite validation result: {row}")
            if abs(result - baseline) / scale > tolerance:
                raise SystemExit(
                    f"validation mismatch for {key}, {row['family']} x{row['lanes']}: "
                    f"baseline={baseline}, result={result}, tolerance={tolerance}"
                )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise SystemExit(f"refusing to write empty summary {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def index_summary(
    rows: list[dict[str, object]], keys: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, object]]:
    return {tuple(str(row[key]) for key in keys): row for row in rows}


def packing_speedups(timing: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline_keys = (
        "distribution",
        "kernel",
        "storage",
        "arithmetic",
        "m",
        "n",
    )
    baselines = {
        tuple(str(row[key]) for key in baseline_keys): float(row["median_time_ms"])
        for row in timing
        if row["family"] == "scalar_single"
    }
    result = []
    for row in timing:
        key = tuple(str(row[field]) for field in baseline_keys)
        baseline = baselines[key]
        current = float(row["median_time_ms"])
        result.append(
            {
                **{field: row[field] for field in TIMING_KEYS},
                "median_time_ms": current,
                "x1_time_ms": baseline,
                "speedup_over_x1": baseline / current,
            }
        )
    return result


def mixed_penalties(timing: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ("distribution", "kernel", "storage", "family", "lanes", "m", "n")
    indexed = index_summary(timing, keys + ("arithmetic",))
    result = []
    for row in timing:
        if row["arithmetic"] != "fp64" or row["storage"] == "fp64_e11m52":
            continue
        native = NARROWEST_ARITHMETIC[str(row["storage"])]
        native_key = tuple(str(row[field]) for field in keys) + (native,)
        if native_key not in indexed:
            continue
        native_row = indexed[native_key]
        mixed_time = float(row["median_time_ms"])
        native_time = float(native_row["median_time_ms"])
        result.append(
            {
                **{field: row[field] for field in TIMING_KEYS},
                "native_arithmetic": native,
                "native_time_ms": native_time,
                "mixed_fp64_time_ms": mixed_time,
                "mixed_over_native_slowdown": mixed_time / native_time,
            }
        )
    return result


def derive_roofs(
    timing: list[dict[str, object]], components: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    stream_loads = [row for row in components if row["component"] == "stream_load"]
    sustainable_gb_s = max(
        float(row["logical_storage_bytes"])
        / (float(row["median_time_ms"]) * 1.0e-3)
        / 1.0e9
        for row in stream_loads
    )
    arithmetic_peak: dict[str, float] = defaultdict(float)
    for row in components:
        if row["component"] != "arithmetic_chain":
            continue
        rate = (
            float(row["useful_flops"])
            / (float(row["median_time_ms"]) * 1.0e-3)
            / 1.0e9
        )
        arithmetic_peak[str(row["arithmetic"])] = max(
            arithmetic_peak[str(row["arithmetic"])], rate
        )

    component_index = index_summary(
        components,
        ("distribution", "component", "storage", "arithmetic", "family", "lanes"),
    )
    roof_rows = []
    floor_rows = []
    for row in timing:
        # Native half2/bfloat162 arithmetic is a separate control.  Its
        # throughput ceiling is not represented by the scalar arithmetic-chain
        # calibration below, so do not publish a misleading roof or resource
        # floor for it.  Complete-kernel timing and NCU data still retain these
        # rows.
        family = str(row["family"])
        if family == "packed_arithmetic":
            continue
        elapsed = float(row["median_time_ms"])
        bytes_ = float(row["logical_storage_bytes"])
        flops = float(row["useful_flops"])
        arithmetic = str(row["arithmetic"])
        memory_floor = bytes_ / (sustainable_gb_s * 1.0e9) * 1.0e3
        compute_floor = flops / (arithmetic_peak[arithmetic] * 1.0e9) * 1.0e3
        classical_floor = max(memory_floor, compute_floor)
        roof_rows.append(
            {
                **{field: row[field] for field in TIMING_KEYS},
                "median_time_ms": elapsed,
                "sustainable_hbm_gb_per_s": sustainable_gb_s,
                "empirical_arithmetic_peak_gflop_per_s": arithmetic_peak[arithmetic],
                "memory_floor_ms": memory_floor,
                "compute_floor_ms": compute_floor,
                "classical_roof_time_ms": classical_floor,
                "roof_gap": elapsed / classical_floor,
                "roof_efficiency": classical_floor / elapsed,
            }
        )

        lanes = str(row["lanes"])
        distribution = str(row["distribution"])
        storage = str(row["storage"])
        logical_input_values = (
            2.0 * float(row["n"])
            if row["kernel"] == "dot"
            else float(row["m"]) * float(row["n"]) + float(row["n"])
        )
        logical_fmas = flops / 2.0

        def scaled_component(
            component: str, component_family: str, component_lanes: str
        ) -> float:
            key = (
                distribution,
                component,
                storage,
                arithmetic if component != "stream_load" else "none",
                component_family,
                component_lanes,
            )
            component_row = component_index[key]
            per_value_ms = float(component_row["median_time_ms"]) / float(
                component_row["logical_values"]
            )
            return per_value_ms * logical_input_values

        load_floor = scaled_component("stream_load", family, lanes)
        decode_floor = scaled_component("stream_decode", family, lanes)
        register_floor = scaled_component(
            "register_decode", "register_resident", lanes
        )
        arithmetic_floor = scaled_component(
            "arithmetic_chain", "independent_chains", lanes
        ) * (logical_fmas / logical_input_values)
        floor_rows.append(
            {
                **{field: row[field] for field in TIMING_KEYS},
                "median_time_ms": elapsed,
                "hbm_floor_ms": memory_floor,
                "stream_load_scaled_component_ms": load_floor,
                "stream_decode_scaled_component_ms": decode_floor,
                "register_decode_scaled_component_ms": register_floor,
                "arithmetic_chain_scaled_component_ms": arithmetic_floor,
                "note": (
                    "isolated_component_equivalents_can_exceed_kernel_time_"
                    "and_do_not_sum"
                ),
            }
        )
    return roof_rows, floor_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rounds", type=int, default=3)
    parser.add_argument("--expected-samples", type=int, default=5)
    args = parser.parse_args()

    timing_rows = read_rows(args.timing)
    component_rows = read_rows(args.components)
    validate_samples(
        timing_rows,
        components=False,
        expected_rounds=args.expected_rounds,
        expected_samples=args.expected_samples,
    )
    validate_samples(
        component_rows,
        components=True,
        expected_rounds=args.expected_rounds,
        expected_samples=args.expected_samples,
    )
    if args.validation:
        validation_rows = read_rows(args.validation)
        validate_samples(
            validation_rows,
            components=False,
            expected_rounds=1,
            expected_samples=1,
        )
        validate_correctness(validation_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_summary = summarize_groups(timing_rows, TIMING_KEYS, "time_ms")
    component_summary = summarize_groups(
        component_rows, COMPONENT_KEYS, "time_ms"
    )
    roofs, floors = derive_roofs(timing_summary, component_summary)
    write_csv(args.output_dir / "timing_summary.csv", timing_summary)
    write_csv(args.output_dir / "component_summary.csv", component_summary)
    write_csv(
        args.output_dir / "packing_speedups.csv", packing_speedups(timing_summary)
    )
    write_csv(
        args.output_dir / "mixed_precision_penalties.csv",
        mixed_penalties(timing_summary),
    )
    write_csv(args.output_dir / "roof_metrics.csv", roofs)
    write_csv(args.output_dir / "resource_floors.csv", floors)
    print(
        f"Validated {len(timing_rows)} timing and {len(component_rows)} component "
        f"samples across {len(timing_summary)} timing and "
        f"{len(component_summary)} component groups"
    )


if __name__ == "__main__":
    main()
