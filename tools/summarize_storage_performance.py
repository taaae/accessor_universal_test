#!/usr/bin/env python3
"""Validate and summarize event timings and Nsight Compute measurements."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


IDENTITY = [
    "gpu",
    "compute_capability",
    "distribution",
    "format",
    "storage_bits",
    "component",
    "lanes",
    "n",
    "m",
    "leading_dimension",
    "blocks",
    "threads",
    "decode_repeats",
]

MODEL_FIELDS = [
    "decoded_values",
    "unique_storage_bytes",
    "requested_storage_bytes",
    "useful_flops",
    "modeled_flops",
    "arithmetic_intensity_unique",
    "arithmetic_intensity_requested",
    "theoretical_hbm_gb_per_s",
    "modeled_fp64_gflop_per_s",
]

EXPECTED_FORMATS = {
    "e1m6",
    "e2m5",
    "e3m4",
    "fp8_e4m3",
    "fp8_e5m2",
    "e1m14",
    "e2m13",
    "e3m12",
    "fp16_e5m10",
    "bf16_e8m7",
    "e11m4",
    "e1m30",
    "e2m29",
    "e3m28",
    "fp32_e8m23",
    "e11m20",
    "fp64_e11m52",
}
EXPECTED_DISTRIBUTIONS = {"uniform_0_1", "normal_0_1"}
EXPECTED_COMPONENTS = {
    "register_decode",
    "stream_load",
    "stream_decode",
    "dot",
    "gemv",
}


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_packed_speedups(
    summary_rows: list[dict[str, object]] | list[dict[str, str]], output_dir: Path
) -> None:
    if not summary_rows:
        raise SystemExit("cannot build packed comparisons from an empty summary")
    packed_key_fields = [
        field
        for field in IDENTITY
        if field not in {"lanes", "blocks"}
    ]
    packed_groups: dict[tuple[str, ...], dict[int, dict[str, object]]] = defaultdict(
        dict
    )
    for source in summary_rows:
        row: dict[str, object] = dict(source)
        key = tuple(str(row[field]) for field in packed_key_fields)
        packed_groups[key][int(row["lanes"])] = row

    packed_rows: list[dict[str, object]] = []
    for key, lanes in sorted(packed_groups.items()):
        if set(lanes) != {1, 2, 4}:
            raise SystemExit(f"missing lane timing for {key}: {sorted(lanes)}")
        scalar_time = float(lanes[1]["median_time_ms"])
        component = str(lanes[1]["component"])

        def throughput(row: dict[str, object]) -> tuple[str, float]:
            if component == "register_decode":
                return (
                    "decoded_values_per_second",
                    float(row["median_decoded_gvalues_per_s"]),
                )
            if component == "stream_load":
                return (
                    "storage_bytes_per_second",
                    float(row["median_unique_storage_gb_per_s"]),
                )
            return (
                "useful_operations_per_second",
                float(row["median_useful_gflop_per_s"]),
            )

        comparison_metric, scalar_throughput = throughput(lanes[1])
        for lane in (1, 2, 4):
            row = dict(zip(packed_key_fields, key))
            current_time = float(lanes[lane]["median_time_ms"])
            current_metric, current_throughput = throughput(lanes[lane])
            if current_metric != comparison_metric:
                raise AssertionError("lane comparison metric changed within group")
            row.update(
                {
                    "lanes": lane,
                    "scalar_time_ms": scalar_time,
                    "packed_time_ms": current_time,
                    "comparison_metric": comparison_metric,
                    "scalar_throughput": scalar_throughput,
                    "packed_throughput": current_throughput,
                    "speedup_vs_x1": current_throughput / scalar_throughput,
                    "time_speedup_vs_x1": scalar_time / current_time,
                    "time_change_percent_vs_x1": 100.0
                    * (current_time / scalar_time - 1.0),
                    "packed_cv_percent": lanes[lane]["cv_percent"],
                }
            )
            packed_rows.append(row)

    with (output_dir / "packed_speedups.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(packed_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(packed_rows)


def summarize_timings(samples_path: Path, output_dir: Path) -> None:
    rows = read_rows(samples_path)
    if not rows:
        raise SystemExit(f"no timing rows in {samples_path}")
    formats = {row["format"] for row in rows}
    distributions = {row["distribution"] for row in rows}
    components = {row["component"] for row in rows}
    if formats != EXPECTED_FORMATS:
        raise SystemExit(
            f"format coverage mismatch: missing={sorted(EXPECTED_FORMATS - formats)}, "
            f"extra={sorted(formats - EXPECTED_FORMATS)}"
        )
    if distributions != EXPECTED_DISTRIBUTIONS:
        raise SystemExit(f"distribution coverage mismatch: {sorted(distributions)}")
    if components != EXPECTED_COMPONENTS:
        raise SystemExit(f"component coverage mismatch: {sorted(components)}")

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in IDENTITY)].append(row)

    summary_rows: list[dict[str, object]] = []
    expected_samples: int | None = None
    for key, group in sorted(grouped.items()):
        times = [float(row["time_ms"]) for row in group]
        if not all(math.isfinite(value) and value > 0.0 for value in times):
            raise SystemExit(f"non-positive or non-finite timing for {key}")
        if expected_samples is None:
            expected_samples = len(times)
        elif len(times) != expected_samples:
            raise SystemExit(
                f"incomplete timing group {key}: {len(times)} vs {expected_samples}"
            )
        rounds = {int(row["round"]) for row in group}
        positions = {int(row["order_position"]) for row in group}
        if len(rounds) < 2 or len(positions) < 2:
            raise SystemExit(f"timings were not interleaved across rounds: {key}")

        median = statistics.median(times)
        mean = statistics.fmean(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        mad = statistics.median(abs(value - median) for value in times)
        first = group[0]
        model = {field: float(first[field]) for field in MODEL_FIELDS}
        seconds = median * 1.0e-3
        result: dict[str, object] = dict(zip(IDENTITY, key))
        result.update(model)
        result.update(
            {
                "sample_count": len(times),
                "median_time_ms": median,
                "mean_time_ms": mean,
                "stdev_time_ms": stdev,
                "cv_percent": 100.0 * stdev / mean,
                "mad_time_ms": mad,
                "p05_time_ms": quantile(times, 0.05),
                "p95_time_ms": quantile(times, 0.95),
                "min_time_ms": min(times),
                "max_time_ms": max(times),
                "median_decoded_gvalues_per_s": model["decoded_values"]
                / seconds
                / 1.0e9,
                "median_unique_storage_gb_per_s": model["unique_storage_bytes"]
                / seconds
                / 1.0e9,
                "median_requested_storage_gb_per_s": model[
                    "requested_storage_bytes"
                ]
                / seconds
                / 1.0e9,
                "median_useful_gflop_per_s": model["useful_flops"]
                / seconds
                / 1.0e9,
            }
        )
        summary_rows.append(result)

    summary_path = output_dir / "timing_summary.csv"
    fields = list(summary_rows[0])
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_packed_speedups(summary_rows, output_dir)

    print(
        f"timings: {len(rows)} samples, {len(summary_rows)} complete groups, "
        f"{expected_samples} samples/group"
    )


def number(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name, "")
        if value not in ("", "N/A", None):
            try:
                return float(value)
            except ValueError:
                pass
    return math.nan


def finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def read_ncu_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        next(reader)  # units
        return [dict(zip(header, row)) for row in reader if row]


def ncu_kernel_row(raw: dict[str, str]) -> dict[str, object]:
    duration_ns = number(raw, "gpu__time_duration.sum")
    read_bytes = number(raw, "dram__bytes_read.sum")
    write_bytes = number(raw, "dram__bytes_write.sum")
    total_bytes = finite_or_zero(read_bytes) + finite_or_zero(write_bytes)
    if not (math.isfinite(read_bytes) and math.isfinite(write_bytes)):
        rate = number(raw, "dram__bytes.sum.per_second")
        total_bytes = (
            rate * duration_ns * 1.0e-9
            if math.isfinite(rate) and math.isfinite(duration_ns)
            else math.nan
        )
    fadd = number(raw, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
    fmul = number(raw, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
    ffma = number(raw, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
    dadd = number(raw, "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum")
    dmul = number(raw, "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum")
    dfma = number(raw, "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum")
    sp_flops = sum(map(finite_or_zero, (fadd, fmul))) + 2 * finite_or_zero(ffma)
    dp_flops = sum(map(finite_or_zero, (dadd, dmul))) + 2 * finite_or_zero(dfma)
    duration_s = duration_ns * 1.0e-9
    return {
        "kernel_name": raw.get("Kernel Name", ""),
        "duration_ms": duration_ns * 1.0e-6,
        "dram_read_bytes": read_bytes,
        "dram_write_bytes": write_bytes,
        "dram_total_bytes": total_bytes,
        "measured_dram_gb_per_s": total_bytes / duration_s / 1.0e9,
        "dram_percent_peak": number(
            raw,
            "gpu__dram_throughput.sum.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        "sm_percent_peak": number(
            raw,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.sum.pct_of_peak_sustained_elapsed",
        ),
        "l1_hit_rate_percent": number(raw, "l1tex__t_sector_hit_rate.pct"),
        "l2_hit_rate_percent": number(raw, "lts__t_sector_hit_rate.pct"),
        "sp_flops_executed": sp_flops,
        "dp_flops_executed": dp_flops,
        "total_flops_executed": sp_flops + dp_flops,
        "registers_per_thread": number(raw, "launch__registers_per_thread"),
        "achieved_occupancy_percent": number(
            raw,
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "smsp__warps_active.avg.pct_of_peak_sustained_active",
        ),
        "block_size": number(raw, "launch__block_size"),
        "grid_size": number(raw, "launch__grid_size"),
        "shared_memory_bytes": number(
            raw,
            "launch__shared_mem_per_block_allocated",
            "launch__shared_mem_per_block",
        ),
        "issue_active_percent": number(
            raw, "smsp__issue_active.avg.pct_of_peak_sustained_active"
        ),
        "pipe_alu_percent": number(
            raw, "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed"
        ),
        "pipe_fma_percent": number(
            raw, "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"
        ),
        "pipe_fp64_percent": number(
            raw, "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed"
        ),
        "pipe_lsu_percent": number(
            raw, "sm__inst_executed_pipe_lsu.sum.pct_of_peak_sustained_elapsed"
        ),
        "pipe_xu_percent": number(
            raw, "sm__inst_executed_pipe_xu.sum.pct_of_peak_sustained_elapsed"
        ),
    }


def bottleneck(dram: float, sm: float, occupancy: float, issue: float) -> str:
    values = (dram, sm, occupancy, issue)
    if not all(math.isfinite(value) for value in values):
        return "insufficient_metrics"
    if dram >= 50.0 and dram >= 1.25 * sm:
        return "memory_bandwidth"
    if sm >= 50.0 and sm >= 1.25 * dram:
        return "compute_or_conversion"
    if dram >= 50.0 and sm >= 50.0:
        return "mixed_memory_compute"
    if occupancy < 25.0:
        return "parallelism_or_occupancy"
    if issue < 25.0:
        return "latency_or_dependencies"
    return "instruction_throughput_or_overhead"


def summarize_profiles(profile_dir: Path, output_dir: Path) -> None:
    raw_paths = sorted(profile_dir.rglob("*_ncu_raw.csv"))
    if not raw_paths:
        raise SystemExit(f"no Nsight Compute raw CSVs below {profile_dir}")
    kernel_rows: list[dict[str, object]] = []
    operation_rows: list[dict[str, object]] = []
    for raw_path in raw_paths:
        stem = raw_path.name.removesuffix("_ncu_raw.csv")
        metadata_path = raw_path.with_name(stem + "_metadata.csv")
        if not metadata_path.exists():
            raise SystemExit(f"missing metadata for {raw_path}")
        metadata = read_rows(metadata_path)
        metadata_formats = {row["format"] for row in metadata}
        if metadata_formats != EXPECTED_FORMATS:
            raise SystemExit(
                f"{metadata_path}: format coverage mismatch; "
                f"missing={sorted(EXPECTED_FORMATS - metadata_formats)}"
            )
        raw = read_ncu_csv(raw_path)
        expected = sum(int(row["kernel_count"]) for row in metadata)
        if len(raw) != expected:
            raise SystemExit(
                f"{raw_path}: expected {expected} profiled kernels, found {len(raw)}"
            )
        cursor = 0
        for operation in metadata:
            count = int(operation["kernel_count"])
            current: list[dict[str, object]] = []
            for index in range(count):
                metrics = ncu_kernel_row(raw[cursor])
                cursor += 1
                row: dict[str, object] = {
                    "report": str(raw_path.relative_to(profile_dir)),
                    "profile_sequence": operation["profile_sequence"],
                    "distribution": operation["distribution"],
                    "format": operation["format"],
                    "storage_bits": operation["storage_bits"],
                    "component": operation["component"],
                    "lanes": operation["lanes"],
                    "n": operation["n"],
                    "m": operation["m"],
                    "operation_kernel_index": index,
                }
                row.update(metrics)
                current.append(row)
                kernel_rows.append(row)

            duration = sum(float(row["duration_ms"]) for row in current)
            total_bytes = sum(float(row["dram_total_bytes"]) for row in current)
            total_flops = sum(float(row["total_flops_executed"]) for row in current)
            duration_weights = [float(row["duration_ms"]) / duration for row in current]

            def weighted(field: str) -> float:
                pairs = [
                    (float(row[field]), weight)
                    for row, weight in zip(current, duration_weights)
                    if math.isfinite(float(row[field]))
                ]
                weight_sum = sum(weight for _, weight in pairs)
                return (
                    sum(value * weight for value, weight in pairs) / weight_sum
                    if weight_sum
                    else math.nan
                )

            duration_s = duration * 1.0e-3
            useful_flops = float(operation["useful_flops"])
            unique_bytes = float(operation["unique_storage_bytes"])
            op = {
                **operation,
                "report": str(raw_path.relative_to(profile_dir)),
                "duration_ms": duration,
                "dram_total_bytes": total_bytes,
                "measured_dram_gb_per_s": total_bytes / duration_s / 1.0e9,
                "inferred_sustained_dram_gb_per_s": (
                    total_bytes
                    / duration_s
                    / 1.0e9
                    / (weighted("dram_percent_peak") / 100.0)
                    if weighted("dram_percent_peak") > 0.0
                    else math.nan
                ),
                "executed_flops": total_flops,
                "executed_gflop_per_s": total_flops / duration_s / 1.0e9,
                "useful_gflop_per_s_profiler_contaminated": useful_flops
                / duration_s
                / 1.0e9,
                "algorithmic_intensity_unique": useful_flops / unique_bytes,
                "measured_executed_intensity": total_flops / total_bytes,
                "dram_percent_peak": weighted("dram_percent_peak"),
                "sm_percent_peak": weighted("sm_percent_peak"),
                "achieved_occupancy_percent": weighted(
                    "achieved_occupancy_percent"
                ),
                "issue_active_percent": weighted("issue_active_percent"),
                "pipe_alu_percent": weighted("pipe_alu_percent"),
                "pipe_fma_percent": weighted("pipe_fma_percent"),
                "pipe_fp64_percent": weighted("pipe_fp64_percent"),
                "pipe_lsu_percent": weighted("pipe_lsu_percent"),
                "pipe_xu_percent": weighted("pipe_xu_percent"),
                "registers_per_thread_max": max(
                    float(row["registers_per_thread"]) for row in current
                ),
            }
            op["bottleneck_class"] = bottleneck(
                float(op["dram_percent_peak"]),
                float(op["sm_percent_peak"]),
                float(op["achieved_occupancy_percent"]),
                float(op["issue_active_percent"]),
            )
            operation_rows.append(op)

    with (output_dir / "profile_kernels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(kernel_rows[0]))
        writer.writeheader()
        writer.writerows(kernel_rows)
    with (output_dir / "profile_operations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(operation_rows[0]))
        writer.writeheader()
        writer.writerows(operation_rows)
    print(
        f"profiles: {len(raw_paths)} reports, {len(kernel_rows)} kernels, "
        f"{len(operation_rows)} operations"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    timing_source = parser.add_mutually_exclusive_group(required=True)
    timing_source.add_argument("--samples", type=Path)
    timing_source.add_argument(
        "--summary",
        type=Path,
        help="reuse an existing timing summary and only rebuild packed comparisons",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.samples is not None:
        summarize_timings(args.samples, args.output_dir)
    else:
        write_packed_speedups(read_rows(args.summary), args.output_dir)
    if args.profile_dir is not None:
        summarize_profiles(args.profile_dir, args.output_dir)


if __name__ == "__main__":
    main()
