#!/usr/bin/env python3
"""Map experiment 018 Nsight Compute kernels back to logical variants."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


IDENTITY_FIELDS = [
    "distribution",
    "kernel",
    "storage",
    "storage_bits",
    "arithmetic",
    "family",
    "lanes",
    "m",
    "n",
    "useful_flops",
    "logical_storage_bytes",
    "modeled_load_instructions",
]

SUM_METRICS = {
    "duration_ns": ("gpu__time_duration.sum",),
    "dram_read_bytes": ("dram__bytes_read.sum",),
    "dram_write_bytes": ("dram__bytes_write.sum",),
    "instructions": ("smsp__inst_executed.sum",),
    "fp32_add_instructions": (
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    ),
    "fp32_mul_instructions": (
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    ),
    "fp32_fma_instructions": (
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    ),
    "fp64_add_instructions": (
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
    ),
    "fp64_mul_instructions": (
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
    ),
    "fp64_fma_instructions": (
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    ),
}

WEIGHTED_METRICS = {
    "dram_percent_peak": (
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.sum.pct_of_peak_sustained_elapsed",
    ),
    "sm_percent_peak": (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.sum.pct_of_peak_sustained_elapsed",
    ),
    "l1_hit_rate_percent": ("l1tex__t_sector_hit_rate.pct",),
    "l2_hit_rate_percent": ("lts__t_sector_hit_rate.pct",),
    "achieved_occupancy_percent": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ),
    "eligible_warps_per_scheduler": (
        "smsp__warps_eligible.avg.per_cycle_active",
    ),
    "long_scoreboard_stall_percent": (
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct",
    ),
    "math_pipe_throttle_percent": (
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.pct",
    ),
    "mio_throttle_percent": (
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.pct",
    ),
}

MAX_METRICS = {
    "registers_per_thread": ("launch__registers_per_thread",),
    "shared_memory_bytes": (
        "launch__shared_mem_per_block_allocated",
        "launch__shared_mem_per_block",
    ),
}


def value(row: dict[str, str], names: tuple[str, ...]) -> float:
    for name in names:
        text = row.get(name, "")
        if text not in ("", "N/A", None):
            try:
                return float(text)
            except ValueError:
                continue
    return math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def read_ncu(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        next(reader)  # Units row.
        rows = [dict(zip(header, row)) for row in reader if row]
    if not rows:
        raise SystemExit(f"{path}: no Nsight Compute rows")
    return rows


def aggregate(metadata: dict[str, str], kernels: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {field: metadata[field] for field in IDENTITY_FIELDS}
    result["kernel_count"] = len(kernels)
    result["kernel_names"] = " | ".join(str(row["kernel_name"]) for row in kernels)
    durations = [float(row["duration_ns"]) for row in kernels]
    total_duration = sum(durations)
    for metric in SUM_METRICS:
        result[metric] = sum(float(row[metric]) for row in kernels)
    for metric in WEIGHTED_METRICS:
        finite = [
            (float(row[metric]), duration)
            for row, duration in zip(kernels, durations)
            if math.isfinite(float(row[metric]))
        ]
        result[metric] = (
            sum(metric_value * duration for metric_value, duration in finite)
            / sum(duration for _, duration in finite)
            if finite
            else math.nan
        )
    for metric in MAX_METRICS:
        finite = [float(row[metric]) for row in kernels if math.isfinite(float(row[metric]))]
        result[metric] = max(finite) if finite else math.nan
    result["duration_ms"] = total_duration * 1.0e-6
    result["dram_total_bytes"] = float(result["dram_read_bytes"]) + float(
        result["dram_write_bytes"]
    )
    result["measured_dram_gb_per_s"] = (
        float(result["dram_total_bytes"]) / (total_duration * 1.0e-9) / 1.0e9
        if total_duration > 0.0
        else math.nan
    )
    logical_values = (
        float(metadata["n"])
        if metadata["kernel"] == "dot"
        else float(metadata["m"]) * float(metadata["n"])
    )
    result["instructions_per_logical_value"] = (
        float(result["instructions"]) / logical_values
    )
    return result


def kernel_row(
    metadata: dict[str, str], raw: dict[str, str], role: str, report: str
) -> dict[str, object]:
    result: dict[str, object] = {field: metadata[field] for field in IDENTITY_FIELDS}
    result.update(
        {
            "report": report,
            "kernel_role": role,
            "kernel_name": raw.get("Kernel Name", ""),
            "block_size": raw.get("Block Size", ""),
            "grid_size": raw.get("Grid Size", ""),
        }
    )
    for target, names in SUM_METRICS.items():
        result[target] = value(raw, names)
    for target, names in WEIGHTED_METRICS.items():
        result[target] = value(raw, names)
    for target, names in MAX_METRICS.items():
        result[target] = value(raw, names)
    result["duration_ms"] = float(result["duration_ns"]) * 1.0e-6
    return result


def write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise SystemExit(f"refusing to write empty {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    operation_rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []
    metadata_paths = sorted(args.profile_dir.glob("**/*_metadata.csv"))
    if not metadata_paths:
        raise SystemExit(f"no metadata CSVs under {args.profile_dir}")
    for metadata_path in metadata_paths:
        stem = metadata_path.name.removesuffix("_metadata.csv")
        raw_path = metadata_path.with_name(stem + "_ncu_raw.csv")
        if not raw_path.exists():
            raise SystemExit(f"missing raw CSV for {metadata_path}")
        metadata_rows = read_csv(metadata_path)
        raw_rows = read_ncu(raw_path)
        expected_kernel_rows = sum(
            2 if row["kernel"] == "dot" else 1 for row in metadata_rows
        )
        if len(raw_rows) != expected_kernel_rows:
            raise SystemExit(
                f"{raw_path}: expected {expected_kernel_rows} kernels from metadata, "
                f"found {len(raw_rows)}"
            )
        offset = 0
        for metadata in metadata_rows:
            count = 2 if metadata["kernel"] == "dot" else 1
            selected_raw = raw_rows[offset : offset + count]
            roles = ("map", "finalize") if count == 2 else ("main",)
            selected_kernels = [
                kernel_row(metadata, raw, role, str(raw_path.relative_to(args.profile_dir)))
                for raw, role in zip(selected_raw, roles)
            ]
            kernel_rows.extend(selected_kernels)
            operation_rows.append(aggregate(metadata, selected_kernels))
            offset += count

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "profile_kernels.csv", kernel_rows)
    write(args.output_dir / "profile_operations.csv", operation_rows)
    print(
        f"Mapped {len(kernel_rows)} profiled kernels to "
        f"{len(operation_rows)} logical operations"
    )


if __name__ == "__main__":
    main()
