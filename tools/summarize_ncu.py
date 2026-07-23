#!/usr/bin/env python3
"""Extract compact kernel and operation metrics from Nsight Compute raw CSV."""

import argparse
import csv
import math
from pathlib import Path


VARIANT_ORDER = [
    "raw_f32",
    "raw_f64",
    "accessor_f32_f32",
    "accessor_f64_f32",
]


def number(row, *names):
    for name in names:
        value = row.get(name, "")
        if value not in ("", None, "N/A"):
            try:
                return float(value)
            except ValueError:
                pass
    return math.nan


def finite_or_zero(value):
    return value if math.isfinite(value) else 0.0


def read_ncu_csv(path):
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        next(reader)  # Units.
        return [dict(zip(header, row)) for row in reader if row]


def metric_row(variant, index, raw):
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
    sp_flops = sum(map(finite_or_zero, (fadd, fmul))) + 2.0 * finite_or_zero(ffma)
    dp_flops = sum(map(finite_or_zero, (dadd, dmul))) + 2.0 * finite_or_zero(dfma)
    total_flops = sp_flops + dp_flops
    duration_s = duration_ns * 1.0e-9

    return {
        "variant": variant,
        "scope": "kernel",
        "kernel_index": index,
        "kernel_name": raw.get("Kernel Name", ""),
        "duration_ms": duration_ns * 1.0e-6,
        "dram_read_bytes": read_bytes,
        "dram_write_bytes": write_bytes,
        "dram_total_bytes": total_bytes,
        "measured_dram_gb_per_s": (
            total_bytes / duration_s / 1.0e9 if duration_s else math.nan
        ),
        "dram_percent_peak": number(
            raw, "gpu__dram_throughput.sum.pct_of_peak_sustained_elapsed"
        ),
        "l1_hit_rate_percent": number(raw, "l1tex__t_sector_hit_rate.pct"),
        "l2_hit_rate_percent": number(raw, "lts__t_sector_hit_rate.pct"),
        "sp_flops_executed": sp_flops,
        "dp_flops_executed": dp_flops,
        "total_flops_executed": total_flops,
        "measured_arithmetic_intensity_flop_per_byte": (
            total_flops / total_bytes if total_bytes else math.nan
        ),
        "measured_gflop_per_s": (
            total_flops / duration_s / 1.0e9 if duration_s else math.nan
        ),
        "registers_per_thread": number(raw, "launch__registers_per_thread"),
        "achieved_occupancy_percent": number(
            raw, "sm__warps_active.avg.pct_of_peak_sustained_active"
        ),
        "block_size": number(raw, "launch__block_size"),
        "grid_size": number(raw, "launch__grid_size"),
        "shared_memory_bytes": number(
            raw,
            "launch__shared_mem_per_block_allocated",
            "launch__shared_mem_per_block",
        ),
    }


FIELDS = [
    "variant",
    "scope",
    "kernel_index",
    "kernel_name",
    "duration_ms",
    "dram_read_bytes",
    "dram_write_bytes",
    "dram_total_bytes",
    "measured_dram_gb_per_s",
    "dram_percent_peak",
    "l1_hit_rate_percent",
    "l2_hit_rate_percent",
    "sp_flops_executed",
    "dp_flops_executed",
    "total_flops_executed",
    "measured_arithmetic_intensity_flop_per_byte",
    "measured_gflop_per_s",
    "registers_per_thread",
    "achieved_occupancy_percent",
    "block_size",
    "grid_size",
    "shared_memory_bytes",
]


def operation_row(variant, kernels):
    duration_ms = sum(row["duration_ms"] for row in kernels)
    total_bytes = sum(row["dram_total_bytes"] for row in kernels)
    read_values = [row["dram_read_bytes"] for row in kernels]
    write_values = [row["dram_write_bytes"] for row in kernels]
    read_bytes = (
        sum(read_values) if all(math.isfinite(value) for value in read_values) else math.nan
    )
    write_bytes = (
        sum(write_values)
        if all(math.isfinite(value) for value in write_values)
        else math.nan
    )
    sp_flops = sum(row["sp_flops_executed"] for row in kernels)
    dp_flops = sum(row["dp_flops_executed"] for row in kernels)
    total_flops = sp_flops + dp_flops
    duration_s = duration_ms * 1.0e-3
    result = {field: "" for field in FIELDS}
    result.update(
        {
            "variant": variant,
            "scope": "operation",
            "kernel_name": "all_dot_kernels",
            "duration_ms": duration_ms,
            "dram_read_bytes": read_bytes,
            "dram_write_bytes": write_bytes,
            "dram_total_bytes": total_bytes,
            "measured_dram_gb_per_s": total_bytes / duration_s / 1.0e9,
            "sp_flops_executed": sp_flops,
            "dp_flops_executed": dp_flops,
            "total_flops_executed": total_flops,
            "measured_arithmetic_intensity_flop_per_byte": (
                total_flops / total_bytes if total_bytes else math.nan
            ),
            "measured_gflop_per_s": total_flops / duration_s / 1.0e9,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_rows = []
    for variant in VARIANT_ORDER:
        paths = sorted((args.profile_dir / variant).glob("*_ncu_raw.csv"))
        if len(paths) != 1:
            raise SystemExit(
                f"expected one raw Nsight Compute CSV for {variant}, found {len(paths)}"
            )
        kernels = [
            metric_row(variant, index, row)
            for index, row in enumerate(read_ncu_csv(paths[0]))
        ]
        output_rows.extend(kernels)
        output_rows.append(operation_row(variant, kernels))

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
