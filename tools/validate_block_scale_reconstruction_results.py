#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

FULL_N = [65536, 1048576, 16777216, 67108864, 268435456]
SMOKE_N = [16384, 1048576]
BLOCKS = [16, 32, 128]
FIELDS = "mode,stage,N,B,scale_type,path,reconstruction,subtotal_mode,dot_accumulation,variant_id,payload_type,grid,threads,round,order,time_ms,result,valid,job_id,node,experiment_id,dataset_id,seed_set_id,source_commit,binary_sha,sass_sha,audit_sha".split(",")


def fail(message: str) -> None:
    raise ValueError(message)


def variants() -> set[tuple[str, int, str, str, str, str, str, str]]:
    out = set()
    for block in BLOCKS:
        out.add((f"reconstruct64_b{block}_s32", block, "fp32", "per_element", "fp64", "none", "fp64", "fp32"))
        out.add((f"reconstruct32_b{block}_s32", block, "fp32", "per_element", "fp32", "none", "fp64", "fp32"))
        out.add((f"reconstruct64_b{block}_s64", block, "fp64", "per_element", "fp64", "none", "fp64", "fp32"))
    out.add(("deferred_local_b128_s32", 128, "fp32", "deferred_local", "fp64", "thread_local_4", "fp64", "fp32"))
    out.add(("deferred_local_b128_s64", 128, "fp64", "deferred_local", "fp64", "thread_local_4", "fp64", "fp32"))
    out.add(("raw_fp32", 0, "none", "baseline", "native_fp32", "none", "fp32", "fp32"))
    out.add(("fp32_to_fp64", 0, "none", "baseline", "widen_fp32", "none", "fp64", "fp32"))
    out.add(("raw_fp64", 0, "none", "baseline", "native_fp64", "none", "fp64", "fp64"))
    return out


def validate(samples: str, manifest_path: str, mode: str | None = None):
    manifest = json.loads(Path(manifest_path).read_text())
    with open(samples, newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames != FIELDS:
            fail("CSV header mismatch")
    if not rows:
        fail("empty CSV")
    mode = mode or manifest.get("mode")
    if {row["mode"] for row in rows} != {mode} or mode not in {"full", "smoke"}:
        fail("mode mismatch")
    sizes = FULL_N if mode == "full" else SMOKE_N
    sample_count = 50 if mode == "full" else 3
    allowed = variants()
    groups = defaultdict(list)
    round_orders = defaultdict(list)
    required = {"source_commit": "commit", "binary_sha": "binary_sha256", "sass_sha": "sass_sha256", "audit_sha": "audit_sha256", "job_id": "job_id", "node": "node"}
    hex40 = re.compile(r"[0-9a-f]{40}")
    hex64 = re.compile(r"[0-9a-f]{64}")
    if not hex40.fullmatch(str(manifest.get("commit", ""))):
        fail("invalid source commit provenance")
    for key in ("binary_sha256", "sass_sha256", "audit_sha256", "environment_sha256"):
        if not hex64.fullmatch(str(manifest.get(key, ""))):
            fail(f"invalid digest provenance {key}")
    if not str(manifest.get("job_id", "")).isdigit() or not str(manifest.get("node", "")):
        fail("invalid job or node provenance")
    for row in rows:
        try:
            key = (row["variant_id"], int(row["B"]), row["scale_type"], row["path"], row["reconstruction"], row["subtotal_mode"], row["dot_accumulation"], row["payload_type"])
            n = int(row["N"])
            round_index = int(row["round"])
            order = int(row["order"])
            time_ms = float(row["time_ms"])
            value = float(row["result"])
        except Exception as error:
            fail(f"malformed row: {error}")
        if key not in allowed:
            fail(f"unknown or inconsistent variant {key}")
        if n not in sizes or row["stage"] not in {"initial", "rerun"}:
            fail("bad N or stage")
        if row["grid"] != "512" or row["threads"] != "256" or row["valid"] != "1":
            fail("geometry or validity mismatch")
        if row["experiment_id"] != "034_block_scale_reconstruction" or row["dataset_id"] != "block-scale-v1" or row["seed_set_id"] != "block-scale-v1":
            fail("experiment or dataset identity mismatch")
        if not math.isfinite(time_ms) or time_ms <= 0 or not math.isfinite(value):
            fail("invalid numeric value")
        for column, manifest_key in required.items():
            if row[column] != str(manifest.get(manifest_key, "")):
                fail(f"manifest binding mismatch {column}")
        group = (row["stage"], n, row["variant_id"])
        groups[group].append(row)
        round_orders[(row["stage"], n, round_index)].append(order)
    initial = {("initial", n, item[0]) for n in sizes for item in allowed}
    if {group for group in groups if group[0] == "initial"} != initial:
        fail("exact initial inventory mismatch")
    expected_initial = len(sizes) * 14 * sample_count
    if len([row for row in rows if row["stage"] == "initial"]) != expected_initial:
        fail(f"expected {expected_initial} initial rows")
    rerun_sizes = {n for stage, n, _ in groups if stage == "rerun"}
    for n in rerun_sizes:
        expected = {("rerun", n, item[0]) for item in allowed}
        if {(stage, size, variant) for stage, size, variant in groups if stage == "rerun" and size == n} != expected:
            fail("incomplete rerun block")
    if mode == "smoke" and rerun_sizes:
        fail("smoke must not contain reruns")
    for group, items in groups.items():
        if len(items) != sample_count or {int(item["round"]) for item in items} != set(range(sample_count)):
            fail(f"case sample inventory mismatch {group}")
        if len({item["result"] for item in items}) != 1:
            fail(f"nondeterministic result {group}")
    for key, orders in round_orders.items():
        if sorted(orders) != list(range(14)):
            fail(f"execution order mismatch {key}")
    if mode == "full":
        for n in sizes:
            drift = False
            for variant in ("raw_fp32", "fp32_to_fp64", "raw_fp64"):
                items = sorted(groups[("initial", n, variant)], key=lambda item: int(item["round"]))
                early = sorted(float(item["time_ms"]) for item in items[:25])
                late = sorted(float(item["time_ms"]) for item in items[25:])
                early_median = early[12]
                late_median = late[12]
                drift |= abs(late_median - early_median) / early_median > 0.05
            if drift and n not in rerun_sizes:
                fail(f"baseline drift requires complete rerun N={n}")
            if not drift and n in rerun_sizes:
                fail(f"rerun without qualifying baseline drift N={n}")
    for stage, n, _ in groups:
        for block in BLOCKS:
            a = groups[(stage, n, f"reconstruct64_b{block}_s32")][0]["result"]
            b = groups[(stage, n, f"reconstruct64_b{block}_s64")][0]["result"]
            if a != b:
                fail(f"reconstruct64 scale-width identity failed {stage} {n} B{block}")
        a = groups[(stage, n, "deferred_local_b128_s32")][0]["result"]
        b = groups[(stage, n, "deferred_local_b128_s64")][0]["result"]
        if a != b:
            fail(f"deferred scale-width identity failed {stage} {n}")
    expected_manifest = {"mode": mode, "sizes": sizes, "variants": 14, "samples": sample_count, "warmups": 10 if mode == "full" else 1, "grid": 512, "threads": 256, "payload": "fp32", "dot_accumulation": "fp64_except_raw_fp32", "access": "scalar_x1", "cache_protocol": "repeat_input_no_explicit_flush", "experiment_id": "034_block_scale_reconstruction", "dataset_id": "block-scale-v1", "seed_set_id": "block-scale-v1", "rounding_reference_cases": 238}
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        fail("manifest experiment identity mismatch")
    print(f"validation passed: {len(rows)} rows, {len(groups)} groups, mode={mode}")
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=["full", "smoke"])
    args = parser.parse_args()
    validate(args.samples, args.manifest, args.mode)


if __name__ == "__main__":
    main()
