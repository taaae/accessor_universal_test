#!/usr/bin/env python3
"""Fail-closed structural audit for experiment 034 timed-kernel SASS."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

FUNCTION = re.compile(r"(?:Function\s*:\s*|\.section\s+\.text\.)(.+?)\s*$")
INSTRUCTION = re.compile(r"/\*([0-9a-fA-F]+)\*/\s+(@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)(?:\s+([^;]*))?;")
RECON64 = re.compile(r"reconstruct64_kernelILi(16|32|128)E([fd])E")
RECON32 = re.compile(r"reconstruct32_kernelILi(16|32|128)EE")
DEFERRED = re.compile(r"deferred_local_b128_kernelI([fd])E")
BASELINES = {"raw32_kernel": "raw_fp32", "widened32_kernel": "fp32_to_fp64", "raw64_kernel": "raw_fp64"}
REDUCERS = {"reduce32_kernel": "reduce32", "reduce64_kernel": "reduce64"}
REGISTER = re.compile(r"\b(?:U?R(?:Z|\d+))(?:\.reuse)?\b")


def parse(text):
    functions = {}
    current = None
    for line in text.splitlines():
        match = FUNCTION.search(line)
        if match:
            current = match.group(1).strip()
            functions.setdefault(current, [])
            continue
        match = INSTRUCTION.search(line)
        if match and current:
            functions[current].append((int(match.group(1), 16), match.group(3), ((match.group(2) or "") + (match.group(4) or "")).strip()))
    return functions


def resources(text):
    output = {}
    current = None
    for line in text.splitlines():
        match = re.search(r"Function properties for (\S+)", line)
        if match:
            current = match.group(1)
            output.setdefault(current, {})
        if current:
            match = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if match:
                output[current].update(spill_stores=int(match.group(1)), spill_loads=int(match.group(2)))
            match = re.search(r"Used (\d+) registers.*?(\d+) bytes smem", line)
            if match:
                output[current].update(registers=int(match.group(1)), shared_bytes=int(match.group(2)))
    return output


def lineage(sequence):
    provenance = {}
    loads = []
    operations = []
    for position, (_, opcode, arguments) in enumerate(sequence):
        registers = [item.replace(".reuse", "") for item in REGISTER.findall(arguments)]
        if not registers:
            continue
        destination, sources = registers[0], registers[1:]
        if opcode.startswith("LDG"):
            tag = f"load{len(loads)}"
            loads.append((position, opcode, destination, arguments))
            provenance[destination] = {tag}
            continue
        source_sets = [set(provenance.get(source, set())) for source in sources]
        if opcode.startswith(("F2F", "FMUL", "DMUL", "FFMA", "DFMA", "FADD", "DADD")):
            merged = set().union(*source_sets)
            provenance[destination] = merged
            operations.append((position, opcode, source_sets, merged, destination, arguments))
        elif destination not in sources:
            provenance.pop(destination, None)
    return loads, operations


def row(symbol, kind, block, scale, sequence, resource):
    opcodes = [item[1] for item in sequence]
    histogram = Counter(opcodes)
    reasons = []
    if any("FTZ" in op for op in opcodes):
        reasons.append("ftz_opcode")
    if any(op.startswith(("LDL", "STL", "CALL")) for op in opcodes):
        reasons.append("spill_or_call")
    if any("DIV" in op for op in opcodes):
        reasons.append("division_opcode")
    if any(op.startswith(("LDG.128", "LDG.E.128")) for op in opcodes):
        reasons.append("vectorized_load")
    loads, operations = lineage(sequence)
    load_opcodes = [item[1] for item in loads]
    loads64 = [op for op in load_opcodes if ".64" in op]
    count = lambda prefix: sum(op.startswith(prefix) for op in opcodes)
    def last_writer(register, before):
        writer = None
        for position, (_, opcode, arguments) in enumerate(sequence[:before]):
            registers = [item.replace(".reuse", "") for item in REGISTER.findall(arguments)]
            if registers and registers[0] == register:
                writer = (position, opcode)
        return writer
    fmuls = [item for item in operations if item[1].startswith("FMUL")]
    dmuls = [item for item in operations if item[1].startswith("DMUL")]
    dfmas = [item for item in operations if item[1].startswith("DFMA")]
    widens = [item for item in operations if item[1].startswith("F2F.F64.F32")]
    expected_pairs = [{"load0", "load2"}, {"load1", "load3"}]
    load_registers = [item[2] for item in loads]
    widen_map = {}
    for item in widens:
        registers = [register.replace(".reuse", "") for register in REGISTER.findall(item[5])]
        if len(registers) >= 2:
            widen_map[registers[1]] = registers[0]
    if kind == "reconstruct32":
        if len(loads) != 4 or loads64 or count("FMUL") != 2 or count("F2F.F64.F32") != 2 or count("DMUL") != 0 or count("DFMA") != 1 or count("SHFL") != 0:
            reasons.append("reconstruct32_shape")
        if [item[3] for item in fmuls] != expected_pairs:
            reasons.append("reconstruct32_operand_pairing")
        fmul_destinations = {item[4] for item in fmuls}
        widen_sources = {
            REGISTER.findall(item[5])[1].replace(".reuse", "")
            for item in widens if len(REGISTER.findall(item[5])) >= 2
        }
        widen_destinations = {item[4] for item in widens}
        dfma_sources = {
            register.replace(".reuse", "")
            for register in (REGISTER.findall(dfmas[0][5])[1:3] if len(dfmas) == 1 else [])
        }
        if fmul_destinations != widen_sources or widen_destinations != dfma_sources:
            reasons.append("reconstruct32_round_widen_chain")
        for item in widens:
            registers = [register.replace(".reuse", "") for register in REGISTER.findall(item[5])]
            if len(registers) >= 2 and last_writer(registers[1], item[0]) not in {(fmul[0], fmul[1]) for fmul in fmuls}:
                reasons.append("reconstruct32_intervening_fmul_write")
        if len(dfmas) == 1:
            registers = [register.replace(".reuse", "") for register in REGISTER.findall(dfmas[0][5])]
            for source in registers[1:3]:
                if last_writer(source, dfmas[0][0]) not in {(widen[0], widen[1]) for widen in widens}:
                    reasons.append("reconstruct32_intervening_widen_write")
        if len(dfmas) != 1 or dfmas[0][2][:2] != expected_pairs:
            reasons.append("reconstruct32_dot_does_not_consume_rounded_operands")
    elif kind == "reconstruct64":
        expected_widens = 4 if scale == "fp32" else 2
        if len(loads) != 4 or len(loads64) != (0 if scale == "fp32" else 2) or count("F2F.F64.F32") != expected_widens or count("DMUL") != 2 or count("DFMA") != 1 or count("FMUL") != 0 or count("SHFL") != 0:
            reasons.append("reconstruct64_shape")
        if [item[3] for item in dmuls] != expected_pairs:
            reasons.append("reconstruct64_operand_pairing")
        if len(load_registers) == 4:
            expected_widen_sources = set(load_registers if scale == "fp32" else load_registers[:2])
            expected_dmul_sources = [
                {widen_map.get(load_registers[0]), widen_map.get(load_registers[2], load_registers[2])},
                {widen_map.get(load_registers[1]), widen_map.get(load_registers[3], load_registers[3])},
            ]
            actual_dmul_sources = []
            for item in dmuls:
                registers = [register.replace(".reuse", "") for register in REGISTER.findall(item[5])]
                actual_dmul_sources.append(set(registers[1:3]))
            if set(widen_map) != expected_widen_sources or actual_dmul_sources != expected_dmul_sources:
                reasons.append("reconstruct64_widen_chain")
            required_writers = {item[4]: (item[0], item[1]) for item in widens}
            for load in loads:
                required_writers.setdefault(load[2], (load[0], load[1]))
            for item in dmuls:
                registers = [register.replace(".reuse", "") for register in REGISTER.findall(item[5])]
                if any(last_writer(source, item[0]) != required_writers.get(source) for source in registers[1:3]):
                    reasons.append("reconstruct64_intervening_operand_write")
            if len(dfmas) == 1:
                registers = [register.replace(".reuse", "") for register in REGISTER.findall(dfmas[0][5])]
                dmul_writers = {item[4]: (item[0], item[1]) for item in dmuls}
                if any(last_writer(source, dfmas[0][0]) != dmul_writers.get(source) for source in registers[1:3]):
                    reasons.append("reconstruct64_intervening_product_write")
        if len(dfmas) != 1 or dfmas[0][2][:2] != expected_pairs:
            reasons.append("reconstruct64_dot_does_not_consume_reconstructed_operands")
    elif kind == "deferred_local":
        expected_widens = 4 if scale == "fp32" else 2
        if block != 128 or len(loads) != 4 or len(loads64) != (0 if scale == "fp32" else 2) or count("F2F.F64.F32") != expected_widens or count("DMUL") != 1 or count("DFMA") != 2 or count("SHFL") != 0:
            reasons.append("deferred_local_shape")
        payload = [item for item in dfmas if len(item[2]) >= 2 and item[2][0] == {"load0"} and item[2][1] == {"load1"}]
        scale_product = [item for item in dmuls if len(item[2]) >= 2 and {frozenset(item[2][0]), frozenset(item[2][1])} == {frozenset({"load2"}), frozenset({"load3"})}]
        final = [item for item in dfmas if len(item[2]) >= 2 and item[2][0] == {"load2", "load3"} and item[2][1] == {"load0", "load1"}]
        if len(payload) != 1 or len(scale_product) != 1 or len(final) != 1 or not payload[0][0] < scale_product[0][0] < final[0][0]:
            reasons.append("deferred_local_subtotal_dataflow")
        if len(load_registers) == 4:
            expected_widen_sources = set(load_registers if scale == "fp32" else load_registers[:2])
            payload_registers = {register.replace(".reuse", "") for register in REGISTER.findall(payload[0][5])[1:3]} if len(payload) == 1 else set()
            scale_registers = {register.replace(".reuse", "") for register in REGISTER.findall(scale_product[0][5])[1:3]} if len(scale_product) == 1 else set()
            expected_payload = {widen_map.get(load_registers[0]), widen_map.get(load_registers[1])}
            expected_scale = {widen_map.get(load_registers[2], load_registers[2]), widen_map.get(load_registers[3], load_registers[3])}
            if set(widen_map) != expected_widen_sources or payload_registers != expected_payload or scale_registers != expected_scale:
                reasons.append("deferred_local_widen_chain")
            required_writers = {item[4]: (item[0], item[1]) for item in widens}
            for load in loads:
                required_writers.setdefault(load[2], (load[0], load[1]))
            for item in payload + scale_product:
                registers = [register.replace(".reuse", "") for register in REGISTER.findall(item[5])]
                if any(last_writer(source, item[0]) != required_writers.get(source) for source in registers[1:3]):
                    reasons.append("deferred_local_intervening_operand_write")
            if len(final) == 1 and len(payload) == 1 and len(scale_product) == 1:
                registers = [register.replace(".reuse", "") for register in REGISTER.findall(final[0][5])]
                final_writers = {payload[0][4]: (payload[0][0], payload[0][1]), scale_product[0][4]: (scale_product[0][0], scale_product[0][1])}
                if any(last_writer(source, final[0][0]) != final_writers.get(source) for source in registers[1:3]):
                    reasons.append("deferred_local_intervening_subtotal_write")
    elif kind == "raw_fp32":
        if len(loads) != 2 or loads64 or count("FFMA") != 1:
            reasons.append("raw_fp32_shape")
        ffmas = [item for item in operations if item[1].startswith("FFMA")]
        if len(ffmas) != 1 or ffmas[0][2][:2] != [{"load0"}, {"load1"}]:
            reasons.append("raw_fp32_load_lineage")
    elif kind == "fp32_to_fp64":
        if len(loads) != 2 or loads64 or count("F2F.F64.F32") != 2 or count("DFMA") != 1:
            reasons.append("widened_fp32_shape")
        if len(dfmas) != 1 or dfmas[0][2][:2] != [{"load0"}, {"load1"}]:
            reasons.append("widened_fp32_load_lineage")
    elif kind == "raw_fp64":
        if len(loads) != 2 or len(loads64) != 2 or count("DFMA") != 1:
            reasons.append("raw_fp64_shape")
        if len(dfmas) != 1 or dfmas[0][2][:2] != [{"load0"}, {"load1"}]:
            reasons.append("raw_fp64_load_lineage")
    elif kind == "reduce32":
        if len(loads) != 1 or not count("FADD"):
            reasons.append("reduce32_shape")
        if not any(item[1].startswith("FADD") and {"load0"} in item[2] for item in operations):
            reasons.append("reduce32_load_lineage")
    elif kind == "reduce64":
        if len(loads) != 1 or not count("DADD"):
            reasons.append("reduce64_shape")
        if not any(item[1].startswith("DADD") and {"load0"} in item[2] for item in operations):
            reasons.append("reduce64_load_lineage")
    for key in ("registers", "spill_loads", "spill_stores", "shared_bytes"):
        if key not in resource:
            reasons.append("missing_resource_" + key)
    if resource.get("spill_loads", 0) or resource.get("spill_stores", 0):
        reasons.append("ptxas_spills")
    return dict(symbol=symbol, kind=kind, B=block, scale_type=scale,
                fmul=count("FMUL"), dmul=count("DMUL"), dfma=count("DFMA"),
                shuffle=count("SHFL"), opcode_histogram=dict(histogram),
                registers=resource.get("registers"), shared_bytes=resource.get("shared_bytes"),
                spill_loads=resource.get("spill_loads"), spill_stores=resource.get("spill_stores"),
                status="pass" if not reasons else "fail", reason=";".join(reasons))


def run(sass, build):
    functions = parse(sass)
    resource_map = resources(build)
    rows = []
    for symbol, sequence in functions.items():
        match = RECON64.search(symbol)
        if match:
            rows.append(row(symbol, "reconstruct64", int(match.group(1)), "fp32" if match.group(2) == "f" else "fp64", sequence, resource_map.get(symbol, {})))
            continue
        match = RECON32.search(symbol)
        if match:
            rows.append(row(symbol, "reconstruct32", int(match.group(1)), "fp32", sequence, resource_map.get(symbol, {})))
            continue
        match = DEFERRED.search(symbol)
        if match:
            rows.append(row(symbol, "deferred_local", 128, "fp32" if match.group(1) == "f" else "fp64", sequence, resource_map.get(symbol, {})))
            continue
        matched = False
        for needle, name in BASELINES.items():
            if needle in symbol:
                rows.append(row(symbol, name, 0, "none", sequence, resource_map.get(symbol, {})))
                matched = True
                break
        if matched:
            continue
        for needle, name in REDUCERS.items():
            if needle in symbol:
                rows.append(row(symbol, name, 0, "none", sequence, resource_map.get(symbol, {})))
                break
    expected = {("reconstruct64", block, scale) for block in (16, 32, 128) for scale in ("fp32", "fp64")}
    expected |= {("reconstruct32", block, "fp32") for block in (16, 32, 128)}
    expected |= {("deferred_local", 128, scale) for scale in ("fp32", "fp64")}
    expected |= {(name, 0, "none") for name in BASELINES.values()}
    expected |= {(name, 0, "none") for name in REDUCERS.values()}
    observed = {(item["kind"], item["B"], item["scale_type"]) for item in rows}
    if len(rows) != 16 or len({item["symbol"] for item in rows}) != 16 or observed != expected:
        raise ValueError(f"timed-kernel inventory mismatch missing={expected-observed} extra={observed-expected}")
    failed = [item for item in rows if item["status"] != "pass"]
    if failed:
        raise ValueError(f"SASS audit failed: {failed}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sass", required=True)
    parser.add_argument("--build-log", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = run(Path(args.sass).read_text(errors="replace"), Path(args.build_log).read_text(errors="replace"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "assembly_audit.json").write_text(json.dumps(rows, indent=2))
    with (output / "assembly_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"SASS audit passed: {len(rows)} timed kernels")


if __name__ == "__main__":
    main()
