"""Static SASS/resource extraction for generated calibration kernels."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .manifest import CASES


FUNCTION_RE = re.compile(r"(?:Function\s*:\s*|\.section\s+\.text\.)(.+?)\s*$")
INSTRUCTION_RE = re.compile(
    r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?P\d+\s+)?"
    r"([A-Z][A-Z0-9_.]*)(?:\s+([^;]*?))?\s*;\s*$"
)
TARGET_RE = re.compile(r"(?:0x)?([0-9a-fA-F]+)\s*;?\s*$")
REGISTER_RE = re.compile(r"\b(?:U?RZ|U?R\d+|P\d+)\b")
CASE_RE = re.compile(r"dot_case_kernel(?:<|ILi)(\d+)")


@dataclass(frozen=True)
class Instruction:
    address: int
    opcode: str
    operands: str
    predicate: bool


def parse_functions(text: str) -> dict[str, list[Instruction]]:
    result: dict[str, list[Instruction]] = {}
    current = ""
    for line in text.splitlines():
        function = FUNCTION_RE.search(line)
        if function:
            current = function.group(1).strip()
            result.setdefault(current, [])
            continue
        instruction = INSTRUCTION_RE.search(line)
        if instruction and current:
            before = line[: instruction.start(2)]
            result[current].append(Instruction(
                int(instruction.group(1), 16), instruction.group(2),
                (instruction.group(3) or "").strip(),
                "@P" in before or "@!P" in before))
    return result


def case_index(function_name: str) -> int | None:
    match = CASE_RE.search(function_name)
    return int(match.group(1)) if match else None


def main_loop(instructions: list[Instruction]) -> list[Instruction]:
    """Return the widest backward-branch region, normally the grid-stride loop."""
    candidates: list[tuple[int, int]] = []
    addresses = {item.address for item in instructions}
    for item in instructions:
        if not (item.opcode.startswith("BRA") or item.opcode.startswith("JMP")):
            continue
        target = TARGET_RE.search(item.operands)
        if not target:
            continue
        address = int(target.group(1), 16)
        if address in addresses and address < item.address:
            candidates.append((address, item.address))
    if not candidates:
        return instructions
    begin, end = max(candidates, key=lambda pair: pair[1] - pair[0])
    return [item for item in instructions if begin <= item.address <= end]


def critical_depth(instructions: Iterable[Instruction]) -> int:
    last_writer: dict[str, int] = {}
    maximum = 0
    no_destination = ("ST", "BRA", "JMP", "EXIT", "RET", "BAR", "MEMBAR")
    for item in instructions:
        registers = REGISTER_RE.findall(item.operands)
        if not registers:
            continue
        has_destination = not item.opcode.startswith(no_destination)
        destination = registers[0] if has_destination else None
        sources = registers[1:] if destination else registers
        depth = 1 + max((last_writer.get(source, 0) for source in sources), default=0)
        if destination and destination not in {"RZ", "URZ"}:
            last_writer[destination] = depth
        maximum = max(maximum, depth)
    return maximum


def _count(opcodes: Iterable[str], prefixes: tuple[str, ...]) -> int:
    return sum(opcode.startswith(prefixes) for opcode in opcodes)


def feature_row(index: int, instructions: list[Instruction]) -> dict[str, float | int | str]:
    case = CASES[index]
    loop = main_loop(instructions)
    opcodes = [item.opcode for item in loop]
    global_loads = _count(opcodes, ("LDG", "LD.E", "LD."))
    shared_loads = _count(opcodes, ("LDS",))
    constant_loads = _count(opcodes, ("LDC",))
    local_loads = _count(opcodes, ("LDL",))
    local_stores = _count(opcodes, ("STL",))
    branch_count = _count(opcodes, ("BRA", "BRX", "JMP"))
    predicated = sum(item.predicate for item in loop)
    probability = case.branch_probability or 0.0
    true_path = 1.0 - (1.0 - probability) ** 32 if probability else 0.0
    false_path = 1.0 - probability ** 32 if probability else 0.0
    lut_entries = 1 << case.lut_index_bits if case.lut_index_bits else 0
    lut_bytes = lut_entries * (24 if case.case_id.endswith("pwqnormal32_8_24") else 16 if case.case_id.endswith("pwlnormal32_16_16") else 8)
    expected_sectors = 0.0
    if case.lut_index_bits and case.lut_memory == "global":
        sectors = max(1, (lut_bytes + 31) // 32)
        expected_sectors = sectors * (1.0 - (1.0 - 1.0 / sectors) ** 32) * case.lut_loads
    return {
        "case_id": case.case_id,
        "split": case.split,
        "group": case.group,
        "function": next((name for name, seq in []), ""),
        "loop_instruction_count": len(loop),
        "integer_alu": _count(opcodes, ("IADD", "UIADD", "LOP", "SHF", "IMAD", "LEA", "BFE", "BFI", "PRMT", "POPC", "FLO")),
        "integer_multiply": _count(opcodes, ("IMAD", "XMAD")),
        "integer_divide": _count(opcodes, ("CALL",)) if case.params.get("op") == "div_u32" else 0,
        "conversion": _count(opcodes, ("I2F", "F2I", "F2F")),
        "fp32": _count(opcodes, ("FADD", "FMUL", "FFMA")),
        "fp64": _count(opcodes, ("DADD", "DMUL", "DFMA")),
        "special": _count(opcodes, ("MUFU", "RRO", "CALL")),
        "global_loads": global_loads,
        "shared_loads": shared_loads,
        "constant_loads": constant_loads,
        "local_loads": local_loads,
        "local_stores": local_stores,
        "branch_count": branch_count,
        "predicated_instruction_count": predicated,
        "expected_true_warp_path": true_path,
        "expected_false_warp_path": false_path,
        "critical_dependency_depth": critical_depth(loop),
        "lut_loads_metadata": case.lut_loads,
        "lut_footprint_bytes": lut_bytes,
        "lut_expected_sectors_per_warp": expected_sectors,
        "branch_probability": probability,
    }


def extract(text: str) -> list[dict[str, float | int | str]]:
    functions = parse_functions(text)
    found: dict[int, tuple[str, list[Instruction]]] = {}
    for name, instructions in functions.items():
        index = case_index(name)
        if index is not None:
            found[index] = (name, instructions)
    missing = sorted(set(range(len(CASES))) - set(found))
    if missing:
        raise ValueError(f"missing generated kernels in SASS: {missing[:12]}")
    rows = []
    for index in range(len(CASES)):
        name, instructions = found[index]
        row = feature_row(index, instructions)
        row["function"] = name
        rows.append(row)
    return rows


def merge_resources(rows: list[dict], resource_csv: Path | None) -> None:
    resources: dict[str, dict[str, str]] = {}
    if resource_csv and resource_csv.exists():
        with resource_csv.open(newline="") as handle:
            resources = {row["case_id"]: row for row in csv.DictReader(handle)}
    for row in rows:
        resource = resources.get(str(row["case_id"]), {})
        registers = int(resource.get("registers", 0))
        static_shared = int(resource.get("static_shared_bytes", 0))
        local_bytes = int(resource.get("local_bytes", 0))
        case = next(case for case in CASES if case.case_id == row["case_id"])
        dynamic_shared = (1 << case.lut_index_bits) * 8 if case.lut_memory == "shared" else (6144 if case.case_id.endswith("pwqnormal32_8_24") else 0)
        blocks_register = 32 if not registers else max(1, 65536 // (registers * 256))
        blocks_shared = 32 if not (static_shared + dynamic_shared + 2048) else max(1, 233472 // (static_shared + dynamic_shared + 2048))
        resident_blocks = min(8, blocks_register, blocks_shared)
        occupancy = min(1.0, resident_blocks * 256 / 2048.0)
        row.update({"registers_per_thread": registers, "static_shared_bytes": static_shared,
                    "dynamic_shared_bytes": dynamic_shared + 2048,
                    "local_bytes_per_thread": local_bytes,
                    "estimated_occupancy": occupancy})


def validate_acceptance(rows: list[dict]) -> list[str]:
    warnings: list[str] = []
    by_id = {row["case_id"]: row for row in rows}
    for case in CASES:
        row = by_id[case.case_id]
        if row["loop_instruction_count"] == 0:
            raise ValueError(f"empty SASS loop: {case.case_id}")
        if case.kind == "lut" and row["global_loads"] + row["shared_loads"] + row["constant_loads"] == 0:
            raise ValueError(f"LUT load optimized away: {case.case_id}")
        if case.kind == "branch" and row["branch_count"] == 0 and row["predicated_instruction_count"] == 0:
            warnings.append(f"branch body has neither explicit branch nor predication: {case.case_id}")
    training_branches = [by_id[case.case_id] for case in CASES if case.split == "train" and case.kind == "branch"]
    if not any(row["branch_count"] > 0 for row in training_branches):
        raise ValueError("branch calibration contains no real SASS branch; increase only long bodies to 32")
    if not any(row["predicated_instruction_count"] > 0 and row["branch_count"] == 0 for row in training_branches):
        warnings.append("branch calibration contains no purely predicated case")
    return warnings


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
