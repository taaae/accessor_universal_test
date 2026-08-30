"""Static SASS/resource extraction for generated calibration kernels."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .manifest import CASES


FUNCTION_RE = re.compile(r"(?:Function\s*:\s*|\.section\s+\.text\.)(.+?)\s*$")
INSTRUCTION_RE = re.compile(
    r"/\*([0-9a-fA-F]+)\*/\s+(?:@!?P\d+\s+)?"
    r"([A-Z][A-Z0-9_.]*)(?:\s+([^;]*?))?\s*;"
    r"(?:\s*/\*.*\*/)?\s*$"
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


def _count_exact(opcodes: Iterable[str], names: tuple[str, ...]) -> int:
    return sum(opcode in names for opcode in opcodes)


def feature_row(index: int, instructions: list[Instruction]) -> dict[str, float | int | str]:
    case = CASES[index]
    loop = main_loop(instructions)
    opcodes = [item.opcode for item in loop]
    global_loads = _count(opcodes, ("LDG", "LD.E"))
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
        expected_sectors = (2.0 * sectors *
                            (1.0 - (1.0 - 1.0 / sectors) ** 32) *
                            case.lut_loads)
    return {
        "case_id": case.case_id,
        "split": case.split,
        "group": case.group,
        "function": next((name for name, seq in []), ""),
        "loop_opcode_signature": hashlib.sha256(" ".join(opcodes).encode()).hexdigest(),
        "loop_instruction_count": len(loop),
        "integer_alu": _count(opcodes, ("IADD", "UIADD", "VIADD", "LOP", "SHF", "IMAD", "LEA", "BFE", "BFI", "PRMT", "POPC", "FLO")),
        # Hopper may encode a two-source integer add as IMAD.IADD. Keep the
        # exact spellings as diagnostics and group their shared add semantics
        # separately. Plain IMAD is the multiply-add family; IMAD.IADD is not.
        "iadd_sass": _count(opcodes, ("IADD3", "UIADD3", "VIADD")) +
                     _count_exact(opcodes, ("IMAD.IADD",)),
        "iadd3_sass": _count(opcodes, ("IADD3", "UIADD3")),
        "viadd_sass": _count(opcodes, ("VIADD",)),
        "lop3_sass": _count(opcodes, ("LOP3",)),
        "shf_sass": _count(opcodes, ("SHF",)),
        "imad_sass": _count_exact(opcodes, ("IMAD",)) + _count(opcodes, ("XMAD",)),
        "integer_multiply": _count_exact(opcodes, ("IMAD",)) + _count(opcodes, ("XMAD",)),
        "integer_divide": _count(opcodes, ("CALL",)) if case.params.get("op") == "div_u32" else 0,
        "conversion": _count(opcodes, ("I2F", "F2I", "F2F")),
        "fp32": sum(opcode.startswith(("FADD", "FMUL", "FFMA")) and ".D2" not in opcode for opcode in opcodes),
        "fp64": _count(opcodes, ("DADD", "DMUL", "DFMA")) +
                sum(opcode.startswith(("FADD.D2", "FMUL.D2", "FFMA.D2")) for opcode in opcodes),
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
    baseline = rows[0]
    delta_fields = ("loop_instruction_count", "integer_alu", "integer_multiply",
                    "integer_divide", "iadd_sass", "iadd3_sass", "viadd_sass", "lop3_sass",
                    "shf_sass", "imad_sass", "conversion", "fp32", "fp64", "special",
                    "global_loads", "shared_loads", "constant_loads", "local_loads",
                    "local_stores", "branch_count", "predicated_instruction_count")
    for row in rows:
        for field in delta_fields:
            row[f"decoder_{field}"] = max(0, int(row[field]) - int(baseline[field]))
        case = next(case for case in CASES if case.case_id == row["case_id"])
        nominal = 0
        primary = "none"
        if case.kind in {"integer", "clz", "int64"}:
            nominal = int(case.params.get("count", 1)); primary = "integer_alu"
        elif case.kind == "numeric":
            nominal = int(case.params["count"]); primary = "conversion"
        elif case.kind == "numeric_chain":
            nominal = 2; primary = "conversion"
        elif case.kind == "fp":
            nominal = int(case.params["count"]); primary = str(case.params["precision"])
        elif case.kind == "latency":
            nominal = int(case.params["fmas"]); primary = "fp64"
        elif case.kind == "special":
            nominal = 1; primary = "special"
        elif case.kind == "lut":
            nominal = int(case.lut_loads); primary = f"{case.lut_memory}_loads"
        # The converter is inlined twice in each loop iteration: once for the
        # left operand and once for the right operand.
        nominal *= 2
        row["nominal_primary_operations"] = nominal
        row["primary_operation_family"] = primary
        if primary.endswith("_loads"):
            memory = primary.removesuffix("_loads")
            actual = row[f"decoder_{memory}_loads"]
        elif primary == "special":
            # CUDA lowers some nominal special operations, notably FP64 exp2,
            # to a visible polynomial instead of MUFU or CALL. Count the
            # complete lowered numeric body while retaining each SASS family
            # in its own predictor feature.
            actual = sum(row[f"decoder_{family}"] for family in
                         ("special", "fp64", "fp32", "conversion",
                          "integer_multiply", "integer_divide"))
        elif primary in {"fp32", "fp64", "conversion", "integer_alu"}:
            actual = row[f"decoder_{primary}"]
        else:
            actual = 0
        row["compiled_primary_operations"] = actual
        row["compiler_count_status"] = ("not_applicable" if nominal == 0 else
                                        "exact" if actual == nominal else "changed")
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
        dynamic_shared = (6144 if case.case_id.endswith("pwqnormal32_8_24") else
                          (1 << case.lut_index_bits) * 8 if case.lut_memory == "shared" else 0)
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
        if case.kind == "lut" and row[f"decoder_{case.lut_memory}_loads"] == 0:
            raise ValueError(f"LUT load optimized away or in wrong memory space: {case.case_id}")
        if case.kind == "branch" and row["decoder_branch_count"] == 0 and row["decoder_predicated_instruction_count"] == 0:
            warnings.append(f"branch body has neither explicit branch nor predication: {case.case_id}")
        if case.kind in {"integer", "clz", "int64", "numeric", "numeric_chain", "fp", "latency", "special"} and row["compiled_primary_operations"] == 0:
            raise ValueError(f"primary operation family optimized away: {case.case_id}")
        if case.kind == "integer":
            required = {"iadd3": "iadd_sass", "lop3": "lop3_sass",
                        "shf": "shf_sass", "imad": "imad_sass"}[case.params["op"]]
            if row[f"decoder_{required}"] == 0:
                raise ValueError(f"requested SASS opcode family missing: {case.case_id}")
        if case.kind == "mixed":
            for family in ("integer_alu", "conversion", "fp64"):
                if row[f"decoder_{family}"] == 0:
                    raise ValueError(f"mixed pipeline lost {family}: {case.case_id}")
        if case.kind == "clz_shift" and row["decoder_integer_alu"] == 0:
            raise ValueError(f"CLZ/shift validation body optimized away: {case.case_id}")
        if case.kind == "branch":
            family = "integer_alu" if case.params["body"] == "integer" else "fp64"
            if row[f"decoder_{family}"] == 0:
                raise ValueError(f"branch body optimized away: {case.case_id}")
        if case.kind == "real" and row["loop_opcode_signature"] == by_id[CASES[0].case_id]["loop_opcode_signature"]:
            raise ValueError(f"real converter opcode stream indistinguishable from bitcast anchor: {case.case_id}")
        if row["compiler_count_status"] == "changed":
            warnings.append(f"compiler changed nominal primary count for {case.case_id}: nominal={row['nominal_primary_operations']} compiled={row['compiled_primary_operations']}")
    training_branches = [by_id[case.case_id] for case in CASES if case.split == "train" and case.kind == "branch"]
    if not any(row["decoder_branch_count"] > 0 for row in training_branches):
        raise ValueError("branch calibration contains no real SASS branch; increase only long bodies to 32")
    if not any(row["decoder_predicated_instruction_count"] > 0 and row["decoder_branch_count"] == 0 for row in training_branches):
        warnings.append("branch calibration contains no purely predicated case")
    return warnings


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
