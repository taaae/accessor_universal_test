#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


PWL_KERNELS = (
    "dot_pwl2_compand32_kernel",
    "dot_pwl4_compand32_kernel",
)
BASELINE_KERNEL = "dot_int32_kernel"


def function_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"Function\s*:\s*([^\n]+)", text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[match.start():end]
    return sections


def instructions(section: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    pattern = re.compile(
        r"/\*[^*]+\*/\s+(?:@!?P[0-9]+\s+)?([A-Z][A-Z0-9_.]+)"
    )
    for line in section.splitlines():
        match = pattern.search(line)
        if match:
            result.append((match.group(1), line))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sass = args.sass.read_text(errors="replace")
    build = args.build_log.read_text(errors="replace")
    source = args.source.read_text(errors="replace")
    sections = function_sections(sass)
    findings: list[str] = []
    passed = True

    nonzero_spills = re.findall(
        r"([1-9][0-9]*) bytes spill stores|([1-9][0-9]*) bytes spill loads", build
    )
    if nonzero_spills:
        passed = False
        findings.append(f"FAIL nonzero ptxas spill report: {nonzero_spills}")
    else:
        findings.append("PASS ptxas reports no nonzero spill loads or stores")

    for function in ("decode_pwl2", "decode_pwl4"):
        match = re.search(
            rf"\b{function}\s*\([^)]*\)\s*\{{(?P<body>.*?)\n\}}",
            source,
            re.DOTALL,
        )
        if match is None:
            passed = False
            findings.append(f"FAIL source audit: could not locate {function}")
            continue
        body = match.group("body")
        if re.search(r"\b(?:table|coefficients?)\b|[A-Za-z_][A-Za-z0-9_]*\s*\[", body):
            passed = False
            findings.append(
                f"FAIL source audit: {function} contains an array or coefficient-table reference"
            )
        else:
            findings.append(
                f"PASS source audit: {function} uses scalar compile-time coefficients only"
            )

    baseline_matches = [
        section for symbol, section in sections.items() if BASELINE_KERNEL in symbol
    ]
    if len(baseline_matches) != 1:
        passed = False
        findings.append(
            f"FAIL {BASELINE_KERNEL}: expected one SASS section, found {len(baseline_matches)}"
        )
        baseline_branch_count = -1
    else:
        baseline_instructions = instructions(baseline_matches[0])
        baseline_branch_count = sum(
            opcode.startswith(("BRA", "BRX", "JMP"))
            for opcode, _ in baseline_instructions
        )
        findings.append(
            f"INFO {BASELINE_KERNEL}: {baseline_branch_count} static branch instructions"
        )

    for name in PWL_KERNELS:
        matching = [section for symbol, section in sections.items() if name in symbol]
        if len(matching) != 1:
            passed = False
            findings.append(f"FAIL {name}: expected one SASS section, found {len(matching)}")
            continue
        section = matching[0]
        decoded = instructions(section)
        opcodes = [opcode for opcode, _ in decoded]
        local_ops = [opcode for opcode in opcodes if opcode.startswith(("LDL", "STL"))]
        branch_indirect = [opcode for opcode in opcodes if opcode.startswith("BRX")]
        branch_count = sum(
            opcode.startswith(("BRA", "BRX", "JMP")) for opcode in opcodes
        )
        global_loads = [opcode for opcode in opcodes if opcode.startswith("LDG")]
        indexed_constant_loads = [
            line for opcode, line in decoded
            if opcode.startswith(("LDC", "ULDC"))
            and re.search(r"c\[[^]]+\]\[[^]]*R[0-9]+", line)
        ]
        select_ops = [opcode for opcode in opcodes if opcode.startswith("SEL")]
        fma_ops = [opcode for opcode in opcodes if "FMA" in opcode]
        if local_ops:
            passed = False
            findings.append(f"FAIL {name}: local-memory operations {local_ops}")
        else:
            findings.append(f"PASS {name}: no local-memory loads or stores")
        if branch_indirect:
            passed = False
            findings.append(f"FAIL {name}: indirect branch operations {branch_indirect}")
        else:
            findings.append(f"PASS {name}: no indirect branch or jump table")
        if baseline_branch_count >= 0 and branch_count > baseline_branch_count:
            passed = False
            findings.append(
                f"FAIL {name}: {branch_count} static branches versus "
                f"{baseline_branch_count} in Int32, indicating extra control flow"
            )
        else:
            findings.append(
                f"PASS {name}: {branch_count} static branches, no more than Int32"
            )
        if len(global_loads) > 2:
            passed = False
            findings.append(
                f"FAIL {name}: {len(global_loads)} global loads, expected only the two code arrays"
            )
        else:
            findings.append(
                f"PASS {name}: {len(global_loads)} static global-load instructions, no coefficient table"
            )
        if indexed_constant_loads:
            passed = False
            findings.append(
                f"FAIL {name}: dynamically indexed constant-memory loads: "
                f"{indexed_constant_loads}"
            )
        else:
            findings.append(
                f"PASS {name}: no dynamically indexed constant-memory load"
            )
        if not select_ops:
            passed = False
            findings.append(f"FAIL {name}: no predicated select instruction found")
        else:
            findings.append(f"PASS {name}: {len(select_ops)} predicated select instructions")
        if len(fma_ops) < 2:
            passed = False
            findings.append(f"FAIL {name}: expected decode and DOT FMAs, found {len(fma_ops)}")
        else:
            findings.append(f"PASS {name}: {len(fma_ops)} FMA-family instructions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(findings) + f"\nall_codegen_checks_passed={int(passed)}\n")
    print(args.output.read_text(), end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
