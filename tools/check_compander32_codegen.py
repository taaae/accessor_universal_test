#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


PWL_KERNELS = (
    "dot_pwl2_compand32_kernel",
    "dot_pwl4_compand32_kernel",
)


def function_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"Function\s*:\s*([^\n]+)", text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[match.start():end]
    return sections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sass = args.sass.read_text(errors="replace")
    build = args.build_log.read_text(errors="replace")
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

    for name in PWL_KERNELS:
        matching = [section for symbol, section in sections.items() if name in symbol]
        if len(matching) != 1:
            passed = False
            findings.append(f"FAIL {name}: expected one SASS section, found {len(matching)}")
            continue
        section = matching[0]
        opcodes = re.findall(r"/\*[^*]+\*/\s+([A-Z0-9_.]+)", section)
        local_ops = [opcode for opcode in opcodes if opcode.startswith(("LDL", "STL"))]
        branch_indirect = [opcode for opcode in opcodes if opcode.startswith("BRX")]
        global_loads = [opcode for opcode in opcodes if opcode.startswith("LDG")]
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
        if len(global_loads) > 2:
            passed = False
            findings.append(
                f"FAIL {name}: {len(global_loads)} global loads, expected only the two code arrays"
            )
        else:
            findings.append(
                f"PASS {name}: {len(global_loads)} static global-load instructions, no coefficient table"
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
