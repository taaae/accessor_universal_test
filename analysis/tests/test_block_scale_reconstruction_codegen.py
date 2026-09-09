import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "tools"))
from check_block_scale_reconstruction_codegen import row

RESOURCE = dict(registers=20, shared_bytes=2048, spill_loads=0, spill_stores=0)


def reconstruct32_sequence():
    return [(0, "LDG.E", "R1, desc"), (1, "LDG.E", "R2, desc"),
            (2, "LDG.E", "R3, desc"), (3, "LDG.E", "R4, desc"),
            (4, "FMUL", "R5, R1, R3"), (5, "FMUL", "R6, R2, R4"),
            (6, "F2F.F64.F32", "R7, R5"), (7, "F2F.F64.F32", "R8, R6"),
            (8, "DFMA", "R9, R7, R8, R9")]


def reconstruct64_sequence():
    return [(0, "LDG.E", "R1, desc"), (1, "LDG.E", "R2, desc"),
            (2, "LDG.E", "R3, desc"), (3, "LDG.E", "R4, desc"),
            (4, "F2F.F64.F32", "R5, R1"), (5, "F2F.F64.F32", "R6, R2"),
            (6, "F2F.F64.F32", "R7, R3"), (7, "F2F.F64.F32", "R8, R4"),
            (8, "DMUL", "R9, R5, R7"), (9, "DMUL", "R10, R6, R8"),
            (10, "DFMA", "R11, R9, R10, R11")]


def deferred_sequence():
    return [(0, "LDG.E", "R1, desc"), (1, "LDG.E", "R2, desc"),
            (2, "F2F.F64.F32", "R5, R1"), (3, "F2F.F64.F32", "R6, R2"),
            (4, "DFMA", "R7, R5, R6, R7"), (5, "LDG.E", "R3, desc"),
            (6, "LDG.E", "R4, desc"), (7, "F2F.F64.F32", "R8, R3"),
            (8, "F2F.F64.F32", "R9, R4"), (9, "DMUL", "R10, R8, R9"),
            (10, "DFMA", "R11, R10, R7, R11")]


def test_reconstruct32_positive():
    assert row("x", "reconstruct32", 16, "fp32", reconstruct32_sequence(), RESOURCE)["status"] == "pass"


def test_reconstruct32_rejects_missing_rounding():
    sequence = reconstruct64_sequence()
    assert row("x", "reconstruct32", 16, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_reconstruct32_rejects_wrong_fmul_pairing():
    sequence = reconstruct32_sequence()
    sequence[4] = (4, "FMUL", "R5, R1, R2")
    assert row("x", "reconstruct32", 16, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_reconstruct32_rejects_unrelated_widens():
    sequence = reconstruct32_sequence()
    sequence[6] = (6, "F2F.F64.F32", "R7, R20")
    sequence[7] = (7, "F2F.F64.F32", "R8, R21")
    assert row("x", "reconstruct32", 16, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_reconstruct32_rejects_intervening_fmul_write():
    sequence = reconstruct32_sequence()
    sequence.insert(5, (5, "FADD", "R5, R5, R20"))
    assert row("x", "reconstruct32", 16, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_reconstruct64_positive():
    assert row("x", "reconstruct64", 128, "fp32", reconstruct64_sequence(), RESOURCE)["status"] == "pass"


def test_reconstruct64_rejects_unrelated_widens():
    sequence = reconstruct64_sequence()
    sequence[4] = (4, "F2F.F64.F32", "R5, R20")
    assert row("x", "reconstruct64", 128, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_reconstruct64_rejects_intervening_widen_write():
    sequence = reconstruct64_sequence()
    sequence.insert(8, (8, "DADD", "R5, R5, R20"))
    assert row("x", "reconstruct64", 128, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_deferred_local_positive():
    assert row("x", "deferred_local", 128, "fp32", deferred_sequence(), RESOURCE)["status"] == "pass"


def test_deferred_local_rejects_shuffle_path():
    sequence = deferred_sequence() + [(11, "SHFL.DOWN", "R12, R11, 0x10, 0x1f")]
    assert row("x", "deferred_local", 128, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_deferred_local_rejects_ignored_subtotal():
    sequence = deferred_sequence()
    sequence[-1] = (10, "DFMA", "R11, R10, R20, R11")
    assert row("x", "deferred_local", 128, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_deferred_local_rejects_unrelated_widens():
    sequence = deferred_sequence()
    sequence[2] = (2, "F2F.F64.F32", "R5, R20")
    assert row("x", "deferred_local", 128, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_deferred_local_rejects_intervening_widen_write():
    sequence = deferred_sequence()
    sequence.insert(4, (4, "DADD", "R5, R5, R20"))
    assert row("x", "deferred_local", 128, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_rejects_ftz_opcode():
    sequence = reconstruct32_sequence()
    sequence[4] = (4, "FMUL.FTZ", "R5, R1, R3")
    assert row("x", "reconstruct32", 16, "fp32", sequence, RESOURCE)["status"] == "fail"


def test_rejects_spill():
    resource = dict(RESOURCE, spill_loads=8)
    assert row("x", "raw_fp64", 0, "none", [(0, "LDG.E.64", "R1, desc"), (1, "LDG.E.64", "R2, desc"), (2, "DFMA", "R3, R1, R2, R3")], resource)["status"] == "fail"


def test_baseline_rejects_unrelated_operands():
    sequence = [(0, "LDG.E.64", "R1, desc"), (1, "LDG.E.64", "R2, desc"),
                (2, "DFMA", "R3, R8, R9, R3")]
    assert row("x", "raw_fp64", 0, "none", sequence, RESOURCE)["status"] == "fail"
