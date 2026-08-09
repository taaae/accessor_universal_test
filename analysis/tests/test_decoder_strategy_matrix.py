import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "analysis" / "decoder_strategy_matrix.csv"

EXPECTED_FORMATS = {
    "e0m1": (2, 0, 1),
    "e1m0": (2, 1, 0),
    "e0m3": (4, 0, 3),
    "e1m2": (4, 1, 2),
    "fp4_e2m1": (4, 2, 1),
    "e3m0": (4, 3, 0),
    "e0m7": (8, 0, 7),
    "e1m6": (8, 1, 6),
    "e2m5": (8, 2, 5),
    "e3m4": (8, 3, 4),
    "fp8_e4m3": (8, 4, 3),
    "fp8_e5m2": (8, 5, 2),
    "e6m1": (8, 6, 1),
    "e7m0": (8, 7, 0),
    "e0m15": (16, 0, 15),
    "e1m14": (16, 1, 14),
    "e2m13": (16, 2, 13),
    "e3m12": (16, 3, 12),
    "e4m11": (16, 4, 11),
    "fp16_e5m10": (16, 5, 10),
    "e6m9": (16, 6, 9),
    "e7m8": (16, 7, 8),
    "bf16_e8m7": (16, 8, 7),
    "e9m6": (16, 9, 6),
    "e10m5": (16, 10, 5),
    "e11m4": (16, 11, 4),
    "e0m31": (32, 0, 31),
    "e1m30": (32, 1, 30),
    "e2m29": (32, 2, 29),
    "e3m28": (32, 3, 28),
    "e4m27": (32, 4, 27),
    "e5m26": (32, 5, 26),
    "e6m25": (32, 6, 25),
    "e7m24": (32, 7, 24),
    "fp32_e8m23": (32, 8, 23),
    "e9m22": (32, 9, 22),
    "e10m21": (32, 10, 21),
    "e11m20": (32, 11, 20),
    "fp64_e11m52": (64, 11, 52),
}


def read_rows():
    with MATRIX.open(newline="") as stream:
        return {row["format"]: row for row in csv.DictReader(stream)}


def test_matrix_covers_the_complete_format_inventory():
    rows = read_rows()
    assert set(rows) == set(EXPECTED_FORMATS)
    for name, expected in EXPECTED_FORMATS.items():
        row = rows[name]
        actual = tuple(
            int(row[field])
            for field in ("storage_bits", "exponent_bits", "fraction_bits")
        )
        assert actual == expected
        assert row["p0_strategies"]


def test_lut_size_metadata_is_consistent():
    rows = read_rows()
    for name, (bits, exponent, fraction) in EXPECTED_FORMATS.items():
        row = rows[name]
        if bits == 64:
            assert row["exact_full_lut_bytes"] == "NA"
            assert row["subnormal_lut_bytes"] == "NA"
            continue

        output_bytes = 4 if fraction <= 20 else 8
        assert int(row["exact_full_lut_bytes"]) == (1 << bits) * output_bytes
        if exponent == 0 or fraction == 0:
            assert int(row["subnormal_lut_bytes"]) == 0
        else:
            assert int(row["subnormal_lut_bytes"]) == (1 << fraction) * output_bytes


def test_inexact_fp32_intermediates_are_not_proposed():
    rows = read_rows()
    for name, row in rows.items():
        if row["exact_fp32_intermediate"] != "no":
            continue
        assert "FP32-BITS" not in row["p0_strategies"]
        assert "FP32-BITS" not in row["p1_strategies"]


def test_every_nonbaseline_format_has_an_implementation_note():
    note_root = ROOT / "docs" / "decoder_strategies"
    original = {
        "e1m6", "e2m5", "e3m4", "fp8_e4m3", "fp8_e5m2",
        "e1m14", "e2m13", "e3m12", "fp16_e5m10", "bf16_e8m7",
        "e11m4", "e1m30", "e2m29", "e3m28", "fp32_e8m23", "e11m20",
    }
    notes = {path.stem for path in note_root.glob("*.md")} - {"README"}
    assert original <= notes
    assert "expanded_formats" in notes


def test_every_nonbaseline_format_is_registered_for_cuda_smoke():
    header = (ROOT / "include" / "format_decoder_strategies.cuh").read_text()
    smoke = (ROOT / "src" / "all_format_strategy_smoke.cu").read_text()
    assert "template <typename Format> struct format_layout" in header
    for name in set(EXPECTED_FORMATS) - {"fp64_e11m52"}:
        assert f"storage::{name}" in smoke


def test_every_non_e2e3_format_is_dispatched_by_the_full_benchmark():
    benchmark = (ROOT / "src" / "all_format_strategy_bench.cu").read_text()
    # Experiment 015 predates the expanded candidates. Experiment 016 is their
    # compilation/correctness gate; a later timing experiment will extend this.
    expected = {
        "e1m6", "e3m4", "fp8_e4m3", "fp8_e5m2", "e1m14", "e2m13",
        "e3m12", "fp16_e5m10", "bf16_e8m7", "e11m4", "e1m30", "e2m29",
        "e3m28", "fp32_e8m23", "e11m20",
    } - {"e2m5", "e3m4"}
    for name in expected:
        assert f"DISPATCH({name})" in benchmark
