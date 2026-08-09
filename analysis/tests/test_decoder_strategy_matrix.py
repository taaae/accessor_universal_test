import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "analysis" / "decoder_strategy_matrix.csv"

EXPECTED_FORMATS = {
    "e1m6": (8, 1, 6),
    "e2m5": (8, 2, 5),
    "e3m4": (8, 3, 4),
    "fp8_e4m3": (8, 4, 3),
    "fp8_e5m2": (8, 5, 2),
    "e1m14": (16, 1, 14),
    "e2m13": (16, 2, 13),
    "e3m12": (16, 3, 12),
    "fp16_e5m10": (16, 5, 10),
    "bf16_e8m7": (16, 8, 7),
    "e11m4": (16, 11, 4),
    "e1m30": (32, 1, 30),
    "e2m29": (32, 2, 29),
    "e3m28": (32, 3, 28),
    "fp32_e8m23": (32, 8, 23),
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
    for name, (bits, _exponent, fraction) in EXPECTED_FORMATS.items():
        row = rows[name]
        if bits == 64:
            assert row["exact_full_lut_bytes"] == "NA"
            assert row["subnormal_lut_bytes"] == "NA"
            continue

        output_bytes = 4 if fraction <= 20 else 8
        assert int(row["exact_full_lut_bytes"]) == (1 << bits) * output_bytes
        assert int(row["subnormal_lut_bytes"]) == (1 << fraction) * output_bytes


def test_inexact_fp32_intermediates_are_not_proposed():
    rows = read_rows()
    for name in ("e11m4", "e1m30", "e2m29", "e3m28", "e11m20", "fp64_e11m52"):
        row = rows[name]
        assert row["exact_fp32_intermediate"] == "no"
        assert "FP32-BITS" not in row["p0_strategies"]
        assert "FP32-BITS" not in row["p1_strategies"]

