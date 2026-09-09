import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "tools"))
from validate_block_scale_reconstruction_results import FIELDS, FULL_N, validate, variants


def fixture(tmp_path):
    manifest = dict(mode="full", sizes=FULL_N, variants=14, samples=50, warmups=10,
                    grid=512, threads=256, payload="fp32",
                    dot_accumulation="fp64_except_raw_fp32", access="scalar_x1",
                    cache_protocol="repeat_input_no_explicit_flush",
                    experiment_id="034_block_scale_reconstruction",
                    dataset_id="block-scale-v1", seed_set_id="block-scale-v1",
                    rounding_reference_cases=238, commit="c" * 40, binary_sha256="b" * 64,
                    sass_sha256="d" * 64, audit_sha256="a" * 64,
                    environment_sha256="e" * 64, job_id="123", node="n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    samples = tmp_path / "samples.csv"
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for n in FULL_N:
            for round_index in range(50):
                for order, item in enumerate(sorted(variants())):
                    variant, block, scale, path, reconstruction, subtotal, accumulation, payload = item
                    result = 2 if variant not in {"raw_fp32", "fp32_to_fp64", "raw_fp64"} else 3
                    writer.writerow(dict(mode="full", stage="initial", N=n, B=block,
                                         scale_type=scale, path=path,
                                         reconstruction=reconstruction,
                                         subtotal_mode=subtotal,
                                         dot_accumulation=accumulation,
                                         variant_id=variant, payload_type=payload,
                                         grid=512, threads=256, round=round_index,
                                         order=order, time_ms=1, result=result, valid=1,
                                         job_id="123", node="n",
                                         experiment_id="034_block_scale_reconstruction",
                                         dataset_id="block-scale-v1",
                                         seed_set_id="block-scale-v1", source_commit="c" * 40,
                                         binary_sha="b" * 64, sass_sha="d" * 64, audit_sha="a" * 64))
    return samples, manifest_path


def load(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def rewrite(path, rows, fields=FIELDS):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_valid(tmp_path):
    samples, manifest = fixture(tmp_path)
    validate(samples, manifest, "full")


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "bad_n", "bad_time", "bad_digest", "bad_order",
    "bad_variant", "nondeterministic", "wrong_reconstruction", "wrong_experiment",
    "wrong_geometry", "truncated_header",
])
def test_rejects(tmp_path, mutation):
    samples, manifest = fixture(tmp_path)
    rows = load(samples)
    fields = FIELDS
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[-1].copy())
    elif mutation == "bad_n":
        rows[0]["N"] = "7"
    elif mutation == "bad_time":
        rows[0]["time_ms"] = "nan"
    elif mutation == "bad_digest":
        rows[0]["binary_sha"] = "wrong"
    elif mutation == "bad_order":
        rows[0]["order"] = "14"
    elif mutation == "bad_variant":
        rows[0]["variant_id"] = "unknown"
    elif mutation == "nondeterministic":
        rows[0]["result"] = "9"
    elif mutation == "wrong_reconstruction":
        rows[0]["reconstruction"] = "wrong"
    elif mutation == "wrong_experiment":
        rows[0]["experiment_id"] = "033_block_scale_dot"
    elif mutation == "wrong_geometry":
        rows[0]["grid"] = "511"
    else:
        fields = FIELDS[:-1]
        rows = [{key: value for key, value in row.items() if key in fields} for row in rows]
    rewrite(samples, rows, fields)
    with pytest.raises(ValueError):
        validate(samples, manifest, "full")


def test_incomplete_rerun_rejected(tmp_path):
    samples, manifest = fixture(tmp_path)
    rows = load(samples)
    rerun = [row.copy() for row in rows if row["N"] == str(FULL_N[0]) and row["variant_id"] != "raw_fp64"]
    for row in rerun:
        row["stage"] = "rerun"
    rewrite(samples, rows + rerun)
    with pytest.raises(ValueError):
        validate(samples, manifest, "full")


def test_drift_without_rerun_rejected(tmp_path):
    samples, manifest = fixture(tmp_path)
    rows = load(samples)
    for row in rows:
        if row["N"] == str(FULL_N[0]) and row["variant_id"] == "raw_fp32" and int(row["round"]) >= 25:
            row["time_ms"] = "2"
    rewrite(samples, rows)
    with pytest.raises(ValueError):
        validate(samples, manifest, "full")


def test_unqualified_complete_rerun_rejected(tmp_path):
    samples, manifest = fixture(tmp_path)
    rows = load(samples)
    rerun = [row.copy() for row in rows if row["N"] == str(FULL_N[0])]
    for row in rerun:
        row["stage"] = "rerun"
    rewrite(samples, rows + rerun)
    with pytest.raises(ValueError):
        validate(samples, manifest, "full")
