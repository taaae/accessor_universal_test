from __future__ import annotations

import math

import pytest

from accuracy_model.formats import format_by_name
from accuracy_model.model import (
    Distribution,
    ScalarMoments,
    dot_prediction,
    gemv_prediction,
    scalar_moments,
)


def test_fp64_storage_is_exact() -> None:
    moments = scalar_moments(
        format_by_name("fp64_e11m52"), Distribution.NORMAL_01
    )
    prediction = dot_prediction(moments, 1 << 20)
    assert moments.scalar_mse == 0.0
    assert prediction["mse"] == 0.0


def test_e1m6_is_uniform_grid_on_uniform_0_1() -> None:
    moments = scalar_moments(format_by_name("e1m6"), Distribution.UNIFORM_01)
    expected = (1.0 / 32.0) ** 2 / 12.0
    assert moments.mean_error == pytest.approx(0.0, abs=1e-16)
    assert moments.scalar_mse == pytest.approx(expected, rel=1e-12)
    assert moments.decoded_zero_probability == pytest.approx(1.0 / 64.0)


def test_symmetric_normal_has_zero_scalar_bias() -> None:
    moments = scalar_moments(format_by_name("e3m4"), Distribution.NORMAL_01)
    assert moments.mean_x == pytest.approx(0.0, abs=1e-15)
    assert moments.mean_error == pytest.approx(0.0, abs=1e-15)


def test_dot_formula_matches_midpoint_summary() -> None:
    d = 1.0 / 3072.0
    moments = ScalarMoments(
        "midpoint4",
        Distribution.UNIFORM_01.value,
        4,
        0,
        4,
        "closed_form_test",
        1.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.0,
        1.0 / 3.0,
        -d,
        d,
        0.5,
    )
    n = 1024
    prediction = dot_prediction(moments, n)
    expected = n * (2.0 * d / 3.0 - d * d)
    assert prediction["mse"] == pytest.approx(expected, rel=1e-14)


def test_gemv_l2_scaling_and_relative_independence_of_m() -> None:
    moments = scalar_moments(format_by_name("e3m4"), Distribution.NORMAL_01)
    small = gemv_prediction(moments, n=1024, m=16)
    large = gemv_prediction(moments, n=1024, m=1024)
    assert large["rms_l2_error"] == pytest.approx(
        small["rms_l2_error"] * math.sqrt(1024 / 16)
    )
    assert large["relative_l2_rms"] == pytest.approx(small["relative_l2_rms"])


def test_e2m5_normal_reports_nonzero_overflow() -> None:
    moments = scalar_moments(format_by_name("e2m5"), Distribution.NORMAL_01)
    assert 1e-5 < moments.overflow_probability < 1e-3
    prediction = dot_prediction(moments, 1 << 16)
    assert prediction["any_nonfinite_input_probability"] > 0.99
