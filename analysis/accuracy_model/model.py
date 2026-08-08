"""Cell-integral and high-resolution analytical accuracy predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import ceil, erf, exp, expm1, isfinite, log1p, pi, sqrt
from typing import Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf as special_erf
from scipy.special import ndtr

from .formats import FORMAT_SPECS, FormatSpec


class Distribution(str, Enum):
    UNIFORM_01 = "uniform_0_1"
    NORMAL_01 = "normal_0_1"


@dataclass(frozen=True)
class IntervalMoments:
    mass: float
    first: float
    second: float
    absolute_first: float


@dataclass(frozen=True)
class ScalarMoments:
    format: str
    distribution: str
    storage_bits: int
    exponent_bits: int
    fraction_bits: int
    method: str
    finite_probability: float
    overflow_probability: float
    saturation_probability: float
    decoded_zero_probability: float
    mean_x: float
    mean_error: float
    second_x: float
    cross_x_error: float
    second_error: float
    mean_abs_x: float

    @property
    def scalar_bias(self) -> float:
        return self.mean_error

    @property
    def scalar_mse(self) -> float:
        return self.second_error

    @property
    def scalar_rmse(self) -> float:
        return sqrt(max(0.0, self.second_error))

    @property
    def mean_q(self) -> float:
        return self.mean_x + self.mean_error

    @property
    def second_q(self) -> float:
        return self.second_x + 2.0 * self.cross_x_error + self.second_error

    @property
    def cross_x_q(self) -> float:
        return self.second_x + self.cross_x_error

    def row(self) -> dict[str, float | int | str]:
        result = asdict(self)
        result.update(
            scalar_bias=self.scalar_bias,
            scalar_mse=self.scalar_mse,
            scalar_rmse=self.scalar_rmse,
            mean_q=self.mean_q,
            second_q=self.second_q,
            cross_x_q=self.cross_x_q,
        )
        return result


def _normal_density(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.exp(-0.5 * values * values) / sqrt(2.0 * pi)
    return np.where(np.isfinite(values), result, 0.0)


def _normal_mass(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty_like(left)
    near_zero = np.maximum(np.abs(left), np.abs(right)) <= 1.0
    result[near_zero] = 0.5 * (
        special_erf(right[near_zero] / sqrt(2.0))
        - special_erf(left[near_zero] / sqrt(2.0))
    )
    positive = (left >= 0.0) & ~near_zero
    negative = (right <= 0.0) & ~near_zero
    middle = ~(near_zero | positive | negative)
    result[positive] = ndtr(-left[positive]) - ndtr(-right[positive])
    result[negative] = ndtr(right[negative]) - ndtr(left[negative])
    result[middle] = ndtr(right[middle]) - ndtr(left[middle])
    return np.maximum(result, 0.0)


def interval_moments(
    distribution: Distribution,
    left: np.ndarray | float,
    right: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if distribution == Distribution.UNIFORM_01:
        lo = np.maximum(left_array, 0.0)
        hi = np.minimum(right_array, 1.0)
        width = np.maximum(hi - lo, 0.0)
        first = np.zeros_like(width)
        second = np.zeros_like(width)
        active = width > 0.0
        first[active] = (hi[active] * hi[active] - lo[active] * lo[active]) / 2.0
        second[active] = (
            hi[active] * hi[active] * hi[active]
            - lo[active] * lo[active] * lo[active]
        ) / 3.0
        return width, first, second

    mass = _normal_mass(left_array, right_array)
    density_left = _normal_density(left_array.copy())
    density_right = _normal_density(right_array.copy())
    first = density_left - density_right
    left_density = np.zeros_like(left_array)
    right_density = np.zeros_like(right_array)
    np.multiply(
        left_array, density_left, out=left_density, where=np.isfinite(left_array)
    )
    np.multiply(
        right_array,
        density_right,
        out=right_density,
        where=np.isfinite(right_array),
    )
    second = mass + left_density - right_density
    return mass, first, np.maximum(second, 0.0)


def _single_interval(
    distribution: Distribution, left: float, right: float
) -> IntervalMoments:
    mass, first, second = interval_moments(distribution, left, right)
    mass_value = float(mass)
    first_value = float(first)
    second_value = float(second)
    if right <= 0.0:
        absolute = -first_value
    elif left >= 0.0:
        absolute = first_value
    else:
        _, negative_first, _ = interval_moments(distribution, left, 0.0)
        _, positive_first, _ = interval_moments(distribution, 0.0, right)
        absolute = -float(negative_first) + float(positive_first)
    return IntervalMoments(mass_value, first_value, second_value, absolute)


def _stable_any_probability(single_probability: float, count: int) -> float:
    if count <= 0 or single_probability <= 0.0:
        return 0.0
    if single_probability >= 1.0:
        return 1.0
    return -expm1(count * log1p(-single_probability))


def _two_sided_tail_probability(
    distribution: Distribution, threshold: float
) -> float:
    positive = _single_interval(distribution, threshold, float("inf")).mass
    negative = _single_interval(distribution, -float("inf"), -threshold).mass
    return positive + negative


def _exact_scalar_moments(
    spec: FormatSpec, distribution: Distribution
) -> ScalarMoments:
    levels = spec.finite_levels()
    midpoints = levels[:-1] + (levels[1:] - levels[:-1]) / 2.0
    threshold = spec.overflow_threshold()
    if spec.overflow_mode == "finite":
        boundaries = np.concatenate(([-np.inf], midpoints, [np.inf]))
    else:
        boundaries = np.concatenate(([-threshold], midpoints, [threshold]))

    left = boundaries[:-1]
    right = boundaries[1:]
    mass, first, second = interval_moments(distribution, left, right)
    active = mass > 0.0
    q = levels[active]
    mass = mass[active]
    first = first[active]
    second = second[active]

    tail_probability = _two_sided_tail_probability(distribution, threshold)
    finite_mass = 1.0 if spec.overflow_mode == "finite" else 1.0 - tail_probability
    scale = 1.0 / finite_mass
    mean_x = float(np.sum(first)) * scale
    second_x = float(np.sum(second)) * scale

    mean_error = float(np.sum(q * mass - first)) * scale
    cross_x_error = float(np.sum(q * first - second)) * scale
    second_error_terms = q * q * mass - 2.0 * q * first + second
    second_error = max(0.0, float(np.sum(second_error_terms)) * scale)

    if distribution == Distribution.NORMAL_01:
        # The source, representable set, rounding cells, overflow, and
        # saturation policies are all sign-symmetric.
        mean_x = 0.0
        mean_error = 0.0
    elif abs(mean_error) < 1.0e-18:
        mean_error = 0.0

    absolute = _single_interval(
        distribution,
        -float("inf") if spec.overflow_mode == "finite" else -threshold,
        float("inf") if spec.overflow_mode == "finite" else threshold,
    )
    mean_abs_x = absolute.absolute_first / finite_mass

    zero_index = np.flatnonzero(levels == 0.0)
    zero_probability = 0.0
    if zero_index.size:
        index = int(zero_index[0])
        zero_mass, _, _ = interval_moments(
            distribution, boundaries[index], boundaries[index + 1]
        )
        zero_probability = float(zero_mass) / finite_mass

    overflow_probability = (
        tail_probability if spec.overflow_mode == "infinity" else 0.0
    )
    saturation_probability = 0.0
    if spec.overflow_mode == "finite":
        saturation_probability = tail_probability

    return ScalarMoments(
        spec.name,
        distribution.value,
        spec.total_bits,
        spec.exponent_bits,
        spec.fraction_bits,
        "exact_cells",
        finite_mass,
        overflow_probability,
        saturation_probability,
        zero_probability,
        mean_x,
        mean_error,
        second_x,
        cross_x_error,
        second_error,
        mean_abs_x,
    )


def _high_resolution_interior_mse(
    spec: FormatSpec,
    distribution: Distribution,
    lower: float,
    upper: float,
) -> float:
    total = 0.0
    emin = spec.minimum_normal_exponent
    emax = spec.maximum_normal_exponent

    minimum_normal = np.ldexp(1.0, emin)
    sub_left = max(lower, -minimum_normal)
    sub_right = min(upper, minimum_normal)
    if sub_left < sub_right:
        probability = _single_interval(distribution, sub_left, sub_right).mass
        step = np.ldexp(1.0, emin - spec.fraction_bits)
        total += probability * step * step / 12.0

    # U(0,1) has no mass above 1. A standard normal has no numerically
    # representable probability outside [-64,64]. Avoid constructing enormous
    # binade endpoints that cannot contribute to the integral.
    relevant_max_exponent = 0 if distribution == Distribution.UNIFORM_01 else 5
    for exponent in range(emin, min(emax, relevant_max_exponent) + 1):
        magnitude_left = np.ldexp(1.0, exponent)
        magnitude_right = np.ldexp(1.0, exponent + 1)
        step = np.ldexp(1.0, exponent - spec.fraction_bits)
        step_squared = step * step
        for a, b in (
            (magnitude_left, magnitude_right),
            (-magnitude_right, -magnitude_left),
        ):
            cell_left = max(lower, a)
            cell_right = min(upper, b)
            if cell_left >= cell_right:
                continue
            probability = _single_interval(distribution, cell_left, cell_right).mass
            if probability > 0.0 and isfinite(step_squared):
                total += probability * step_squared / 12.0
    return total


def _tail_error_moments(
    distribution: Distribution,
    threshold: float,
    maximum: float,
) -> tuple[float, float, float]:
    mean_error = 0.0
    cross_x_error = 0.0
    second_error = 0.0
    for left, right, q in (
        (threshold, float("inf"), maximum),
        (-float("inf"), -threshold, -maximum),
    ):
        interval = _single_interval(distribution, left, right)
        mean_error += q * interval.mass - interval.first
        cross_x_error += q * interval.first - interval.second
        second_error += (
            q * q * interval.mass - 2.0 * q * interval.first + interval.second
        )
    return mean_error, cross_x_error, max(0.0, second_error)


def _approximate_scalar_moments(
    spec: FormatSpec, distribution: Distribution
) -> ScalarMoments:
    threshold = spec.overflow_threshold()
    if distribution == Distribution.UNIFORM_01:
        lower, upper = 0.0, min(1.0, threshold)
    else:
        lower, upper = -threshold, threshold

    interior = _single_interval(distribution, lower, upper)
    interior_mse = _high_resolution_interior_mse(
        spec, distribution, lower, upper
    )
    zero_threshold = 0.5 * np.ldexp(
        1.0, spec.minimum_normal_exponent - spec.fraction_bits
    )
    zero_interval = _single_interval(
        distribution, -zero_threshold, zero_threshold
    )

    if spec.overflow_mode == "infinity":
        overflow_probability = _two_sided_tail_probability(distribution, threshold)
        finite_mass = 1.0 - overflow_probability
        scale = 1.0 / finite_mass
        mean_x = interior.first * scale
        second_x = interior.second * scale
        mean_abs_x = interior.absolute_first * scale
        return ScalarMoments(
            spec.name,
            distribution.value,
            spec.total_bits,
            spec.exponent_bits,
            spec.fraction_bits,
            "high_resolution",
            finite_mass,
            overflow_probability,
            0.0,
            zero_interval.mass / finite_mass,
            mean_x,
            0.0,
            second_x,
            -interior_mse * scale,
            interior_mse * scale,
            mean_abs_x,
        )

    full = _single_interval(distribution, -float("inf"), float("inf"))
    tail_mean, tail_cross, tail_second = _tail_error_moments(
        distribution, threshold, spec.maximum_finite()
    )
    saturation_probability = _two_sided_tail_probability(distribution, threshold)
    return ScalarMoments(
        spec.name,
        distribution.value,
        spec.total_bits,
        spec.exponent_bits,
        spec.fraction_bits,
        "high_resolution",
        1.0,
        0.0,
        saturation_probability,
        zero_interval.mass,
        full.first,
        tail_mean,
        full.second,
        -interior_mse + tail_cross,
        interior_mse + tail_second,
        full.absolute_first,
    )


def scalar_moments(
    spec: FormatSpec, distribution: Distribution
) -> ScalarMoments:
    if spec.family == "identity":
        source = _single_interval(distribution, -float("inf"), float("inf"))
        return ScalarMoments(
            spec.name,
            distribution.value,
            spec.total_bits,
            spec.exponent_bits,
            spec.fraction_bits,
            "identity",
            1.0,
            0.0,
            0.0,
            0.0,
            source.first,
            0.0,
            source.second,
            0.0,
            0.0,
            source.absolute_first,
        )
    if spec.exact_cell_model:
        return _exact_scalar_moments(spec, distribution)
    return _approximate_scalar_moments(spec, distribution)


def _folded_normal_mean(mean: float, sigma: float) -> float:
    if sigma == 0.0:
        return abs(mean)
    ratio = abs(mean) / sigma
    return (
        sigma * sqrt(2.0 / pi) * exp(-0.5 * ratio * ratio)
        + abs(mean) * erf(ratio / sqrt(2.0))
    )


def _folded_normal_quantile(mean: float, sigma: float, probability: float) -> float:
    if sigma == 0.0:
        return abs(mean)

    def cdf(value: float) -> float:
        return float(
            ndtr((value - mean) / sigma) - ndtr((-value - mean) / sigma)
        )

    upper = abs(mean) + 10.0 * sigma
    while cdf(upper) < probability:
        upper *= 2.0
    return brentq(lambda value: cdf(value) - probability, 0.0, upper)


def _product_error_moments(moments: ScalarMoments) -> tuple[float, float, float]:
    mx = moments.mean_x
    me = moments.mean_error
    x2 = moments.second_x
    xe = moments.cross_x_error
    e2 = moments.second_error
    mean = 2.0 * mx * me + me * me
    second = (
        2.0 * x2 * e2
        + e2 * e2
        + 2.0 * xe * xe
        + 4.0 * xe * e2
    )
    variance = max(0.0, second - mean * mean)
    return mean, second, variance


def dot_prediction(moments: ScalarMoments, n: int) -> dict[str, float | int | str]:
    term_bias, term_second, term_variance = _product_error_moments(moments)
    bias = n * term_bias
    variance = n * term_variance
    mse = variance + bias * bias
    rmse = sqrt(max(0.0, mse))
    sigma = sqrt(max(0.0, variance))

    reference_second = (
        n * moments.second_x * moments.second_x
        + n * (n - 1) * moments.mean_x**4
    )
    reference_rms = sqrt(max(0.0, reference_second))
    expected_normalizer = n * moments.mean_abs_x * moments.mean_abs_x
    relative_rms = rmse / reference_rms if reference_rms else 0.0
    normalized_rmse = rmse / expected_normalizer if expected_normalizer else 0.0

    if moments.distribution == Distribution.UNIFORM_01.value:
        typical_condition = 1.0
    else:
        reference_sigma = sqrt(n) * moments.second_x
        median_abs_reference = 0.6744897501960817 * reference_sigma
        typical_condition = (
            expected_normalizer / median_abs_reference
            if median_abs_reference
            else float("inf")
        )

    return {
        "kernel": "dot",
        "distribution": moments.distribution,
        "format": moments.format,
        "storage_bits": moments.storage_bits,
        "method": moments.method,
        "n": n,
        "m": 1,
        "finite_conditioned": moments.overflow_probability > 0.0,
        "term_bias": term_bias,
        "term_error_second_moment": term_second,
        "bias": bias,
        "mse": mse,
        "rmse": rmse,
        "approx_mean_absolute_error": _folded_normal_mean(bias, sigma),
        "approx_p95_absolute_error": _folded_normal_quantile(bias, sigma, 0.95),
        "reference_rms": reference_rms,
        "relative_rms": relative_rms,
        "expected_absolute_product_sum": expected_normalizer,
        "normalized_rmse_proxy": normalized_rmse,
        "typical_condition_proxy": typical_condition,
        "any_nonfinite_input_probability": _stable_any_probability(
            moments.overflow_probability, 2 * n
        ),
        "any_saturated_input_probability": _stable_any_probability(
            moments.saturation_probability, 2 * n
        ),
    }


def gemv_prediction(
    moments: ScalarMoments, n: int, m: int = 1024
) -> dict[str, float | int | str]:
    row = dot_prediction(moments, n)
    row_mse = float(row["mse"])
    reference_row_second = float(row["reference_rms"]) ** 2
    l2_rms = sqrt(m * row_mse)
    reference_l2_rms = sqrt(m * reference_row_second)
    return {
        "kernel": "gemv",
        "distribution": moments.distribution,
        "format": moments.format,
        "storage_bits": moments.storage_bits,
        "method": moments.method,
        "n": n,
        "m": m,
        "finite_conditioned": moments.overflow_probability > 0.0,
        "row_bias": row["bias"],
        "row_mse": row_mse,
        "row_rmse": row["rmse"],
        "approx_row_mean_absolute_error": row["approx_mean_absolute_error"],
        "approx_row_p95_absolute_error": row["approx_p95_absolute_error"],
        "normalized_row_rmse_proxy": row["normalized_rmse_proxy"],
        "rms_l2_error": l2_rms,
        "reference_l2_rms": reference_l2_rms,
        "relative_l2_rms": (
            l2_rms / reference_l2_rms if reference_l2_rms else 0.0
        ),
        "typical_row_condition_proxy": row["typical_condition_proxy"],
        # Marginal probability for one output row.  This is the quantity that
        # can be compared with the fraction of non-finite simulated outputs.
        "row_nonfinite_input_probability": _stable_any_probability(
            moments.overflow_probability, 2 * n
        ),
        "row_saturated_input_probability": _stable_any_probability(
            moments.saturation_probability, 2 * n
        ),
        # Probability that at least one value anywhere in the complete GEMV
        # input is exceptional.  This is an operation-level diagnostic.
        "any_nonfinite_input_probability": _stable_any_probability(
            moments.overflow_probability, n * (m + 1)
        ),
        "any_saturated_input_probability": _stable_any_probability(
            moments.saturation_probability, n * (m + 1)
        ),
    }


def gamma_bound(steps: int, unit_roundoff: float = 2.0**-53) -> float:
    product = steps * unit_roundoff
    if product >= 1.0:
        return float("inf")
    return product / (1.0 - product)


def _lane_combine_depth(lanes: int) -> int:
    return {1: 0, 2: 1, 4: 2}[lanes]


def dot_rounding_steps(
    n: int,
    lanes: int,
    threads: int = 256,
    block_cap: int = 2112,
) -> tuple[int, int]:
    packs = ceil(n / lanes)
    blocks = max(1, min(ceil(packs / threads), block_cap))
    first_thread_fmas = ceil(packs / (blocks * threads))
    first_path = first_thread_fmas + _lane_combine_depth(lanes) + 8
    final_thread_additions = ceil(blocks / threads)
    return first_path + final_thread_additions + 8, blocks


def gemv_rounding_steps(n: int, lanes: int, threads: int = 256) -> int:
    packs = ceil(n / lanes)
    thread_fmas = ceil(packs / threads)
    return thread_fmas + _lane_combine_depth(lanes) + 8


def arithmetic_bound_rows(
    dot_sizes: Iterable[int],
    gemv_sizes: Iterable[int],
    lanes_values: Iterable[int] = (1, 2, 4),
    threads: int = 256,
    dot_block_cap: int = 2112,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for lanes in lanes_values:
        for n in dot_sizes:
            steps, blocks = dot_rounding_steps(
                n, lanes, threads=threads, block_cap=dot_block_cap
            )
            rows.append(
                {
                    "kernel": "dot",
                    "n": n,
                    "lanes": lanes,
                    "threads": threads,
                    "blocks": blocks,
                    "rounding_steps": steps,
                    "fp64_normalized_error_bound": gamma_bound(steps),
                }
            )
        for n in gemv_sizes:
            steps = gemv_rounding_steps(n, lanes, threads=threads)
            rows.append(
                {
                    "kernel": "gemv",
                    "n": n,
                    "lanes": lanes,
                    "threads": threads,
                    "blocks": 1,
                    "rounding_steps": steps,
                    "fp64_normalized_error_bound": gamma_bound(steps),
                }
            )
    return rows


def build_scalar_rows(
    formats: Iterable[FormatSpec] = FORMAT_SPECS,
    distributions: Iterable[Distribution] = tuple(Distribution),
) -> tuple[list[ScalarMoments], list[dict[str, float | int | str]]]:
    moments = [
        scalar_moments(spec, distribution)
        for distribution in distributions
        for spec in formats
    ]
    return moments, [item.row() for item in moments]


def build_kernel_rows(
    moments: Iterable[ScalarMoments],
    dot_sizes: Iterable[int],
    gemv_sizes: Iterable[int],
    gemv_m: int = 1024,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    dot_values = tuple(dot_sizes)
    gemv_values = tuple(gemv_sizes)
    for item in moments:
        rows.extend(dot_prediction(item, n) for n in dot_values)
        rows.extend(gemv_prediction(item, n, gemv_m) for n in gemv_values)
    return rows
