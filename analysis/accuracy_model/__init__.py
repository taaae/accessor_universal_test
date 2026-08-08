"""Analytical quantization-error model for storage DOT and GEMV."""

from .formats import FORMAT_SPECS, FormatSpec
from .model import (
    Distribution,
    ScalarMoments,
    arithmetic_bound_rows,
    build_kernel_rows,
    build_scalar_rows,
    dot_prediction,
    gemv_prediction,
    scalar_moments,
)

__all__ = [
    "FORMAT_SPECS",
    "Distribution",
    "FormatSpec",
    "ScalarMoments",
    "arithmetic_bound_rows",
    "build_kernel_rows",
    "build_scalar_rows",
    "dot_prediction",
    "gemv_prediction",
    "scalar_moments",
]
