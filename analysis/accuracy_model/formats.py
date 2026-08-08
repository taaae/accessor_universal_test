"""Format descriptors matching include/storage_formats.hpp and CUDA wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ldexp

import numpy as np


@dataclass(frozen=True)
class FormatSpec:
    name: str
    total_bits: int
    exponent_bits: int
    fraction_bits: int
    overflow_mode: str
    family: str = "ieee"

    @property
    def exponent_bias(self) -> int:
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def maximum_exponent_field(self) -> int:
        if self.family == "e4m3fn":
            return 15
        if self.overflow_mode == "finite":
            return (1 << self.exponent_bits) - 1
        return (1 << self.exponent_bits) - 2

    @property
    def minimum_normal_exponent(self) -> int:
        return 1 - self.exponent_bias

    @property
    def maximum_normal_exponent(self) -> int:
        return self.maximum_exponent_field - self.exponent_bias

    @property
    def exact_cell_model(self) -> bool:
        return self.family == "identity" or self.fraction_bits <= 14

    def maximum_finite(self) -> float:
        if self.family == "identity":
            return float.fromhex("0x1.fffffffffffffp+1023")
        if self.family == "e4m3fn":
            return 448.0
        return ldexp(2.0 - ldexp(1.0, -self.fraction_bits),
                     self.maximum_normal_exponent)

    def overflow_threshold(self) -> float:
        """RNE boundary above which a larger-than-finite code is required."""
        if self.family == "identity":
            return float("inf")
        if self.family == "e4m3fn":
            return 464.0
        return ldexp(2.0 - ldexp(1.0, -self.fraction_bits - 1),
                     self.maximum_normal_exponent)

    def positive_finite_levels(self) -> np.ndarray:
        """Return every positive finite decoded value for exact-cell formats."""
        if self.family == "identity" or not self.exact_cell_model:
            raise ValueError(f"{self.name} is not enumerated")

        fraction_count = 1 << self.fraction_bits
        values: list[float] = []

        subnormal_step = ldexp(1.0,
                               self.minimum_normal_exponent - self.fraction_bits)
        values.extend(k * subnormal_step for k in range(1, fraction_count))

        if self.family == "e4m3fn":
            for exponent_field in range(1, 15):
                exponent = exponent_field - self.exponent_bias
                values.extend(
                    ldexp(1.0 + k / fraction_count, exponent)
                    for k in range(fraction_count)
                )
            exponent = 15 - self.exponent_bias
            values.extend(
                ldexp(1.0 + k / fraction_count, exponent)
                for k in range(7)
            )
        else:
            for exponent_field in range(1, self.maximum_exponent_field + 1):
                exponent = exponent_field - self.exponent_bias
                values.extend(
                    ldexp(1.0 + k / fraction_count, exponent)
                    for k in range(fraction_count)
                )

        return np.asarray(values, dtype=np.float64)

    def finite_levels(self) -> np.ndarray:
        positive = self.positive_finite_levels()
        return np.concatenate((-positive[::-1], np.asarray([0.0]), positive))


FORMAT_SPECS: tuple[FormatSpec, ...] = (
    FormatSpec("e1m6", 8, 1, 6, "finite"),
    FormatSpec("e2m5", 8, 2, 5, "infinity"),
    FormatSpec("e3m4", 8, 3, 4, "infinity"),
    FormatSpec("fp8_e4m3", 8, 4, 3, "finite", "e4m3fn"),
    FormatSpec("fp8_e5m2", 8, 5, 2, "finite"),
    FormatSpec("e1m14", 16, 1, 14, "finite"),
    FormatSpec("e2m13", 16, 2, 13, "infinity"),
    FormatSpec("e3m12", 16, 3, 12, "infinity"),
    FormatSpec("fp16_e5m10", 16, 5, 10, "infinity"),
    FormatSpec("bf16_e8m7", 16, 8, 7, "infinity"),
    FormatSpec("e11m4", 16, 11, 4, "infinity"),
    FormatSpec("e1m30", 32, 1, 30, "finite"),
    FormatSpec("e2m29", 32, 2, 29, "infinity"),
    FormatSpec("e3m28", 32, 3, 28, "infinity"),
    FormatSpec("fp32_e8m23", 32, 8, 23, "infinity"),
    FormatSpec("e11m20", 32, 11, 20, "infinity"),
    FormatSpec("fp64_e11m52", 64, 11, 52, "identity", "identity"),
)


def format_by_name(name: str) -> FormatSpec:
    for spec in FORMAT_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)
