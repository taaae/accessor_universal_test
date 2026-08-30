#ifndef ACCESSOR_UNIVERSAL_TEST_DYADIC_NORMAL32_CORE_HPP_
#define ACCESSOR_UNIVERSAL_TEST_DYADIC_NORMAL32_CORE_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "lut_decomposition_core.hpp"

#if defined(__CUDACC__)
#define AUT_DN32_HD __host__ __device__
#define AUT_DN32_INLINE __forceinline__
#else
#define AUT_DN32_HD
#define AUT_DN32_INLINE inline
#endif

namespace aut::dyadic_normal32 {

inline constexpr std::uint32_t magnitude_mask = 0x7fffffffu;
inline constexpr std::uint32_t payload_mask = 0x3fffffffu;
inline constexpr std::uint64_t fp64_exponent_one = 0x3ff0000000000000ull;
inline constexpr std::uint64_t fp64_mantissa_mask = 0x000fffffffffffffull;
inline constexpr std::size_t segment_count = 32;
inline constexpr double source_sigma = 1.0;
inline constexpr double density_sigma = 1.7320508075688772935;

struct alignas(16) segment_coefficients {
  double start;
  double step;
};

static_assert(sizeof(segment_coefficients) == 16);
static_assert(alignof(segment_coefficients) == 16);

AUT_DN32_HD AUT_DN32_INLINE std::uint32_t
segment_from_rank(std::uint32_t rank) {
  const auto inverted = ~(rank << 1);
#if defined(__CUDA_ARCH__)
  return static_cast<std::uint32_t>(__clz(inverted));
#else
  return static_cast<std::uint32_t>(__builtin_clz(inverted));
#endif
}

AUT_DN32_HD AUT_DN32_INLINE std::uint32_t
payload_for_segment(std::uint32_t rank, std::uint32_t segment) {
  return rank & (payload_mask >> segment);
}

AUT_DN32_HD AUT_DN32_INLINE std::uint32_t
rank_from_segment_payload_unchecked(std::uint32_t segment,
                                    std::uint32_t payload) {
  const auto prefix = (0xffffffffu << (31u - segment)) & magnitude_mask;
  return prefix | (payload & (payload_mask >> segment));
}

inline std::uint32_t rank_from_segment_payload(std::uint32_t segment,
                                               std::uint32_t payload) {
  if (segment >= segment_count) {
    throw std::invalid_argument("DyadicNormal32 segment must be in [0,31]");
  }
  const auto mask = payload_mask >> segment;
  if ((payload & ~mask) != 0) {
    throw std::invalid_argument("DyadicNormal32 payload does not fit segment");
  }
  return rank_from_segment_payload_unchecked(segment, payload);
}

inline double half_normal_density_boundary(std::uint32_t boundary) {
  if (boundary > segment_count) {
    throw std::invalid_argument("DyadicNormal32 boundary must be in [0,32]");
  }
  if (boundary == 0) {
    return 0.0;
  }
  const auto target_tail = std::ldexp(1.0, -static_cast<int>(boundary));
  double low = 0.0;
  double high = 16.0;
  const auto scale = std::sqrt(6.0);
  for (int iteration = 0; iteration < 160; ++iteration) {
    const auto middle = 0.5 * (low + high);
    const auto tail = std::erfc(middle / scale);
    if (tail > target_tail) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return 0.5 * (low + high);
}

inline std::array<segment_coefficients, segment_count> make_coefficients() {
  std::array<segment_coefficients, segment_count> result{};
  std::array<double, segment_count + 1> boundaries{};
  for (std::uint32_t boundary = 0; boundary <= segment_count; ++boundary) {
    boundaries[boundary] = half_normal_density_boundary(boundary);
  }
  for (std::uint32_t segment = 0; segment < segment_count - 1; ++segment) {
    const auto levels = std::ldexp(1.0, 30 - static_cast<int>(segment));
    const auto step = (boundaries[segment + 1] - boundaries[segment]) / levels;
    result[segment] = {boundaries[segment] + 0.5 * step, step};
  }
  result[31] = {boundaries[32], 0.0};
  return result;
}

inline std::array<segment_coefficients, segment_count>
make_bitcast_coefficients() {
  const auto linear = make_coefficients();
  std::array<segment_coefficients, segment_count> result{};
  for (std::uint32_t segment = 0; segment < segment_count - 1; ++segment) {
    const auto levels = std::ldexp(1.0, 30 - static_cast<int>(segment));
    const auto span = linear[segment].step * levels;
    result[segment] = {linear[segment].start - span, span};
  }
  result[31] = linear[31];
  return result;
}

AUT_DN32_HD AUT_DN32_INLINE std::uint64_t
bitcast_coordinate_bits(std::uint32_t rank, std::uint32_t segment) {
  const auto fraction =
      (static_cast<std::uint64_t>(rank) << (22u + segment)) &
      fp64_mantissa_mask;
  return fp64_exponent_one | fraction;
}

inline double bitcast_coordinate(std::uint32_t rank, std::uint32_t segment) {
  const auto bits = bitcast_coordinate_bits(rank, segment);
  double result{};
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline double
decode(std::uint32_t code,
       const std::array<segment_coefficients, segment_count> &coefficients) {
  const auto rank = code & magnitude_mask;
  const auto segment = segment_from_rank(rank);
  const auto payload = payload_for_segment(rank, segment);
  const auto magnitude =
      std::fma(static_cast<double>(payload), coefficients[segment].step,
               coefficients[segment].start);
  return (code >> 31) != 0 ? -magnitude : magnitude;
}

inline double decode_bitcast(
    std::uint32_t code,
    const std::array<segment_coefficients, segment_count> &coefficients) {
  const auto rank = code & magnitude_mask;
  const auto segment = segment_from_rank(rank);
  const auto coordinate = bitcast_coordinate(rank, segment);
  const auto magnitude =
      std::fma(coordinate, coefficients[segment].step,
               coefficients[segment].start);
  return (code >> 31) != 0 ? -magnitude : magnitude;
}

inline double uniform_unique_segments() {
  return lut_decomposition::uniform_unique_indices(segment_count);
}

inline double expected_unique_segments_for_probabilities(
    const std::array<double, segment_count> &probabilities) {
  double result{};
  for (const auto probability : probabilities) {
    result += 1.0 - std::pow(1.0 - probability, lut_decomposition::warp_width);
  }
  return result;
}

inline std::array<double, segment_count>
genuine_standard_normal_segment_probabilities() {
  std::array<double, segment_count> result{};
  constexpr auto inverse_sqrt_two = 0.70710678118654752440;
  for (std::uint32_t segment = 0; segment < segment_count - 1; ++segment) {
    const auto lower = half_normal_density_boundary(segment);
    const auto upper = half_normal_density_boundary(segment + 1);
    result[segment] = std::erfc(lower * inverse_sqrt_two) -
                      std::erfc(upper * inverse_sqrt_two);
  }
  result[31] = std::erfc(half_normal_density_boundary(31) * inverse_sqrt_two);
  return result;
}

inline double genuine_standard_normal_dispersion() {
  const auto unique = expected_unique_segments_for_probabilities(
      genuine_standard_normal_segment_probabilities());
  return lut_decomposition::normalized_lookup_dispersion(segment_count, unique);
}

} // namespace aut::dyadic_normal32

#undef AUT_DN32_HD
#undef AUT_DN32_INLINE

#endif // ACCESSOR_UNIVERSAL_TEST_DYADIC_NORMAL32_CORE_HPP_
