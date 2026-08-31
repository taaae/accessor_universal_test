#ifndef ACCESSOR_UNIVERSAL_TEST_COMPANDER32_CORE_HPP_
#define ACCESSOR_UNIVERSAL_TEST_COMPANDER32_CORE_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__CUDACC__)
#define AUT_COMP32_HD __host__ __device__
#define AUT_COMP32_INLINE __forceinline__
#else
#define AUT_COMP32_HD
#define AUT_COMP32_INLINE inline
#endif

namespace aut::compander32 {

inline constexpr double range = 8.0;
inline constexpr double q = 2147483647.0;
inline constexpr double alpha = 0.65;
inline constexpr std::int32_t minimum_code = -2147483647;
inline constexpr std::int32_t maximum_code = 2147483647;

inline constexpr double int_value_scale = range / q;
inline constexpr double quadratic_value_scale = range / (q * q);
inline constexpr double blended_value_scale = range / q;

inline constexpr double pwl2_boundary = 1.168;
inline constexpr double pwl4_boundary_0 = 0.552;
inline constexpr double pwl4_boundary_1 = 1.168;
inline constexpr double pwl4_boundary_2 = 1.992;

AUT_COMP32_HD AUT_COMP32_INLINE double clamp_source(double value) {
  return value < -range ? -range : value > range ? range : value;
}

AUT_COMP32_HD AUT_COMP32_INLINE std::int32_t
round_symmetric_code(double normalized) {
  const auto magnitude = ::nearbyint(::fmin(::fabs(normalized), 1.0) * q);
  const auto signed_value = ::copysign(magnitude, normalized);
  return static_cast<std::int32_t>(signed_value);
}

AUT_COMP32_HD AUT_COMP32_INLINE std::int32_t encode_integer(double value) {
  return round_symmetric_code(clamp_source(value) / range);
}

AUT_COMP32_HD AUT_COMP32_INLINE std::int32_t encode_quadratic(double value) {
  const auto normalized = clamp_source(value) / range;
  const auto coordinate = ::sqrt(::fabs(normalized));
  return round_symmetric_code(::copysign(coordinate, normalized));
}

AUT_COMP32_HD AUT_COMP32_INLINE double
invert_blended_quadratic_magnitude(double magnitude) {
  if (magnitude <= 0.0) {
    return 0.0;
  }
  const auto linear = 1.0 - alpha;
  return (::sqrt(linear * linear + 4.0 * alpha * magnitude) - linear) /
         (2.0 * alpha);
}

AUT_COMP32_HD AUT_COMP32_INLINE std::int32_t
encode_blended_quadratic(double value) {
  const auto normalized = clamp_source(value) / range;
  const auto coordinate =
      invert_blended_quadratic_magnitude(::fabs(normalized));
  return round_symmetric_code(::copysign(coordinate, normalized));
}

AUT_COMP32_HD AUT_COMP32_INLINE double
invert_blended_cubic_magnitude(double magnitude) {
  double low = 0.0;
  double high = 1.0;
#pragma unroll 6
  for (int iteration = 0; iteration < 42; ++iteration) {
    const auto middle = 0.5 * (low + high);
    const auto mapped = middle * ((1.0 - alpha) + alpha * middle * middle);
    if (mapped < magnitude) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return 0.5 * (low + high);
}

AUT_COMP32_HD AUT_COMP32_INLINE std::int32_t
encode_blended_cubic(double value) {
  const auto normalized = clamp_source(value) / range;
  const auto coordinate = invert_blended_cubic_magnitude(::fabs(normalized));
  return round_symmetric_code(::copysign(coordinate, normalized));
}

AUT_COMP32_HD AUT_COMP32_INLINE double decode_integer(std::int32_t code) {
  return static_cast<double>(code);
}

AUT_COMP32_HD AUT_COMP32_INLINE double decode_quadratic(std::int32_t code) {
  const auto value = static_cast<double>(code);
  return value * ::fabs(value);
}

AUT_COMP32_HD AUT_COMP32_INLINE double
decode_blended_quadratic(std::int32_t code) {
  const auto value = static_cast<double>(code);
  return value * ::fma(alpha / q, ::fabs(value), 1.0 - alpha);
}

AUT_COMP32_HD AUT_COMP32_INLINE double
decode_blended_cubic(std::int32_t code) {
  const auto value = static_cast<double>(code);
  const auto square = value * value;
  return value * ::fma(alpha / (q * q), square, 1.0 - alpha);
}

AUT_COMP32_HD AUT_COMP32_INLINE std::uint64_t double_bits(double value) {
#if defined(__CUDA_ARCH__)
  return static_cast<std::uint64_t>(__double_as_longlong(value));
#else
  std::uint64_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
#endif
}

AUT_COMP32_HD AUT_COMP32_INLINE double from_double_bits(std::uint64_t bits) {
#if defined(__CUDA_ARCH__)
  return __longlong_as_double(static_cast<long long>(bits));
#else
  double value{};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

AUT_COMP32_HD AUT_COMP32_INLINE double apply_sign(double magnitude,
                                                  std::uint32_t code) {
  return from_double_bits(double_bits(magnitude) |
                          (static_cast<std::uint64_t>(code >> 31) << 63));
}

AUT_COMP32_HD AUT_COMP32_INLINE std::uint32_t
encode_pwl2(double value) {
  constexpr std::uint32_t levels = std::uint32_t{1} << 30;
  const auto clipped = clamp_source(value);
  const auto magnitude = ::fabs(clipped);
  const auto segment = static_cast<std::uint32_t>(magnitude >= pwl2_boundary);
  const auto lower = segment == 0 ? 0.0 : pwl2_boundary;
  const auto upper = segment == 0 ? pwl2_boundary : range;
  auto payload = static_cast<std::uint64_t>(
      ::floor((magnitude - lower) * static_cast<double>(levels) /
              (upper - lower)));
  payload = payload >= levels ? levels - 1 : payload;
  const auto sign = static_cast<std::uint32_t>(clipped < 0.0) << 31;
  return sign | (segment << 30) | static_cast<std::uint32_t>(payload);
}

AUT_COMP32_HD AUT_COMP32_INLINE double decode_pwl2(std::uint32_t code) {
  constexpr double levels = static_cast<double>(std::uint64_t{1} << 30);
  const auto segment = (code >> 30) & 1u;
  const auto payload = code & 0x3fffffffu;
  const auto lower = segment == 0 ? 0.0 : pwl2_boundary;
  const auto span = segment == 0 ? pwl2_boundary : range - pwl2_boundary;
  const auto step = span / levels;
  const auto magnitude = ::fma(static_cast<double>(payload), step,
                               lower + 0.5 * step);
  return apply_sign(magnitude, code);
}

AUT_COMP32_HD AUT_COMP32_INLINE std::uint32_t
encode_pwl4(double value) {
  constexpr std::uint32_t levels = std::uint32_t{1} << 29;
  const auto clipped = clamp_source(value);
  const auto magnitude = ::fabs(clipped);
  const auto segment =
      static_cast<std::uint32_t>(magnitude >= pwl4_boundary_0) +
      static_cast<std::uint32_t>(magnitude >= pwl4_boundary_1) +
      static_cast<std::uint32_t>(magnitude >= pwl4_boundary_2);
  const auto lower = segment == 0   ? 0.0
                     : segment == 1 ? pwl4_boundary_0
                     : segment == 2 ? pwl4_boundary_1
                                    : pwl4_boundary_2;
  const auto upper = segment == 0   ? pwl4_boundary_0
                     : segment == 1 ? pwl4_boundary_1
                     : segment == 2 ? pwl4_boundary_2
                                    : range;
  auto payload = static_cast<std::uint64_t>(
      ::floor((magnitude - lower) * static_cast<double>(levels) /
              (upper - lower)));
  payload = payload >= levels ? levels - 1 : payload;
  const auto sign = static_cast<std::uint32_t>(clipped < 0.0) << 31;
  return sign | (segment << 29) | static_cast<std::uint32_t>(payload);
}

AUT_COMP32_HD AUT_COMP32_INLINE double decode_pwl4(std::uint32_t code) {
  constexpr double levels = static_cast<double>(std::uint64_t{1} << 29);
  const auto segment = (code >> 29) & 3u;
  const auto payload = code & 0x1fffffffu;
  const auto lower = segment == 0   ? 0.0
                     : segment == 1 ? pwl4_boundary_0
                     : segment == 2 ? pwl4_boundary_1
                                    : pwl4_boundary_2;
  const auto span = segment == 0   ? pwl4_boundary_0
                    : segment == 1 ? pwl4_boundary_1 - pwl4_boundary_0
                    : segment == 2 ? pwl4_boundary_2 - pwl4_boundary_1
                                   : range - pwl4_boundary_2;
  const auto step = span / levels;
  const auto magnitude = ::fma(static_cast<double>(payload), step,
                               lower + 0.5 * step);
  return apply_sign(magnitude, code);
}

} // namespace aut::compander32

#undef AUT_COMP32_HD
#undef AUT_COMP32_INLINE

#endif // ACCESSOR_UNIVERSAL_TEST_COMPANDER32_CORE_HPP_
