#ifndef ACCESSOR_UNIVERSAL_TEST_NORMAL32_FORMATS_HPP_
#define ACCESSOR_UNIVERSAL_TEST_NORMAL32_FORMATS_HPP_

#include "storage_formats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__)
#define AUT_NORMAL32_HD __host__ __device__
#define AUT_NORMAL32_INLINE __forceinline__
#else
#define AUT_NORMAL32_HD
#define AUT_NORMAL32_INLINE inline
#endif

namespace aut::normal32 {

inline constexpr int code_bits = 32;
inline constexpr double normal_sigma =
    static_cast<double>(std::numeric_limits<float>::max()) / 4.0;
inline constexpr double normal_cutoff_sigma = 4.0;

inline constexpr double qn_a = 1.59577102;
inline constexpr double qn_b = 7.38651020;
inline constexpr double qn_max_sigma = qn_a + qn_b;
inline constexpr double qn_integer_scale = 2147483648.0;

inline constexpr int pwl_segment_bits = 16;
inline constexpr int pwl_local_bits = 16;
inline constexpr std::size_t pwl_segment_count = std::size_t{1}
                                                 << pwl_segment_bits;

inline constexpr int pwq_segment_bits = 8;
inline constexpr int pwq_local_bits = 24;
inline constexpr std::size_t pwq_segment_count = std::size_t{1}
                                                 << pwq_segment_bits;

struct pwl_coeff {
  double a{};
  double b{};
};

struct pwq_coeff {
  double a{};
  double b{};
  double c{};
};

struct codebook_tables {
  std::vector<pwl_coeff> pwl;
  std::vector<pwq_coeff> pwq;
};

inline double standard_normal_cdf(double value) {
  return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

inline double inverse_standard_normal_cdf(double probability) {
  if (!(probability > 0.0 && probability < 1.0)) {
    throw std::invalid_argument("normal quantile requires 0 < p < 1");
  }
  double lower = -12.0;
  double upper = 12.0;
  for (int iteration = 0; iteration < 64; ++iteration) {
    const auto middle = 0.5 * (lower + upper);
    if (standard_normal_cdf(middle) < probability) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return 0.5 * (lower + upper);
}

inline double compander_value(double probability, double sigma) {
  return sigma * std::sqrt(3.0) * inverse_standard_normal_cdf(probability);
}

inline std::vector<double> compander_boundaries(std::size_t segments,
                                                double sigma) {
  if (segments == 0 || !(sigma > 0.0)) {
    throw std::invalid_argument("invalid Normal32 table parameters");
  }
  // The ideal 32-bit codebook uses the centers of 2^32 equal compander
  // intervals. Keeping the outer boundaries at the first and last centers
  // makes the two tail segments finite.
  constexpr double code_count = 4294967296.0;
  constexpr double outer_probability = 0.5 / code_count;
  constexpr double retained_probability = 1.0 - 2.0 * outer_probability;

  std::vector<double> result(segments + 1);
  for (std::size_t index = 0; index <= segments; ++index) {
    const auto coordinate =
        static_cast<double>(index) / static_cast<double>(segments);
    const auto probability =
        outer_probability + retained_probability * coordinate;
    result[index] = compander_value(probability, sigma);
  }
  if ((segments & 1u) == 0) {
    result[segments / 2] = 0.0;
  }
  return result;
}

inline codebook_tables build_codebook_tables(double sigma = normal_sigma) {
  codebook_tables result;

  const auto pwl_edges = compander_boundaries(pwl_segment_count, sigma);
  result.pwl.resize(pwl_segment_count);
  constexpr double pwl_positions =
      static_cast<double>(std::uint64_t{1} << pwl_local_bits);
  for (std::size_t segment = 0; segment < pwl_segment_count; ++segment) {
    const auto left = pwl_edges[segment];
    const auto right = pwl_edges[segment + 1];
    result.pwl[segment] = {0.5 * (left + right),
                           (right - left) / pwl_positions};
  }

  const auto pwq_edges = compander_boundaries(pwq_segment_count, sigma);
  result.pwq.resize(pwq_segment_count);
  constexpr double half_positions =
      static_cast<double>(std::uint64_t{1} << (pwq_local_bits - 1));
  constexpr double full_positions = 2.0 * half_positions;
  for (std::size_t segment = 0; segment < pwq_segment_count; ++segment) {
    const auto left = pwq_edges[segment];
    const auto right = pwq_edges[segment + 1];
    const auto center_probability = (static_cast<double>(segment) + 0.5) /
                                    static_cast<double>(pwq_segment_count);
    const auto middle = compander_value(center_probability, sigma);
    const auto linear = (right - left) / full_positions;
    const auto quadratic =
        (left + right - 2.0 * middle) / (2.0 * half_positions * half_positions);
    result.pwq[segment] = {middle, linear, quadratic};
  }
  return result;
}

inline std::uint32_t encode_compander(double value,
                                      double sigma = normal_sigma) {
  if (!std::isfinite(value) || !(sigma > 0.0)) {
    throw std::invalid_argument("Normal32 encoder requires a finite value");
  }
  const auto normalized = value / (sigma * std::sqrt(3.0));
  const auto probability = standard_normal_cdf(normalized);
  constexpr double code_count = 4294967296.0;
  const auto scaled = std::clamp(probability * code_count, 0.0,
                                 std::nextafter(code_count, 0.0));
  return static_cast<std::uint32_t>(static_cast<std::uint64_t>(scaled));
}

inline std::uint32_t encode_pwl(double value, double sigma = normal_sigma) {
  const auto rank = encode_compander(value, sigma);
  constexpr auto local_mask = (std::uint32_t{1} << pwl_local_bits) - 1u;
  constexpr auto local_half = std::uint32_t{1} << (pwl_local_bits - 1);
  const auto segment = rank >> pwl_local_bits;
  const auto local_rank = rank & local_mask;
  const auto local_bits = (local_rank - local_half) & local_mask;
  return (segment << pwl_local_bits) | local_bits;
}

inline std::uint32_t encode_pwq(double value, double sigma = normal_sigma) {
  const auto rank = encode_compander(value, sigma);
  constexpr auto local_mask = (std::uint32_t{1} << pwq_local_bits) - 1u;
  constexpr auto local_half = std::uint32_t{1} << (pwq_local_bits - 1);
  const auto segment = rank >> pwq_local_bits;
  const auto local_rank = rank & local_mask;
  const auto local_bits = (local_rank - local_half) & local_mask;
  return (segment << pwq_local_bits) | local_bits;
}

inline std::int32_t signed_local_24(std::uint32_t code) {
  const auto local = code & 0x00ffffffu;
  return (local & 0x00800000u) != 0
             ? static_cast<std::int32_t>(local | 0xff000000u)
             : static_cast<std::int32_t>(local);
}

inline double decode_pwl(std::uint32_t code,
                         const std::vector<pwl_coeff> &table) {
  if (table.size() != pwl_segment_count) {
    throw std::invalid_argument("wrong PWLNormal32 table size");
  }
  const auto segment = code >> pwl_local_bits;
  const auto local = static_cast<std::int16_t>(code & 0xffffu);
  const auto coefficient = table[segment];
  return std::fma(static_cast<double>(local), coefficient.b, coefficient.a);
}

inline double decode_pwq(std::uint32_t code,
                         const std::vector<pwq_coeff> &table) {
  if (table.size() != pwq_segment_count) {
    throw std::invalid_argument("wrong PWQNormal32 table size");
  }
  const auto segment = code >> pwq_local_bits;
  const auto local = static_cast<double>(signed_local_24(code));
  const auto coefficient = table[segment];
  return std::fma(local, std::fma(local, coefficient.c, coefficient.b),
                  coefficient.a);
}

inline std::int32_t encode_qn32(double value, double sigma = normal_sigma) {
  if (!std::isfinite(value) || !(sigma > 0.0)) {
    throw std::invalid_argument("QN32 encoder requires a finite value");
  }
  const auto z = std::abs(value) / sigma;
  const auto t =
      (std::sqrt(qn_a * qn_a + 4.0 * qn_b * z) - qn_a) / (2.0 * qn_b);
  const auto magnitude = std::nearbyint(std::min(t, 1.0) * qn_integer_scale);
  const auto signed_value = std::copysign(magnitude, value);
  const auto clipped =
      std::clamp(signed_value,
                 static_cast<double>(std::numeric_limits<std::int32_t>::min()),
                 static_cast<double>(std::numeric_limits<std::int32_t>::max()));
  return static_cast<std::int32_t>(clipped);
}

inline double decode_qn32(std::int32_t code, double sigma = normal_sigma) {
  const auto value = static_cast<double>(code);
  const auto c1 = sigma * qn_a / qn_integer_scale;
  const auto c2 = sigma * qn_b / (qn_integer_scale * qn_integer_scale);
  return value * std::fma(std::abs(value), c2, c1);
}

template <int ExponentBits, int FractionBits>
AUT_NORMAL32_HD AUT_NORMAL32_INLINE std::uint64_t encode_ieee(double value) {
  static_assert(ExponentBits > 0 && ExponentBits <= 11);
  static_assert(FractionBits > 0 && FractionBits < 52);
  constexpr int total_bits = 1 + ExponentBits + FractionBits;
  constexpr auto exponent_mask = (std::uint64_t{1} << ExponentBits) - 1u;
  constexpr auto fraction_mask = (std::uint64_t{1} << FractionBits) - 1u;
  constexpr int exponent_bias = (1 << (ExponentBits - 1)) - 1;
  constexpr auto maximum_normal_exponent = exponent_mask - 1u;

  const auto source = storage::detail::double_bits(value);
  const auto sign = source >> 63;
  const auto source_exponent = (source >> 52) & 0x7ffu;
  const auto source_fraction = source & ((std::uint64_t{1} << 52) - 1u);
  const auto sign_field = sign << (total_bits - 1);

  if (source_exponent == 0x7ffu) {
    if (source_fraction == 0) {
      return sign_field | (exponent_mask << FractionBits);
    }
    auto payload = source_fraction >> (52 - FractionBits);
    if (payload == 0) {
      payload = 1;
    }
    return sign_field | (exponent_mask << FractionBits) |
           (payload & fraction_mask);
  }
  if (source_exponent == 0 && source_fraction == 0) {
    return sign_field;
  }
  if (source_exponent == 0) {
    return sign_field;
  }

  const auto unbiased = static_cast<int>(source_exponent) - 1023;
  auto target_exponent = unbiased + exponent_bias;
  if (target_exponent <= 0) {
    const auto significand = (std::uint64_t{1} << 52) | source_fraction;
    const auto shift = 53 - unbiased - exponent_bias - FractionBits;
    const auto subnormal = shift >= 64
                               ? std::uint64_t{0}
                               : storage::detail::round_right_shift_even(
                                     significand, static_cast<unsigned>(shift));
    if (subnormal >= (std::uint64_t{1} << FractionBits)) {
      return sign_field | (std::uint64_t{1} << FractionBits);
    }
    return sign_field | subnormal;
  }

  auto target_fraction = storage::detail::round_right_shift_even(
      source_fraction, 52 - FractionBits);
  if (target_fraction == (std::uint64_t{1} << FractionBits)) {
    target_fraction = 0;
    ++target_exponent;
  }
  if (target_exponent > static_cast<int>(maximum_normal_exponent)) {
    return sign_field | (exponent_mask << FractionBits);
  }
  return sign_field |
         (static_cast<std::uint64_t>(target_exponent) << FractionBits) |
         target_fraction;
}

template <int ExponentBits, int FractionBits>
AUT_NORMAL32_HD AUT_NORMAL32_INLINE double decode_ieee(std::uint64_t raw) {
  static_assert(ExponentBits > 0 && ExponentBits <= 11);
  static_assert(FractionBits > 0 && FractionBits < 52);
  constexpr int total_bits = 1 + ExponentBits + FractionBits;
  constexpr auto exponent_mask = (std::uint64_t{1} << ExponentBits) - 1u;
  constexpr auto fraction_mask = (std::uint64_t{1} << FractionBits) - 1u;
  constexpr int exponent_bias = (1 << (ExponentBits - 1)) - 1;

  const auto sign = raw >> (total_bits - 1);
  const auto exponent = (raw >> FractionBits) & exponent_mask;
  const auto fraction = raw & fraction_mask;
  const auto sign64 = sign << 63;
  if (exponent == 0) {
    if (fraction == 0) {
      return storage::detail::double_from_bits(sign64);
    }
    const auto leading = storage::detail::highest_set_bit(fraction);
    const auto unbiased = leading + 1 - exponent_bias - FractionBits;
    const auto exponent64 = static_cast<std::uint64_t>(unbiased + 1023);
    const auto fraction64 = (fraction - (std::uint64_t{1} << leading))
                            << (52 - leading);
    return storage::detail::double_from_bits(sign64 | (exponent64 << 52) |
                                             fraction64);
  }
  if (exponent == exponent_mask) {
    return storage::detail::double_from_bits(sign64 |
                                             (std::uint64_t{0x7ff} << 52) |
                                             (fraction << (52 - FractionBits)));
  }
  const auto unbiased = static_cast<int>(exponent) - exponent_bias;
  const auto exponent64 = static_cast<std::uint64_t>(unbiased + 1023);
  return storage::detail::double_from_bits(sign64 | (exponent64 << 52) |
                                           (fraction << (52 - FractionBits)));
}

template <int Bits>
AUT_NORMAL32_HD AUT_NORMAL32_INLINE constexpr std::uint64_t raw_mask() {
  static_assert(Bits > 0 && Bits < 64);
  return (std::uint64_t{1} << Bits) - 1u;
}

template <int Bits>
AUT_NORMAL32_HD AUT_NORMAL32_INLINE constexpr std::size_t
dense_word_count(std::size_t values) {
  static_assert(Bits >= 32 && Bits < 64);
  return (values * static_cast<std::size_t>(Bits) + 31u) / 32u + 2u;
}

template <int Bits>
AUT_NORMAL32_HD AUT_NORMAL32_INLINE constexpr std::size_t
dense_data_bytes(std::size_t values) {
  return (values * static_cast<std::size_t>(Bits) + 7u) / 8u;
}

template <int Bits>
AUT_NORMAL32_HD AUT_NORMAL32_INLINE std::uint64_t
load_dense(const std::uint32_t *words, std::size_t index) {
  static_assert(Bits >= 32 && Bits < 64);
  const auto bit = index * static_cast<std::size_t>(Bits);
  const auto word = bit >> 5;
  const auto shift = static_cast<unsigned>(bit & 31u);
  const auto low = static_cast<std::uint64_t>(words[word]) |
                   (static_cast<std::uint64_t>(words[word + 1]) << 32);
  auto result = low >> shift;
  if (shift + Bits > 64) {
    result |= static_cast<std::uint64_t>(words[word + 2]) << (64 - shift);
  }
  return result & raw_mask<Bits>();
}

} // namespace aut::normal32

#undef AUT_NORMAL32_HD
#undef AUT_NORMAL32_INLINE

#endif // ACCESSOR_UNIVERSAL_TEST_NORMAL32_FORMATS_HPP_
