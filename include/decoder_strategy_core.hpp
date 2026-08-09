#ifndef ACCESSOR_UNIVERSAL_TEST_DECODER_STRATEGY_CORE_HPP_
#define ACCESSOR_UNIVERSAL_TEST_DECODER_STRATEGY_CORE_HPP_

#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__CUDACC__)
#define AUT_DECODER_HD __host__ __device__
#define AUT_DECODER_INLINE __forceinline__
#else
#define AUT_DECODER_HD
#define AUT_DECODER_INLINE inline
#endif

namespace aut::decoder {

enum class special_policy { ieee, finite_all, e4m3fn, fp64_prefix };

template <int TotalBits, int ExponentBits, special_policy Policy>
struct binary_layout {
  static constexpr int total_bits = TotalBits;
  static constexpr int exponent_bits = ExponentBits;
  static constexpr int fraction_bits = TotalBits - ExponentBits - 1;
  static constexpr int exponent_bias = (1 << (ExponentBits - 1)) - 1;
  static constexpr special_policy policy = Policy;
};

using e1m6_layout = binary_layout<8, 1, special_policy::finite_all>;
using e2m5_layout = binary_layout<8, 2, special_policy::ieee>;
using e3m4_layout = binary_layout<8, 3, special_policy::ieee>;
using fp8_e4m3_layout = binary_layout<8, 4, special_policy::e4m3fn>;
using fp8_e5m2_layout = binary_layout<8, 5, special_policy::ieee>;
using e1m14_layout = binary_layout<16, 1, special_policy::finite_all>;
using e2m13_layout = binary_layout<16, 2, special_policy::ieee>;
using e3m12_layout = binary_layout<16, 3, special_policy::ieee>;
using fp16_e5m10_layout = binary_layout<16, 5, special_policy::ieee>;
using bf16_e8m7_layout = binary_layout<16, 8, special_policy::ieee>;
using e11m4_layout = binary_layout<16, 11, special_policy::fp64_prefix>;
using e1m30_layout = binary_layout<32, 1, special_policy::finite_all>;
using e2m29_layout = binary_layout<32, 2, special_policy::ieee>;
using e3m28_layout = binary_layout<32, 3, special_policy::ieee>;
using fp32_e8m23_layout = binary_layout<32, 8, special_policy::ieee>;
using e11m20_layout = binary_layout<32, 11, special_policy::fp64_prefix>;

struct fp64_words {
  std::uint32_t high{};
  std::uint32_t low{};
};

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE constexpr std::uint32_t raw_mask() {
  if constexpr (Layout::total_bits == 32) {
    return 0xffffffffu;
  } else {
    return (std::uint32_t{1} << Layout::total_bits) - 1;
  }
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE constexpr std::uint32_t exponent_mask() {
  return (std::uint32_t{1} << Layout::exponent_bits) - 1;
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE constexpr std::uint32_t fraction_mask() {
  return (std::uint32_t{1} << Layout::fraction_bits) - 1;
}

AUT_DECODER_HD AUT_DECODER_INLINE int highest_set_bit(std::uint32_t value) {
#if defined(__CUDA_ARCH__)
  return 31 - __clz(value);
#elif defined(__GNUC__) || defined(__clang__)
  return 31 - __builtin_clz(value);
#else
  int result = -1;
  while (value != 0) {
    value >>= 1;
    ++result;
  }
  return result;
#endif
}

AUT_DECODER_HD AUT_DECODER_INLINE fp64_words
place_fraction(std::uint32_t high, std::uint32_t fraction,
               int source_fraction_bits) {
  if (source_fraction_bits <= 20) {
    high |= fraction << (20 - source_fraction_bits);
    return {high, 0};
  }
  high |= fraction >> (source_fraction_bits - 20);
  return {high, fraction << (52 - source_fraction_bits)};
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE fp64_words
normal_magnitude_words(std::uint32_t exponent, std::uint32_t fraction) {
  const auto exponent64 = static_cast<std::uint32_t>(
      static_cast<int>(exponent) - Layout::exponent_bias + 1023);
  return place_fraction(exponent64 << 20, fraction, Layout::fraction_bits);
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE fp64_words
special_magnitude_words(std::uint32_t fraction) {
  return place_fraction(0x7ff00000u, fraction, Layout::fraction_bits);
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE fp64_words
subnormal_magnitude_words(std::uint32_t fraction) {
  const auto safe_fraction = fraction | 1u;
  const auto leading = highest_set_bit(safe_fraction);
  const auto unbiased =
      leading + 1 - Layout::exponent_bias - Layout::fraction_bits;
  const auto exponent64 = static_cast<std::uint32_t>(unbiased + 1023);
  const auto remainder = fraction - (std::uint32_t{1} << leading);
  auto result = place_fraction(exponent64 << 20, remainder, leading);
  const auto nonzero_mask =
      0u - static_cast<std::uint32_t>(fraction != 0);
  result.high &= nonzero_mask;
  result.low &= nonzero_mask;
  return result;
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE bool is_special(std::uint32_t exponent,
                                                  std::uint32_t fraction) {
  if constexpr (Layout::policy == special_policy::ieee) {
    return exponent == exponent_mask<Layout>();
  } else if constexpr (Layout::policy == special_policy::e4m3fn) {
    return exponent == exponent_mask<Layout>() &&
           fraction == fraction_mask<Layout>();
  } else {
    return false;
  }
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE fp64_words
decode_words_branchy(std::uint32_t raw) {
  static_assert(Layout::policy != special_policy::fp64_prefix);
  raw &= raw_mask<Layout>();
  const auto sign = raw >> (Layout::total_bits - 1);
  const auto exponent =
      (raw >> Layout::fraction_bits) & exponent_mask<Layout>();
  const auto fraction = raw & fraction_mask<Layout>();

  fp64_words result{};
  if (exponent == 0) {
    result = subnormal_magnitude_words<Layout>(fraction);
  } else if (is_special<Layout>(exponent, fraction)) {
    if constexpr (Layout::policy == special_policy::e4m3fn) {
      result = {0x7ff80000u, 0};
    } else {
      result = special_magnitude_words<Layout>(fraction);
    }
  } else {
    result = normal_magnitude_words<Layout>(exponent, fraction);
  }
  result.high |= sign << 31;
  return result;
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE fp64_words
decode_words_masked(std::uint32_t raw) {
  static_assert(Layout::policy != special_policy::fp64_prefix);
  raw &= raw_mask<Layout>();
  const auto sign = raw >> (Layout::total_bits - 1);
  const auto exponent =
      (raw >> Layout::fraction_bits) & exponent_mask<Layout>();
  const auto fraction = raw & fraction_mask<Layout>();

  const auto subnormal = subnormal_magnitude_words<Layout>(fraction);
  const auto normal = normal_magnitude_words<Layout>(exponent, fraction);
  auto special = special_magnitude_words<Layout>(fraction);
  if constexpr (Layout::policy == special_policy::e4m3fn) {
    special = {0x7ff80000u, 0};
  }

  const auto subnormal_mask =
      0u - static_cast<std::uint32_t>(exponent == 0);
  const auto special_mask =
      0u - static_cast<std::uint32_t>(is_special<Layout>(exponent, fraction));
  const auto normal_mask = ~(subnormal_mask | special_mask);
  return {(sign << 31) | (subnormal.high & subnormal_mask) |
              (normal.high & normal_mask) | (special.high & special_mask),
          (subnormal.low & subnormal_mask) | (normal.low & normal_mask) |
              (special.low & special_mask)};
}

template <int FractionBits>
AUT_DECODER_HD AUT_DECODER_INLINE fp64_words
decode_prefix_words(std::uint32_t raw) {
  static_assert(FractionBits == 4 || FractionBits == 20);
  if constexpr (FractionBits == 4) {
    return {raw << 16, 0};
  } else {
    return {raw, 0};
  }
}

AUT_DECODER_HD AUT_DECODER_INLINE double words_to_double(fp64_words words) {
#if defined(__CUDA_ARCH__)
  return __hiloint2double(static_cast<int>(words.high),
                          static_cast<int>(words.low));
#else
  const auto bits = (static_cast<std::uint64_t>(words.high) << 32) | words.low;
  double value{};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

AUT_DECODER_HD AUT_DECODER_INLINE float bits_to_float(std::uint32_t bits) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(bits);
#else
  float value{};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE std::uint32_t
decode_fp32_bits(std::uint32_t raw) {
  static_assert(Layout::policy != special_policy::fp64_prefix);
  static_assert(Layout::exponent_bits <= 8);
  static_assert(Layout::fraction_bits <= 23);
  raw &= raw_mask<Layout>();

  if constexpr (Layout::exponent_bits == 8 &&
                Layout::policy == special_policy::ieee) {
    return raw << (32 - Layout::total_bits);
  }

  const auto sign = raw >> (Layout::total_bits - 1);
  const auto exponent =
      (raw >> Layout::fraction_bits) & exponent_mask<Layout>();
  const auto fraction = raw & fraction_mask<Layout>();
  if (exponent == 0) {
    if (fraction == 0) {
      return sign << 31;
    }
    const auto leading = highest_set_bit(fraction);
    const auto unbiased =
        leading + 1 - Layout::exponent_bias - Layout::fraction_bits;
    const auto exponent32 = static_cast<std::uint32_t>(unbiased + 127);
    const auto remainder = fraction - (std::uint32_t{1} << leading);
    return (sign << 31) | (exponent32 << 23) |
           (remainder << (23 - leading));
  }
  if (is_special<Layout>(exponent, fraction)) {
    if constexpr (Layout::policy == special_policy::e4m3fn) {
      return (sign << 31) | 0x7fc00000u;
    }
    return (sign << 31) | 0x7f800000u |
           (fraction << (23 - Layout::fraction_bits));
  }
  const auto exponent32 = static_cast<std::uint32_t>(
      static_cast<int>(exponent) - Layout::exponent_bias + 127);
  return (sign << 31) | (exponent32 << 23) |
         (fraction << (23 - Layout::fraction_bits));
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE double decode_via_fp32(std::uint32_t raw) {
  return static_cast<double>(bits_to_float(decode_fp32_bits<Layout>(raw)));
}

template <typename Layout>
AUT_DECODER_HD AUT_DECODER_INLINE double decode_e1_integer(std::uint32_t raw) {
  static_assert(Layout::exponent_bits == 1);
  static_assert(Layout::policy == special_policy::finite_all);
  raw &= raw_mask<Layout>();
  const auto sign = raw >> (Layout::total_bits - 1);
  const auto magnitude = raw & (raw_mask<Layout>() >> 1);
  const auto scale_exponent = 1 - Layout::fraction_bits + 1023;
  const auto scale = words_to_double(
      {static_cast<std::uint32_t>(scale_exponent) << 20, 0});
  const auto value = static_cast<double>(magnitude) * scale;
  return sign != 0 ? -value : value;
}

} // namespace aut::decoder

#undef AUT_DECODER_HD
#undef AUT_DECODER_INLINE

#endif // ACCESSOR_UNIVERSAL_TEST_DECODER_STRATEGY_CORE_HPP_
