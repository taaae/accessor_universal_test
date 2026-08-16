#ifndef ACCESSOR_UNIVERSAL_TEST_LNS_BENCHMARK_CORE_HPP_
#define ACCESSOR_UNIVERSAL_TEST_LNS_BENCHMARK_CORE_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define AUT_LNS_HD __host__ __device__
#define AUT_LNS_INLINE __forceinline__
#else
#define AUT_LNS_HD
#define AUT_LNS_INLINE inline
#endif

namespace aut::lns {

/**
 * Universal-style logarithmic number.
 *
 * The high bit is the value sign.  The remaining B-1 bits are an R-bit
 * fixed-point, two's-complement base-2 logarithm.  The most-negative log
 * code is reserved: positive sign means zero and negative sign means NaN.
 * This matches the special-value encoding used by Universal's lns type.
 */
template <int Bits, int FractionBits> struct format {
  static_assert(Bits >= 2 && Bits <= 32);
  static_assert(FractionBits >= 0 && FractionBits < Bits - 1);

  static constexpr int total_bits = Bits;
  static constexpr int log_fraction_bits = FractionBits;
  static constexpr int log_integer_bits = Bits - 1 - FractionBits;
  static constexpr int exponent_bits = log_integer_bits;
  static constexpr int fraction_bits = log_fraction_bits;
};

#define AUT_DEFINE_LNS_FORMAT(name_, bits_, fraction_)                       \
  struct name_ : format<bits_, fraction_> {                                 \
    static constexpr const char *name = #name_;                             \
  }

AUT_DEFINE_LNS_FORMAT(lns4_r0, 4, 0);
AUT_DEFINE_LNS_FORMAT(lns4_r1, 4, 1);
AUT_DEFINE_LNS_FORMAT(lns6_r2, 6, 2);
AUT_DEFINE_LNS_FORMAT(lns8_r2, 8, 2);
AUT_DEFINE_LNS_FORMAT(lns8_r3, 8, 3);
AUT_DEFINE_LNS_FORMAT(lns8_r4, 8, 4);
AUT_DEFINE_LNS_FORMAT(lns8_r5, 8, 5);
AUT_DEFINE_LNS_FORMAT(lns10_r4, 10, 4);
AUT_DEFINE_LNS_FORMAT(lns12_r6, 12, 6);
AUT_DEFINE_LNS_FORMAT(lns16_r4, 16, 4);
AUT_DEFINE_LNS_FORMAT(lns16_r7, 16, 7);
AUT_DEFINE_LNS_FORMAT(lns16_r10, 16, 10);
AUT_DEFINE_LNS_FORMAT(lns16_r11, 16, 11);
AUT_DEFINE_LNS_FORMAT(lns16_r12, 16, 12);
AUT_DEFINE_LNS_FORMAT(lns16_r13, 16, 13);
AUT_DEFINE_LNS_FORMAT(lns32_r20, 32, 20);
AUT_DEFINE_LNS_FORMAT(lns32_r23, 32, 23);
AUT_DEFINE_LNS_FORMAT(lns32_r28, 32, 28);

#undef AUT_DEFINE_LNS_FORMAT

template <int Bits> struct unsigned_storage;
template <> struct unsigned_storage<8> { using type = std::uint8_t; };
template <> struct unsigned_storage<16> { using type = std::uint16_t; };
template <> struct unsigned_storage<32> { using type = std::uint32_t; };

template <typename Format>
using padded_storage_t = typename unsigned_storage<
    (Format::total_bits <= 8 ? 8 : Format::total_bits <= 16 ? 16 : 32)>::type;

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE constexpr std::uint32_t raw_mask() {
  if constexpr (Format::total_bits == 32) {
    return 0xffffffffu;
  } else {
    return (std::uint32_t{1} << Format::total_bits) - 1u;
  }
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE constexpr std::uint32_t log_mask() {
  return (std::uint32_t{1} << (Format::total_bits - 1)) - 1u;
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE constexpr std::uint32_t sign_mask() {
  return std::uint32_t{1} << (Format::total_bits - 1);
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE constexpr std::uint32_t special_log_code() {
  return std::uint32_t{1} << (Format::total_bits - 2);
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE constexpr std::int32_t minimum_finite_log_code() {
  return -static_cast<std::int32_t>(special_log_code<Format>()) + 1;
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE constexpr std::int32_t maximum_finite_log_code() {
  return static_cast<std::int32_t>(special_log_code<Format>()) - 1;
}

// A fused product adds two finite log codes, each in
// [-special+1, special-1], so the sum spans [-2*special+2, 2*special-2].
// Biasing by 2*special-2 maps that onto [0, 4*special-4], which fits a table
// of 4*special == 2^total_bits entries -- the same size as the full lookup
// over single codes, and the reason a fused product table is affordable.
template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE constexpr std::int32_t product_log_bias() {
  return 2 * static_cast<std::int32_t>(special_log_code<Format>()) - 2;
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE constexpr std::size_t product_log_entries() {
  return std::size_t{1} << Format::total_bits;
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE bool is_special(std::uint32_t raw) {
  return (raw & log_mask<Format>()) == special_log_code<Format>();
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE bool is_zero(std::uint32_t raw) {
  return is_special<Format>(raw) && (raw & sign_mask<Format>()) == 0;
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE bool is_nan(std::uint32_t raw) {
  return is_special<Format>(raw) && (raw & sign_mask<Format>()) != 0;
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE bool sign(std::uint32_t raw) {
  return (raw & sign_mask<Format>()) != 0;
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE std::int32_t log_code(std::uint32_t raw) {
  constexpr int width = Format::total_bits - 1;
  auto code = raw & log_mask<Format>();
  if ((code & (std::uint32_t{1} << (width - 1))) != 0) {
    code |= ~log_mask<Format>();
  }
  return static_cast<std::int32_t>(code);
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE std::uint32_t make_raw(bool negative,
                                                 std::int32_t code) {
  return (negative ? sign_mask<Format>() : 0u) |
         (static_cast<std::uint32_t>(code) & log_mask<Format>());
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE std::uint32_t zero_raw() {
  return special_log_code<Format>();
}

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE std::uint32_t nan_raw() {
  return sign_mask<Format>() | special_log_code<Format>();
}

template <typename Target>
AUT_LNS_HD AUT_LNS_INLINE Target exp2_target(Target value) {
  if constexpr (std::is_same_v<Target, float>) {
    return ::exp2f(value);
  } else {
    return ::exp2(value);
  }
}

template <typename Format, typename Target>
AUT_LNS_HD AUT_LNS_INLINE Target decode(std::uint32_t raw) {
  raw &= raw_mask<Format>();
  if (is_zero<Format>(raw)) {
    return Target{0};
  }
  if (is_nan<Format>(raw)) {
    return std::numeric_limits<Target>::quiet_NaN();
  }
  constexpr Target scale =
      static_cast<Target>(std::uint32_t{1} << Format::log_fraction_bits);
  auto value = exp2_target(static_cast<Target>(log_code<Format>(raw)) / scale);
  return sign<Format>(raw) ? -value : value;
}

template <typename Format, typename Source>
AUT_LNS_HD AUT_LNS_INLINE std::uint32_t encode(Source value) {
  const auto wide_value = static_cast<double>(value);
  // Avoid relying on whether a given host standard library also exports the
  // C math classification functions into the global namespace.  These tests
  // are valid in both the host and CUDA device passes.
  if (wide_value != wide_value) {
    return nan_raw<Format>();
  }
  constexpr auto finite_limit = std::numeric_limits<double>::max();
  if (wide_value > finite_limit || wide_value < -finite_limit) {
    return make_raw<Format>(value < Source{0},
                            maximum_finite_log_code<Format>());
  }
  if (value == Source{0}) {
    return zero_raw<Format>();
  }
  const auto negative = value < Source{0};
  const auto magnitude = negative ? -static_cast<double>(value)
                                  : static_cast<double>(value);
  constexpr auto scale = std::uint64_t{1} << Format::log_fraction_bits;
  const auto scaled_log =
      ::log2(magnitude) * static_cast<double>(scale);
  const auto minimum = static_cast<std::int64_t>(
      minimum_finite_log_code<Format>());
  const auto maximum = static_cast<std::int64_t>(
      maximum_finite_log_code<Format>());
  // Universal's saturating LNS maps values at or below the half-step below
  // minpos to zero.  Values between that threshold and minpos round/clamp to
  // minpos instead of wrapping through the reserved code.
  if (scaled_log <= static_cast<double>(minimum) - 0.5) {
    return zero_raw<Format>();
  }
  auto rounded = static_cast<std::int64_t>(::nearbyint(scaled_log));
  rounded = rounded < minimum ? minimum : rounded > maximum ? maximum : rounded;
  return make_raw<Format>(negative, static_cast<std::int32_t>(rounded));
}

template <typename Format>
struct product_code {
  bool negative{};
  bool zero{};
  bool nan{};
  std::int64_t log{};
};

template <typename Format>
AUT_LNS_HD AUT_LNS_INLINE product_code<Format>
multiply_codes(std::uint32_t left, std::uint32_t right) {
  if (is_nan<Format>(left) || is_nan<Format>(right)) {
    return {false, false, true, 0};
  }
  if (is_zero<Format>(left) || is_zero<Format>(right)) {
    return {sign<Format>(left) != sign<Format>(right), true, false, 0};
  }
  return {sign<Format>(left) != sign<Format>(right), false, false,
          static_cast<std::int64_t>(log_code<Format>(left)) +
              static_cast<std::int64_t>(log_code<Format>(right))};
}

template <typename Format, typename Target>
AUT_LNS_HD AUT_LNS_INLINE Target decode_product(product_code<Format> product) {
  if (product.nan) {
    return std::numeric_limits<Target>::quiet_NaN();
  }
  if (product.zero) {
    return product.negative ? -Target{0} : Target{0};
  }
  constexpr Target scale =
      static_cast<Target>(std::uint32_t{1} << Format::log_fraction_bits);
  auto value = exp2_target(static_cast<Target>(product.log) / scale);
  return product.negative ? -value : value;
}

template <typename Format, typename Target>
AUT_LNS_HD AUT_LNS_INLINE Target multiply_fused(std::uint32_t left,
                                                std::uint32_t right) {
  return decode_product<Format, Target>(multiply_codes<Format>(left, right));
}

} // namespace aut::lns

#endif // ACCESSOR_UNIVERSAL_TEST_LNS_BENCHMARK_CORE_HPP_
