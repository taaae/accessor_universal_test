#ifndef ACCESSOR_UNIVERSAL_TEST_STORAGE_FORMATS_HPP_
#define ACCESSOR_UNIVERSAL_TEST_STORAGE_FORMATS_HPP_

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define AUT_STORAGE_HD __host__ __device__
#define AUT_STORAGE_INLINE __forceinline__
#else
#define AUT_STORAGE_HD
#define AUT_STORAGE_INLINE inline
#endif

namespace aut::storage {

namespace detail {

template <int Bits> struct uint_for;
template <> struct uint_for<8> {
  using type = std::uint8_t;
};
template <> struct uint_for<16> {
  using type = std::uint16_t;
};
template <> struct uint_for<32> {
  using type = std::uint32_t;
};

AUT_STORAGE_HD AUT_STORAGE_INLINE std::uint64_t double_bits(double value) {
#if defined(__CUDA_ARCH__)
  return static_cast<std::uint64_t>(__double_as_longlong(value));
#else
  std::uint64_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
#endif
}

AUT_STORAGE_HD AUT_STORAGE_INLINE double double_from_bits(std::uint64_t bits) {
#if defined(__CUDA_ARCH__)
  return __longlong_as_double(static_cast<long long>(bits));
#else
  double value{};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

AUT_STORAGE_HD AUT_STORAGE_INLINE int highest_set_bit(std::uint64_t value) {
#if defined(__CUDA_ARCH__)
  return 63 - __clzll(value);
#elif defined(__GNUC__) || defined(__clang__)
  return 63 - __builtin_clzll(value);
#else
  int result = -1;
  while (value != 0) {
    ++result;
    value >>= 1;
  }
  return result;
#endif
}

AUT_STORAGE_HD AUT_STORAGE_INLINE std::uint64_t
round_right_shift_even(std::uint64_t value, unsigned shift) {
  if (shift == 0) {
    return value;
  }
  if (shift >= 64) {
    return 0;
  }
  const auto quotient = value >> shift;
  const auto mask = (std::uint64_t{1} << shift) - 1;
  const auto remainder = value & mask;
  const auto halfway = std::uint64_t{1} << (shift - 1);
  const auto increment =
      remainder > halfway || (remainder == halfway && (quotient & 1u) != 0);
  return quotient + static_cast<std::uint64_t>(increment);
}

} // namespace detail

/**
 * A compact IEEE-like format stored in exactly 8, 16, or 32 bits.
 *
 * The non-finite family reserves the all-ones exponent for infinity and NaN.
 * The finite family reclaims it for normal finite values and saturates
 * overflow. Both families use round-to-nearest-even and gradual underflow.
 */
template <int TotalBits, int ExponentBits, bool Finite> struct binary_format {
  static constexpr int total_bits = TotalBits;
  static constexpr int exponent_bits = ExponentBits;
  static constexpr int fraction_bits = TotalBits - ExponentBits - 1;
  static constexpr bool finite = Finite;
};

struct e1m6 : binary_format<8, 1, true> {
  static constexpr const char *name = "e1m6";
};
struct e2m5 : binary_format<8, 2, false> {
  static constexpr const char *name = "e2m5";
};
struct e3m4 : binary_format<8, 3, false> {
  static constexpr const char *name = "e3m4";
};

struct e1m14 : binary_format<16, 1, true> {
  static constexpr const char *name = "e1m14";
};
struct e2m13 : binary_format<16, 2, false> {
  static constexpr const char *name = "e2m13";
};
struct e3m12 : binary_format<16, 3, false> {
  static constexpr const char *name = "e3m12";
};

struct e1m30 : binary_format<32, 1, true> {
  static constexpr const char *name = "e1m30";
};
struct e2m29 : binary_format<32, 2, false> {
  static constexpr const char *name = "e2m29";
};
struct e3m28 : binary_format<32, 3, false> {
  static constexpr const char *name = "e3m28";
};

template <int FractionBits> struct fp64_prefix {
  static_assert(FractionBits == 4 || FractionBits == 20,
                "the experiment uses only the 16- and 32-bit FP64 prefixes");
  static constexpr int total_bits = 12 + FractionBits;
  static constexpr int exponent_bits = 11;
  static constexpr int fraction_bits = FractionBits;
  static constexpr bool finite = false;
};

struct e11m4 : fp64_prefix<4> {
  static constexpr const char *name = "e11m4";
};
struct e11m20 : fp64_prefix<20> {
  static constexpr const char *name = "e11m20";
};

template <typename Format> struct codec;

template <int TotalBits, int ExponentBits, bool Finite>
struct codec<binary_format<TotalBits, ExponentBits, Finite>> {
  using format_type = binary_format<TotalBits, ExponentBits, Finite>;
  using storage_type = typename detail::uint_for<TotalBits>::type;

  static constexpr int fraction_bits = format_type::fraction_bits;
  static constexpr std::uint64_t fraction_mask =
      (std::uint64_t{1} << fraction_bits) - 1;
  static constexpr std::uint64_t exponent_mask =
      (std::uint64_t{1} << ExponentBits) - 1;
  static constexpr int exponent_bias = (1 << (ExponentBits - 1)) - 1;
  static constexpr std::uint64_t sign_mask = std::uint64_t{1}
                                             << (TotalBits - 1);
  static constexpr std::uint64_t maximum_normal_exponent =
      Finite ? exponent_mask : exponent_mask - 1;

  AUT_STORAGE_HD AUT_STORAGE_INLINE static storage_type encode(double value) {
    const auto source = detail::double_bits(value);
    const auto sign = source >> 63;
    const auto source_exponent = (source >> 52) & 0x7ffu;
    const auto source_fraction = source & ((std::uint64_t{1} << 52) - 1);
    const auto sign_field = sign << (TotalBits - 1);

    if (source_exponent == 0x7ffu) {
      if constexpr (Finite) {
        return static_cast<storage_type>(
            sign_field | (maximum_normal_exponent << fraction_bits) |
            fraction_mask);
      } else {
        if (source_fraction == 0) {
          return static_cast<storage_type>(sign_field |
                                           (exponent_mask << fraction_bits));
        }
        auto payload = source_fraction >> (52 - fraction_bits);
        if (payload == 0) {
          payload = 1;
        }
        return static_cast<storage_type>(sign_field |
                                         (exponent_mask << fraction_bits) |
                                         (payload & fraction_mask));
      }
    }

    if (source_exponent == 0) {
      // Every binary64 subnormal is far below the range of the E1/E2/E3
      // formats used by this experiment.
      return static_cast<storage_type>(sign_field);
    }

    const auto unbiased_exponent = static_cast<int>(source_exponent) - 1023;
    auto target_exponent = unbiased_exponent + exponent_bias;

    if (target_exponent <= 0) {
      const auto significand = (std::uint64_t{1} << 52) | source_fraction;
      const auto shift = static_cast<unsigned>(53 - unbiased_exponent -
                                               exponent_bias - fraction_bits);
      const auto subnormal = detail::round_right_shift_even(significand, shift);
      if (subnormal == 0) {
        return static_cast<storage_type>(sign_field);
      }
      if (subnormal >= (std::uint64_t{1} << fraction_bits)) {
        return static_cast<storage_type>(sign_field |
                                         (std::uint64_t{1} << fraction_bits));
      }
      return static_cast<storage_type>(sign_field | subnormal);
    }

    auto target_fraction =
        detail::round_right_shift_even(source_fraction, 52 - fraction_bits);
    if (target_fraction == (std::uint64_t{1} << fraction_bits)) {
      target_fraction = 0;
      ++target_exponent;
    }

    if (target_exponent > static_cast<int>(maximum_normal_exponent)) {
      if constexpr (Finite) {
        return static_cast<storage_type>(
            sign_field | (maximum_normal_exponent << fraction_bits) |
            fraction_mask);
      } else {
        return static_cast<storage_type>(sign_field |
                                         (exponent_mask << fraction_bits));
      }
    }

    return static_cast<storage_type>(
        sign_field |
        (static_cast<std::uint64_t>(target_exponent) << fraction_bits) |
        target_fraction);
  }

  AUT_STORAGE_HD AUT_STORAGE_INLINE static double decode(storage_type stored) {
    const auto raw = static_cast<std::uint64_t>(stored);
    const auto sign = raw >> (TotalBits - 1);
    const auto exponent = (raw >> fraction_bits) & exponent_mask;
    const auto fraction = raw & fraction_mask;
    const auto sign64 = sign << 63;

    if (exponent == 0) {
      if (fraction == 0) {
        return detail::double_from_bits(sign64);
      }
      const auto leading = detail::highest_set_bit(fraction);
      const auto unbiased = leading + 1 - exponent_bias - fraction_bits;
      const auto exponent64 = static_cast<std::uint64_t>(unbiased + 1023);
      const auto fraction64 = (fraction - (std::uint64_t{1} << leading))
                              << (52 - leading);
      return detail::double_from_bits(sign64 | (exponent64 << 52) | fraction64);
    }

    if constexpr (!Finite) {
      if (exponent == exponent_mask) {
        const auto fraction64 = fraction << (52 - fraction_bits);
        return detail::double_from_bits(sign64 | (std::uint64_t{0x7ff} << 52) |
                                        fraction64);
      }
    }

    const auto unbiased = static_cast<int>(exponent) - exponent_bias;
    const auto exponent64 = static_cast<std::uint64_t>(unbiased + 1023);
    const auto fraction64 = fraction << (52 - fraction_bits);
    return detail::double_from_bits(sign64 | (exponent64 << 52) | fraction64);
  }
};

template <typename Format>
struct inherited_binary_codec
    : codec<binary_format<Format::total_bits, Format::exponent_bits,
                          Format::finite>> {};

template <> struct codec<e1m6> : inherited_binary_codec<e1m6> {};
template <> struct codec<e2m5> : inherited_binary_codec<e2m5> {};
template <> struct codec<e3m4> : inherited_binary_codec<e3m4> {};
template <> struct codec<e1m14> : inherited_binary_codec<e1m14> {};
template <> struct codec<e2m13> : inherited_binary_codec<e2m13> {};
template <> struct codec<e3m12> : inherited_binary_codec<e3m12> {};
template <> struct codec<e1m30> : inherited_binary_codec<e1m30> {};
template <> struct codec<e2m29> : inherited_binary_codec<e2m29> {};
template <> struct codec<e3m28> : inherited_binary_codec<e3m28> {};

template <int FractionBits> struct prefix_codec {
  using storage_type = typename detail::uint_for<12 + FractionBits>::type;
  static constexpr int discarded_bits = 52 - FractionBits;

  AUT_STORAGE_HD AUT_STORAGE_INLINE static storage_type encode(double value) {
    const auto bits = detail::double_bits(value);
    const auto exponent = (bits >> 52) & 0x7ffu;
    const auto fraction = bits & ((std::uint64_t{1} << 52) - 1);
    auto retained = bits >> discarded_bits;

    if (exponent == 0x7ffu) {
      if (fraction != 0 &&
          (retained & ((std::uint64_t{1} << FractionBits) - 1)) == 0) {
        retained |= 1;
      }
      return static_cast<storage_type>(retained);
    }

    const auto discarded_mask = (std::uint64_t{1} << discarded_bits) - 1;
    const auto discarded = bits & discarded_mask;
    const auto halfway = std::uint64_t{1} << (discarded_bits - 1);
    if (discarded > halfway || (discarded == halfway && (retained & 1u) != 0)) {
      ++retained;
    }
    return static_cast<storage_type>(retained);
  }

  AUT_STORAGE_HD AUT_STORAGE_INLINE static double decode(storage_type stored) {
    return detail::double_from_bits(static_cast<std::uint64_t>(stored)
                                    << discarded_bits);
  }
};

template <> struct codec<e11m4> : prefix_codec<4> {};
template <> struct codec<e11m20> : prefix_codec<20> {};

template <typename Format>
using storage_type_t = typename codec<Format>::storage_type;

template <typename Format>
AUT_STORAGE_HD AUT_STORAGE_INLINE storage_type_t<Format> encode(double value) {
  return codec<Format>::encode(value);
}

template <typename Format>
AUT_STORAGE_HD AUT_STORAGE_INLINE double decode(storage_type_t<Format> value) {
  return codec<Format>::decode(value);
}

static_assert(sizeof(storage_type_t<e1m6>) == 1);
static_assert(sizeof(storage_type_t<e11m4>) == 2);
static_assert(sizeof(storage_type_t<e11m20>) == 4);
static_assert(std::is_trivially_copyable_v<storage_type_t<e3m28>>);

} // namespace aut::storage

#undef AUT_STORAGE_HD
#undef AUT_STORAGE_INLINE

#endif // ACCESSOR_UNIVERSAL_TEST_STORAGE_FORMATS_HPP_
