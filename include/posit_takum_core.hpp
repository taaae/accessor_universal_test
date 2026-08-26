#ifndef ACCESSOR_UNIVERSAL_TEST_POSIT_TAKUM_CORE_HPP_
#define ACCESSOR_UNIVERSAL_TEST_POSIT_TAKUM_CORE_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define AUT_PT_HD __host__ __device__
#define AUT_PT_INLINE __forceinline__
#else
#define AUT_PT_HD
#define AUT_PT_INLINE inline
#endif

namespace aut::pt {

enum class family { posit, takum_linear, takum_log };
enum class arithmetic { fp32, fp64 };
enum class strategy { direct, full_lut_global, full_lut_shared };

template <int Bits> AUT_PT_HD constexpr std::uint32_t mask() {
  static_assert(Bits >= 2 && Bits <= 32);
  if constexpr (Bits == 32) {
    return 0xffffffffu;
  } else {
    return (std::uint32_t{1} << Bits) - 1u;
  }
}

template <int Bits> AUT_PT_HD constexpr std::uint32_t sign_mask() {
  return std::uint32_t{1} << (Bits - 1);
}

template <int Bits>
AUT_PT_HD AUT_PT_INLINE std::uint32_t magnitude_bits(std::uint32_t raw) {
  raw &= mask<Bits>();
  return (raw & sign_mask<Bits>()) ? ((~raw + 1u) & mask<Bits>()) : raw;
}

AUT_PT_HD AUT_PT_INLINE int leading_zeros(std::uint32_t value) {
#if defined(__CUDA_ARCH__)
  return __clz(value);
#else
  return value == 0u ? 32 : __builtin_clz(value);
#endif
}

template <typename Float> struct ieee_traits;
template <> struct ieee_traits<float> {
  using uint_type = std::uint32_t;
  static constexpr int fraction_bits = 23;
  static constexpr int exponent_bits = 8;
  static constexpr int bias = 127;
  static constexpr int min_exponent = -126;
  static constexpr int max_exponent = 127;
};
template <> struct ieee_traits<double> {
  using uint_type = std::uint64_t;
  static constexpr int fraction_bits = 52;
  static constexpr int exponent_bits = 11;
  static constexpr int bias = 1023;
  static constexpr int min_exponent = -1022;
  static constexpr int max_exponent = 1023;
};

template <typename Float>
AUT_PT_HD AUT_PT_INLINE Float from_bits(typename ieee_traits<Float>::uint_type bits) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<Float, float>) {
    return __uint_as_float(bits);
  } else {
    return __longlong_as_double(static_cast<long long>(bits));
  }
#else
  Float value{};
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

template <typename Float>
AUT_PT_HD AUT_PT_INLINE typename ieee_traits<Float>::uint_type to_bits(Float value) {
#if defined(__CUDA_ARCH__)
  if constexpr (std::is_same_v<Float, float>) {
    return __float_as_uint(value);
  } else {
    return static_cast<std::uint64_t>(__double_as_longlong(value));
  }
#else
  typename ieee_traits<Float>::uint_type bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
#endif
}

template <typename Float> AUT_PT_HD AUT_PT_INLINE Float quiet_nan() {
  using uint_type = typename ieee_traits<Float>::uint_type;
  constexpr int p = ieee_traits<Float>::fraction_bits;
  const auto exponent =
      ((uint_type{1} << ieee_traits<Float>::exponent_bits) - 1u) << p;
  return from_bits<Float>(exponent | (uint_type{1} << (p - 1)));
}

AUT_PT_HD AUT_PT_INLINE std::uint64_t round_shift_right(std::uint64_t value,
                                                        int shift) {
  if (shift <= 0) {
    return shift <= -63 ? 0u : value << (-shift);
  }
  if (shift >= 64) {
    return 0u;
  }
  const auto truncated = value >> shift;
  const auto remainder_mask = (std::uint64_t{1} << shift) - 1u;
  const auto remainder = value & remainder_mask;
  const auto halfway = std::uint64_t{1} << (shift - 1);
  return truncated +
         static_cast<std::uint64_t>(remainder > halfway ||
                                    (remainder == halfway && (truncated & 1u)));
}

// Encode sign * (significand / 2^source_fraction_bits) * 2^exponent into the
// target IEEE type. The integer significand includes its leading one.
template <typename Float>
AUT_PT_HD AUT_PT_INLINE Float assemble_binary(bool sign, int exponent,
                                               std::uint64_t significand,
                                               int source_fraction_bits) {
  using traits = ieee_traits<Float>;
  using uint_type = typename traits::uint_type;
  constexpr int p = traits::fraction_bits;
  constexpr auto sign_bit = uint_type{1} << (p + traits::exponent_bits);
  const auto encoded_sign = sign ? sign_bit : uint_type{0};

  if (significand == 0u) {
    return from_bits<Float>(encoded_sign);
  }
  if (exponent > traits::max_exponent) {
    const auto infinity = ((uint_type{1} << traits::exponent_bits) - 1u) << p;
    return from_bits<Float>(encoded_sign | infinity);
  }

  if (exponent >= traits::min_exponent) {
    std::uint64_t rounded = source_fraction_bits > p
                                ? round_shift_right(significand,
                                                    source_fraction_bits - p)
                                : significand << (p - source_fraction_bits);
    if (rounded == (std::uint64_t{1} << (p + 1))) {
      rounded >>= 1;
      ++exponent;
      if (exponent > traits::max_exponent) {
        const auto infinity =
            ((uint_type{1} << traits::exponent_bits) - 1u) << p;
        return from_bits<Float>(encoded_sign | infinity);
      }
    }
    const auto exponent_field =
        static_cast<uint_type>(exponent + traits::bias) << p;
    const auto fraction_field = static_cast<uint_type>(
        rounded & ((std::uint64_t{1} << p) - 1u));
    return from_bits<Float>(encoded_sign | exponent_field | fraction_field);
  }

  // Subnormal units are 2^(min_exponent-p).
  const int right_shift = source_fraction_bits - exponent +
                          traits::min_exponent - p;
  std::uint64_t rounded = right_shift > 0
                              ? round_shift_right(significand, right_shift)
                              : (right_shift <= -63
                                     ? 0u
                                     : significand << (-right_shift));
  if (rounded >= (std::uint64_t{1} << p)) {
    return from_bits<Float>(encoded_sign | (uint_type{1} << p));
  }
  return from_bits<Float>(encoded_sign | static_cast<uint_type>(rounded));
}

template <int Bits, int Es, typename Float>
AUT_PT_HD AUT_PT_INLINE Float decode_posit(std::uint32_t raw) {
  static_assert(Es >= 0 && Es <= 3);
  raw &= mask<Bits>();
  if (raw == 0u) {
    return Float{0};
  }
  if (raw == sign_mask<Bits>()) {
    return quiet_nan<Float>();
  }

  const bool sign = (raw & sign_mask<Bits>()) != 0u;
  const auto mag = magnitude_bits<Bits>(raw);
  const auto aligned = mag << (33 - Bits);
  const bool regime_one = (aligned & 0x80000000u) != 0u;
  int run = regime_one ? leading_zeros(~aligned) : leading_zeros(aligned);
  if (run > Bits - 1) {
    run = Bits - 1;
  }
  const bool terminated = run < Bits - 1;
  const int consumed = run + (terminated ? 1 : 0);
  int remaining = Bits - 1 - consumed;
  const int exponent_taken = remaining < Es ? remaining : Es;
  const int fraction_bits = remaining - exponent_taken;
  const auto tail_mask = consumed == Bits - 1
                             ? 0u
                             : ((std::uint32_t{1} << (Bits - 1 - consumed)) - 1u);
  const auto tail = mag & tail_mask;
  const auto exponent_raw = exponent_taken == 0
                                ? 0u
                                : (tail >> fraction_bits) &
                                      ((std::uint32_t{1} << exponent_taken) - 1u);
  const auto exponent_bits = exponent_raw << (Es - exponent_taken);
  const auto fraction = fraction_bits == 0
                            ? 0u
                            : tail & ((std::uint32_t{1} << fraction_bits) - 1u);
  const int regime = regime_one ? run - 1 : -run;
  const int exponent = regime * (1 << Es) + static_cast<int>(exponent_bits);
  const auto significand = (std::uint64_t{1} << fraction_bits) | fraction;
  return assemble_binary<Float>(sign, exponent, significand, fraction_bits);
}

template <int Bits> struct takum_fields {
  bool sign{};
  unsigned direction{};
  unsigned regime_code{};
  unsigned characteristic_bits{};
  int characteristic{};
  int tail_bits{};
  std::uint32_t tail{};
};

template <int Bits>
AUT_PT_HD AUT_PT_INLINE takum_fields<Bits> split_takum(std::uint32_t raw) {
  raw &= mask<Bits>();
  const bool sign = (raw & sign_mask<Bits>()) != 0u;
  const auto mag = magnitude_bits<Bits>(raw);
  const unsigned dr = (mag >> (Bits - 5)) & 0xfu;
  const unsigned direction = dr >> 3;
  const unsigned regime_code = dr & 7u;
  const unsigned characteristic_bits = direction ? regime_code : 7u - regime_code;
  constexpr unsigned available = Bits - 5;
  const unsigned stored_characteristic =
      characteristic_bits < available ? characteristic_bits : available;
  const int tail_bits = static_cast<int>(available - stored_characteristic);
  const auto stored = stored_characteristic == 0
                          ? 0u
                          : (mag >> tail_bits) &
                                ((std::uint32_t{1} << stored_characteristic) - 1u);
  const auto characteristic_payload =
      stored << (characteristic_bits - stored_characteristic);
  const int bias = direction ? (static_cast<int>(std::uint32_t{1}
                                                 << characteristic_bits) -
                                1)
                             : (1 - static_cast<int>(std::uint32_t{1}
                                                    << (characteristic_bits + 1)));
  const int characteristic = bias + static_cast<int>(characteristic_payload);
  const auto tail = tail_bits == 0
                        ? 0u
                        : mag & ((std::uint32_t{1} << tail_bits) - 1u);
  return {sign, direction, regime_code, characteristic_bits, characteristic,
          tail_bits, tail};
}

template <int Bits, typename Float>
AUT_PT_HD AUT_PT_INLINE Float decode_linear_takum(std::uint32_t raw) {
  raw &= mask<Bits>();
  if (raw == 0u) {
    return Float{0};
  }
  if (raw == sign_mask<Bits>()) {
    return quiet_nan<Float>();
  }
  const auto fields = split_takum<Bits>(raw);
  const auto significand =
      (std::uint64_t{1} << fields.tail_bits) | fields.tail;
  return assemble_binary<Float>(fields.sign, fields.characteristic, significand,
                                fields.tail_bits);
}

template <int Bits, typename Float>
AUT_PT_HD AUT_PT_INLINE Float decode_log_takum(std::uint32_t raw) {
  raw &= mask<Bits>();
  if (raw == 0u) {
    return Float{0};
  }
  if (raw == sign_mask<Bits>()) {
    return quiet_nan<Float>();
  }
  const auto fields = split_takum<Bits>(raw);
  const double fraction = fields.tail_bits == 0
                              ? 0.0
                              : static_cast<double>(fields.tail) /
                                    static_cast<double>(std::uint64_t{1}
                                                        << fields.tail_bits);
  const double natural_exponent =
      (static_cast<double>(fields.characteristic) + fraction) * 0.5;
  Float magnitude{};
#if defined(__CUDA_ARCH__)
  magnitude = static_cast<Float>(exp(natural_exponent));
#else
  magnitude = static_cast<Float>(std::exp(natural_exponent));
#endif
  return fields.sign ? -magnitude : magnitude;
}

template <family Family, int Bits, int Es, typename Float>
AUT_PT_HD AUT_PT_INLINE Float decode(std::uint32_t raw) {
  if constexpr (Family == family::posit) {
    return decode_posit<Bits, Es, Float>(raw);
  } else if constexpr (Family == family::takum_linear) {
    return decode_linear_takum<Bits, Float>(raw);
  } else {
    return decode_log_takum<Bits, Float>(raw);
  }
}

template <family Family, int Bits, int Es>
inline long double decode_long_double(std::uint32_t raw) {
  if constexpr (Family == family::posit) {
    return static_cast<long double>(decode_posit<Bits, Es, double>(raw));
  } else if constexpr (Family == family::takum_linear) {
    return static_cast<long double>(decode_linear_takum<Bits, double>(raw));
  } else {
    if ((raw & mask<Bits>()) == 0u) {
      return 0.0L;
    }
    if ((raw & mask<Bits>()) == sign_mask<Bits>()) {
      return std::numeric_limits<long double>::quiet_NaN();
    }
    const auto fields = split_takum<Bits>(raw);
    const long double fraction =
        fields.tail_bits == 0
            ? 0.0L
            : static_cast<long double>(fields.tail) /
                  static_cast<long double>(std::uint64_t{1} << fields.tail_bits);
    return (fields.sign ? -1.0L : 1.0L) *
           std::exp((static_cast<long double>(fields.characteristic) + fraction) /
                    2.0L);
  }
}

template <family Family, int Bits, int Es>
inline std::uint32_t encode_positive_log2(long double q) {
  const std::uint32_t high = sign_mask<Bits>() - 1u;
  std::uint32_t lo = 1u;
  std::uint32_t hi = high;
  while (lo < hi) {
    const auto mid = lo + ((hi - lo) >> 1);
    const auto value = decode_long_double<Family, Bits, Es>(mid);
    const auto mid_q = std::log2(value);
    if (mid_q < q) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  if (lo == 1u) {
    return lo;
  }
  const auto lower_q =
      std::log2(decode_long_double<Family, Bits, Es>(lo - 1u));
  const auto upper_q = std::log2(decode_long_double<Family, Bits, Es>(lo));
  return (q - lower_q <= upper_q - q) ? lo - 1u : lo;
}

template <int Bits>
AUT_PT_HD AUT_PT_INLINE std::uint32_t apply_sign(std::uint32_t magnitude,
                                                 bool negative) {
  magnitude &= mask<Bits>();
  return negative ? ((~magnitude + 1u) & mask<Bits>()) : magnitude;
}

} // namespace aut::pt

#undef AUT_PT_HD
#undef AUT_PT_INLINE

#endif
