#ifndef ACCESSOR_UNIVERSAL_TEST_E2E3_DECODER_STRATEGIES_CUH_
#define ACCESSOR_UNIVERSAL_TEST_E2E3_DECODER_STRATEGIES_CUH_

#include "storage_formats.hpp"

#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace aut::e2e3_strategies {

constexpr int block_threads = 256;

enum class decode_kind {
  generic_fp64,
  branchless_fp32,
  lut_fp32,
  lut_fp64,
  lut_prefix,
  direct_fp64_bits,
  decomposed_bits,
  direct_fp64_words_branchy,
  direct_fp64_words_masked,
  lut_subnormal,
  lut_high_word,
  lut_high_word_swizzled,
};

enum class table_location { global_read_only, shared };

template <decode_kind Kind, int Lanes,
          table_location Location = table_location::global_read_only,
          bool Pipelined = false>
struct strategy {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8);
  static constexpr decode_kind kind = Kind;
  static constexpr int lanes = Lanes;
  static constexpr table_location location = Location;
  static constexpr bool pipelined = Pipelined;
};

struct table_bundle {
  const float *fp32{};
  const double *fp64{};
  const std::uint16_t *prefix16{};
  const std::uint32_t *prefix32{};
  const std::uint64_t *exponent_prefix{};
  const std::uint32_t *high_word{};
  const std::uint32_t *subnormal_high_word{};
};

template <typename Format>
inline constexpr bool supported_format_v =
    std::is_same_v<Format, storage::e2m5> ||
    std::is_same_v<Format, storage::e3m4>;

template <typename Strategy>
inline constexpr bool uses_lookup_v =
    Strategy::kind == decode_kind::lut_fp32 ||
    Strategy::kind == decode_kind::lut_fp64 ||
    Strategy::kind == decode_kind::lut_prefix ||
    Strategy::kind == decode_kind::decomposed_bits ||
    Strategy::kind == decode_kind::lut_subnormal ||
    Strategy::kind == decode_kind::lut_high_word ||
    Strategy::kind == decode_kind::lut_high_word_swizzled;

template <typename Format, typename Strategy>
inline constexpr std::size_t lookup_entry_bytes_v = [] {
  if constexpr (Strategy::kind == decode_kind::lut_fp32) {
    return sizeof(float);
  } else if constexpr (Strategy::kind == decode_kind::lut_fp64 ||
                       Strategy::kind == decode_kind::decomposed_bits) {
    return sizeof(std::uint64_t);
  } else if constexpr (Strategy::kind == decode_kind::lut_prefix) {
    return Format::fraction_bits == 4 ? sizeof(std::uint16_t)
                                      : sizeof(std::uint32_t);
  } else if constexpr (Strategy::kind == decode_kind::lut_subnormal ||
                       Strategy::kind == decode_kind::lut_high_word ||
                       Strategy::kind == decode_kind::lut_high_word_swizzled) {
    return sizeof(std::uint32_t);
  } else {
    return std::size_t{0};
  }
}();

template <typename Format, typename Strategy>
inline constexpr std::size_t shared_table_bytes_v = [] {
  if constexpr (Strategy::location != table_location::shared) {
    return std::size_t{0};
  } else {
    static_assert(Strategy::kind == decode_kind::lut_fp32 ||
                  Strategy::kind == decode_kind::lut_fp64 ||
                  Strategy::kind == decode_kind::lut_prefix ||
                  Strategy::kind == decode_kind::lut_subnormal ||
                  Strategy::kind == decode_kind::lut_high_word ||
                  Strategy::kind == decode_kind::lut_high_word_swizzled);
    if constexpr (Strategy::kind == decode_kind::lut_subnormal) {
      return (std::size_t{1} << Format::fraction_bits) * sizeof(std::uint32_t);
    } else if constexpr (Strategy::kind ==
                         decode_kind::lut_high_word_swizzled) {
      // Four copies with one padding word between copies rotate their bank
      // mappings. Each eight-lane group in a warp selects a different copy.
      return 4 * 257 * sizeof(std::uint32_t);
    } else {
      return 256 * lookup_entry_bytes_v<Format, Strategy>;
    }
  }
}();

template <int Lanes> struct decoded_packet {
  double values[Lanes];
};

template <int Lanes> struct source_packet {
  std::uint32_t low{};
  std::uint32_t high{};
};

template <int Lanes>
__device__ __forceinline__ source_packet<Lanes>
load_source_packet(const std::uint8_t *values) {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8);
  if constexpr (Lanes == 1) {
    return {static_cast<std::uint32_t>(*values), 0};
  } else if constexpr (Lanes == 2) {
    return {static_cast<std::uint32_t>(
                *reinterpret_cast<const std::uint16_t *>(values)),
            0};
  } else if constexpr (Lanes == 4) {
    return {*reinterpret_cast<const std::uint32_t *>(values), 0};
  } else {
    return {*reinterpret_cast<const std::uint32_t *>(values),
            *reinterpret_cast<const std::uint32_t *>(values + 4)};
  }
}

template <int Lane, int Lanes>
__device__ __forceinline__ std::uint32_t raw_lane(source_packet<Lanes> packet) {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8);
  static_assert(Lane >= 0 && Lane < Lanes);
  if constexpr (Lane < 4) {
    return (packet.low >> (8 * Lane)) & 0xffu;
  } else {
    return (packet.high >> (8 * (Lane - 4))) & 0xffu;
  }
}

/** Exact E2M5/E3M4 finite decoding through FP32 arithmetic. */
template <typename Format>
__device__ __forceinline__ float branchless_fp32_decode(std::uint32_t raw) {
  static_assert(supported_format_v<Format>);
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_bits = Format::exponent_bits;
  constexpr auto exponent_mask = (std::uint32_t{1} << exponent_bits) - 1;
  constexpr auto fraction_mask = (std::uint32_t{1} << fraction_bits) - 1;
  constexpr auto exponent_bias = (1 << (exponent_bits - 1)) - 1;

  raw &= 0xffu;
  const auto sign = raw >> 7;
  const auto exponent = (raw >> fraction_bits) & exponent_mask;
  const auto fraction = raw & fraction_mask;
  const auto normal = static_cast<std::uint32_t>(exponent != 0);
  const auto significand = fraction | (normal << fraction_bits);
  const auto effective_exponent =
      exponent + static_cast<unsigned>(exponent == 0);
  const auto scale_exponent =
      static_cast<int>(effective_exponent) - exponent_bias - fraction_bits;
  const auto scale_bits = static_cast<std::uint32_t>(scale_exponent + 127)
                          << 23;
  const auto finite_magnitude =
      __uint2float_rn(significand) * __uint_as_float(scale_bits);
  const auto finite_bits = __float_as_uint(finite_magnitude) | (sign << 31);

  const auto special_bits =
      (sign << 31) | 0x7f800000u | (fraction << (23 - fraction_bits));
  const auto special_mask =
      0u - static_cast<std::uint32_t>(exponent == exponent_mask);
  return __uint_as_float((finite_bits & ~special_mask) |
                         (special_bits & special_mask));
}

template <typename Format>
__device__ __forceinline__ std::uint64_t
subnormal_magnitude_bits(std::uint32_t fraction) {
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_bits = Format::exponent_bits;
  constexpr auto exponent_bias = (1 << (exponent_bits - 1)) - 1;
  const auto safe_fraction = fraction | 1u;
  const auto leading = 31 - __clz(safe_fraction);
  const auto unbiased = leading + 1 - exponent_bias - fraction_bits;
  const auto exponent64 = static_cast<std::uint64_t>(unbiased + 1023);
  const auto remainder = fraction - (std::uint32_t{1} << leading);
  return (exponent64 << 52) |
         (static_cast<std::uint64_t>(remainder) << (52 - leading));
}

/** Exact branch-free construction of the final FP64 representation. */
template <typename Format>
__device__ __forceinline__ std::uint64_t
direct_fp64_decode_bits(std::uint32_t raw) {
  static_assert(supported_format_v<Format>);
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_bits = Format::exponent_bits;
  constexpr auto exponent_mask = (std::uint32_t{1} << exponent_bits) - 1;
  constexpr auto fraction_mask = (std::uint32_t{1} << fraction_bits) - 1;
  constexpr auto exponent_bias = (1 << (exponent_bits - 1)) - 1;

  raw &= 0xffu;
  const auto sign_bits = static_cast<std::uint64_t>(raw >> 7) << 63;
  const auto exponent = (raw >> fraction_bits) & exponent_mask;
  const auto fraction = raw & fraction_mask;

  const auto subnormal_bits = subnormal_magnitude_bits<Format>(fraction);
  const auto normal_exponent = static_cast<std::uint64_t>(
      static_cast<int>(exponent) - exponent_bias + 1023);
  const auto normal_bits =
      (normal_exponent << 52) |
      (static_cast<std::uint64_t>(fraction) << (52 - fraction_bits));
  const auto special_bits =
      (std::uint64_t{0x7ff} << 52) |
      (static_cast<std::uint64_t>(fraction) << (52 - fraction_bits));

  const auto subnormal_mask =
      std::uint64_t{0} -
      static_cast<std::uint64_t>(exponent == 0 && fraction != 0);
  const auto normal_mask =
      std::uint64_t{0} -
      static_cast<std::uint64_t>(exponent != 0 && exponent != exponent_mask);
  const auto special_mask =
      std::uint64_t{0} - static_cast<std::uint64_t>(exponent == exponent_mask);
  const auto magnitude = (subnormal_bits & subnormal_mask) |
                         (normal_bits & normal_mask) |
                         (special_bits & special_mask);
  return sign_bits | magnitude;
}

__device__ __forceinline__ double fp64_from_bits(std::uint64_t bits) {
  return __longlong_as_double(static_cast<long long>(bits));
}

__device__ __forceinline__ double fp64_from_words(std::uint32_t high,
                                                  std::uint32_t low = 0) {
  return __hiloint2double(static_cast<int>(high), static_cast<int>(low));
}

template <typename Format>
__device__ __forceinline__ std::uint32_t fp64_sign_high(std::uint32_t raw) {
  static_assert(supported_format_v<Format>);
  return (raw & 0x80u) << 24;
}

template <typename Format>
__device__ __forceinline__ std::uint32_t
normal_magnitude_high(std::uint32_t exponent, std::uint32_t fraction) {
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_bits = Format::exponent_bits;
  constexpr auto exponent_bias = (1 << (exponent_bits - 1)) - 1;
  const auto exponent64 = static_cast<std::uint32_t>(
      static_cast<int>(exponent) - exponent_bias + 1023);
  return (exponent64 << 20) | (fraction << (20 - fraction_bits));
}

template <typename Format>
__device__ __forceinline__ std::uint32_t
special_magnitude_high(std::uint32_t fraction) {
  return 0x7ff00000u | (fraction << (20 - Format::fraction_bits));
}

template <typename Format>
__device__ __forceinline__ std::uint32_t
subnormal_magnitude_high_masked(std::uint32_t fraction) {
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_bits = Format::exponent_bits;
  constexpr auto exponent_bias = (1 << (exponent_bits - 1)) - 1;
  const auto safe_fraction = fraction | 1u;
  const auto leading = 31 - __clz(safe_fraction);
  const auto unbiased = leading + 1 - exponent_bias - fraction_bits;
  const auto exponent64 = static_cast<std::uint32_t>(unbiased + 1023);
  const auto remainder = fraction - (std::uint32_t{1} << leading);
  const auto magnitude = (exponent64 << 20) | (remainder << (20 - leading));
  const auto nonzero_mask = 0u - static_cast<std::uint32_t>(fraction != 0);
  return magnitude & nonzero_mask;
}

template <typename Format>
__device__ __forceinline__ std::uint32_t
subnormal_magnitude_high_branchy(std::uint32_t fraction) {
  if (fraction == 0) {
    return 0;
  }
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_bits = Format::exponent_bits;
  constexpr auto exponent_bias = (1 << (exponent_bits - 1)) - 1;
  const auto leading = 31 - __clz(fraction);
  const auto unbiased = leading + 1 - exponent_bias - fraction_bits;
  const auto exponent64 = static_cast<std::uint32_t>(unbiased + 1023);
  const auto remainder = fraction - (std::uint32_t{1} << leading);
  return (exponent64 << 20) | (remainder << (20 - leading));
}

template <typename Format>
__device__ __forceinline__ double
direct_fp64_words_branchy_decode(std::uint32_t raw) {
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_mask =
      (std::uint32_t{1} << Format::exponent_bits) - 1;
  constexpr auto fraction_mask = (std::uint32_t{1} << fraction_bits) - 1;
  raw &= 0xffu;
  const auto exponent = (raw >> fraction_bits) & exponent_mask;
  const auto fraction = raw & fraction_mask;
  std::uint32_t magnitude{};
  if (exponent == 0) {
    magnitude = subnormal_magnitude_high_branchy<Format>(fraction);
  } else if (exponent == exponent_mask) {
    magnitude = special_magnitude_high<Format>(fraction);
  } else {
    magnitude = normal_magnitude_high<Format>(exponent, fraction);
  }
  return fp64_from_words(fp64_sign_high<Format>(raw) | magnitude);
}

template <typename Format>
__device__ __forceinline__ double
direct_fp64_words_masked_decode(std::uint32_t raw) {
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_mask =
      (std::uint32_t{1} << Format::exponent_bits) - 1;
  constexpr auto fraction_mask = (std::uint32_t{1} << fraction_bits) - 1;
  raw &= 0xffu;
  const auto exponent = (raw >> fraction_bits) & exponent_mask;
  const auto fraction = raw & fraction_mask;
  const auto subnormal = subnormal_magnitude_high_masked<Format>(fraction);
  const auto normal = normal_magnitude_high<Format>(exponent, fraction);
  const auto special = special_magnitude_high<Format>(fraction);
  const auto subnormal_mask = 0u - static_cast<std::uint32_t>(exponent == 0);
  const auto normal_mask = 0u - static_cast<std::uint32_t>(
                                    exponent != 0 && exponent != exponent_mask);
  const auto special_mask =
      0u - static_cast<std::uint32_t>(exponent == exponent_mask);
  const auto magnitude = (subnormal & subnormal_mask) | (normal & normal_mask) |
                         (special & special_mask);
  return fp64_from_words(fp64_sign_high<Format>(raw) | magnitude);
}

template <typename Format>
__device__ __forceinline__ double
decomposed_decode(std::uint32_t raw, const std::uint64_t *exponent_prefix) {
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_bits = Format::exponent_bits;
  constexpr auto exponent_mask = (std::uint32_t{1} << exponent_bits) - 1;
  constexpr auto fraction_mask = (std::uint32_t{1} << fraction_bits) - 1;

  raw &= 0xffu;
  const auto exponent = (raw >> fraction_bits) & exponent_mask;
  const auto fraction = raw & fraction_mask;
  const auto regular_bits =
      static_cast<std::uint64_t>(
          __ldg(reinterpret_cast<const unsigned long long *>(exponent_prefix) +
                (raw >> fraction_bits))) |
      (static_cast<std::uint64_t>(fraction) << (52 - fraction_bits));
  const auto sign_bits = static_cast<std::uint64_t>(raw >> 7) << 63;
  const auto subnormal_nonzero =
      std::uint64_t{0} - static_cast<std::uint64_t>(fraction != 0);
  const auto subnormal_bits =
      sign_bits |
      (subnormal_magnitude_bits<Format>(fraction) & subnormal_nonzero);
  const auto exponent_zero_mask =
      std::uint64_t{0} - static_cast<std::uint64_t>(exponent == 0);
  return fp64_from_bits((subnormal_bits & exponent_zero_mask) |
                        (regular_bits & ~exponent_zero_mask));
}

template <typename Format, typename Strategy>
__device__ __forceinline__ table_bundle
stage_shared_table(table_bundle tables, std::uint64_t *shared_words) {
  if constexpr (Strategy::location == table_location::shared) {
    static_assert(Strategy::kind == decode_kind::lut_fp32 ||
                  Strategy::kind == decode_kind::lut_fp64 ||
                  Strategy::kind == decode_kind::lut_prefix ||
                  Strategy::kind == decode_kind::lut_subnormal ||
                  Strategy::kind == decode_kind::lut_high_word ||
                  Strategy::kind == decode_kind::lut_high_word_swizzled);
    if constexpr (Strategy::kind == decode_kind::lut_fp32) {
      auto *destination = reinterpret_cast<float *>(shared_words);
      destination[threadIdx.x] = __ldg(tables.fp32 + threadIdx.x);
      tables.fp32 = destination;
    } else if constexpr (Strategy::kind == decode_kind::lut_fp64) {
      auto *destination = reinterpret_cast<double *>(shared_words);
      destination[threadIdx.x] = __ldg(tables.fp64 + threadIdx.x);
      tables.fp64 = destination;
    } else if constexpr (Format::fraction_bits == 4) {
      if constexpr (Strategy::kind == decode_kind::lut_prefix) {
        auto *destination = reinterpret_cast<std::uint16_t *>(shared_words);
        destination[threadIdx.x] = __ldg(tables.prefix16 + threadIdx.x);
        tables.prefix16 = destination;
      } else if constexpr (Strategy::kind == decode_kind::lut_subnormal) {
        auto *destination = reinterpret_cast<std::uint32_t *>(shared_words);
        constexpr auto count = std::size_t{1} << Format::fraction_bits;
        if (threadIdx.x < count) {
          destination[threadIdx.x] =
              __ldg(tables.subnormal_high_word + threadIdx.x);
        }
        tables.subnormal_high_word = destination;
      } else if constexpr (Strategy::kind == decode_kind::lut_high_word) {
        auto *destination = reinterpret_cast<std::uint32_t *>(shared_words);
        destination[threadIdx.x] = __ldg(tables.high_word + threadIdx.x);
        tables.high_word = destination;
      } else {
        static_assert(Strategy::kind == decode_kind::lut_high_word_swizzled);
        auto *destination = reinterpret_cast<std::uint32_t *>(shared_words);
        const auto value = __ldg(tables.high_word + threadIdx.x);
#pragma unroll
        for (int copy = 0; copy < 4; ++copy) {
          destination[copy * 257 + threadIdx.x] = value;
        }
        tables.high_word = destination;
      }
    } else if constexpr (Strategy::kind == decode_kind::lut_prefix) {
      auto *destination = reinterpret_cast<std::uint32_t *>(shared_words);
      destination[threadIdx.x] = __ldg(tables.prefix32 + threadIdx.x);
      tables.prefix32 = destination;
    } else if constexpr (Strategy::kind == decode_kind::lut_subnormal) {
      auto *destination = reinterpret_cast<std::uint32_t *>(shared_words);
      constexpr auto count = std::size_t{1} << Format::fraction_bits;
      if (threadIdx.x < count) {
        destination[threadIdx.x] =
            __ldg(tables.subnormal_high_word + threadIdx.x);
      }
      tables.subnormal_high_word = destination;
    } else if constexpr (Strategy::kind == decode_kind::lut_high_word) {
      auto *destination = reinterpret_cast<std::uint32_t *>(shared_words);
      destination[threadIdx.x] = __ldg(tables.high_word + threadIdx.x);
      tables.high_word = destination;
    } else {
      static_assert(Strategy::kind == decode_kind::lut_high_word_swizzled);
      auto *destination = reinterpret_cast<std::uint32_t *>(shared_words);
      const auto value = __ldg(tables.high_word + threadIdx.x);
#pragma unroll
      for (int copy = 0; copy < 4; ++copy) {
        destination[copy * 257 + threadIdx.x] = value;
      }
      tables.high_word = destination;
    }
    __syncthreads();
  }
  return tables;
}

template <typename Format, typename Strategy>
__device__ __forceinline__ double decode_raw(std::uint32_t raw,
                                             table_bundle tables) {
  static_assert(supported_format_v<Format>);
  if constexpr (Strategy::kind == decode_kind::generic_fp64) {
    return storage::decode<Format>(static_cast<std::uint8_t>(raw));
  } else if constexpr (Strategy::kind == decode_kind::branchless_fp32) {
    return static_cast<double>(branchless_fp32_decode<Format>(raw));
  } else if constexpr (Strategy::kind == decode_kind::lut_fp32) {
    if constexpr (Strategy::location == table_location::shared) {
      return static_cast<double>(tables.fp32[raw & 0xffu]);
    } else {
      return static_cast<double>(__ldg(tables.fp32 + (raw & 0xffu)));
    }
  } else if constexpr (Strategy::kind == decode_kind::lut_fp64) {
    if constexpr (Strategy::location == table_location::shared) {
      return tables.fp64[raw & 0xffu];
    } else {
      return __ldg(tables.fp64 + (raw & 0xffu));
    }
  } else if constexpr (Strategy::kind == decode_kind::lut_prefix) {
    if constexpr (Format::fraction_bits == 4) {
      std::uint16_t prefix{};
      if constexpr (Strategy::location == table_location::shared) {
        prefix = tables.prefix16[raw & 0xffu];
      } else {
        prefix = __ldg(tables.prefix16 + (raw & 0xffu));
      }
      return fp64_from_bits(static_cast<std::uint64_t>(prefix) << 48);
    } else {
      std::uint32_t prefix{};
      if constexpr (Strategy::location == table_location::shared) {
        prefix = tables.prefix32[raw & 0xffu];
      } else {
        prefix = __ldg(tables.prefix32 + (raw & 0xffu));
      }
      return fp64_from_bits(static_cast<std::uint64_t>(prefix) << 47);
    }
  } else if constexpr (Strategy::kind ==
                       decode_kind::direct_fp64_words_branchy) {
    return direct_fp64_words_branchy_decode<Format>(raw);
  } else if constexpr (Strategy::kind ==
                       decode_kind::direct_fp64_words_masked) {
    return direct_fp64_words_masked_decode<Format>(raw);
  } else if constexpr (Strategy::kind == decode_kind::lut_subnormal) {
    constexpr auto fraction_bits = Format::fraction_bits;
    constexpr auto exponent_mask =
        (std::uint32_t{1} << Format::exponent_bits) - 1;
    constexpr auto fraction_mask = (std::uint32_t{1} << fraction_bits) - 1;
    raw &= 0xffu;
    const auto exponent = (raw >> fraction_bits) & exponent_mask;
    const auto fraction = raw & fraction_mask;
    std::uint32_t magnitude{};
    if (exponent == 0) {
      if constexpr (Strategy::location == table_location::shared) {
        magnitude = tables.subnormal_high_word[fraction];
      } else {
        magnitude = __ldg(tables.subnormal_high_word + fraction);
      }
    } else if (exponent == exponent_mask) {
      magnitude = special_magnitude_high<Format>(fraction);
    } else {
      magnitude = normal_magnitude_high<Format>(exponent, fraction);
    }
    return fp64_from_words(fp64_sign_high<Format>(raw) | magnitude);
  } else if constexpr (Strategy::kind == decode_kind::lut_high_word) {
    std::uint32_t high{};
    if constexpr (Strategy::location == table_location::shared) {
      high = tables.high_word[raw & 0xffu];
    } else {
      high = __ldg(tables.high_word + (raw & 0xffu));
    }
    return fp64_from_words(high);
  } else if constexpr (Strategy::kind == decode_kind::lut_high_word_swizzled) {
    static_assert(Strategy::location == table_location::shared);
    const auto warp_lane = threadIdx.x & 31u;
    const auto copy = warp_lane >> 3;
    return fp64_from_words(tables.high_word[copy * 257 + (raw & 0xffu)]);
  } else if constexpr (Strategy::kind == decode_kind::direct_fp64_bits) {
    return fp64_from_bits(direct_fp64_decode_bits<Format>(raw));
  } else {
    static_assert(Strategy::kind == decode_kind::decomposed_bits);
    return decomposed_decode<Format>(raw, tables.exponent_prefix);
  }
}

template <typename Format, typename Strategy, std::size_t... Lane>
__device__ __forceinline__ decoded_packet<Strategy::lanes>
decode_packet_impl(source_packet<Strategy::lanes> packet, table_bundle tables,
                   std::index_sequence<Lane...>) {
  return {{decode_raw<Format, Strategy>(
      raw_lane<static_cast<int>(Lane)>(packet), tables)...}};
}

template <typename Format, typename Strategy>
__device__ __forceinline__ decoded_packet<Strategy::lanes>
decode_packet(source_packet<Strategy::lanes> packet, table_bundle tables) {
  return decode_packet_impl<Format, Strategy>(
      packet, tables, std::make_index_sequence<Strategy::lanes>{});
}

template <int Lanes>
__device__ __forceinline__ void
accumulate_values(double (&sums)[Lanes], const decoded_packet<Lanes> &values) {
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    sums[lane] += values.values[lane];
  }
}

template <int Lanes>
__device__ __forceinline__ void
accumulate_products(double (&sums)[Lanes], const decoded_packet<Lanes> &left,
                    const decoded_packet<Lanes> &right) {
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    sums[lane] = fma(left.values[lane], right.values[lane], sums[lane]);
  }
}

template <int Lanes>
__device__ __forceinline__ double combine(double (&sums)[Lanes]) {
  double result{};
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    result += sums[lane];
  }
  return result;
}

template <typename Format, typename Strategy>
__global__ void decode_all_codes(const std::uint8_t *codes, table_bundle tables,
                                 double *output) {
  extern __shared__ std::uint64_t shared_words[];
  tables = stage_shared_table<Format, Strategy>(tables, shared_words);
  constexpr auto lanes = Strategy::lanes;
  const auto pack = static_cast<std::size_t>(threadIdx.x);
  if (pack * lanes >= 256) {
    return;
  }
  const auto values = decode_packet<Format, Strategy>(
      load_source_packet<lanes>(codes + pack * lanes), tables);
#pragma unroll
  for (int lane = 0; lane < lanes; ++lane) {
    output[pack * lanes + lane] = values.values[lane];
  }
}

template <typename Format, typename Strategy>
__global__ void dot_map_reduce(const std::uint8_t *left,
                               const std::uint8_t *right, std::size_t count,
                               table_bundle tables, double *partials) {
  using block_reduce = cub::BlockReduce<double, block_threads>;
  __shared__ typename block_reduce::TempStorage reduction_storage;
  extern __shared__ std::uint64_t shared_words[];
  tables = stage_shared_table<Format, Strategy>(tables, shared_words);

  constexpr auto lanes = Strategy::lanes;
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto pack_count = count / lanes;
  double sums[lanes]{};

  if constexpr (Strategy::pipelined) {
    auto pack = first;
    if (pack < pack_count) {
      auto left_packet = load_source_packet<lanes>(left + pack * lanes);
      auto right_packet = load_source_packet<lanes>(right + pack * lanes);
      while (true) {
        const auto next_pack = pack + stride;
        source_packet<lanes> next_left{};
        source_packet<lanes> next_right{};
        if (next_pack < pack_count) {
          next_left = load_source_packet<lanes>(left + next_pack * lanes);
          next_right = load_source_packet<lanes>(right + next_pack * lanes);
        }
        accumulate_products(
            sums, decode_packet<Format, Strategy>(left_packet, tables),
            decode_packet<Format, Strategy>(right_packet, tables));
        if (next_pack >= pack_count) {
          break;
        }
        pack = next_pack;
        left_packet = next_left;
        right_packet = next_right;
      }
    }
  } else {
    for (auto pack = first; pack < pack_count; pack += stride) {
      accumulate_products(
          sums,
          decode_packet<Format, Strategy>(
              load_source_packet<lanes>(left + pack * lanes), tables),
          decode_packet<Format, Strategy>(
              load_source_packet<lanes>(right + pack * lanes), tables));
    }
  }

  const auto tail = static_cast<std::size_t>(lanes) * pack_count + first;
  if (tail < count) {
    sums[0] = fma(decode_raw<Format, Strategy>(left[tail], tables),
                  decode_raw<Format, Strategy>(right[tail], tables), sums[0]);
  }
  const auto reduced = block_reduce(reduction_storage).Sum(combine(sums));
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename Format, typename Strategy>
__global__ void gemv(const std::uint8_t *matrix, const std::uint8_t *vector,
                     std::size_t rows, std::size_t columns,
                     std::size_t leading_dimension, table_bundle tables,
                     double *result) {
  using block_reduce = cub::BlockReduce<double, block_threads>;
  __shared__ typename block_reduce::TempStorage reduction_storage;
  extern __shared__ std::uint64_t shared_words[];
  tables = stage_shared_table<Format, Strategy>(tables, shared_words);

  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  constexpr auto lanes = Strategy::lanes;
  const auto *row_values = matrix + row * leading_dimension;
  const auto pack_count = columns / lanes;
  const auto first = static_cast<std::size_t>(threadIdx.x);
  const auto stride = static_cast<std::size_t>(blockDim.x);
  double sums[lanes]{};

  if constexpr (Strategy::pipelined) {
    auto pack = first;
    if (pack < pack_count) {
      auto matrix_packet = load_source_packet<lanes>(row_values + pack * lanes);
      auto vector_packet = load_source_packet<lanes>(vector + pack * lanes);
      while (true) {
        const auto next_pack = pack + stride;
        source_packet<lanes> next_matrix{};
        source_packet<lanes> next_vector{};
        if (next_pack < pack_count) {
          next_matrix =
              load_source_packet<lanes>(row_values + next_pack * lanes);
          next_vector = load_source_packet<lanes>(vector + next_pack * lanes);
        }
        accumulate_products(
            sums, decode_packet<Format, Strategy>(matrix_packet, tables),
            decode_packet<Format, Strategy>(vector_packet, tables));
        if (next_pack >= pack_count) {
          break;
        }
        pack = next_pack;
        matrix_packet = next_matrix;
        vector_packet = next_vector;
      }
    }
  } else {
    for (auto pack = first; pack < pack_count; pack += stride) {
      accumulate_products(
          sums,
          decode_packet<Format, Strategy>(
              load_source_packet<lanes>(row_values + pack * lanes), tables),
          decode_packet<Format, Strategy>(
              load_source_packet<lanes>(vector + pack * lanes), tables));
    }
  }

  const auto tail = static_cast<std::size_t>(lanes) * pack_count + first;
  if (tail < columns) {
    sums[0] = fma(decode_raw<Format, Strategy>(row_values[tail], tables),
                  decode_raw<Format, Strategy>(vector[tail], tables), sums[0]);
  }
  const auto reduced = block_reduce(reduction_storage).Sum(combine(sums));
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
}

} // namespace aut::e2e3_strategies

#endif // ACCESSOR_UNIVERSAL_TEST_E2E3_DECODER_STRATEGIES_CUH_
