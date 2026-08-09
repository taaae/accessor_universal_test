#ifndef ACCESSOR_UNIVERSAL_TEST_FORMAT_DECODER_STRATEGIES_CUH_
#define ACCESSOR_UNIVERSAL_TEST_FORMAT_DECODER_STRATEGIES_CUH_

#include "cuda_storage_formats.cuh"
#include "decoder_strategy_core.hpp"

#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace aut::format_strategies {

inline constexpr int block_threads = 256;

enum class decode_kind {
  generic,
  direct_words_branchy,
  direct_words_masked,
  fp32_bits,
  fixed_integer,
  e1_integer,
  exponent_only,
  prefix_word,
  full_high_lut,
  subnormal_high_lut,
  prefix_high_lut,
  native_direct,
  native_fp32,
  native_packed,
  native_half2,
  pair_high_lut,
  quad_high_lut,
  warp_high_lut,
  full_high_lut_swizzled,
};

enum class table_location { global_read_only, shared };
enum class unpack_kind { shift_mask, byte_permute };

template <decode_kind Kind, int Lanes,
          table_location Location = table_location::global_read_only,
          unpack_kind Unpack = unpack_kind::shift_mask>
struct strategy {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8);
  static constexpr decode_kind kind = Kind;
  static constexpr int lanes = Lanes;
  static constexpr table_location location = Location;
  static constexpr unpack_kind unpack = Unpack;
};

template <typename Format> struct format_policy {
  static constexpr auto value =
      Format::exponent_bits == 0
          ? decoder::special_policy::fixed_e0
          : (Format::exponent_bits == 11
                 ? decoder::special_policy::fp64_prefix
                 : (Format::finite ? decoder::special_policy::finite_all
                                   : decoder::special_policy::ieee));
};

template <> struct format_policy<storage::fp8_e4m3> {
  static constexpr auto value = decoder::special_policy::e4m3fn;
};

template <typename Format> struct format_layout {
  using type = decoder::binary_layout<Format::total_bits,
                                      Format::exponent_bits,
                                      format_policy<Format>::value>;
};
template <typename Format>
using format_layout_t = typename format_layout<Format>::type;

struct table_bundle {
  const std::uint32_t *full_high{};
  const std::uint32_t *subnormal_high{};
  const std::uint32_t *prefix_high{};
  const uint2 *pair_high{};
  const uint4 *quad_high{};
  std::uint32_t warp_high{};
};

template <typename Strategy>
inline constexpr bool uses_table_v =
    Strategy::kind == decode_kind::full_high_lut ||
    Strategy::kind == decode_kind::subnormal_high_lut ||
    Strategy::kind == decode_kind::prefix_high_lut ||
    Strategy::kind == decode_kind::pair_high_lut ||
    Strategy::kind == decode_kind::quad_high_lut ||
    Strategy::kind == decode_kind::warp_high_lut ||
    Strategy::kind == decode_kind::full_high_lut_swizzled;

template <typename Format, typename Strategy>
inline constexpr std::size_t table_entries_v = [] {
  using layout = format_layout_t<Format>;
  if constexpr (Strategy::kind == decode_kind::full_high_lut) {
    return std::size_t{1} << layout::total_bits;
  } else if constexpr (Strategy::kind ==
                       decode_kind::full_high_lut_swizzled) {
    static_assert(layout::total_bits == 8);
    return std::size_t{1} << 8;
  } else if constexpr (Strategy::kind == decode_kind::subnormal_high_lut) {
    return std::size_t{1} << layout::fraction_bits;
  } else if constexpr (Strategy::kind == decode_kind::prefix_high_lut) {
    return std::size_t{1} << (layout::exponent_bits + 1);
  } else if constexpr (Strategy::kind == decode_kind::pair_high_lut) {
    return std::size_t{1} << (2 * layout::total_bits);
  } else if constexpr (Strategy::kind == decode_kind::quad_high_lut) {
    static_assert(layout::total_bits == 2);
    return std::size_t{1} << 8;
  } else if constexpr (Strategy::kind == decode_kind::warp_high_lut) {
    static_assert(layout::total_bits <= 4);
    return std::size_t{1} << layout::total_bits;
  } else {
    return std::size_t{0};
  }
}();

template <typename Format, typename Strategy>
inline constexpr std::size_t shared_table_bytes_v =
    Strategy::location != table_location::shared
        ? std::size_t{0}
        : (Strategy::kind == decode_kind::full_high_lut_swizzled
               ? 4 * 257 * sizeof(std::uint32_t)
               : (Strategy::kind == decode_kind::pair_high_lut
                      ? table_entries_v<Format, Strategy> * sizeof(uint2)
                      : (Strategy::kind == decode_kind::quad_high_lut
                             ? table_entries_v<Format, Strategy> *
                                   sizeof(uint4)
                             : table_entries_v<Format, Strategy> *
                                   sizeof(std::uint32_t))));

template <int Lanes> struct source_packet {
  std::uint32_t words[8]{};
};

template <typename Format>
using device_storage_t =
    std::conditional_t<(Format::total_bits < 8), std::uint8_t,
                       storage::storage_type_t<Format>>;

template <typename Format>
inline constexpr std::size_t packed_storage_count(std::size_t logical_count) {
  if constexpr (Format::total_bits < 8) {
    return (logical_count * Format::total_bits + 7) / 8;
  } else {
    return logical_count;
  }
}

template <typename Format, int Lanes>
__device__ __forceinline__ source_packet<Lanes>
load_source_packet(const device_storage_t<Format> *values,
                   std::size_t logical_offset) {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8);
  constexpr auto bytes = sizeof(storage::storage_type_t<Format>);
  source_packet<Lanes> result{};
  if constexpr (Format::total_bits < 8) {
    constexpr auto packet_bits = Format::total_bits * Lanes;
    const auto bit_offset = logical_offset * Format::total_bits;
    const auto *raw = values + bit_offset / 8;
    const auto shift = static_cast<unsigned>(bit_offset % 8);
    if (shift != 0) {
      // Row starts need not be byte aligned. Gather at most five bytes, then
      // normalize the requested packet to the low bits of words[0].
      constexpr auto maximum_bytes = (packet_bits + 7 + 7) / 8;
      const auto bytes_needed = (shift + packet_bits + 7) / 8;
      std::uint64_t gathered{};
#pragma unroll
      for (int byte = 0; byte < maximum_bytes; ++byte) {
        if (byte < bytes_needed) {
          gathered |= static_cast<std::uint64_t>(raw[byte]) << (8 * byte);
        }
      }
      result.words[0] = static_cast<std::uint32_t>(gathered >> shift);
    } else if constexpr (packet_bits <= 8) {
      result.words[0] = raw[0];
    } else if constexpr (packet_bits <= 16) {
      result.words[0] = *reinterpret_cast<const std::uint16_t *>(raw);
    } else {
      static_assert(packet_bits <= 32);
      result.words[0] = *reinterpret_cast<const std::uint32_t *>(raw);
    }
  } else if constexpr (bytes == 1) {
    const auto *raw = reinterpret_cast<const std::uint8_t *>(values +
                                                             logical_offset);
    if constexpr (Lanes == 1) {
      result.words[0] = raw[0];
    } else if constexpr (Lanes == 2) {
      result.words[0] = *reinterpret_cast<const std::uint16_t *>(raw);
    } else if constexpr (Lanes == 4) {
      result.words[0] = *reinterpret_cast<const std::uint32_t *>(raw);
    } else {
      const auto packed = *reinterpret_cast<const uint2 *>(raw);
      result.words[0] = packed.x;
      result.words[1] = packed.y;
    }
  } else if constexpr (bytes == 2) {
    const auto *raw = reinterpret_cast<const std::uint8_t *>(values +
                                                             logical_offset);
    if constexpr (Lanes == 1) {
      result.words[0] = *reinterpret_cast<const std::uint16_t *>(raw);
    } else if constexpr (Lanes == 2) {
      result.words[0] = *reinterpret_cast<const std::uint32_t *>(raw);
    } else if constexpr (Lanes == 4) {
      const auto packed = *reinterpret_cast<const uint2 *>(raw);
      result.words[0] = packed.x;
      result.words[1] = packed.y;
    } else {
      const auto packed = *reinterpret_cast<const uint4 *>(raw);
      result.words[0] = packed.x;
      result.words[1] = packed.y;
      result.words[2] = packed.z;
      result.words[3] = packed.w;
    }
  } else {
    static_assert(bytes == 4);
    const auto *raw = reinterpret_cast<const std::uint8_t *>(values +
                                                             logical_offset);
    if constexpr (Lanes == 1) {
      result.words[0] = *reinterpret_cast<const std::uint32_t *>(raw);
    } else if constexpr (Lanes == 2) {
      const auto packed = *reinterpret_cast<const uint2 *>(raw);
      result.words[0] = packed.x;
      result.words[1] = packed.y;
    } else if constexpr (Lanes == 4) {
      const auto packed = *reinterpret_cast<const uint4 *>(raw);
      result.words[0] = packed.x;
      result.words[1] = packed.y;
      result.words[2] = packed.z;
      result.words[3] = packed.w;
    } else {
      const auto low = *reinterpret_cast<const uint4 *>(raw);
      const auto high = *reinterpret_cast<const uint4 *>(raw + 16);
      result.words[0] = low.x;
      result.words[1] = low.y;
      result.words[2] = low.z;
      result.words[3] = low.w;
      result.words[4] = high.x;
      result.words[5] = high.y;
      result.words[6] = high.z;
      result.words[7] = high.w;
    }
  }
  return result;
}

template <typename Format, int Lane, int Lanes,
          unpack_kind Unpack = unpack_kind::shift_mask>
__device__ __forceinline__ std::uint32_t
raw_lane(const source_packet<Lanes> &packet) {
  constexpr auto bytes = sizeof(storage::storage_type_t<Format>);
  static_assert(Lane >= 0 && Lane < Lanes);
  if constexpr (Format::total_bits < 8) {
    return (packet.words[0] >> (Format::total_bits * Lane)) &
           ((std::uint32_t{1} << Format::total_bits) - 1);
  } else if constexpr (bytes == 1) {
    if constexpr (Unpack == unpack_kind::byte_permute) {
      return __byte_perm(packet.words[Lane / 4], 0u,
                         0x4440u | static_cast<unsigned>(Lane % 4));
    } else {
      return (packet.words[Lane / 4] >> (8 * (Lane % 4))) & 0xffu;
    }
  } else if constexpr (bytes == 2) {
    return (packet.words[Lane / 2] >> (16 * (Lane % 2))) & 0xffffu;
  } else {
    return packet.words[Lane];
  }
}

template <typename Format>
__device__ __forceinline__ std::uint32_t
raw_from_storage(storage::storage_type_t<Format> value);

template <typename Format>
__device__ __forceinline__ std::uint32_t
load_raw_code(const device_storage_t<Format> *values,
              std::size_t logical_index) {
  if constexpr (Format::total_bits < 8) {
    const auto bit_offset = logical_index * Format::total_bits;
    return (values[bit_offset / 8] >> (bit_offset % 8)) &
           ((std::uint32_t{1} << Format::total_bits) - 1);
  } else {
    return raw_from_storage<Format>(values[logical_index]);
  }
}

template <typename Format>
__device__ __forceinline__ storage::storage_type_t<Format>
storage_from_raw(std::uint32_t raw) {
  static_assert(std::is_integral_v<storage::storage_type_t<Format>>);
  return static_cast<storage::storage_type_t<Format>>(raw);
}

template <typename Format>
__device__ __forceinline__ std::uint32_t
raw_from_storage(storage::storage_type_t<Format> value) {
  static_assert(std::is_integral_v<storage::storage_type_t<Format>>);
  return static_cast<std::uint32_t>(value);
}

template <>
__device__ __forceinline__ storage::storage_type_t<storage::fp8_e4m3>
storage_from_raw<storage::fp8_e4m3>(std::uint32_t raw) {
  storage::storage_type_t<storage::fp8_e4m3> value;
  value.__x = static_cast<__nv_fp8_storage_t>(raw);
  return value;
}

template <>
__device__ __forceinline__ std::uint32_t raw_from_storage<storage::fp8_e4m3>(
    storage::storage_type_t<storage::fp8_e4m3> value) {
  return static_cast<std::uint32_t>(value.__x);
}

template <>
__device__ __forceinline__ storage::storage_type_t<storage::fp8_e5m2>
storage_from_raw<storage::fp8_e5m2>(std::uint32_t raw) {
  storage::storage_type_t<storage::fp8_e5m2> value;
  value.__x = static_cast<__nv_fp8_storage_t>(raw);
  return value;
}

template <>
__device__ __forceinline__ std::uint32_t raw_from_storage<storage::fp8_e5m2>(
    storage::storage_type_t<storage::fp8_e5m2> value) {
  return static_cast<std::uint32_t>(value.__x);
}

template <>
__device__ __forceinline__ storage::storage_type_t<storage::fp16_e5m10>
storage_from_raw<storage::fp16_e5m10>(std::uint32_t raw) {
  return __half{__half_raw{static_cast<unsigned short>(raw)}};
}

template <>
__device__ __forceinline__ std::uint32_t raw_from_storage<storage::fp16_e5m10>(
    storage::storage_type_t<storage::fp16_e5m10> value) {
  return static_cast<std::uint32_t>(__half_as_ushort(value));
}

template <>
__device__ __forceinline__ storage::storage_type_t<storage::bf16_e8m7>
storage_from_raw<storage::bf16_e8m7>(std::uint32_t raw) {
  return __nv_bfloat16{
      __nv_bfloat16_raw{static_cast<unsigned short>(raw)}};
}

template <>
__device__ __forceinline__ std::uint32_t raw_from_storage<storage::bf16_e8m7>(
    storage::storage_type_t<storage::bf16_e8m7> value) {
  return static_cast<std::uint32_t>(__bfloat16_as_ushort(value));
}

template <>
__device__ __forceinline__ storage::storage_type_t<storage::fp32_e8m23>
storage_from_raw<storage::fp32_e8m23>(std::uint32_t raw) {
  return __uint_as_float(raw);
}

template <>
__device__ __forceinline__ std::uint32_t raw_from_storage<storage::fp32_e8m23>(
    storage::storage_type_t<storage::fp32_e8m23> value) {
  return __float_as_uint(value);
}

template <typename Format, typename Strategy>
__device__ __forceinline__ table_bundle
stage_shared_table(table_bundle tables, std::uint32_t *shared) {
  if constexpr (Strategy::location == table_location::shared) {
    static_assert(uses_table_v<Strategy>);
    if constexpr (Strategy::kind == decode_kind::pair_high_lut) {
      auto *target = reinterpret_cast<uint2 *>(shared);
      for (std::size_t i = threadIdx.x; i < table_entries_v<Format, Strategy>;
           i += blockDim.x) {
        target[i] = __ldg(tables.pair_high + i);
      }
      __syncthreads();
      tables.pair_high = target;
      return tables;
    } else if constexpr (Strategy::kind == decode_kind::quad_high_lut) {
      auto *target = reinterpret_cast<uint4 *>(shared);
      for (std::size_t i = threadIdx.x; i < table_entries_v<Format, Strategy>;
           i += blockDim.x) {
        target[i] = __ldg(tables.quad_high + i);
      }
      __syncthreads();
      tables.quad_high = target;
      return tables;
    }
    const std::uint32_t *source{};
    if constexpr (Strategy::kind == decode_kind::full_high_lut ||
                  Strategy::kind == decode_kind::full_high_lut_swizzled) {
      source = tables.full_high;
    } else if constexpr (Strategy::kind == decode_kind::subnormal_high_lut) {
      source = tables.subnormal_high;
    } else if constexpr (Strategy::kind == decode_kind::prefix_high_lut) {
      source = tables.prefix_high;
    }
    for (std::size_t i = threadIdx.x; i < table_entries_v<Format, Strategy>;
         i += blockDim.x) {
      const auto value = __ldg(source + i);
      if constexpr (Strategy::kind == decode_kind::full_high_lut_swizzled) {
#pragma unroll
        for (int copy = 0; copy < 4; ++copy) {
          shared[copy * 257 + i] = value;
        }
      } else {
        shared[i] = value;
      }
    }
    __syncthreads();
    if constexpr (Strategy::kind == decode_kind::full_high_lut ||
                  Strategy::kind == decode_kind::full_high_lut_swizzled) {
      tables.full_high = shared;
    } else if constexpr (Strategy::kind == decode_kind::subnormal_high_lut) {
      tables.subnormal_high = shared;
    } else {
      tables.prefix_high = shared;
    }
  }
  return tables;
}

template <typename Format, typename Strategy>
__device__ __forceinline__ table_bundle
prepare_thread_table(table_bundle tables) {
  if constexpr (Strategy::kind == decode_kind::warp_high_lut) {
    constexpr auto entries = 1u << Format::total_bits;
    const auto warp_lane = threadIdx.x & 31u;
    tables.warp_high = warp_lane < entries
                           ? __ldg(tables.full_high + warp_lane)
                           : 0u;
  }
  return tables;
}

template <typename Strategy>
__device__ __forceinline__ std::uint32_t
lookup_high(const std::uint32_t *table, std::size_t index) {
  if constexpr (Strategy::location == table_location::shared) {
    return table[index];
  } else {
    return __ldg(table + index);
  }
}

__device__ __forceinline__ double decode_fp4_native(std::uint32_t raw) {
  const auto half_raw = __nv_cvt_fp4_to_halfraw(
      static_cast<__nv_fp4_storage_t>(raw & 0xfu), __NV_E2M1);
  return static_cast<double>(__half2float(__half{half_raw}));
}

template <typename Format, typename Strategy>
__device__ __forceinline__ double decode_raw(std::uint32_t raw,
                                             table_bundle tables) {
  using layout = format_layout_t<Format>;
  if constexpr (Strategy::kind == decode_kind::generic) {
    return storage::decode<Format>(storage_from_raw<Format>(raw));
  } else if constexpr (Strategy::kind == decode_kind::direct_words_branchy) {
    return decoder::words_to_double(decoder::decode_words_branchy<layout>(raw));
  } else if constexpr (Strategy::kind == decode_kind::direct_words_masked) {
    return decoder::words_to_double(decoder::decode_words_masked<layout>(raw));
  } else if constexpr (Strategy::kind == decode_kind::fp32_bits) {
    return decoder::decode_via_fp32<layout>(raw);
  } else if constexpr (Strategy::kind == decode_kind::fixed_integer) {
    return decoder::decode_fixed_integer<layout>(raw);
  } else if constexpr (Strategy::kind == decode_kind::e1_integer) {
    return decoder::decode_e1_integer<layout>(raw);
  } else if constexpr (Strategy::kind == decode_kind::exponent_only) {
    return decoder::decode_exponent_only<layout>(raw);
  } else if constexpr (Strategy::kind == decode_kind::native_direct) {
    if constexpr (std::is_same_v<Format, storage::fp4_e2m1>) {
      return decode_fp4_native(raw);
    } else {
      return static_cast<double>(storage_from_raw<Format>(raw));
    }
  } else if constexpr (Strategy::kind == decode_kind::native_fp32) {
    if constexpr (std::is_same_v<Format, storage::fp4_e2m1>) {
      return decode_fp4_native(raw);
    } else {
      return static_cast<double>(
          static_cast<float>(storage_from_raw<Format>(raw)));
    }
  } else if constexpr (Strategy::kind == decode_kind::native_packed ||
                       Strategy::kind == decode_kind::native_half2) {
    // Scalar tail fallback for a packed native strategy.
    if constexpr (std::is_same_v<Format, storage::fp4_e2m1>) {
      return decode_fp4_native(raw);
    } else {
      return static_cast<double>(
          static_cast<float>(storage_from_raw<Format>(raw)));
    }
  } else if constexpr (Strategy::kind == decode_kind::prefix_word) {
    return decoder::words_to_double(
        decoder::decode_prefix_words<layout::fraction_bits>(raw));
  } else if constexpr (Strategy::kind == decode_kind::full_high_lut) {
    return decoder::words_to_double(
        {lookup_high<Strategy>(tables.full_high,
                               raw & decoder::raw_mask<layout>()),
         0});
  } else if constexpr (Strategy::kind == decode_kind::warp_high_lut ||
                       Strategy::kind == decode_kind::quad_high_lut) {
    // Scalar/tail fallback. Full packets use the register-shuffle or byte
    // quad implementation below.
    return decoder::words_to_double(
        {__ldg(tables.full_high +
               (raw & decoder::raw_mask<layout>())),
         0});
  } else if constexpr (Strategy::kind ==
                       decode_kind::full_high_lut_swizzled) {
    static_assert(Strategy::location == table_location::shared);
    const auto copy = (threadIdx.x & 31u) >> 3;
    return decoder::words_to_double(
        {tables.full_high[copy * 257 +
                          (raw & decoder::raw_mask<layout>())],
         0});
  } else if constexpr (Strategy::kind == decode_kind::subnormal_high_lut) {
    raw &= decoder::raw_mask<layout>();
    const auto exponent =
        (raw >> layout::fraction_bits) & decoder::exponent_mask<layout>();
    const auto fraction = raw & decoder::fraction_mask<layout>();
    decoder::fp64_words words{};
    if (exponent == 0) {
      words.high = lookup_high<Strategy>(tables.subnormal_high, fraction);
    } else if (decoder::is_special<layout>(exponent, fraction)) {
      words = decoder::special_magnitude_words<layout>(fraction);
    } else {
      words = decoder::normal_magnitude_words<layout>(exponent, fraction);
    }
    words.high |= (raw >> (layout::total_bits - 1)) << 31;
    return decoder::words_to_double(words);
  } else if constexpr (Strategy::kind == decode_kind::pair_high_lut) {
    return decoder::words_to_double(
        {lookup_high<Strategy>(tables.full_high,
                               raw & decoder::raw_mask<layout>()),
         0});
  } else {
    static_assert(Strategy::kind == decode_kind::prefix_high_lut,
                  "native decoders are specialized per format");
    raw &= decoder::raw_mask<layout>();
    const auto exponent =
        (raw >> layout::fraction_bits) & decoder::exponent_mask<layout>();
    const auto fraction = raw & decoder::fraction_mask<layout>();
    if (exponent == 0) {
      auto words = decoder::subnormal_magnitude_words<layout>(fraction);
      words.high |= (raw >> (layout::total_bits - 1)) << 31;
      return decoder::words_to_double(words);
    }
    const auto prefix = raw >> layout::fraction_bits;
    auto high = lookup_high<Strategy>(tables.prefix_high, prefix);
    if constexpr (layout::fraction_bits <= 20) {
      high |= fraction << (20 - layout::fraction_bits);
      return decoder::words_to_double({high, 0});
    } else {
      return decoder::words_to_double(
          {high | (fraction >> (layout::fraction_bits - 20)),
           fraction << (52 - layout::fraction_bits)});
    }
  }
}

template <int Lanes> struct decoded_packet {
  double values[Lanes];
};

template <int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_e4m3_native_packet(const source_packet<Lanes> &packet) {
  decoded_packet<Lanes> result{};
  if constexpr (Lanes == 1) {
    result.values[0] = static_cast<double>(
        static_cast<float>(storage_from_raw<storage::fp8_e4m3>(
            raw_lane<storage::fp8_e4m3, 0>(packet))));
  } else if constexpr (Lanes == 2) {
    __nv_fp8x2_e4m3 packed;
    packed.__x = static_cast<__nv_fp8x2_storage_t>(packet.words[0]);
    const auto values = static_cast<float2>(packed);
    result.values[0] = static_cast<double>(values.x);
    result.values[1] = static_cast<double>(values.y);
  } else {
#pragma unroll
    for (int group = 0; group < Lanes / 4; ++group) {
      __nv_fp8x4_e4m3 packed;
      packed.__x = static_cast<__nv_fp8x4_storage_t>(packet.words[group]);
      const auto values = static_cast<float4>(packed);
      result.values[4 * group] = static_cast<double>(values.x);
      result.values[4 * group + 1] = static_cast<double>(values.y);
      result.values[4 * group + 2] = static_cast<double>(values.z);
      result.values[4 * group + 3] = static_cast<double>(values.w);
    }
  }
  return result;
}

template <int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_e4m3_half2_packet(const source_packet<Lanes> &packet) {
  static_assert(Lanes == 2 || Lanes == 4 || Lanes == 8);
  decoded_packet<Lanes> result{};
#pragma unroll
  for (int group = 0; group < Lanes / 2; ++group) {
    const auto word = packet.words[group / 2] >> (16 * (group % 2));
    __nv_fp8x2_e4m3 packed;
    packed.__x = static_cast<__nv_fp8x2_storage_t>(word);
    const auto values = __half22float2(static_cast<__half2>(packed));
    result.values[2 * group] = static_cast<double>(values.x);
    result.values[2 * group + 1] = static_cast<double>(values.y);
  }
  return result;
}

template <int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_e5m2_native_packet(const source_packet<Lanes> &packet) {
  decoded_packet<Lanes> result{};
  if constexpr (Lanes == 1) {
    result.values[0] = static_cast<double>(
        static_cast<float>(storage_from_raw<storage::fp8_e5m2>(
            raw_lane<storage::fp8_e5m2, 0>(packet))));
  } else if constexpr (Lanes == 2) {
    __nv_fp8x2_e5m2 packed;
    packed.__x = static_cast<__nv_fp8x2_storage_t>(packet.words[0]);
    const auto values = static_cast<float2>(packed);
    result.values[0] = static_cast<double>(values.x);
    result.values[1] = static_cast<double>(values.y);
  } else {
#pragma unroll
    for (int group = 0; group < Lanes / 4; ++group) {
      __nv_fp8x4_e5m2 packed;
      packed.__x = static_cast<__nv_fp8x4_storage_t>(packet.words[group]);
      const auto values = static_cast<float4>(packed);
      result.values[4 * group] = static_cast<double>(values.x);
      result.values[4 * group + 1] = static_cast<double>(values.y);
      result.values[4 * group + 2] = static_cast<double>(values.z);
      result.values[4 * group + 3] = static_cast<double>(values.w);
    }
  }
  return result;
}

template <int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_e5m2_half2_packet(const source_packet<Lanes> &packet) {
  static_assert(Lanes == 2 || Lanes == 4 || Lanes == 8);
  decoded_packet<Lanes> result{};
#pragma unroll
  for (int group = 0; group < Lanes / 2; ++group) {
    const auto word = packet.words[group / 2] >> (16 * (group % 2));
    __nv_fp8x2_e5m2 packed;
    packed.__x = static_cast<__nv_fp8x2_storage_t>(word);
    const auto values = __half22float2(static_cast<__half2>(packed));
    result.values[2 * group] = static_cast<double>(values.x);
    result.values[2 * group + 1] = static_cast<double>(values.y);
  }
  return result;
}

template <int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_fp16_native_packet(const source_packet<Lanes> &packet) {
  decoded_packet<Lanes> result{};
  if constexpr (Lanes == 1) {
    result.values[0] = static_cast<double>(__half2float(
        storage_from_raw<storage::fp16_e5m10>(packet.words[0])));
  } else {
#pragma unroll
    for (int group = 0; group < Lanes / 2; ++group) {
      const auto word = packet.words[group];
      const __half2_raw raw{static_cast<unsigned short>(word),
                            static_cast<unsigned short>(word >> 16)};
      const auto values = __half22float2(__half2{raw});
      result.values[2 * group] = static_cast<double>(values.x);
      result.values[2 * group + 1] = static_cast<double>(values.y);
    }
  }
  return result;
}

template <int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_bf16_native_packet(const source_packet<Lanes> &packet) {
  decoded_packet<Lanes> result{};
  if constexpr (Lanes == 1) {
    result.values[0] = static_cast<double>(__bfloat162float(
        storage_from_raw<storage::bf16_e8m7>(packet.words[0])));
  } else {
#pragma unroll
    for (int group = 0; group < Lanes / 2; ++group) {
      const auto word = packet.words[group];
      const __nv_bfloat162_raw raw{static_cast<unsigned short>(word),
                                   static_cast<unsigned short>(word >> 16)};
      const auto values = __bfloat1622float2(__nv_bfloat162{raw});
      result.values[2 * group] = static_cast<double>(values.x);
      result.values[2 * group + 1] = static_cast<double>(values.y);
    }
  }
  return result;
}

template <int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_fp32_native_packet(const source_packet<Lanes> &packet) {
  decoded_packet<Lanes> result{};
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    result.values[lane] =
        static_cast<double>(__uint_as_float(packet.words[lane]));
  }
  return result;
}

template <typename Format, int Lanes>
__device__ __forceinline__ std::uint32_t
runtime_raw_lane(const source_packet<Lanes> &packet, int lane) {
  if constexpr (Format::total_bits < 8) {
    return (packet.words[0] >> (Format::total_bits * lane)) &
           ((std::uint32_t{1} << Format::total_bits) - 1);
  } else if constexpr (Format::total_bits == 8) {
    return (packet.words[lane / 4] >> (8 * (lane % 4))) & 0xffu;
  } else if constexpr (Format::total_bits == 16) {
    return (packet.words[lane / 2] >> (16 * (lane % 2))) & 0xffffu;
  } else {
    return packet.words[lane];
  }
}

template <int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_fp4_native_packet(const source_packet<Lanes> &packet) {
  decoded_packet<Lanes> result{};
  if constexpr (Lanes == 1) {
    result.values[0] = decode_fp4_native(packet.words[0]);
  } else {
#pragma unroll
    for (int pair = 0; pair < Lanes / 2; ++pair) {
      const auto packed =
          runtime_raw_lane<storage::fp4_e2m1>(packet, 2 * pair) |
          (runtime_raw_lane<storage::fp4_e2m1>(packet, 2 * pair + 1) << 4);
      const auto half_raw = __nv_cvt_fp4x2_to_halfraw2(
          static_cast<__nv_fp4x2_storage_t>(packed), __NV_E2M1);
      const auto values = __half22float2(__half2{half_raw});
      result.values[2 * pair] = static_cast<double>(values.x);
      result.values[2 * pair + 1] = static_cast<double>(values.y);
    }
  }
  return result;
}

template <typename Format, typename Strategy, int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_pair_packet(const source_packet<Lanes> &packet, table_bundle tables) {
  static_assert(Format::total_bits <= 8);
  static_assert(Lanes == 2 || Lanes == 4 || Lanes == 8);
  decoded_packet<Lanes> result{};
#pragma unroll
  for (int pair = 0; pair < Lanes / 2; ++pair) {
    const auto low = runtime_raw_lane<Format>(packet, 2 * pair);
    const auto high = runtime_raw_lane<Format>(packet, 2 * pair + 1);
    const auto index = low + (high << Format::total_bits);
    const auto words = [&] {
      if constexpr (Strategy::location == table_location::shared) {
        return tables.pair_high[index];
      } else {
        return __ldg(tables.pair_high + index);
      }
    }();
    result.values[2 * pair] = decoder::words_to_double({words.x, 0});
    result.values[2 * pair + 1] = decoder::words_to_double({words.y, 0});
  }
  return result;
}

template <typename Strategy, int Lanes>
__device__ __forceinline__ decoded_packet<Lanes>
decode_quad_packet(const source_packet<Lanes> &packet, table_bundle tables) {
  static_assert(Lanes == 4 || Lanes == 8);
  decoded_packet<Lanes> result{};
#pragma unroll
  for (int group = 0; group < Lanes / 4; ++group) {
    const auto packed_byte = (packet.words[0] >> (8 * group)) & 0xffu;
    const auto words = [&] {
      if constexpr (Strategy::location == table_location::shared) {
        return tables.quad_high[packed_byte];
      } else {
        return __ldg(tables.quad_high + packed_byte);
      }
    }();
    result.values[4 * group] = decoder::words_to_double({words.x, 0});
    result.values[4 * group + 1] = decoder::words_to_double({words.y, 0});
    result.values[4 * group + 2] = decoder::words_to_double({words.z, 0});
    result.values[4 * group + 3] = decoder::words_to_double({words.w, 0});
  }
  return result;
}

template <typename Format, typename Strategy>
__device__ __forceinline__ decoded_packet<Strategy::lanes>
decode_warp_lut_packet(const source_packet<Strategy::lanes> &packet,
                       table_bundle tables) {
  static_assert(Format::total_bits <= 4);
  decoded_packet<Strategy::lanes> result{};
  const auto active = __activemask();
#pragma unroll
  for (int lane = 0; lane < Strategy::lanes; ++lane) {
    const auto raw = runtime_raw_lane<Format>(packet, lane);
    // Every active lane executes the shuffle. If the requested owner lane is
    // absent in a short tail warp, use the ordinary read-only-cache lookup.
    const auto shuffled = __shfl_sync(active, tables.warp_high, raw);
    const auto high = (active & (1u << raw)) != 0
                          ? shuffled
                          : __ldg(tables.full_high + raw);
    result.values[lane] = decoder::words_to_double({high, 0});
  }
  return result;
}

template <typename Format, typename Strategy, std::size_t... Lane>
__device__ __forceinline__ decoded_packet<Strategy::lanes>
decode_packet_impl(const source_packet<Strategy::lanes> &packet,
                   table_bundle tables, std::index_sequence<Lane...>) {
  return {{decode_raw<Format, Strategy>(
      raw_lane<Format, static_cast<int>(Lane), Strategy::lanes,
               Strategy::unpack>(packet),
      tables)...}};
}

template <typename Format, typename Strategy>
__device__ __forceinline__ decoded_packet<Strategy::lanes>
decode_packet(const source_packet<Strategy::lanes> &packet,
              table_bundle tables) {
  if constexpr (Strategy::kind == decode_kind::native_packed) {
    static_assert(std::is_same_v<Format, storage::fp4_e2m1> ||
                  std::is_same_v<Format, storage::fp8_e4m3> ||
                  std::is_same_v<Format, storage::fp8_e5m2> ||
                  std::is_same_v<Format, storage::fp16_e5m10> ||
                  std::is_same_v<Format, storage::bf16_e8m7> ||
                  std::is_same_v<Format, storage::fp32_e8m23>);
    if constexpr (std::is_same_v<Format, storage::fp4_e2m1>) {
      return decode_fp4_native_packet(packet);
    } else if constexpr (std::is_same_v<Format, storage::fp8_e4m3>) {
      return decode_e4m3_native_packet(packet);
    } else if constexpr (std::is_same_v<Format, storage::fp8_e5m2>) {
      return decode_e5m2_native_packet(packet);
    } else if constexpr (std::is_same_v<Format, storage::fp16_e5m10>) {
      return decode_fp16_native_packet(packet);
    } else if constexpr (std::is_same_v<Format, storage::bf16_e8m7>) {
      return decode_bf16_native_packet(packet);
    } else {
      return decode_fp32_native_packet(packet);
    }
  } else if constexpr (Strategy::kind == decode_kind::native_half2) {
    static_assert(std::is_same_v<Format, storage::fp8_e4m3> ||
                  std::is_same_v<Format, storage::fp8_e5m2>);
    if constexpr (std::is_same_v<Format, storage::fp8_e4m3>) {
      return decode_e4m3_half2_packet(packet);
    } else {
      return decode_e5m2_half2_packet(packet);
    }
  } else if constexpr (Strategy::kind == decode_kind::pair_high_lut) {
    return decode_pair_packet<Format, Strategy>(packet, tables);
  } else if constexpr (Strategy::kind == decode_kind::quad_high_lut) {
    static_assert(std::is_same_v<Format, storage::e0m1> ||
                  std::is_same_v<Format, storage::e1m0>);
    return decode_quad_packet<Strategy>(packet, tables);
  } else if constexpr (Strategy::kind == decode_kind::warp_high_lut) {
    return decode_warp_lut_packet<Format, Strategy>(packet, tables);
  } else {
    return decode_packet_impl<Format, Strategy>(
        packet, tables, std::make_index_sequence<Strategy::lanes>{});
  }
}

template <typename Format, typename Strategy>
__global__ void decode_codes(const device_storage_t<Format> *codes,
                             std::size_t count, table_bundle tables,
                             double *output) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto *shared_table = reinterpret_cast<std::uint32_t *>(shared_bytes);
  tables = stage_shared_table<Format, Strategy>(tables, shared_table);
  tables = prepare_thread_table<Format, Strategy>(tables);
  constexpr auto lanes = Strategy::lanes;
  const auto pack = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    static_cast<std::size_t>(threadIdx.x);
  const auto offset = pack * lanes;
  if (offset >= count) {
    return;
  }
  if (offset + lanes <= count) {
    const auto values = decode_packet<Format, Strategy>(
        load_source_packet<Format, lanes>(codes, offset), tables);
#pragma unroll
    for (int lane = 0; lane < lanes; ++lane) {
      output[offset + lane] = values.values[lane];
    }
  } else {
    for (std::size_t index = offset; index < count; ++index) {
      output[index] = decode_raw<Format, Strategy>(
          load_raw_code<Format>(codes, index), tables);
    }
  }
}

template <int Lanes>
__device__ __forceinline__ double combine_sums(double (&sums)[Lanes]) {
  double result{};
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    result += sums[lane];
  }
  return result;
}

template <typename Format, typename Strategy>
__global__ void dot_map_reduce(
    const device_storage_t<Format> *left,
    const device_storage_t<Format> *right, std::size_t count,
    table_bundle tables, double *partials) {
  using block_reduce = cub::BlockReduce<double, block_threads>;
  __shared__ typename block_reduce::TempStorage reduction_storage;
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto *shared_table = reinterpret_cast<std::uint32_t *>(shared_bytes);
  tables = stage_shared_table<Format, Strategy>(tables, shared_table);
  tables = prepare_thread_table<Format, Strategy>(tables);

  constexpr auto lanes = Strategy::lanes;
  const auto first = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     static_cast<std::size_t>(threadIdx.x);
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto pack_count = count / lanes;
  double sums[lanes]{};
  for (auto pack = first; pack < pack_count; pack += stride) {
    const auto offset = pack * lanes;
    const auto left_values = decode_packet<Format, Strategy>(
        load_source_packet<Format, lanes>(left, offset), tables);
    const auto right_values = decode_packet<Format, Strategy>(
        load_source_packet<Format, lanes>(right, offset), tables);
#pragma unroll
    for (int lane = 0; lane < lanes; ++lane) {
      sums[lane] = fma(left_values.values[lane], right_values.values[lane],
                       sums[lane]);
    }
  }

  const auto tail = pack_count * lanes + first;
  if (tail < count) {
    sums[0] = fma(decode_raw<Format, Strategy>(
                      load_raw_code<Format>(left, tail), tables),
                  decode_raw<Format, Strategy>(
                      load_raw_code<Format>(right, tail), tables),
                  sums[0]);
  }
  const auto sum = block_reduce(reduction_storage).Sum(combine_sums(sums));
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = sum;
  }
}

template <typename Format, typename Strategy>
__global__ void gemv(const device_storage_t<Format> *matrix,
                     const device_storage_t<Format> *vector,
                     std::size_t rows, std::size_t columns,
                     std::size_t leading_dimension, table_bundle tables,
                     double *result) {
  using block_reduce = cub::BlockReduce<double, block_threads>;
  __shared__ typename block_reduce::TempStorage reduction_storage;
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto *shared_table = reinterpret_cast<std::uint32_t *>(shared_bytes);
  tables = stage_shared_table<Format, Strategy>(tables, shared_table);
  tables = prepare_thread_table<Format, Strategy>(tables);

  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  constexpr auto lanes = Strategy::lanes;
  const auto row_offset = row * leading_dimension;
  const auto pack_count = columns / lanes;
  double sums[lanes]{};
  for (auto pack = static_cast<std::size_t>(threadIdx.x); pack < pack_count;
       pack += blockDim.x) {
    const auto offset = pack * lanes;
    const auto matrix_values = decode_packet<Format, Strategy>(
        load_source_packet<Format, lanes>(matrix, row_offset + offset), tables);
    const auto vector_values = decode_packet<Format, Strategy>(
        load_source_packet<Format, lanes>(vector, offset), tables);
#pragma unroll
    for (int lane = 0; lane < lanes; ++lane) {
      sums[lane] = fma(matrix_values.values[lane], vector_values.values[lane],
                       sums[lane]);
    }
  }

  const auto tail = pack_count * lanes + threadIdx.x;
  if (tail < columns) {
    sums[0] = fma(decode_raw<Format, Strategy>(
                      load_raw_code<Format>(matrix, row_offset + tail), tables),
                  decode_raw<Format, Strategy>(
                      load_raw_code<Format>(vector, tail), tables),
                  sums[0]);
  }
  const auto sum = block_reduce(reduction_storage).Sum(combine_sums(sums));
  if (threadIdx.x == 0) {
    result[row] = sum;
  }
}

} // namespace aut::format_strategies

#endif // ACCESSOR_UNIVERSAL_TEST_FORMAT_DECODER_STRATEGIES_CUH_
