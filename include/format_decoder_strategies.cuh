#ifndef ACCESSOR_UNIVERSAL_TEST_FORMAT_DECODER_STRATEGIES_CUH_
#define ACCESSOR_UNIVERSAL_TEST_FORMAT_DECODER_STRATEGIES_CUH_

#include "cuda_storage_formats.cuh"
#include "decoder_strategy_core.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace aut::format_strategies {

enum class decode_kind {
  generic,
  direct_words_branchy,
  direct_words_masked,
  fp32_bits,
  e1_integer,
  prefix_word,
  full_high_lut,
  subnormal_high_lut,
  prefix_high_lut,
  native_direct,
  native_fp32,
  native_packed,
};

enum class table_location { global_read_only, shared };

template <decode_kind Kind, int Lanes,
          table_location Location = table_location::global_read_only>
struct strategy {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8);
  static constexpr decode_kind kind = Kind;
  static constexpr int lanes = Lanes;
  static constexpr table_location location = Location;
};

template <typename Format> struct format_layout;
template <> struct format_layout<storage::e1m6> {
  using type = decoder::e1m6_layout;
};
template <typename Format>
using format_layout_t = typename format_layout<Format>::type;

struct table_bundle {
  const std::uint32_t *full_high{};
  const std::uint32_t *subnormal_high{};
  const std::uint32_t *prefix_high{};
};

template <typename Strategy>
inline constexpr bool uses_table_v =
    Strategy::kind == decode_kind::full_high_lut ||
    Strategy::kind == decode_kind::subnormal_high_lut ||
    Strategy::kind == decode_kind::prefix_high_lut;

template <typename Format, typename Strategy>
inline constexpr std::size_t table_entries_v = [] {
  using layout = format_layout_t<Format>;
  if constexpr (Strategy::kind == decode_kind::full_high_lut) {
    return std::size_t{1} << layout::total_bits;
  } else if constexpr (Strategy::kind == decode_kind::subnormal_high_lut) {
    return std::size_t{1} << layout::fraction_bits;
  } else if constexpr (Strategy::kind == decode_kind::prefix_high_lut) {
    return std::size_t{1} << (layout::exponent_bits + 1);
  } else {
    return std::size_t{0};
  }
}();

template <typename Format, typename Strategy>
inline constexpr std::size_t shared_table_bytes_v =
    Strategy::location == table_location::shared
        ? table_entries_v<Format, Strategy> * sizeof(std::uint32_t)
        : std::size_t{0};

template <int Lanes> struct source_packet {
  std::uint32_t words[8]{};
};

template <typename Format, int Lanes>
__device__ __forceinline__ source_packet<Lanes>
load_source_packet(const storage::storage_type_t<Format> *values) {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8);
  constexpr auto bytes = sizeof(storage::storage_type_t<Format>);
  const auto *raw = reinterpret_cast<const std::uint8_t *>(values);
  source_packet<Lanes> result{};
  if constexpr (bytes == 1) {
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

template <typename Format, int Lane, int Lanes>
__device__ __forceinline__ std::uint32_t
raw_lane(const source_packet<Lanes> &packet) {
  constexpr auto bytes = sizeof(storage::storage_type_t<Format>);
  static_assert(Lane >= 0 && Lane < Lanes);
  if constexpr (bytes == 1) {
    return (packet.words[Lane / 4] >> (8 * (Lane % 4))) & 0xffu;
  } else if constexpr (bytes == 2) {
    return (packet.words[Lane / 2] >> (16 * (Lane % 2))) & 0xffffu;
  } else {
    return packet.words[Lane];
  }
}

template <typename Format>
__device__ __forceinline__ storage::storage_type_t<Format>
storage_from_raw(std::uint32_t raw) {
  static_assert(std::is_integral_v<storage::storage_type_t<Format>>);
  return static_cast<storage::storage_type_t<Format>>(raw);
}

template <typename Format, typename Strategy>
__device__ __forceinline__ table_bundle
stage_shared_table(table_bundle tables, std::uint32_t *shared) {
  if constexpr (Strategy::location == table_location::shared) {
    static_assert(uses_table_v<Strategy>);
    const std::uint32_t *source{};
    if constexpr (Strategy::kind == decode_kind::full_high_lut) {
      source = tables.full_high;
    } else if constexpr (Strategy::kind == decode_kind::subnormal_high_lut) {
      source = tables.subnormal_high;
    } else {
      source = tables.prefix_high;
    }
    for (std::size_t i = threadIdx.x; i < table_entries_v<Format, Strategy>;
         i += blockDim.x) {
      shared[i] = __ldg(source + i);
    }
    __syncthreads();
    if constexpr (Strategy::kind == decode_kind::full_high_lut) {
      tables.full_high = shared;
    } else if constexpr (Strategy::kind == decode_kind::subnormal_high_lut) {
      tables.subnormal_high = shared;
    } else {
      tables.prefix_high = shared;
    }
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
  } else if constexpr (Strategy::kind == decode_kind::e1_integer) {
    return decoder::decode_e1_integer<layout>(raw);
  } else if constexpr (Strategy::kind == decode_kind::prefix_word) {
    return decoder::words_to_double(
        decoder::decode_prefix_words<layout::fraction_bits>(raw));
  } else if constexpr (Strategy::kind == decode_kind::full_high_lut) {
    return decoder::words_to_double(
        {lookup_high<Strategy>(tables.full_high,
                               raw & decoder::raw_mask<layout>()),
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
  } else {
    static_assert(Strategy::kind == decode_kind::prefix_high_lut,
                  "native decoders are specialized per format");
    const auto prefix = raw >> layout::fraction_bits;
    auto high = lookup_high<Strategy>(tables.prefix_high, prefix);
    const auto fraction = raw & decoder::fraction_mask<layout>();
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

template <typename Format, typename Strategy, std::size_t... Lane>
__device__ __forceinline__ decoded_packet<Strategy::lanes>
decode_packet_impl(const source_packet<Strategy::lanes> &packet,
                   table_bundle tables, std::index_sequence<Lane...>) {
  return {{decode_raw<Format, Strategy>(
      raw_lane<Format, static_cast<int>(Lane)>(packet), tables)...}};
}

template <typename Format, typename Strategy>
__device__ __forceinline__ decoded_packet<Strategy::lanes>
decode_packet(const source_packet<Strategy::lanes> &packet,
              table_bundle tables) {
  return decode_packet_impl<Format, Strategy>(
      packet, tables, std::make_index_sequence<Strategy::lanes>{});
}

template <typename Format, typename Strategy>
__global__ void decode_codes(const storage::storage_type_t<Format> *codes,
                             std::size_t count, table_bundle tables,
                             double *output) {
  extern __shared__ std::uint32_t shared_table[];
  tables = stage_shared_table<Format, Strategy>(tables, shared_table);
  constexpr auto lanes = Strategy::lanes;
  const auto pack = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    static_cast<std::size_t>(threadIdx.x);
  const auto offset = pack * lanes;
  if (offset >= count) {
    return;
  }
  if (offset + lanes <= count) {
    const auto values = decode_packet<Format, Strategy>(
        load_source_packet<Format, lanes>(codes + offset), tables);
#pragma unroll
    for (int lane = 0; lane < lanes; ++lane) {
      output[offset + lane] = values.values[lane];
    }
  } else {
    for (std::size_t index = offset; index < count; ++index) {
      output[index] = decode_raw<Format, Strategy>(
          static_cast<std::uint32_t>(codes[index]), tables);
    }
  }
}

} // namespace aut::format_strategies

#endif // ACCESSOR_UNIVERSAL_TEST_FORMAT_DECODER_STRATEGIES_CUH_
