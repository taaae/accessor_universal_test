#ifndef ACCESSOR_UNIVERSAL_TEST_BITWIDTH_BENCHMARK_CORE_HPP_
#define ACCESSOR_UNIVERSAL_TEST_BITWIDTH_BENCHMARK_CORE_HPP_

#include "decoder_strategy_core.hpp"
#include "storage_formats.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__CUDACC__)
#define AUT_BITWIDTH_HD __host__ __device__
#define AUT_BITWIDTH_INLINE __forceinline__
#else
#define AUT_BITWIDTH_HD
#define AUT_BITWIDTH_INLINE inline
#endif

namespace aut::bitwidth {

enum class compute_kind { fp32, fp64 };
enum class storage_layout { dense, padded };
enum class access_method { scalar, thread_packet, cooperative_shuffle };
enum class decoder_kind {
  generic_reference,
  direct_branchy,
  direct_masked,
  fixed_integer,
  e1_integer,
  exponent_only,
  full_lut_global,
  full_lut_shared,
  prefix_lut_global,
  prefix_lut_shared,
  subnormal_lut_global,
  subnormal_lut_shared,
  native_scalar,
  native_packed
};

template <compute_kind Kind> struct compute_traits;
template <> struct compute_traits<compute_kind::fp32> {
  using type = float;
  static constexpr const char *name = "fp32";
};
template <> struct compute_traits<compute_kind::fp64> {
  using type = double;
  static constexpr const char *name = "fp64";
};
template <compute_kind Kind>
using compute_t = typename compute_traits<Kind>::type;

template <typename Format> struct format_layout {
  static constexpr auto policy = [] {
    if constexpr (Format::exponent_bits == 0) {
      return decoder::special_policy::fixed_e0;
    } else if constexpr (Format::finite) {
      return decoder::special_policy::finite_all;
    } else {
      return decoder::special_policy::ieee;
    }
  }();
  using type = decoder::binary_layout<Format::total_bits,
                                      Format::exponent_bits, policy>;
};
template <typename Format>
using format_layout_t = typename format_layout<Format>::type;

template <int Bits> struct dense_geometry {
  static_assert(Bits >= 2 && Bits <= 32);
  static constexpr int gcd = [] {
    int left = Bits;
    int right = 32;
    while (right != 0) {
      const auto remainder = left % right;
      left = right;
      right = remainder;
    }
    return left;
  }();
  static constexpr int values_per_aligned_chunk = 32 / gcd;
  static constexpr int words_per_aligned_chunk = Bits / gcd;
  static constexpr int chunk_bits = values_per_aligned_chunk * Bits;
};

template <int Bits>
AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE constexpr std::uint32_t raw_mask() {
  if constexpr (Bits == 32) {
    return 0xffffffffu;
  } else {
    return (std::uint32_t{1} << Bits) - 1u;
  }
}

template <int Bits>
AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE std::uint32_t
load_dense_scalar(const std::uint32_t *words, std::size_t index) {
  const auto bit = index * static_cast<std::size_t>(Bits);
  const auto word = bit >> 5;
  const auto shift = static_cast<unsigned>(bit & 31u);
  const auto pair = static_cast<std::uint64_t>(words[word]) |
                    (static_cast<std::uint64_t>(words[word + 1]) << 32);
  return static_cast<std::uint32_t>(pair >> shift) & raw_mask<Bits>();
}

template <int Bits>
AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE void
or_dense_scalar(std::uint32_t *words, std::size_t index, std::uint32_t raw) {
  const auto bit = index * static_cast<std::size_t>(Bits);
  const auto word = bit >> 5;
  const auto shift = static_cast<unsigned>(bit & 31u);
  const auto placed = static_cast<std::uint64_t>(raw & raw_mask<Bits>())
                      << shift;
  words[word] |= static_cast<std::uint32_t>(placed);
  if (shift + Bits > 32) {
    words[word + 1] |= static_cast<std::uint32_t>(placed >> 32);
  }
}

template <int Bits>
constexpr std::size_t dense_word_count(std::size_t values) {
  // One zero guard word lets every scalar decoder use the same two-word path.
  return (values * static_cast<std::size_t>(Bits) + 31u) / 32u + 1u;
}

template <int Bits>
constexpr std::size_t dense_data_bytes(std::size_t values) {
  return (values * static_cast<std::size_t>(Bits) + 7u) / 8u;
}

template <typename Format>
using padded_storage_t = storage::storage_type_t<Format>;

template <typename Format>
AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE std::uint32_t
raw_from_padded(padded_storage_t<Format> value) {
  return static_cast<std::uint32_t>(value) & raw_mask<Format::total_bits>();
}

AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE std::uint32_t float_bits(float value) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(value);
#else
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
#endif
}

template <typename Format>
AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE double decode_direct_fp64_branchy(
    std::uint32_t raw) {
  using layout = format_layout_t<Format>;
  if constexpr (Format::exponent_bits == 11) {
    constexpr auto shift = 52 - Format::fraction_bits;
    const auto bits = static_cast<std::uint64_t>(raw & raw_mask<Format::total_bits>())
                      << shift;
    return decoder::words_to_double(
        {static_cast<std::uint32_t>(bits >> 32),
         static_cast<std::uint32_t>(bits)});
  } else {
    return decoder::words_to_double(decoder::decode_words_branchy<layout>(raw));
  }
}

template <typename Format>
AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE double decode_direct_fp64_masked(
    std::uint32_t raw) {
  using layout = format_layout_t<Format>;
  if constexpr (Format::exponent_bits == 11) {
    return decode_direct_fp64_branchy<Format>(raw);
  } else {
    return decoder::words_to_double(decoder::decode_words_masked<layout>(raw));
  }
}

template <typename Format>
AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE float decode_direct_fp32(std::uint32_t raw) {
  using layout = format_layout_t<Format>;
  static_assert(Format::exponent_bits <= 8,
                "the exact direct FP32 path is used only for E<=8");
  static_assert(Format::fraction_bits <= 23,
                "the exact direct FP32 path is used only for M<=23");
  return decoder::bits_to_float(decoder::decode_fp32_bits<layout>(raw));
}

template <typename Format, compute_kind Compute>
AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE compute_t<Compute>
decode_reference(std::uint32_t raw) {
  using stored = storage::storage_type_t<Format>;
  return static_cast<compute_t<Compute>>(
      storage::decode<Format>(static_cast<stored>(raw)));
}

template <typename Format, compute_kind Compute, decoder_kind Decoder>
AUT_BITWIDTH_HD AUT_BITWIDTH_INLINE compute_t<Compute>
decode_raw(std::uint32_t raw) {
  using layout = format_layout_t<Format>;
  if constexpr (Decoder == decoder_kind::generic_reference) {
    return decode_reference<Format, Compute>(raw);
  } else if constexpr (Decoder == decoder_kind::direct_branchy) {
    if constexpr (Compute == compute_kind::fp32) {
      return decode_direct_fp32<Format>(raw);
    } else {
      return decode_direct_fp64_branchy<Format>(raw);
    }
  } else if constexpr (Decoder == decoder_kind::direct_masked) {
    if constexpr (Compute == compute_kind::fp32) {
      return decode_direct_fp32<Format>(raw);
    } else {
      return decode_direct_fp64_masked<Format>(raw);
    }
  } else if constexpr (Decoder == decoder_kind::fixed_integer) {
    static_assert(Format::exponent_bits == 0);
    if constexpr (Compute == compute_kind::fp32) {
      const auto sign = raw >> (Format::total_bits - 1);
      const auto magnitude = raw & raw_mask<Format::fraction_bits>();
      const auto value = static_cast<float>(magnitude) /
                         static_cast<float>(std::uint32_t{1}
                                            << Format::fraction_bits);
      return sign == 0 ? value : -value;
    } else {
      return decoder::decode_fixed_integer<layout>(raw);
    }
  } else if constexpr (Decoder == decoder_kind::e1_integer) {
    static_assert(Format::exponent_bits == 1 && Format::finite);
    if constexpr (Compute == compute_kind::fp32) {
      const auto sign = raw >> (Format::total_bits - 1);
      const auto magnitude = raw & raw_mask<Format::total_bits - 1>();
      const auto value = static_cast<float>(magnitude) /
                         static_cast<float>(std::uint32_t{1}
                                            << (Format::fraction_bits - 1));
      return sign == 0 ? value : -value;
    } else {
      return decoder::decode_e1_integer<layout>(raw);
    }
  } else if constexpr (Decoder == decoder_kind::exponent_only) {
    static_assert(Format::fraction_bits == 0);
    if constexpr (Compute == compute_kind::fp32) {
      return decode_direct_fp32<Format>(raw);
    } else {
      return decoder::decode_exponent_only<layout>(raw);
    }
  } else {
    static_assert(Decoder == decoder_kind::generic_reference,
                  "table and native decoders need a supplied table/API");
  }
}

} // namespace aut::bitwidth

#undef AUT_BITWIDTH_HD
#undef AUT_BITWIDTH_INLINE

#endif // ACCESSOR_UNIVERSAL_TEST_BITWIDTH_BENCHMARK_CORE_HPP_
