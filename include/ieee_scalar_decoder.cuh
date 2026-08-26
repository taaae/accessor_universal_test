#ifndef ACCESSOR_UNIVERSAL_TEST_IEEE_SCALAR_DECODER_CUH_
#define ACCESSOR_UNIVERSAL_TEST_IEEE_SCALAR_DECODER_CUH_

#include "bitwidth_benchmark_kernels.cuh"
#include "posit_takum_core.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aut::pt {

template <typename Format, bitwidth::compute_kind Compute,
          bitwidth::decoder_kind Decoder>
struct ieee_scalar_decoder {
  using Float = bitwidth::compute_t<Compute>;
  const Float *table{};

  __device__ __forceinline__ const Float *prepare(Float *shared) const {
    if constexpr (Decoder == bitwidth::decoder_kind::full_lut_shared) {
      constexpr std::size_t entries = std::size_t{1} << Format::total_bits;
      for (std::size_t index = threadIdx.x; index < entries;
           index += blockDim.x) {
        shared[index] = table[index];
      }
      __syncthreads();
      return shared;
    } else {
      return table;
    }
  }

  __device__ __forceinline__ Float value(std::uint32_t raw,
                                         const Float *active_table) const {
    namespace bw = bitwidth;
    if constexpr (Decoder == bw::decoder_kind::full_lut_global ||
                  Decoder == bw::decoder_kind::full_lut_shared) {
      return active_table[raw];
    } else if constexpr (Decoder == bw::decoder_kind::prefix_lut_global) {
      using layout = bw::format_layout_t<Format>;
      const auto fraction = raw & decoder::fraction_mask<layout>();
      const auto exponent =
          (raw >> Format::fraction_bits) & decoder::exponent_mask<layout>();
      if (exponent == 0 || decoder::is_special<layout>(exponent, fraction)) {
        return bw::decode_raw<Format, Compute,
                              bw::decoder_kind::direct_branchy>(raw);
      }
      const auto prefix = raw >> Format::fraction_bits;
      auto bits = pt::to_bits(active_table[prefix]);
      if constexpr (Compute == bw::compute_kind::fp32) {
        bits |= static_cast<std::uint32_t>(fraction)
                << (23 - Format::fraction_bits);
      } else {
        bits |= static_cast<std::uint64_t>(fraction)
                << (52 - Format::fraction_bits);
      }
      return pt::from_bits<Float>(bits);
    } else if constexpr (Decoder == bw::decoder_kind::subnormal_lut_global) {
      using layout = bw::format_layout_t<Format>;
      const auto fraction = raw & decoder::fraction_mask<layout>();
      const auto exponent =
          (raw >> Format::fraction_bits) & decoder::exponent_mask<layout>();
      if (exponent != 0) {
        return bw::decode_raw<Format, Compute,
                              bw::decoder_kind::direct_branchy>(raw);
      }
      const auto source_sign = raw >> (Format::total_bits - 1);
      auto bits = pt::to_bits(active_table[fraction]);
      bits |= static_cast<typename ieee_traits<Float>::uint_type>(source_sign)
              << (ieee_traits<Float>::fraction_bits +
                  ieee_traits<Float>::exponent_bits);
      return pt::from_bits<Float>(bits);
    } else {
      return bw::decode_with_table<Format, Compute, Decoder>(raw, nullptr);
    }
  }
};

template <bitwidth::decoder_kind Decoder>
inline constexpr bool ieee_shared_v =
    Decoder == bitwidth::decoder_kind::full_lut_shared;

} // namespace aut::pt

#endif
