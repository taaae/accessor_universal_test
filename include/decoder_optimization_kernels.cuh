#ifndef ACCESSOR_UNIVERSAL_TEST_DECODER_OPTIMIZATION_KERNELS_CUH_
#define ACCESSOR_UNIVERSAL_TEST_DECODER_OPTIMIZATION_KERNELS_CUH_

#include "storage_performance_kernels.cuh"

#include <cub/block/block_reduce.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aut::decoder_optimization {

struct current_x1 {
  static constexpr int lanes = 1;
  static constexpr const char *name = "current_x1";
};

struct current_x4 {
  static constexpr int lanes = 4;
  static constexpr const char *name = "current_x4";
};

struct branchless_x4 {
  static constexpr int lanes = 4;
  static constexpr const char *name = "branchless_x4";
};

struct lut_x1 {
  static constexpr int lanes = 1;
  static constexpr const char *name = "lut_x1";
};

struct lut_x4 {
  static constexpr int lanes = 4;
  static constexpr const char *name = "lut_x4";
};

template <typename Strategy>
inline constexpr bool uses_lut_v =
    std::is_same_v<Strategy, lut_x1> || std::is_same_v<Strategy, lut_x4>;

template <typename Format>
inline constexpr bool is_optimized_format_v =
    std::is_same_v<Format, storage::e2m5> ||
    std::is_same_v<Format, storage::e3m4>;

/**
 * Decode E2M5/E3M4 through exact FP32 arithmetic.
 *
 * Every finite value of these formats is exactly representable in FP32. The
 * source byte is split into an integer significand and a power-of-two scale,
 * avoiding the generic decoder's leading-bit search and FP64 bit assembly.
 * The final FP32-to-FP64 conversion is exact for all finite values. A mask
 * selects the infinity/NaN representation without control-flow branches.
 */
template <typename Format>
__device__ __forceinline__ float branchless_decode_byte(std::uint32_t raw) {
  static_assert(is_optimized_format_v<Format>);
  constexpr auto fraction_bits = Format::fraction_bits;
  constexpr auto exponent_bits = Format::exponent_bits;
  constexpr auto exponent_mask = (std::uint32_t{1} << exponent_bits) - 1;
  constexpr auto fraction_mask = (std::uint32_t{1} << fraction_bits) - 1;
  constexpr auto exponent_bias = (1 << (exponent_bits - 1)) - 1;

  raw &= 0xffu;
  const auto sign = raw >> 7;
  const auto exponent = (raw >> fraction_bits) & exponent_mask;
  const auto fraction = raw & fraction_mask;
  const auto has_normal_exponent = static_cast<std::uint32_t>(exponent != 0);
  const auto significand = fraction | (has_normal_exponent << fraction_bits);
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
__device__ __forceinline__ double scalar_from_bits(std::uint64_t bits) {
  return storage::decode<Format>(performance::storage_from_bits<Format>(bits));
}

template <typename Format, typename Strategy>
__device__ __forceinline__ kernels::fp64x4
decode_register_packet(const performance::register_packet &packet,
                       const float *lut) {
  constexpr auto lanes = Strategy::lanes;
  if constexpr (std::is_same_v<Strategy, branchless_x4>) {
    static_assert(is_optimized_format_v<Format>);
    return {static_cast<double>(branchless_decode_byte<Format>(packet.x)),
            static_cast<double>(branchless_decode_byte<Format>(packet.y)),
            static_cast<double>(branchless_decode_byte<Format>(packet.z)),
            static_cast<double>(branchless_decode_byte<Format>(packet.w))};
  } else if constexpr (uses_lut_v<Strategy>) {
    static_assert(is_optimized_format_v<Format>);
    kernels::fp64x4 result{static_cast<double>(__ldg(lut + packet.x)), 0.0, 0.0,
                           0.0};
    if constexpr (lanes == 4) {
      result.y = static_cast<double>(__ldg(lut + packet.y));
      result.z = static_cast<double>(__ldg(lut + packet.z));
      result.w = static_cast<double>(__ldg(lut + packet.w));
    }
    return result;
  } else {
    kernels::fp64x4 result{scalar_from_bits<Format>(packet.x), 0.0, 0.0, 0.0};
    if constexpr (lanes == 4) {
      result.y = scalar_from_bits<Format>(packet.y);
      result.z = scalar_from_bits<Format>(packet.z);
      result.w = scalar_from_bits<Format>(packet.w);
    }
    return result;
  }
}

template <typename Format, typename Strategy>
__device__ __forceinline__ kernels::fp64x4
load_decoded(const storage::storage_type_t<Format> *values, const float *lut) {
  if constexpr (std::is_same_v<Strategy, current_x1>) {
    return {storage::decode<Format>(values[0]), 0.0, 0.0, 0.0};
  } else if constexpr (std::is_same_v<Strategy, current_x4>) {
    const auto decoded = storage::packed_decoder<Format>::load4(values);
    return {decoded.x, decoded.y, decoded.z, decoded.w};
  } else if constexpr (std::is_same_v<Strategy, branchless_x4>) {
    static_assert(is_optimized_format_v<Format>);
    const auto packed = *reinterpret_cast<const std::uint32_t *>(values);
    return {static_cast<double>(branchless_decode_byte<Format>(packed)),
            static_cast<double>(branchless_decode_byte<Format>(packed >> 8)),
            static_cast<double>(branchless_decode_byte<Format>(packed >> 16)),
            static_cast<double>(branchless_decode_byte<Format>(packed >> 24))};
  } else {
    static_assert(uses_lut_v<Strategy> && is_optimized_format_v<Format>);
    if constexpr (Strategy::lanes == 1) {
      const auto raw = static_cast<std::uint32_t>(values[0]);
      return {static_cast<double>(__ldg(lut + raw)), 0.0, 0.0, 0.0};
    } else {
      const auto packed = *reinterpret_cast<const std::uint32_t *>(values);
      return {static_cast<double>(__ldg(lut + (packed & 0xffu))),
              static_cast<double>(__ldg(lut + ((packed >> 8) & 0xffu))),
              static_cast<double>(__ldg(lut + ((packed >> 16) & 0xffu))),
              static_cast<double>(__ldg(lut + (packed >> 24)))};
    }
  }
}

template <typename Format, typename Strategy>
__global__ void register_decode(const storage::storage_type_t<Format> *values,
                                std::size_t count, int repeats,
                                const float *lut, double *sink) {
  constexpr auto lanes = Strategy::lanes;
  const auto thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto offset = thread * lanes;
  if (offset + lanes > count) {
    offset %= count - lanes + 1;
  }
  auto packet =
      performance::load_register_packet<Format, lanes>(values + offset);
  kernels::fp64x4 sums{};
#pragma unroll 1
  for (int repeat = 0; repeat < repeats; ++repeat) {
    performance::compiler_barrier<Format, lanes>(packet);
    performance::accumulate_values<lanes>(
        sums, decode_register_packet<Format, Strategy>(packet, lut));
  }
  sink[thread] = kernels::combine_sums<lanes>(sums);
}

template <typename Format, typename Strategy>
__global__ void
stream_load_decode(const storage::storage_type_t<Format> *values,
                   std::size_t count, const float *lut, double *block_sums) {
  constexpr auto lanes = Strategy::lanes;
  using block_reduce = cub::BlockReduce<double, performance::block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto pack_count = count / lanes;
  kernels::fp64x4 sums{};
  for (auto pack = first; pack < pack_count; pack += stride) {
    performance::accumulate_values<lanes>(
        sums, load_decoded<Format, Strategy>(values + pack * lanes, lut));
  }
  const auto tail = static_cast<std::size_t>(lanes) * pack_count + first;
  if (tail < count) {
    sums.x += storage::decode<Format>(values[tail]);
  }
  const auto reduced =
      block_reduce(temporary).Sum(kernels::combine_sums<lanes>(sums));
  if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = reduced;
  }
}

template <typename Format, typename Strategy>
__global__ void dot_map_reduce(const storage::storage_type_t<Format> *left,
                               const storage::storage_type_t<Format> *right,
                               std::size_t count, const float *lut,
                               double *partials) {
  constexpr auto lanes = Strategy::lanes;
  using block_reduce =
      cub::BlockReduce<double, kernels::reduction_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto pack_count = count / lanes;
  kernels::fp64x4 sums{};
  for (auto pack = first; pack < pack_count; pack += stride) {
    const auto offset = static_cast<std::size_t>(lanes) * pack;
    const auto left_values = load_decoded<Format, Strategy>(left + offset, lut);
    const auto right_values =
        load_decoded<Format, Strategy>(right + offset, lut);
    kernels::accumulate_product<lanes>(sums, left_values, right_values);
  }
  const auto tail = static_cast<std::size_t>(lanes) * pack_count + first;
  if (tail < count) {
    sums.x = fma(storage::decode<Format>(left[tail]),
                 storage::decode<Format>(right[tail]), sums.x);
  }
  const auto reduced =
      block_reduce(temporary).Sum(kernels::combine_sums<lanes>(sums));
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename Format, typename Strategy>
__global__ void gemv(const storage::storage_type_t<Format> *matrix,
                     const storage::storage_type_t<Format> *vector,
                     std::size_t rows, std::size_t columns,
                     std::size_t leading_dimension, const float *lut,
                     double *result) {
  constexpr auto lanes = Strategy::lanes;
  using block_reduce =
      cub::BlockReduce<double, kernels::reduction_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const auto *row_values = matrix + row * leading_dimension;
  const auto pack_count = columns / lanes;
  kernels::fp64x4 sums{};
  for (auto pack = static_cast<std::size_t>(threadIdx.x); pack < pack_count;
       pack += blockDim.x) {
    const auto offset = static_cast<std::size_t>(lanes) * pack;
    const auto matrix_values =
        load_decoded<Format, Strategy>(row_values + offset, lut);
    const auto vector_values =
        load_decoded<Format, Strategy>(vector + offset, lut);
    kernels::accumulate_product<lanes>(sums, matrix_values, vector_values);
  }
  const auto tail = static_cast<std::size_t>(lanes) * pack_count + threadIdx.x;
  if (tail < columns) {
    sums.x = fma(storage::decode<Format>(row_values[tail]),
                 storage::decode<Format>(vector[tail]), sums.x);
  }
  const auto reduced =
      block_reduce(temporary).Sum(kernels::combine_sums<lanes>(sums));
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
}

template <typename Format, typename Strategy>
__global__ void decode_all_codes(const storage::storage_type_t<Format> *values,
                                 const float *lut, double *decoded) {
  constexpr auto lanes = Strategy::lanes;
  const auto pack =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pack * lanes >= 256) {
    return;
  }
  const auto result =
      load_decoded<Format, Strategy>(values + pack * lanes, lut);
  decoded[pack * lanes] = result.x;
  if constexpr (lanes == 4) {
    decoded[pack * lanes + 1] = result.y;
    decoded[pack * lanes + 2] = result.z;
    decoded[pack * lanes + 3] = result.w;
  }
}

} // namespace aut::decoder_optimization

#endif // ACCESSOR_UNIVERSAL_TEST_DECODER_OPTIMIZATION_KERNELS_CUH_
