#ifndef ACCESSOR_UNIVERSAL_TEST_LNS_BENCHMARK_KERNELS_CUH_
#define ACCESSOR_UNIVERSAL_TEST_LNS_BENCHMARK_KERNELS_CUH_

#include "bitwidth_benchmark_core.hpp"
#include "lns_benchmark_core.hpp"
#include "lns_decoder_strategies.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace aut::lns_strategy {

using bitwidth::access_method;
using bitwidth::storage_layout;

template <typename Format, storage_layout Layout> struct storage_view {
  const std::uint32_t *dense{};
  const lns::padded_storage_t<Format> *padded{};

  __device__ __forceinline__ std::uint32_t raw(std::size_t index) const {
    if constexpr (Layout == storage_layout::dense) {
      return bitwidth::load_dense_scalar<Format::total_bits>(dense, index);
    } else {
      return static_cast<std::uint32_t>(padded[index]) &
             lns::raw_mask<Format>();
    }
  }

  template <int Lanes>
  __device__ __forceinline__ void raw_packet(
      std::size_t base, std::uint32_t (&output)[Lanes]) const {
    if constexpr (Layout == storage_layout::padded) {
      struct alignas(sizeof(lns::padded_storage_t<Format>) * Lanes >= 16
                         ? 16
                         : sizeof(lns::padded_storage_t<Format>) * Lanes)
          packet {
        lns::padded_storage_t<Format> lane[Lanes];
      };
      const auto loaded = *reinterpret_cast<const packet *>(padded + base);
#pragma unroll
      for (int lane = 0; lane < Lanes; ++lane) {
        output[lane] = static_cast<std::uint32_t>(loaded.lane[lane]) &
                       lns::raw_mask<Format>();
      }
    } else {
#pragma unroll
      for (int lane = 0; lane < Lanes; ++lane) {
        output[lane] = bitwidth::load_dense_scalar<Format::total_bits>(
            dense, base + lane);
      }
    }
  }
};

template <typename Value>
__device__ __forceinline__ Value block_sum(Value value, Value *shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  return shared[0];
}

template <typename Format, decoder_kind Decoder>
inline constexpr std::size_t table_bytes_v =
    table_entries_v<Format, Decoder>;

template <typename Format, compute_kind Compute, decoder_kind Decoder>
inline constexpr std::size_t shared_table_bytes_v =
    uses_shared_table_v<Decoder>
        ? table_entries_v<Format, Decoder> * sizeof(compute_t<Compute>)
        : 0;

template <typename Format, compute_kind Compute, storage_layout Layout,
          int Lanes, decoder_kind Decoder, multiply_kind Multiply>
__global__ void dot_thread_kernel(storage_view<Format, Layout> left,
                                  storage_view<Format, Layout> right,
                                  std::size_t count,
                                  table_bundle<Compute> tables,
                                  compute_t<Compute> *partials) {
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<compute_t<Compute> *>(dynamic_shared);
  const auto context =
      prepare_context<Format, Compute, Decoder>(tables, table_shared);
  auto *reduce_shared = reinterpret_cast<compute_t<Compute> *>(
      dynamic_shared + shared_table_bytes_v<Format, Compute, Decoder>);

  compute_t<Compute> sum{};
  const auto packets = (count + Lanes - 1) / Lanes;
  for (auto packet = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       packet < packets;
       packet += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto base = packet * Lanes;
    std::uint32_t left_raw[Lanes]{};
    std::uint32_t right_raw[Lanes]{};
    left.template raw_packet<Lanes>(base, left_raw);
    right.template raw_packet<Lanes>(base, right_raw);
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      if (base + lane < count) {
        sum += multiply<Format, Compute, Decoder, Multiply>(
            left_raw[lane], right_raw[lane], context);
      }
    }
  }
  const auto reduced = block_sum(sum, reduce_shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename Compute>
__global__ void finalize_dot_kernel(const Compute *partials, std::size_t count,
                                    Compute *result) {
  __shared__ Compute shared[256];
  Compute sum{};
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    sum += partials[index];
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    *result = reduced;
  }
}

template <typename Format, compute_kind Compute, storage_layout Layout,
          int Lanes, decoder_kind Decoder, multiply_kind Multiply>
__global__ void gemv_thread_kernel(storage_view<Format, Layout> matrix,
                                   storage_view<Format, Layout> vector,
                                   std::size_t rows, std::size_t columns,
                                   table_bundle<Compute> tables,
                                   compute_t<Compute> *result) {
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<compute_t<Compute> *>(dynamic_shared);
  const auto context =
      prepare_context<Format, Compute, Decoder>(tables, table_shared);
  auto *reduce_shared = reinterpret_cast<compute_t<Compute> *>(
      dynamic_shared + shared_table_bytes_v<Format, Compute, Decoder>);
  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  compute_t<Compute> sum{};
  for (auto base = static_cast<std::size_t>(threadIdx.x) * Lanes;
       base < columns; base += static_cast<std::size_t>(blockDim.x) * Lanes) {
    std::uint32_t matrix_raw[Lanes]{};
    std::uint32_t vector_raw[Lanes]{};
    matrix.template raw_packet<Lanes>(row * columns + base, matrix_raw);
    vector.template raw_packet<Lanes>(base, vector_raw);
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      if (base + lane < columns) {
        sum += multiply<Format, Compute, Decoder, Multiply>(
            matrix_raw[lane], vector_raw[lane], context);
      }
    }
  }
  const auto reduced = block_sum(sum, reduce_shared);
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
}

template <int Bits> struct cooperative_geometry;

template <> struct cooperative_geometry<4> {
  static constexpr int values = 8;
  static constexpr int words = 1;
  static constexpr int consumers = 8;
  static constexpr int values_per_consumer = 1;
};
template <> struct cooperative_geometry<6> {
  static constexpr int values = 16;
  static constexpr int words = 3;
  static constexpr int consumers = 16;
  static constexpr int values_per_consumer = 1;
};
template <> struct cooperative_geometry<10> {
  static constexpr int values = 16;
  static constexpr int words = 5;
  static constexpr int consumers = 16;
  static constexpr int values_per_consumer = 1;
};
template <> struct cooperative_geometry<12> {
  static constexpr int values = 8;
  static constexpr int words = 3;
  static constexpr int consumers = 8;
  static constexpr int values_per_consumer = 1;
};

template <int Bits>
inline constexpr bool cooperative_supported_v =
    Bits == 4 || Bits == 6 || Bits == 10 || Bits == 12;

template <typename Format>
__device__ __forceinline__ std::uint32_t cooperative_raw(
    const std::uint32_t *words, std::size_t chunk, int lane, int output) {
  using geometry = cooperative_geometry<Format::total_bits>;
  const auto local_value = lane * geometry::values_per_consumer + output;
  const auto local_bit = local_value * Format::total_bits;
  const auto low_word = local_bit >> 5;
  const auto shift = local_bit & 31;
  const auto subgroup_lane = threadIdx.x & (geometry::consumers - 1);
  const auto loaded = subgroup_lane < geometry::words
                          ? words[chunk * geometry::words + subgroup_lane]
                          : 0u;
  const auto active = __activemask();
  const auto low = __shfl_sync(active, loaded, low_word, geometry::consumers);
  const auto high =
      __shfl_sync(active, loaded, low_word + 1, geometry::consumers);
  const auto pair = static_cast<std::uint64_t>(low) |
                    (static_cast<std::uint64_t>(high) << 32);
  return static_cast<std::uint32_t>(pair >> shift) &
         lns::raw_mask<Format>();
}

template <typename Format, compute_kind Compute, decoder_kind Decoder,
          multiply_kind Multiply>
__global__ void dot_cooperative_kernel(const std::uint32_t *left,
                                       const std::uint32_t *right,
                                       std::size_t count,
                                       table_bundle<Compute> tables,
                                       compute_t<Compute> *partials) {
  using geometry = cooperative_geometry<Format::total_bits>;
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<compute_t<Compute> *>(dynamic_shared);
  const auto context =
      prepare_context<Format, Compute, Decoder>(tables, table_shared);
  auto *reduce_shared = reinterpret_cast<compute_t<Compute> *>(
      dynamic_shared + shared_table_bytes_v<Format, Compute, Decoder>);

  const auto global_thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                             threadIdx.x;
  const auto group = global_thread / geometry::consumers;
  const auto groups = static_cast<std::size_t>(gridDim.x) * blockDim.x /
                      geometry::consumers;
  const auto lane = threadIdx.x & (geometry::consumers - 1);
  const auto chunks = (count + geometry::values - 1) / geometry::values;
  compute_t<Compute> sum{};
  for (auto chunk = group; chunk < chunks; chunk += groups) {
#pragma unroll
    for (int output = 0; output < geometry::values_per_consumer; ++output) {
      const auto index = chunk * geometry::values +
                         lane * geometry::values_per_consumer + output;
      const auto left_raw = cooperative_raw<Format>(left, chunk, lane, output);
      const auto right_raw = cooperative_raw<Format>(right, chunk, lane, output);
      if (index < count) {
        sum += multiply<Format, Compute, Decoder, Multiply>(
            left_raw, right_raw, context);
      }
    }
  }
  const auto reduced = block_sum(sum, reduce_shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename Format, compute_kind Compute, decoder_kind Decoder,
          multiply_kind Multiply>
__global__ void gemv_cooperative_kernel(const std::uint32_t *matrix,
                                        const std::uint32_t *vector,
                                        std::size_t rows,
                                        std::size_t columns,
                                        table_bundle<Compute> tables,
                                        compute_t<Compute> *result) {
  using geometry = cooperative_geometry<Format::total_bits>;
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<compute_t<Compute> *>(dynamic_shared);
  const auto context =
      prepare_context<Format, Compute, Decoder>(tables, table_shared);
  auto *reduce_shared = reinterpret_cast<compute_t<Compute> *>(
      dynamic_shared + shared_table_bytes_v<Format, Compute, Decoder>);
  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const auto lane = threadIdx.x & (geometry::consumers - 1);
  const auto group = threadIdx.x / geometry::consumers;
  const auto groups = blockDim.x / geometry::consumers;
  const auto chunks = (columns + geometry::values - 1) / geometry::values;
  compute_t<Compute> sum{};
  for (auto chunk = static_cast<std::size_t>(group); chunk < chunks;
       chunk += groups) {
    const auto column = chunk * geometry::values + lane;
    const auto matrix_chunk = (row * columns) / geometry::values + chunk;
    const auto matrix_raw =
        cooperative_raw<Format>(matrix, matrix_chunk, lane, 0);
    const auto vector_raw = cooperative_raw<Format>(vector, chunk, lane, 0);
    if (column < columns) {
      sum += multiply<Format, Compute, Decoder, Multiply>(
          matrix_raw, vector_raw, context);
    }
  }
  const auto reduced = block_sum(sum, reduce_shared);
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
}

template <typename Format>
__global__ void encode_padded_kernel(const double *source,
                                     lns::padded_storage_t<Format> *destination,
                                     std::size_t count) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    destination[index] = static_cast<lns::padded_storage_t<Format>>(
        lns::encode<Format>(source[index]));
  }
}

template <typename Format>
__global__ void pack_dense_kernel(
    const lns::padded_storage_t<Format> *source, std::uint32_t *destination,
    std::size_t count) {
  using geometry = bitwidth::dense_geometry<Format::total_bits>;
  const auto chunks =
      (count + geometry::values_per_aligned_chunk - 1) /
      geometry::values_per_aligned_chunk;
  for (auto chunk = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       chunk < chunks;
       chunk += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::uint32_t words[geometry::words_per_aligned_chunk]{};
#pragma unroll
    for (int value = 0; value < geometry::values_per_aligned_chunk; ++value) {
      const auto index = chunk * geometry::values_per_aligned_chunk + value;
      const auto raw = index < count
                           ? static_cast<std::uint32_t>(source[index]) &
                                 lns::raw_mask<Format>()
                           : 0u;
      const auto bit = value * Format::total_bits;
      const auto word = bit >> 5;
      const auto shift = bit & 31;
      const auto placed = static_cast<std::uint64_t>(raw) << shift;
      words[word] |= static_cast<std::uint32_t>(placed);
      if (shift + Format::total_bits > 32) {
        words[word + 1] |= static_cast<std::uint32_t>(placed >> 32);
      }
    }
#pragma unroll
    for (int word = 0; word < geometry::words_per_aligned_chunk; ++word) {
      destination[chunk * geometry::words_per_aligned_chunk + word] =
          words[word];
    }
  }
}

template <typename Format, compute_kind Compute, decoder_kind Decoder>
__global__ void validate_decode_kernel(const std::uint32_t *raw,
                                       std::size_t count,
                                       table_bundle<Compute> tables,
                                       compute_t<Compute> *output) {
  extern __shared__ unsigned char dynamic_shared[];
  auto *shared = reinterpret_cast<compute_t<Compute> *>(dynamic_shared);
  const auto context =
      prepare_context<Format, Compute, Decoder>(tables, shared);
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    output[index] = decode_raw<Format, Compute, Decoder>(raw[index], context);
  }
}

} // namespace aut::lns_strategy

#endif // ACCESSOR_UNIVERSAL_TEST_LNS_BENCHMARK_KERNELS_CUH_
