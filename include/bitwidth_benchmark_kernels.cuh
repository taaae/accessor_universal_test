#ifndef ACCESSOR_UNIVERSAL_TEST_BITWIDTH_BENCHMARK_KERNELS_CUH_
#define ACCESSOR_UNIVERSAL_TEST_BITWIDTH_BENCHMARK_KERNELS_CUH_

#include "bitwidth_benchmark_core.hpp"

#include <cuda_fp6.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aut::bitwidth {

template <typename Format> struct native_fp6_traits {
  static constexpr bool supported = false;
};

template <> struct native_fp6_traits<storage::e2m3> {
  static constexpr bool supported = true;
  static constexpr auto interpretation = __NV_E2M3;
};

template <> struct native_fp6_traits<storage::e3m2> {
  static constexpr bool supported = true;
  static constexpr auto interpretation = __NV_E3M2;
};

struct decoder_tables {
  const std::uint32_t *full{};
  const std::uint32_t *prefix{};
  const std::uint32_t *subnormal{};
};

template <typename Format, storage_layout Layout> struct storage_view {
  const std::uint32_t *dense{};
  const padded_storage_t<Format> *padded{};

  __device__ __forceinline__ std::uint32_t raw(std::size_t index) const {
    if constexpr (Layout == storage_layout::dense) {
      return load_dense_scalar<Format::total_bits>(dense, index);
    } else {
      return raw_from_padded<Format>(padded[index]);
    }
  }

  template <int Lanes>
  __device__ __forceinline__ void raw_packet(std::size_t base,
                                              std::uint32_t (&out)[Lanes]) const {
    if constexpr (Layout == storage_layout::padded) {
      struct alignas((sizeof(padded_storage_t<Format>) * Lanes >= 16)
                         ? 16
                         : sizeof(padded_storage_t<Format>) * Lanes) packet {
        padded_storage_t<Format> lane[Lanes];
      };
      const auto loaded =
          *reinterpret_cast<const packet *>(padded + base);
#pragma unroll
      for (int lane = 0; lane < Lanes; ++lane) {
        out[lane] = raw_from_padded<Format>(loaded.lane[lane]);
      }
    } else {
      // The values share the same compact bitstream segment.  ptxas can reuse
      // overlapping word loads; cooperative access below makes that reuse
      // explicit across threads.
#pragma unroll
      for (int lane = 0; lane < Lanes; ++lane) {
        out[lane] = load_dense_scalar<Format::total_bits>(dense, base + lane);
      }
    }
  }
};

template <typename Format, decoder_kind Decoder>
constexpr std::size_t table_entries() {
  if constexpr (Decoder == decoder_kind::full_lut_global ||
                Decoder == decoder_kind::full_lut_shared) {
    return std::size_t{1} << Format::total_bits;
  } else if constexpr (Decoder == decoder_kind::prefix_lut_global ||
                       Decoder == decoder_kind::prefix_lut_shared) {
    return std::size_t{1} << (Format::exponent_bits + 1);
  } else if constexpr (Decoder == decoder_kind::subnormal_lut_global ||
                       Decoder == decoder_kind::subnormal_lut_shared) {
    return std::size_t{1} << Format::fraction_bits;
  } else {
    return 0;
  }
}

template <decoder_kind Decoder> constexpr bool uses_shared_table() {
  return Decoder == decoder_kind::full_lut_shared ||
         Decoder == decoder_kind::prefix_lut_shared ||
         Decoder == decoder_kind::subnormal_lut_shared;
}

template <typename Format, decoder_kind Decoder>
__device__ __forceinline__ const std::uint32_t *
prepare_table(decoder_tables tables, std::uint32_t *shared) {
  const std::uint32_t *global{};
  if constexpr (Decoder == decoder_kind::full_lut_global ||
                Decoder == decoder_kind::full_lut_shared) {
    global = tables.full;
  } else if constexpr (Decoder == decoder_kind::prefix_lut_global ||
                       Decoder == decoder_kind::prefix_lut_shared) {
    global = tables.prefix;
  } else if constexpr (Decoder == decoder_kind::subnormal_lut_global ||
                       Decoder == decoder_kind::subnormal_lut_shared) {
    global = tables.subnormal;
  }
  if constexpr (uses_shared_table<Decoder>()) {
    for (std::size_t index = threadIdx.x;
         index < table_entries<Format, Decoder>(); index += blockDim.x) {
      shared[index] = global[index];
    }
    __syncthreads();
    return shared;
  } else {
    return global;
  }
}

template <typename Format, compute_kind Compute>
__device__ __forceinline__ compute_t<Compute>
value_from_high_word(std::uint32_t word) {
  if constexpr (Compute == compute_kind::fp32) {
    return decoder::bits_to_float(word);
  } else {
    return decoder::words_to_double({word, 0});
  }
}

template <typename Format, compute_kind Compute, decoder_kind Decoder>
__device__ __forceinline__ compute_t<Compute>
decode_with_table(std::uint32_t raw, const std::uint32_t *table) {
  using layout = format_layout_t<Format>;
  if constexpr (Decoder == decoder_kind::full_lut_global ||
                Decoder == decoder_kind::full_lut_shared) {
    return value_from_high_word<Format, Compute>(table[raw]);
  } else if constexpr (Decoder == decoder_kind::prefix_lut_global ||
                       Decoder == decoder_kind::prefix_lut_shared) {
    static_assert(Format::exponent_bits > 0);
    const auto fraction = raw & decoder::fraction_mask<layout>();
    const auto exponent =
        (raw >> Format::fraction_bits) & decoder::exponent_mask<layout>();
    if (exponent == 0 || decoder::is_special<layout>(exponent, fraction)) {
      return decode_raw<Format, Compute, decoder_kind::direct_branchy>(raw);
    }
    const auto prefix = raw >> Format::fraction_bits;
    if constexpr (Compute == compute_kind::fp32) {
      return decoder::bits_to_float(table[prefix] |
                                    (fraction << (23 - Format::fraction_bits)));
    } else if constexpr (Format::fraction_bits <= 20) {
      return decoder::words_to_double(
          {table[prefix] |
               (fraction << (20 - Format::fraction_bits)),
           0});
    } else {
      return decoder::words_to_double(
          {table[prefix] | (fraction >> (Format::fraction_bits - 20)),
           fraction << (52 - Format::fraction_bits)});
    }
  } else if constexpr (Decoder == decoder_kind::subnormal_lut_global ||
                       Decoder == decoder_kind::subnormal_lut_shared) {
    static_assert(Format::fraction_bits > 0);
    const auto fraction = raw & decoder::fraction_mask<layout>();
    const auto exponent =
        (raw >> Format::fraction_bits) & decoder::exponent_mask<layout>();
    if (exponent != 0) {
      return decode_raw<Format, Compute, decoder_kind::direct_branchy>(raw);
    }
    const auto sign = raw >> (Format::total_bits - 1);
    const auto high = table[fraction] | (sign << 31);
    return value_from_high_word<Format, Compute>(high);
  } else if constexpr (Decoder == decoder_kind::native_scalar) {
    static_assert(native_fp6_traits<Format>::supported);
    const __half converted(__nv_cvt_fp6_to_halfraw(
        static_cast<__nv_fp6_storage_t>(raw),
        native_fp6_traits<Format>::interpretation));
    return static_cast<compute_t<Compute>>(__half2float(converted));
  } else {
    return decode_raw<Format, Compute, Decoder>(raw);
  }
}

template <typename Format, compute_kind Compute, decoder_kind Decoder,
          int Lanes>
__device__ __forceinline__ void
decode_packet_values(const std::uint32_t (&raw)[Lanes],
                     const std::uint32_t *table,
                     compute_t<Compute> (&values)[Lanes]) {
  if constexpr (Decoder == decoder_kind::native_packed) {
    static_assert(native_fp6_traits<Format>::supported);
    static_assert(Lanes % 2 == 0);
#pragma unroll
    for (int lane = 0; lane < Lanes; lane += 2) {
      const auto packed = static_cast<__nv_fp6x2_storage_t>(
          raw[lane] | (raw[lane + 1] << 8));
      const __half2 converted(__nv_cvt_fp6x2_to_halfraw2(
          packed, native_fp6_traits<Format>::interpretation));
      const auto pair = __half22float2(converted);
      values[lane] = static_cast<compute_t<Compute>>(pair.x);
      values[lane + 1] = static_cast<compute_t<Compute>>(pair.y);
    }
  } else {
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      values[lane] = decode_with_table<Format, Compute, Decoder>(raw[lane], table);
    }
  }
}

template <typename T>
__device__ __forceinline__ T block_sum(T value, T *shared) {
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

template <typename Format, compute_kind Compute, storage_layout Layout,
          int Lanes, decoder_kind Decoder>
__global__ void dot_thread_kernel(storage_view<Format, Layout> left,
                                  storage_view<Format, Layout> right,
                                  std::size_t count, decoder_tables tables,
                                  compute_t<Compute> *partials) {
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<std::uint32_t *>(dynamic_shared);
  auto *table = prepare_table<Format, Decoder>(tables, table_shared);
  auto *reduce_shared = reinterpret_cast<compute_t<Compute> *>(
      dynamic_shared + table_entries<Format, Decoder>() *
                           (uses_shared_table<Decoder>() ? sizeof(std::uint32_t)
                                                        : 0));

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
    compute_t<Compute> left_values[Lanes]{};
    compute_t<Compute> right_values[Lanes]{};
    decode_packet_values<Format, Compute, Decoder>(left_raw, table, left_values);
    decode_packet_values<Format, Compute, Decoder>(right_raw, table, right_values);
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      if (base + lane < count) {
        sum = left_values[lane] * right_values[lane] + sum;
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
          int Lanes, decoder_kind Decoder>
__global__ void gemv_thread_kernel(storage_view<Format, Layout> matrix,
                                   storage_view<Format, Layout> vector,
                                   std::size_t rows, std::size_t columns,
                                   decoder_tables tables,
                                   compute_t<Compute> *result) {
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<std::uint32_t *>(dynamic_shared);
  auto *table = prepare_table<Format, Decoder>(tables, table_shared);
  auto *reduce_shared = reinterpret_cast<compute_t<Compute> *>(
      dynamic_shared + table_entries<Format, Decoder>() *
                           (uses_shared_table<Decoder>() ? sizeof(std::uint32_t)
                                                        : 0));
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
    compute_t<Compute> matrix_values[Lanes]{};
    compute_t<Compute> vector_values[Lanes]{};
    decode_packet_values<Format, Compute, Decoder>(matrix_raw, table,
                                                    matrix_values);
    decode_packet_values<Format, Compute, Decoder>(vector_raw, table,
                                                    vector_values);
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      if (base + lane < columns) {
        sum = matrix_values[lane] * vector_values[lane] + sum;
      }
    }
  }
  const auto reduced = block_sum(sum, reduce_shared);
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
}

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
  const auto low = __shfl_sync(active, loaded, low_word,
                               geometry::consumers);
  std::uint32_t high{};
  if (shift + Format::total_bits > 32) {
    high = __shfl_sync(active, loaded, low_word + 1,
                       geometry::consumers);
  }
  const auto pair = static_cast<std::uint64_t>(low) |
                    (static_cast<std::uint64_t>(high) << 32);
  return static_cast<std::uint32_t>(pair >> shift) &
         raw_mask<Format::total_bits>();
}

template <typename Format, compute_kind Compute, decoder_kind Decoder>
__global__ void dot_cooperative_kernel(const std::uint32_t *left,
                                       const std::uint32_t *right,
                                       std::size_t count,
                                       decoder_tables tables,
                                       compute_t<Compute> *partials) {
  using geometry = cooperative_geometry<Format::total_bits>;
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<std::uint32_t *>(dynamic_shared);
  auto *table = prepare_table<Format, Decoder>(tables, table_shared);
  auto *reduce_shared = reinterpret_cast<compute_t<Compute> *>(
      dynamic_shared + table_entries<Format, Decoder>() *
                           (uses_shared_table<Decoder>() ? sizeof(std::uint32_t)
                                                        : 0));
  const auto global_thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                             threadIdx.x;
  const auto group = global_thread / geometry::consumers;
  const auto groups = static_cast<std::size_t>(gridDim.x) * blockDim.x /
                      geometry::consumers;
  const auto lane = threadIdx.x & (geometry::consumers - 1);
  const auto chunk_count = (count + geometry::values - 1) / geometry::values;
  compute_t<Compute> sum{};
  for (auto chunk = group; chunk < chunk_count; chunk += groups) {
#pragma unroll
    for (int output = 0; output < geometry::values_per_consumer; ++output) {
      const auto index = chunk * geometry::values +
                         lane * geometry::values_per_consumer + output;
      const auto left_raw = cooperative_raw<Format>(left, chunk, lane, output);
      const auto right_raw = cooperative_raw<Format>(right, chunk, lane, output);
      if (index < count) {
        const auto a =
            decode_with_table<Format, Compute, Decoder>(left_raw, table);
        const auto b =
            decode_with_table<Format, Compute, Decoder>(right_raw, table);
        sum = a * b + sum;
      }
    }
  }
  const auto reduced = block_sum(sum, reduce_shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename Format, compute_kind Compute, decoder_kind Decoder>
__global__ void gemv_cooperative_kernel(const std::uint32_t *matrix,
                                        const std::uint32_t *vector,
                                        std::size_t rows,
                                        std::size_t columns,
                                        decoder_tables tables,
                                        compute_t<Compute> *result) {
  using geometry = cooperative_geometry<Format::total_bits>;
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<std::uint32_t *>(dynamic_shared);
  auto *table = prepare_table<Format, Decoder>(tables, table_shared);
  auto *reduce_shared = reinterpret_cast<compute_t<Compute> *>(
      dynamic_shared + table_entries<Format, Decoder>() *
                           (uses_shared_table<Decoder>() ? sizeof(std::uint32_t)
                                                        : 0));
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
#pragma unroll
    for (int output = 0; output < geometry::values_per_consumer; ++output) {
      const auto column = chunk * geometry::values +
                          lane * geometry::values_per_consumer + output;
      const auto matrix_chunk =
          (row * columns) / geometry::values + chunk;
      const auto matrix_raw = cooperative_raw<Format>(
          matrix, matrix_chunk, lane, output);
      const auto vector_raw =
          cooperative_raw<Format>(vector, chunk, lane, output);
      if (column < columns) {
        const auto a =
            decode_with_table<Format, Compute, Decoder>(matrix_raw, table);
        const auto b =
            decode_with_table<Format, Compute, Decoder>(vector_raw, table);
        sum = a * b + sum;
      }
    }
  }
  const auto reduced = block_sum(sum, reduce_shared);
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
}

template <typename Format>
__global__ void encode_padded_kernel(const double *source,
                                     padded_storage_t<Format> *destination,
                                     std::size_t count) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    destination[index] = storage::encode<Format>(source[index]);
  }
}

template <typename Format>
__global__ void pack_dense_kernel(const padded_storage_t<Format> *source,
                                  std::uint32_t *destination,
                                  std::size_t count) {
  using geometry = dense_geometry<Format::total_bits>;
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
      const auto raw = index < count ? raw_from_padded<Format>(source[index]) : 0;
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
      destination[chunk * geometry::words_per_aligned_chunk + word] = words[word];
    }
  }
}

template <typename Target>
__global__ void cast_source_kernel(const double *source, Target *destination,
                                   std::size_t count) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    destination[index] = static_cast<Target>(source[index]);
  }
}

template <typename Target, int Lanes>
__global__ void raw_dot_kernel(const Target *left, const Target *right,
                               std::size_t count, Target *partials) {
  __shared__ Target shared[256];
  Target sum{};
  const auto packets = (count + Lanes - 1) / Lanes;
  for (auto packet = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
       packet < packets;
       packet += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto base = packet * Lanes;
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      if (base + lane < count) {
        sum = left[base + lane] * right[base + lane] + sum;
      }
    }
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename Target, int Lanes>
__global__ void raw_gemv_kernel(const Target *matrix, const Target *vector,
                                std::size_t rows, std::size_t columns,
                                Target *result) {
  __shared__ Target shared[256];
  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  Target sum{};
  for (auto base = static_cast<std::size_t>(threadIdx.x) * Lanes;
       base < columns; base += static_cast<std::size_t>(blockDim.x) * Lanes) {
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      if (base + lane < columns) {
        sum = matrix[row * columns + base + lane] * vector[base + lane] + sum;
      }
    }
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
}

} // namespace aut::bitwidth

#endif // ACCESSOR_UNIVERSAL_TEST_BITWIDTH_BENCHMARK_KERNELS_CUH_
