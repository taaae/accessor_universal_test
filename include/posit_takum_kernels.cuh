#ifndef ACCESSOR_UNIVERSAL_TEST_POSIT_TAKUM_KERNELS_CUH_
#define ACCESSOR_UNIVERSAL_TEST_POSIT_TAKUM_KERNELS_CUH_

#include "posit_takum_core.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace aut::pt {

template <int Bits> struct storage_view {
  const void *data{};

  __device__ __forceinline__ std::uint32_t raw(std::size_t index) const {
    if constexpr (Bits == 8) {
      return static_cast<const std::uint8_t *>(data)[index];
    } else if constexpr (Bits == 16) {
      return static_cast<const std::uint16_t *>(data)[index];
    } else if constexpr (Bits == 32) {
      return static_cast<const std::uint32_t *>(data)[index];
    } else {
      static_assert(Bits == 14);
      const auto *words = static_cast<const std::uint32_t *>(data);
      const auto bit = index * std::size_t{14};
      const auto word = bit >> 5;
      const auto shift = static_cast<unsigned>(bit & 31u);
      const auto pair = static_cast<std::uint64_t>(words[word]) |
                        (static_cast<std::uint64_t>(words[word + 1]) << 32);
      return static_cast<std::uint32_t>(pair >> shift) & 0x3fffu;
    }
  }
};

template <typename Float>
__device__ __forceinline__ Float block_sum(Float value, Float *shared) {
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

template <family Family, int Bits, int Es, typename Float, strategy Strategy>
struct alternative_decoder {
  const Float *global_table{};

  __device__ __forceinline__ const Float *prepare(Float *shared_table) const {
    if constexpr (Strategy == strategy::full_lut_shared) {
      constexpr std::size_t entries = std::size_t{1} << Bits;
      for (std::size_t index = threadIdx.x; index < entries;
           index += blockDim.x) {
        shared_table[index] = global_table[index];
      }
      __syncthreads();
      return shared_table;
    } else {
      return global_table;
    }
  }

  __device__ __forceinline__ Float value(std::uint32_t raw,
                                         const Float *table) const {
    if constexpr (Strategy == strategy::direct) {
      return decode<Family, Bits, Es, Float>(raw);
    } else {
      return table[raw];
    }
  }
};

template <typename Decoder, int Bits, typename Float, bool SharedTable>
__global__ void dot_kernel(storage_view<Bits> left, storage_view<Bits> right,
                           std::size_t count, Decoder decoder,
                           Float *partials) {
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<Float *>(dynamic_shared);
  const auto *table = decoder.prepare(table_shared);
  constexpr std::size_t table_entries =
      SharedTable ? (std::size_t{1} << Bits) : 0u;
  auto *reduce_shared = reinterpret_cast<Float *>(
      dynamic_shared + table_entries * sizeof(Float));

  Float sum{};
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto a = decoder.value(left.raw(index), table);
    const auto b = decoder.value(right.raw(index), table);
    sum = a * b + sum;
  }
  const auto reduced = block_sum(sum, reduce_shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename Float>
__global__ void finalize_dot_kernel(const Float *partials, std::size_t count,
                                    Float *result) {
  __shared__ Float shared[256];
  Float sum{};
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    sum += partials[index];
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    *result = reduced;
  }
}

template <typename Decoder, int Bits, typename Float, bool SharedTable>
__global__ void gemv_kernel(storage_view<Bits> matrix,
                            storage_view<Bits> vector, std::size_t rows,
                            std::size_t columns, Decoder decoder,
                            Float *result) {
  extern __shared__ unsigned char dynamic_shared[];
  auto *table_shared = reinterpret_cast<Float *>(dynamic_shared);
  const auto *table = decoder.prepare(table_shared);
  constexpr std::size_t table_entries =
      SharedTable ? (std::size_t{1} << Bits) : 0u;
  auto *reduce_shared = reinterpret_cast<Float *>(
      dynamic_shared + table_entries * sizeof(Float));

  const auto row = static_cast<std::size_t>(blockIdx.x);
  Float sum{};
  if (row < rows) {
    for (std::size_t column = threadIdx.x; column < columns;
         column += blockDim.x) {
      const auto a = decoder.value(matrix.raw(row * columns + column), table);
      const auto b = decoder.value(vector.raw(column), table);
      sum = a * b + sum;
    }
  }
  const auto reduced = block_sum(sum, reduce_shared);
  if (threadIdx.x == 0 && row < rows) {
    result[row] = reduced;
  }
}

template <int Bits>
__global__ void validate_load_kernel(storage_view<Bits> input,
                                     std::uint32_t *output,
                                     std::size_t count) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    output[index] = input.raw(index);
  }
}

template <family Family, int Bits, int Es, typename Float>
__global__ void validate_decode_kernel(storage_view<Bits> input, Float *output,
                                       std::size_t count) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    output[index] = decode<Family, Bits, Es, Float>(input.raw(index));
  }
}

template <typename Decoder, int Bits, typename Float, bool SharedTable>
__global__ void validate_generic_decoder_kernel(storage_view<Bits> input,
                                                Decoder decoder,
                                                Float *output,
                                                std::size_t count) {
  extern __shared__ unsigned char dynamic_shared[];
  auto *shared = reinterpret_cast<Float *>(dynamic_shared);
  const auto *table = decoder.prepare(shared);
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    output[index] = decoder.value(input.raw(index), table);
  }
}

} // namespace aut::pt

#endif
