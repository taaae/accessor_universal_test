#ifndef ACCESSOR_UNIVERSAL_TEST_STORAGE_PERFORMANCE_KERNELS_CUH_
#define ACCESSOR_UNIVERSAL_TEST_STORAGE_PERFORMANCE_KERNELS_CUH_

#include "storage_kernels.cuh"

#include <cub/block/block_reduce.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aut::performance {

inline constexpr int block_threads = kernels::reduction_block_threads;

struct register_packet {
  std::uint64_t x{};
  std::uint64_t y{};
  std::uint64_t z{};
  std::uint64_t w{};
};

template <typename Format>
__device__ __forceinline__ std::uint64_t
load_scalar_bits(const storage::storage_type_t<Format> *values,
                 std::size_t index) {
  if constexpr (sizeof(storage::storage_type_t<Format>) == 1) {
    return reinterpret_cast<const std::uint8_t *>(values)[index];
  } else if constexpr (sizeof(storage::storage_type_t<Format>) == 2) {
    return reinterpret_cast<const std::uint16_t *>(values)[index];
  } else if constexpr (sizeof(storage::storage_type_t<Format>) == 4) {
    return reinterpret_cast<const std::uint32_t *>(values)[index];
  } else {
    return reinterpret_cast<const std::uint64_t *>(values)[index];
  }
}

template <typename Format, int Lanes>
__device__ __forceinline__ register_packet
load_register_packet(const storage::storage_type_t<Format> *values) {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4);
  register_packet result{};
  if constexpr (Lanes == 1) {
    result.x = load_scalar_bits<Format>(values, 0);
  } else if constexpr (Lanes == 2 &&
                       sizeof(storage::storage_type_t<Format>) == 1) {
    const auto packed = *reinterpret_cast<const std::uint16_t *>(values);
    result.x = packed & 0xffu;
    result.y = packed >> 8;
  } else if constexpr (Lanes == 2 &&
                       sizeof(storage::storage_type_t<Format>) == 2) {
    const auto packed = *reinterpret_cast<const std::uint32_t *>(values);
    result.x = packed & 0xffffu;
    result.y = packed >> 16;
  } else if constexpr (Lanes == 2 &&
                       sizeof(storage::storage_type_t<Format>) == 4) {
    const auto packed = *reinterpret_cast<const uint2 *>(values);
    result.x = packed.x;
    result.y = packed.y;
  } else if constexpr (Lanes == 2) {
    const auto packed = *reinterpret_cast<const ulonglong2 *>(values);
    result.x = packed.x;
    result.y = packed.y;
  } else if constexpr (sizeof(storage::storage_type_t<Format>) == 1) {
    const auto packed = *reinterpret_cast<const std::uint32_t *>(values);
    result.x = packed & 0xffu;
    result.y = (packed >> 8) & 0xffu;
    result.z = (packed >> 16) & 0xffu;
    result.w = packed >> 24;
  } else if constexpr (sizeof(storage::storage_type_t<Format>) == 2) {
    const auto packed = *reinterpret_cast<const uint2 *>(values);
    result.x = packed.x & 0xffffu;
    result.y = packed.x >> 16;
    result.z = packed.y & 0xffffu;
    result.w = packed.y >> 16;
  } else if constexpr (sizeof(storage::storage_type_t<Format>) == 4) {
    const auto packed = *reinterpret_cast<const uint4 *>(values);
    result.x = packed.x;
    result.y = packed.y;
    result.z = packed.z;
    result.w = packed.w;
  } else {
    const auto low = *reinterpret_cast<const ulonglong2 *>(values);
    const auto high = *reinterpret_cast<const ulonglong2 *>(values + 2);
    result.x = low.x;
    result.y = low.y;
    result.z = high.x;
    result.w = high.y;
  }
  return result;
}

template <typename Format>
__device__ __forceinline__ storage::storage_type_t<Format>
storage_from_bits(std::uint64_t bits) {
  using value_type = storage::storage_type_t<Format>;
  if constexpr (std::is_same_v<Format, storage::fp8_e4m3> ||
                std::is_same_v<Format, storage::fp8_e5m2>) {
    value_type result;
    result.__x = static_cast<__nv_fp8_storage_t>(bits);
    return result;
  } else if constexpr (std::is_same_v<Format, storage::fp16_e5m10>) {
    const __half_raw raw{static_cast<unsigned short>(bits)};
    return value_type{raw};
  } else if constexpr (std::is_same_v<Format, storage::bf16_e8m7>) {
    const __nv_bfloat16_raw raw{static_cast<unsigned short>(bits)};
    return value_type{raw};
  } else if constexpr (std::is_same_v<Format, storage::fp32_e8m23>) {
    return __uint_as_float(static_cast<unsigned>(bits));
  } else if constexpr (std::is_same_v<Format, storage::fp64_e11m52>) {
    return __longlong_as_double(static_cast<long long>(bits));
  } else {
    return static_cast<value_type>(bits);
  }
}

__device__ __forceinline__ void compiler_barrier(std::uint64_t &value,
                                                 bool wide) {
  if (wide) {
    asm volatile("" : "+l"(value));
  } else {
    auto narrow = static_cast<std::uint32_t>(value);
    asm volatile("" : "+r"(narrow));
    value = narrow;
  }
}

template <typename Format, int Lanes>
__device__ __forceinline__ void compiler_barrier(register_packet &packet) {
  constexpr bool wide = Format::total_bits == 64;
  compiler_barrier(packet.x, wide);
  if constexpr (Lanes >= 2) {
    compiler_barrier(packet.y, wide);
  }
  if constexpr (Lanes == 4) {
    compiler_barrier(packet.z, wide);
    compiler_barrier(packet.w, wide);
  }
}

template <typename Format, int Lanes>
__device__ __forceinline__ kernels::fp64x4
decode_register_packet(const register_packet &packet) {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4);
  if constexpr (std::is_same_v<Format, storage::fp8_e4m3> && Lanes >= 2) {
    if constexpr (Lanes == 2) {
      __nv_fp8x2_e4m3 packed;
      packed.__x =
          static_cast<__nv_fp8x2_storage_t>(packet.x | (packet.y << 8));
      const auto values = static_cast<float2>(packed);
      return {values.x, values.y, 0.0, 0.0};
    } else {
      __nv_fp8x4_e4m3 packed;
      packed.__x = static_cast<__nv_fp8x4_storage_t>(
          packet.x | (packet.y << 8) | (packet.z << 16) | (packet.w << 24));
      const auto values = static_cast<float4>(packed);
      return {values.x, values.y, values.z, values.w};
    }
  } else if constexpr (std::is_same_v<Format, storage::fp8_e5m2> &&
                       Lanes >= 2) {
    if constexpr (Lanes == 2) {
      __nv_fp8x2_e5m2 packed;
      packed.__x =
          static_cast<__nv_fp8x2_storage_t>(packet.x | (packet.y << 8));
      const auto values = static_cast<float2>(packed);
      return {values.x, values.y, 0.0, 0.0};
    } else {
      __nv_fp8x4_e5m2 packed;
      packed.__x = static_cast<__nv_fp8x4_storage_t>(
          packet.x | (packet.y << 8) | (packet.z << 16) | (packet.w << 24));
      const auto values = static_cast<float4>(packed);
      return {values.x, values.y, values.z, values.w};
    }
  } else if constexpr (std::is_same_v<Format, storage::fp16_e5m10> &&
                       Lanes >= 2) {
    const auto low_word = static_cast<std::uint32_t>(packet.x) |
                          (static_cast<std::uint32_t>(packet.y) << 16);
    const auto low = __half22float2(storage::detail::half2_from_word(low_word));
    if constexpr (Lanes == 2) {
      return {low.x, low.y, 0.0, 0.0};
    } else {
      const auto high_word = static_cast<std::uint32_t>(packet.z) |
                             (static_cast<std::uint32_t>(packet.w) << 16);
      const auto high =
          __half22float2(storage::detail::half2_from_word(high_word));
      return {low.x, low.y, high.x, high.y};
    }
  } else if constexpr (std::is_same_v<Format, storage::bf16_e8m7> &&
                       Lanes >= 2) {
    const auto low_word = static_cast<std::uint32_t>(packet.x) |
                          (static_cast<std::uint32_t>(packet.y) << 16);
    const auto low =
        __bfloat1622float2(storage::detail::bfloat162_from_word(low_word));
    if constexpr (Lanes == 2) {
      return {low.x, low.y, 0.0, 0.0};
    } else {
      const auto high_word = static_cast<std::uint32_t>(packet.z) |
                             (static_cast<std::uint32_t>(packet.w) << 16);
      const auto high =
          __bfloat1622float2(storage::detail::bfloat162_from_word(high_word));
      return {low.x, low.y, high.x, high.y};
    }
  } else {
    kernels::fp64x4 result{
        storage::decode<Format>(storage_from_bits<Format>(packet.x)), 0.0, 0.0,
        0.0};
    if constexpr (Lanes >= 2) {
      result.y = storage::decode<Format>(storage_from_bits<Format>(packet.y));
    }
    if constexpr (Lanes == 4) {
      result.z = storage::decode<Format>(storage_from_bits<Format>(packet.z));
      result.w = storage::decode<Format>(storage_from_bits<Format>(packet.w));
    }
    return result;
  }
}

template <int Lanes>
__device__ __forceinline__ void accumulate_values(kernels::fp64x4 &sums,
                                                  const kernels::fp64x4 &v) {
  sums.x += v.x;
  if constexpr (Lanes >= 2) {
    sums.y += v.y;
  }
  if constexpr (Lanes == 4) {
    sums.z += v.z;
    sums.w += v.w;
  }
}

/**
 * Conversion-throughput microbenchmark. Each thread loads one packet before
 * the loop. Empty inline-assembly barriers prevent loop-invariant decode
 * hoisting without generating GPU instructions, so timed work is register
 * decode plus the FP64 additions needed to keep results live.
 */
template <typename Format, int Lanes>
__global__ void register_decode(const storage::storage_type_t<Format> *values,
                                std::size_t count, int repeats, double *sink) {
  const auto thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto offset = thread * Lanes;
  if (offset + Lanes > count) {
    offset %= count - Lanes + 1;
  }
  auto packet = load_register_packet<Format, Lanes>(values + offset);
  kernels::fp64x4 sums{};
#pragma unroll 1
  for (int repeat = 0; repeat < repeats; ++repeat) {
    compiler_barrier<Format, Lanes>(packet);
    accumulate_values<Lanes>(sums,
                             decode_register_packet<Format, Lanes>(packet));
  }
  sink[thread] = kernels::combine_sums<Lanes>(sums);
}

struct xor_values {
  __device__ __forceinline__ unsigned long long
  operator()(unsigned long long left, unsigned long long right) const {
    return left ^ right;
  }
};

template <typename Format, int Lanes>
__device__ __forceinline__ unsigned long long
packet_checksum(const storage::storage_type_t<Format> *values) {
  const auto packet = load_register_packet<Format, Lanes>(values);
  auto result = static_cast<unsigned long long>(packet.x);
  if constexpr (Lanes >= 2) {
    result = (result << 7) ^ packet.y;
  }
  if constexpr (Lanes == 4) {
    result = (result << 11) ^ packet.z;
    result = (result << 13) ^ packet.w;
  }
  return result;
}

/** Stream storage bytes without decoding them. */
template <typename Format, int Lanes>
__global__ void stream_load(const storage::storage_type_t<Format> *values,
                            std::size_t count,
                            unsigned long long *block_checksums) {
  using block_reduce = cub::BlockReduce<unsigned long long, block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto pack_count = count / Lanes;
  unsigned long long checksum{};
  for (auto pack = first; pack < pack_count; pack += stride) {
    checksum ^= packet_checksum<Format, Lanes>(values + pack * Lanes);
  }
  const auto tail = static_cast<std::size_t>(Lanes) * pack_count + first;
  if (tail < count) {
    checksum ^= load_scalar_bits<Format>(values, tail);
  }
  const auto reduced = block_reduce(temporary).Reduce(checksum, xor_values{});
  if (threadIdx.x == 0) {
    block_checksums[blockIdx.x] = reduced;
  }
}

/** Stream the same storage bytes, decode to FP64, and reduce their values. */
template <typename Format, int Lanes>
__global__ void
stream_load_decode(const storage::storage_type_t<Format> *values,
                   std::size_t count, double *block_sums) {
  using block_reduce = cub::BlockReduce<double, block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto pack_count = count / Lanes;
  kernels::fp64x4 sums{};
  for (auto pack = first; pack < pack_count; pack += stride) {
    accumulate_values<Lanes>(
        sums, kernels::load_decoded<Format, Lanes>(values + pack * Lanes));
  }
  const auto tail = static_cast<std::size_t>(Lanes) * pack_count + first;
  if (tail < count) {
    sums.x += storage::decode<Format>(values[tail]);
  }
  const auto reduced =
      block_reduce(temporary).Sum(kernels::combine_sums<Lanes>(sums));
  if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = reduced;
  }
}

} // namespace aut::performance

#endif // ACCESSOR_UNIVERSAL_TEST_STORAGE_PERFORMANCE_KERNELS_CUH_
