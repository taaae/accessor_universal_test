#ifndef ACCESSOR_UNIVERSAL_TEST_PRECISION_PACKING_KERNELS_CUH_
#define ACCESSOR_UNIVERSAL_TEST_PRECISION_PACKING_KERNELS_CUH_

#include "format_decoder_strategies.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aut::precision_packing {

namespace fs = format_strategies;

inline constexpr int block_threads = 256;

enum class arithmetic_kind { fp16, bf16, fp32, fp64 };
enum class access_kind { scalar_single, scalar_unrolled, vector_packet };

template <arithmetic_kind Kind> struct arithmetic;

template <> struct arithmetic<arithmetic_kind::fp16> {
  using type = __half;
  __device__ __forceinline__ static type zero() { return __float2half(0.0f); }
  __device__ __forceinline__ static type from_float(float value) {
    return __float2half(value);
  }
  __device__ __forceinline__ static type add(type left, type right) {
    return __hadd(left, right);
  }
  __device__ __forceinline__ static type fma(type left, type right, type sum) {
    return __hfma(left, right, sum);
  }
  __device__ __forceinline__ static double to_double(type value) {
    return static_cast<double>(__half2float(value));
  }
};

template <> struct arithmetic<arithmetic_kind::bf16> {
  using type = __nv_bfloat16;
  __device__ __forceinline__ static type zero() {
    return __float2bfloat16(0.0f);
  }
  __device__ __forceinline__ static type from_float(float value) {
    return __float2bfloat16(value);
  }
  __device__ __forceinline__ static type add(type left, type right) {
    return __hadd(left, right);
  }
  __device__ __forceinline__ static type fma(type left, type right, type sum) {
    return __hfma(left, right, sum);
  }
  __device__ __forceinline__ static double to_double(type value) {
    return static_cast<double>(__bfloat162float(value));
  }
};

template <> struct arithmetic<arithmetic_kind::fp32> {
  using type = float;
  __device__ __forceinline__ static type zero() { return 0.0f; }
  __device__ __forceinline__ static type from_float(float value) {
    return value;
  }
  __device__ __forceinline__ static type add(type left, type right) {
    return left + right;
  }
  __device__ __forceinline__ static type fma(type left, type right, type sum) {
    return fmaf(left, right, sum);
  }
  __device__ __forceinline__ static double to_double(type value) {
    return static_cast<double>(value);
  }
};

template <> struct arithmetic<arithmetic_kind::fp64> {
  using type = double;
  __device__ __forceinline__ static type zero() { return 0.0; }
  __device__ __forceinline__ static type from_float(float value) {
    return static_cast<double>(value);
  }
  __device__ __forceinline__ static type add(type left, type right) {
    return left + right;
  }
  __device__ __forceinline__ static type fma(type left, type right, type sum) {
    return ::fma(left, right, sum);
  }
  __device__ __forceinline__ static double to_double(type value) {
    return value;
  }
};

template <arithmetic_kind Kind>
using arithmetic_t = typename arithmetic<Kind>::type;

template <typename Format>
using device_storage_t =
    std::conditional_t<(Format::total_bits < 8), std::uint8_t,
                       storage::storage_type_t<Format>>;

template <typename Format>
inline constexpr std::size_t storage_bytes(std::size_t logical_count) {
  if constexpr (Format::total_bits < 8) {
    return (logical_count * Format::total_bits + 7) / 8;
  } else {
    return logical_count * sizeof(storage::storage_type_t<Format>);
  }
}

template <typename Format>
__device__ __forceinline__ std::uint32_t
load_scalar_raw(const device_storage_t<Format> *values,
                std::size_t logical_index) {
  if constexpr (std::is_same_v<Format, storage::fp64_e11m52>) {
    // FP64 bypasses this raw-code path and is handled directly below.
    return 0;
  } else {
    return fs::load_raw_code<Format>(values, logical_index);
  }
}

template <typename Format, arithmetic_kind Arithmetic>
__device__ __forceinline__ arithmetic_t<Arithmetic>
decode_storage_as(storage::storage_type_t<Format> value) {
  using ops = arithmetic<Arithmetic>;
  if constexpr (Arithmetic == arithmetic_kind::fp16 &&
                std::is_same_v<Format, storage::fp16_e5m10>) {
    return value;
  } else if constexpr (Arithmetic == arithmetic_kind::bf16 &&
                       std::is_same_v<Format, storage::bf16_e8m7>) {
    return value;
  } else if constexpr (Arithmetic == arithmetic_kind::fp32 &&
                       std::is_same_v<Format, storage::fp32_e8m23>) {
    return value;
  } else if constexpr (Arithmetic == arithmetic_kind::fp64 &&
                       std::is_same_v<Format, storage::fp64_e11m52>) {
    return value;
  } else if constexpr (std::is_same_v<Format, storage::fp8_e4m3> ||
                       std::is_same_v<Format, storage::fp8_e5m2>) {
    if constexpr (Arithmetic == arithmetic_kind::fp16) {
      return static_cast<__half>(value);
    } else {
      return ops::from_float(static_cast<float>(value));
    }
  } else if constexpr (std::is_same_v<Format, storage::fp16_e5m10>) {
    return ops::from_float(__half2float(value));
  } else if constexpr (std::is_same_v<Format, storage::bf16_e8m7>) {
    return ops::from_float(__bfloat162float(value));
  } else if constexpr (std::is_same_v<Format, storage::fp32_e8m23>) {
    return ops::from_float(value);
  } else {
    return ops::from_float(static_cast<float>(storage::decode<Format>(value)));
  }
}

template <typename Format, arithmetic_kind Arithmetic>
__device__ __forceinline__ arithmetic_t<Arithmetic>
decode_raw_as(std::uint32_t raw) {
  return decode_storage_as<Format, Arithmetic>(
      fs::storage_from_raw<Format>(raw));
}

template <typename Format, arithmetic_kind Arithmetic>
__device__ __forceinline__ arithmetic_t<Arithmetic>
load_scalar(const device_storage_t<Format> *values, std::size_t logical_index) {
  if constexpr (std::is_same_v<Format, storage::fp64_e11m52>) {
    static_assert(Arithmetic == arithmetic_kind::fp64);
    return values[logical_index];
  } else {
    return decode_raw_as<Format, Arithmetic>(
        load_scalar_raw<Format>(values, logical_index));
  }
}

template <typename Value, int Lanes> struct value_packet {
  Value lane[Lanes];
};

template <typename Format, arithmetic_kind Arithmetic, int Lane, int Lanes>
__device__ __forceinline__ arithmetic_t<Arithmetic>
decode_packet_lane(const fs::source_packet<Lanes> &packet) {
  return decode_raw_as<Format, Arithmetic>(
      fs::raw_lane<Format, Lane, Lanes>(packet));
}

template <typename Format, arithmetic_kind Arithmetic, int Lanes>
__device__ __forceinline__ value_packet<arithmetic_t<Arithmetic>, Lanes>
load_vector_packet(const device_storage_t<Format> *values,
                   std::size_t logical_offset) {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8);
  value_packet<arithmetic_t<Arithmetic>, Lanes> result{};
  if constexpr (std::is_same_v<Format, storage::fp64_e11m52>) {
    static_assert(Arithmetic == arithmetic_kind::fp64);
    if constexpr (Lanes == 1) {
      result.lane[0] = values[logical_offset];
    } else {
#pragma unroll
      for (int pair = 0; pair < Lanes / 2; ++pair) {
        const auto loaded = *reinterpret_cast<const double2 *>(
            values + logical_offset + 2 * pair);
        result.lane[2 * pair] = loaded.x;
        result.lane[2 * pair + 1] = loaded.y;
      }
    }
  } else {
    const auto packet =
        fs::load_source_packet<Format, Lanes>(values, logical_offset);
    result.lane[0] = decode_packet_lane<Format, Arithmetic, 0>(packet);
    if constexpr (Lanes >= 2) {
      result.lane[1] = decode_packet_lane<Format, Arithmetic, 1>(packet);
    }
    if constexpr (Lanes >= 4) {
      result.lane[2] = decode_packet_lane<Format, Arithmetic, 2>(packet);
      result.lane[3] = decode_packet_lane<Format, Arithmetic, 3>(packet);
    }
    if constexpr (Lanes == 8) {
      result.lane[4] = decode_packet_lane<Format, Arithmetic, 4>(packet);
      result.lane[5] = decode_packet_lane<Format, Arithmetic, 5>(packet);
      result.lane[6] = decode_packet_lane<Format, Arithmetic, 6>(packet);
      result.lane[7] = decode_packet_lane<Format, Arithmetic, 7>(packet);
    }
  }
  return result;
}

template <typename Format, arithmetic_kind Arithmetic, access_kind Access,
          int Lanes>
__device__ __forceinline__ value_packet<arithmetic_t<Arithmetic>, Lanes>
load_values(const device_storage_t<Format> *values,
            std::size_t logical_offset) {
  static_assert((Access == access_kind::scalar_single && Lanes == 1) ||
                Access != access_kind::scalar_single);
  if constexpr (Access == access_kind::vector_packet) {
    return load_vector_packet<Format, Arithmetic, Lanes>(values,
                                                         logical_offset);
  } else {
    value_packet<arithmetic_t<Arithmetic>, Lanes> result{};
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      result.lane[lane] =
          load_scalar<Format, Arithmetic>(values, logical_offset + lane);
    }
    return result;
  }
}

template <arithmetic_kind Arithmetic>
__device__ __forceinline__ arithmetic_t<Arithmetic>
block_sum(arithmetic_t<Arithmetic> value) {
  __shared__ arithmetic_t<Arithmetic> shared[block_threads];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = block_threads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] = arithmetic<Arithmetic>::add(
          shared[threadIdx.x], shared[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  return shared[0];
}

template <arithmetic_kind Arithmetic, int Lanes>
__device__ __forceinline__ arithmetic_t<Arithmetic>
combine_accumulators(arithmetic_t<Arithmetic> (&sums)[Lanes]) {
  auto result = sums[0];
#pragma unroll
  for (int lane = 1; lane < Lanes; ++lane) {
    result = arithmetic<Arithmetic>::add(result, sums[lane]);
  }
  return result;
}

template <typename Format, arithmetic_kind Arithmetic, access_kind Access,
          int Lanes>
__global__ void dot_map(const device_storage_t<Format> *left,
                        const device_storage_t<Format> *right,
                        std::size_t count, arithmetic_t<Arithmetic> *partials) {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4 || Lanes == 8);
  const auto thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto packet_count = count / Lanes;
  arithmetic_t<Arithmetic> sums[Lanes];
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    sums[lane] = arithmetic<Arithmetic>::zero();
  }
  for (auto packet_index = thread; packet_index < packet_count;
       packet_index += stride) {
    const auto logical = packet_index * Lanes;
    const auto a =
        load_values<Format, Arithmetic, Access, Lanes>(left, logical);
    const auto b =
        load_values<Format, Arithmetic, Access, Lanes>(right, logical);
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      sums[lane] =
          arithmetic<Arithmetic>::fma(a.lane[lane], b.lane[lane], sums[lane]);
    }
  }
  const auto tail = packet_count * Lanes + thread;
  if (tail < count) {
    const auto a = load_scalar<Format, Arithmetic>(left, tail);
    const auto b = load_scalar<Format, Arithmetic>(right, tail);
    sums[0] = arithmetic<Arithmetic>::fma(a, b, sums[0]);
  }
  const auto total = block_sum<Arithmetic>(combine_accumulators(sums));
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = total;
  }
}

template <arithmetic_kind Arithmetic>
__global__ void dot_finalize(const arithmetic_t<Arithmetic> *partials,
                             std::size_t count, double *result) {
  double sum{};
  for (auto index = static_cast<std::size_t>(threadIdx.x); index < count;
       index += blockDim.x) {
    sum += arithmetic<Arithmetic>::to_double(partials[index]);
  }
  __shared__ double shared[block_threads];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = block_threads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *result = shared[0];
  }
}

template <typename Format, arithmetic_kind Arithmetic, access_kind Access,
          int Lanes>
__global__ void gemv(const device_storage_t<Format> *matrix,
                     const device_storage_t<Format> *vector, std::size_t rows,
                     std::size_t columns, std::size_t stride, double *result) {
  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  const auto packet_count = columns / Lanes;
  arithmetic_t<Arithmetic> sums[Lanes];
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    sums[lane] = arithmetic<Arithmetic>::zero();
  }
  for (auto packet_index = static_cast<std::size_t>(threadIdx.x);
       packet_index < packet_count; packet_index += blockDim.x) {
    const auto column = packet_index * Lanes;
    const auto a = load_values<Format, Arithmetic, Access, Lanes>(
        matrix, row * stride + column);
    const auto x =
        load_values<Format, Arithmetic, Access, Lanes>(vector, column);
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      sums[lane] =
          arithmetic<Arithmetic>::fma(a.lane[lane], x.lane[lane], sums[lane]);
    }
  }
  const auto tail = packet_count * Lanes + threadIdx.x;
  if (tail < columns) {
    const auto a = load_scalar<Format, Arithmetic>(matrix, row * stride + tail);
    const auto x = load_scalar<Format, Arithmetic>(vector, tail);
    sums[0] = arithmetic<Arithmetic>::fma(a, x, sums[0]);
  }
  const auto total = block_sum<Arithmetic>(combine_accumulators(sums));
  if (threadIdx.x == 0) {
    result[row] = arithmetic<Arithmetic>::to_double(total);
  }
}

template <typename Format>
__global__ void encode_values(const double *source,
                              device_storage_t<Format> *destination,
                              std::size_t logical_count) {
  const auto thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  if constexpr (Format::total_bits < 8) {
    constexpr auto values_per_byte = 8 / Format::total_bits;
    const auto byte_count =
        (logical_count + values_per_byte - 1) / values_per_byte;
    for (auto byte = thread; byte < byte_count; byte += stride) {
      std::uint8_t packed{};
#pragma unroll
      for (int lane = 0; lane < values_per_byte; ++lane) {
        const auto logical = byte * values_per_byte + lane;
        if (logical < logical_count) {
          const auto code = static_cast<std::uint8_t>(
              storage::encode<Format>(source[logical]));
          packed |=
              static_cast<std::uint8_t>(code << (lane * Format::total_bits));
        }
      }
      destination[byte] = packed;
    }
  } else {
    for (auto index = thread; index < logical_count; index += stride) {
      destination[index] = storage::encode<Format>(source[index]);
    }
  }
}

template <typename Format>
__device__ __forceinline__ std::uint64_t
scalar_raw_checksum(const device_storage_t<Format> *values,
                    std::size_t logical_index) {
  if constexpr (std::is_same_v<Format, storage::fp64_e11m52>) {
    return static_cast<std::uint64_t>(
        __double_as_longlong(values[logical_index]));
  } else {
    return load_scalar_raw<Format>(values, logical_index);
  }
}

template <typename Format, access_kind Access, int Lanes>
__device__ __forceinline__ std::uint64_t
packet_checksum(const device_storage_t<Format> *values,
                std::size_t logical_offset) {
  std::uint64_t checksum{};
  if constexpr (Access == access_kind::vector_packet &&
                std::is_same_v<Format, storage::fp64_e11m52>) {
    if constexpr (Lanes == 1) {
      checksum = scalar_raw_checksum<Format>(values, logical_offset);
    } else {
#pragma unroll
      for (int pair = 0; pair < Lanes / 2; ++pair) {
        const auto loaded = *reinterpret_cast<const double2 *>(
            values + logical_offset + 2 * pair);
        checksum ^= static_cast<std::uint64_t>(__double_as_longlong(loaded.x));
        checksum = (checksum << 7) ^
                   static_cast<std::uint64_t>(__double_as_longlong(loaded.y));
      }
    }
  } else if constexpr (Access == access_kind::vector_packet) {
    const auto packet =
        fs::load_source_packet<Format, Lanes>(values, logical_offset);
    constexpr auto word_count = (Format::total_bits * Lanes + 31) / 32;
#pragma unroll
    for (int word = 0; word < word_count; ++word) {
      checksum = (checksum << 11) ^ packet.words[word];
    }
  } else {
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      checksum = (checksum << 7) ^
                 scalar_raw_checksum<Format>(values, logical_offset + lane);
    }
  }
  return checksum;
}

__device__ __forceinline__ std::uint64_t block_xor(std::uint64_t value) {
  __shared__ std::uint64_t shared[block_threads];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = block_threads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] ^= shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  return shared[0];
}

template <typename Format, access_kind Access, int Lanes>
__global__ void stream_load(const device_storage_t<Format> *values,
                            std::size_t count, std::uint64_t *block_checksums) {
  const auto thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto packet_count = count / Lanes;
  std::uint64_t checksum{};
  for (auto packet = thread; packet < packet_count; packet += stride) {
    checksum ^= packet_checksum<Format, Access, Lanes>(values, packet * Lanes);
  }
  const auto tail = packet_count * Lanes + thread;
  if (tail < count) {
    checksum ^= scalar_raw_checksum<Format>(values, tail);
  }
  const auto total = block_xor(checksum);
  if (threadIdx.x == 0) {
    block_checksums[blockIdx.x] = total;
  }
}

template <typename Format, arithmetic_kind Arithmetic, access_kind Access,
          int Lanes>
__global__ void stream_decode(const device_storage_t<Format> *values,
                              std::size_t count, double *block_sums) {
  const auto thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto packet_count = count / Lanes;
  arithmetic_t<Arithmetic> sums[Lanes];
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    sums[lane] = arithmetic<Arithmetic>::zero();
  }
  for (auto packet = thread; packet < packet_count; packet += stride) {
    const auto decoded =
        load_values<Format, Arithmetic, Access, Lanes>(values, packet * Lanes);
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      sums[lane] = arithmetic<Arithmetic>::add(sums[lane], decoded.lane[lane]);
    }
  }
  const auto tail = packet_count * Lanes + thread;
  if (tail < count) {
    sums[0] = arithmetic<Arithmetic>::add(
        sums[0], load_scalar<Format, Arithmetic>(values, tail));
  }
  const auto total = block_sum<Arithmetic>(combine_accumulators(sums));
  if (threadIdx.x == 0) {
    block_sums[blockIdx.x] = arithmetic<Arithmetic>::to_double(total);
  }
}

template <typename Format, int Lanes> struct register_raw_packet {
  std::uint64_t lane[Lanes];
};

template <typename Format, int Lanes>
__device__ __forceinline__ register_raw_packet<Format, Lanes>
load_register_raw_packet(const device_storage_t<Format> *values,
                         std::size_t logical_offset) {
  register_raw_packet<Format, Lanes> result{};
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    result.lane[lane] =
        scalar_raw_checksum<Format>(values, logical_offset + lane);
  }
  return result;
}

template <typename Format>
__device__ __forceinline__ void register_barrier(std::uint64_t &raw) {
  if constexpr (std::is_same_v<Format, storage::fp64_e11m52>) {
    asm volatile("" : "+l"(raw));
  } else {
    auto narrow = static_cast<std::uint32_t>(raw);
    asm volatile("" : "+r"(narrow));
    raw = narrow;
  }
}

template <typename Format, arithmetic_kind Arithmetic>
__device__ __forceinline__ arithmetic_t<Arithmetic>
decode_register_raw(std::uint64_t raw) {
  if constexpr (std::is_same_v<Format, storage::fp64_e11m52>) {
    static_assert(Arithmetic == arithmetic_kind::fp64);
    return __longlong_as_double(static_cast<long long>(raw));
  } else {
    return decode_raw_as<Format, Arithmetic>(static_cast<std::uint32_t>(raw));
  }
}

template <typename Format, arithmetic_kind Arithmetic, int Lanes>
__global__ void register_decode(const device_storage_t<Format> *values,
                                std::size_t count, int repeats,
                                double *thread_sums) {
  const auto thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto logical = thread * Lanes;
  if (logical + Lanes > count) {
    logical %= count - Lanes + 1;
  }
  auto packet = load_register_raw_packet<Format, Lanes>(values, logical);
  arithmetic_t<Arithmetic> sums[Lanes];
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane) {
    sums[lane] = arithmetic<Arithmetic>::zero();
  }
#pragma unroll 1
  for (int repeat = 0; repeat < repeats; ++repeat) {
#pragma unroll
    for (int lane = 0; lane < Lanes; ++lane) {
      register_barrier<Format>(packet.lane[lane]);
      sums[lane] = arithmetic<Arithmetic>::add(
          sums[lane],
          decode_register_raw<Format, Arithmetic>(packet.lane[lane]));
    }
  }
  thread_sums[thread] = arithmetic<Arithmetic>::to_double(
      combine_accumulators<Arithmetic, Lanes>(sums));
}

template <arithmetic_kind Arithmetic, int Chains>
__global__ void arithmetic_chain(int repeats, double *thread_sums) {
  static_assert(Chains == 1 || Chains == 2 || Chains == 4 || Chains == 8);
  const auto thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto source = static_cast<float>((thread & 31u) + 1u) * (1.0f / 64.0f);
  asm volatile("" : "+f"(source));
  const auto value = arithmetic<Arithmetic>::from_float(source);
  arithmetic_t<Arithmetic> sums[Chains];
#pragma unroll
  for (int chain = 0; chain < Chains; ++chain) {
    sums[chain] = arithmetic<Arithmetic>::from_float(
        static_cast<float>(chain + 1) * (1.0f / 256.0f));
  }
#pragma unroll 1
  for (int repeat = 0; repeat < repeats; ++repeat) {
#pragma unroll
    for (int chain = 0; chain < Chains; ++chain) {
      sums[chain] = arithmetic<Arithmetic>::fma(value, value, sums[chain]);
    }
  }
  thread_sums[thread] = arithmetic<Arithmetic>::to_double(
      combine_accumulators<Arithmetic, Chains>(sums));
}

// Packed arithmetic is intentionally separate from vector-packet access so a
// timing change cannot be attributed to both at once.
template <typename Format>
struct supports_packed_arithmetic : std::false_type {};
template <>
struct supports_packed_arithmetic<storage::fp16_e5m10> : std::true_type {};
template <>
struct supports_packed_arithmetic<storage::bf16_e8m7> : std::true_type {};

template <typename Format>
inline constexpr bool supports_packed_arithmetic_v =
    supports_packed_arithmetic<Format>::value;

template <typename Format, int Lanes>
__global__ void packed_arithmetic_dot_map(
    const device_storage_t<Format> *left, const device_storage_t<Format> *right,
    std::size_t count, storage::storage_type_t<Format> *partials) {
  static_assert(supports_packed_arithmetic_v<Format>);
  static_assert(Lanes == 2 || Lanes == 4 || Lanes == 8);
  using pair_type =
      std::conditional_t<std::is_same_v<Format, storage::fp16_e5m10>, __half2,
                         __nv_bfloat162>;
  pair_type sums[Lanes / 2];
#pragma unroll
  for (int pair = 0; pair < Lanes / 2; ++pair) {
    if constexpr (std::is_same_v<Format, storage::fp16_e5m10>) {
      const auto zero = __float2half(0.0f);
      sums[pair] = __halves2half2(zero, zero);
    } else {
      const auto zero = __float2bfloat16(0.0f);
      sums[pair] = __halves2bfloat162(zero, zero);
    }
  }
  const auto thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto packet_count = count / Lanes;
  for (auto packet = thread; packet < packet_count; packet += stride) {
    const auto logical = packet * Lanes;
#pragma unroll
    for (int pair = 0; pair < Lanes / 2; ++pair) {
      const auto a =
          *reinterpret_cast<const pair_type *>(left + logical + 2 * pair);
      const auto b =
          *reinterpret_cast<const pair_type *>(right + logical + 2 * pair);
      sums[pair] = __hfma2(a, b, sums[pair]);
    }
  }
  storage::storage_type_t<Format> scalar_sum;
  if constexpr (std::is_same_v<Format, storage::fp16_e5m10>) {
    scalar_sum = __hadd(__low2half(sums[0]), __high2half(sums[0]));
#pragma unroll
    for (int pair = 1; pair < Lanes / 2; ++pair) {
      scalar_sum = __hadd(
          scalar_sum, __hadd(__low2half(sums[pair]), __high2half(sums[pair])));
    }
    const auto total = block_sum<arithmetic_kind::fp16>(scalar_sum);
    if (threadIdx.x == 0) {
      partials[blockIdx.x] = total;
    }
  } else {
    scalar_sum = __hadd(__low2bfloat16(sums[0]), __high2bfloat16(sums[0]));
#pragma unroll
    for (int pair = 1; pair < Lanes / 2; ++pair) {
      scalar_sum = __hadd(scalar_sum, __hadd(__low2bfloat16(sums[pair]),
                                             __high2bfloat16(sums[pair])));
    }
    const auto total = block_sum<arithmetic_kind::bf16>(scalar_sum);
    if (threadIdx.x == 0) {
      partials[blockIdx.x] = total;
    }
  }
}

template <typename Format, int Lanes>
__global__ void packed_arithmetic_gemv(const device_storage_t<Format> *matrix,
                                       const device_storage_t<Format> *vector,
                                       std::size_t rows, std::size_t columns,
                                       std::size_t stride, double *result) {
  static_assert(supports_packed_arithmetic_v<Format>);
  static_assert(Lanes == 2 || Lanes == 4 || Lanes == 8);
  using pair_type =
      std::conditional_t<std::is_same_v<Format, storage::fp16_e5m10>, __half2,
                         __nv_bfloat162>;
  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  pair_type sums[Lanes / 2];
#pragma unroll
  for (int pair = 0; pair < Lanes / 2; ++pair) {
    if constexpr (std::is_same_v<Format, storage::fp16_e5m10>) {
      const auto zero = __float2half(0.0f);
      sums[pair] = __halves2half2(zero, zero);
    } else {
      const auto zero = __float2bfloat16(0.0f);
      sums[pair] = __halves2bfloat162(zero, zero);
    }
  }
  const auto packet_count = columns / Lanes;
  for (auto packet = static_cast<std::size_t>(threadIdx.x);
       packet < packet_count; packet += blockDim.x) {
    const auto column = packet * Lanes;
#pragma unroll
    for (int pair = 0; pair < Lanes / 2; ++pair) {
      const auto a = *reinterpret_cast<const pair_type *>(
          matrix + row * stride + column + 2 * pair);
      const auto x =
          *reinterpret_cast<const pair_type *>(vector + column + 2 * pair);
      sums[pair] = __hfma2(a, x, sums[pair]);
    }
  }
  storage::storage_type_t<Format> scalar_sum;
  if constexpr (std::is_same_v<Format, storage::fp16_e5m10>) {
    scalar_sum = __hadd(__low2half(sums[0]), __high2half(sums[0]));
#pragma unroll
    for (int pair = 1; pair < Lanes / 2; ++pair) {
      scalar_sum = __hadd(
          scalar_sum, __hadd(__low2half(sums[pair]), __high2half(sums[pair])));
    }
    const auto total = block_sum<arithmetic_kind::fp16>(scalar_sum);
    if (threadIdx.x == 0) {
      result[row] = arithmetic<arithmetic_kind::fp16>::to_double(total);
    }
  } else {
    scalar_sum = __hadd(__low2bfloat16(sums[0]), __high2bfloat16(sums[0]));
#pragma unroll
    for (int pair = 1; pair < Lanes / 2; ++pair) {
      scalar_sum = __hadd(scalar_sum, __hadd(__low2bfloat16(sums[pair]),
                                             __high2bfloat16(sums[pair])));
    }
    const auto total = block_sum<arithmetic_kind::bf16>(scalar_sum);
    if (threadIdx.x == 0) {
      result[row] = arithmetic<arithmetic_kind::bf16>::to_double(total);
    }
  }
}

} // namespace aut::precision_packing

#endif // ACCESSOR_UNIVERSAL_TEST_PRECISION_PACKING_KERNELS_CUH_
