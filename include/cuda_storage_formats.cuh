#ifndef ACCESSOR_UNIVERSAL_TEST_CUDA_STORAGE_FORMATS_CUH_
#define ACCESSOR_UNIVERSAL_TEST_CUDA_STORAGE_FORMATS_CUH_

#include "storage_formats.hpp"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector_types.h>

#include <cstdint>

namespace aut::storage {

struct fp8_e4m3 {
  static constexpr int total_bits = 8;
  static constexpr int exponent_bits = 4;
  static constexpr int fraction_bits = 3;
  static constexpr bool finite = true;
  static constexpr const char *name = "fp8_e4m3";
};
struct fp8_e5m2 {
  static constexpr int total_bits = 8;
  static constexpr int exponent_bits = 5;
  static constexpr int fraction_bits = 2;
  static constexpr bool finite = false;
  static constexpr const char *name = "fp8_e5m2";
};
struct fp16_e5m10 {
  static constexpr int total_bits = 16;
  static constexpr int exponent_bits = 5;
  static constexpr int fraction_bits = 10;
  static constexpr bool finite = false;
  static constexpr const char *name = "fp16_e5m10";
};
struct bf16_e8m7 {
  static constexpr int total_bits = 16;
  static constexpr int exponent_bits = 8;
  static constexpr int fraction_bits = 7;
  static constexpr bool finite = false;
  static constexpr const char *name = "bf16_e8m7";
};
struct fp32_e8m23 {
  static constexpr int total_bits = 32;
  static constexpr int exponent_bits = 8;
  static constexpr int fraction_bits = 23;
  static constexpr bool finite = false;
  static constexpr const char *name = "fp32_e8m23";
};
struct fp64_e11m52 {
  static constexpr int total_bits = 64;
  static constexpr int exponent_bits = 11;
  static constexpr int fraction_bits = 52;
  static constexpr bool finite = false;
  static constexpr const char *name = "fp64_e11m52";
};

template <> struct codec<fp8_e4m3> {
  using storage_type = __nv_fp8_e4m3;
  __host__ __device__ __forceinline__ static storage_type encode(double value) {
    return storage_type{value};
  }
  __host__ __device__ __forceinline__ static double decode(storage_type value) {
    return static_cast<double>(static_cast<float>(value));
  }
};

template <> struct codec<fp8_e5m2> {
  using storage_type = __nv_fp8_e5m2;
  __host__ __device__ __forceinline__ static storage_type encode(double value) {
    return storage_type{value};
  }
  __host__ __device__ __forceinline__ static double decode(storage_type value) {
    return static_cast<double>(static_cast<float>(value));
  }
};

template <> struct codec<fp16_e5m10> {
  using storage_type = __half;
  __host__ __device__ __forceinline__ static storage_type encode(double value) {
    return __double2half(value);
  }
  __host__ __device__ __forceinline__ static double decode(storage_type value) {
    return static_cast<double>(__half2float(value));
  }
};

template <> struct codec<bf16_e8m7> {
  using storage_type = __nv_bfloat16;
  __host__ __device__ __forceinline__ static storage_type encode(double value) {
    return __double2bfloat16(value);
  }
  __host__ __device__ __forceinline__ static double decode(storage_type value) {
    return static_cast<double>(__bfloat162float(value));
  }
};

template <> struct codec<fp32_e8m23> {
  using storage_type = float;
  __host__ __device__ __forceinline__ static storage_type encode(double value) {
    return static_cast<float>(value);
  }
  __host__ __device__ __forceinline__ static double decode(storage_type value) {
    return static_cast<double>(value);
  }
};

template <> struct codec<fp64_e11m52> {
  using storage_type = double;
  __host__ __device__ __forceinline__ static storage_type encode(double value) {
    return value;
  }
  __host__ __device__ __forceinline__ static double decode(storage_type value) {
    return value;
  }
};

struct double4_values {
  double x;
  double y;
  double z;
  double w;
};

namespace detail {

template <typename Format> struct generic_packed_decoder {
  using storage_type = storage_type_t<Format>;

  __device__ __forceinline__ static double2 load2(const storage_type *values) {
    if constexpr (sizeof(storage_type) == 1) {
      const auto packed = *reinterpret_cast<const std::uint16_t *>(values);
      return {decode<Format>(static_cast<storage_type>(packed)),
              decode<Format>(static_cast<storage_type>(packed >> 8))};
    } else if constexpr (sizeof(storage_type) == 2) {
      const auto packed = *reinterpret_cast<const std::uint32_t *>(values);
      return {decode<Format>(static_cast<storage_type>(packed)),
              decode<Format>(static_cast<storage_type>(packed >> 16))};
    } else {
      const auto packed = *reinterpret_cast<const uint2 *>(values);
      return {decode<Format>(static_cast<storage_type>(packed.x)),
              decode<Format>(static_cast<storage_type>(packed.y))};
    }
  }

  __device__ __forceinline__ static double4_values
  load4(const storage_type *values) {
    if constexpr (sizeof(storage_type) == 1) {
      const auto packed = *reinterpret_cast<const std::uint32_t *>(values);
      return {decode<Format>(static_cast<storage_type>(packed)),
              decode<Format>(static_cast<storage_type>(packed >> 8)),
              decode<Format>(static_cast<storage_type>(packed >> 16)),
              decode<Format>(static_cast<storage_type>(packed >> 24))};
    } else if constexpr (sizeof(storage_type) == 2) {
      const auto packed = *reinterpret_cast<const uint2 *>(values);
      return {decode<Format>(static_cast<storage_type>(packed.x)),
              decode<Format>(static_cast<storage_type>(packed.x >> 16)),
              decode<Format>(static_cast<storage_type>(packed.y)),
              decode<Format>(static_cast<storage_type>(packed.y >> 16))};
    } else {
      const auto packed = *reinterpret_cast<const uint4 *>(values);
      return {decode<Format>(static_cast<storage_type>(packed.x)),
              decode<Format>(static_cast<storage_type>(packed.y)),
              decode<Format>(static_cast<storage_type>(packed.z)),
              decode<Format>(static_cast<storage_type>(packed.w))};
    }
  }
};

__device__ __forceinline__ __half2 half2_from_word(std::uint32_t word) {
  const __half2_raw raw{static_cast<unsigned short>(word),
                        static_cast<unsigned short>(word >> 16)};
  return __half2{raw};
}

__device__ __forceinline__ __nv_bfloat162
bfloat162_from_word(std::uint32_t word) {
  const __nv_bfloat162_raw raw{static_cast<unsigned short>(word),
                               static_cast<unsigned short>(word >> 16)};
  return __nv_bfloat162{raw};
}

} // namespace detail

template <typename Format>
struct packed_decoder : detail::generic_packed_decoder<Format> {};

template <> struct packed_decoder<fp8_e4m3> {
  using storage_type = storage_type_t<fp8_e4m3>;

  __device__ __forceinline__ static double2 load2(const storage_type *values) {
    __nv_fp8x2_e4m3 packed;
    packed.__x = *reinterpret_cast<const __nv_fp8x2_storage_t *>(values);
    const auto result = static_cast<float2>(packed);
    return {static_cast<double>(result.x), static_cast<double>(result.y)};
  }

  __device__ __forceinline__ static double4_values
  load4(const storage_type *values) {
    __nv_fp8x4_e4m3 packed;
    packed.__x = *reinterpret_cast<const __nv_fp8x4_storage_t *>(values);
    const auto result = static_cast<float4>(packed);
    return {static_cast<double>(result.x), static_cast<double>(result.y),
            static_cast<double>(result.z), static_cast<double>(result.w)};
  }
};

template <> struct packed_decoder<fp8_e5m2> {
  using storage_type = storage_type_t<fp8_e5m2>;

  __device__ __forceinline__ static double2 load2(const storage_type *values) {
    __nv_fp8x2_e5m2 packed;
    packed.__x = *reinterpret_cast<const __nv_fp8x2_storage_t *>(values);
    const auto result = static_cast<float2>(packed);
    return {static_cast<double>(result.x), static_cast<double>(result.y)};
  }

  __device__ __forceinline__ static double4_values
  load4(const storage_type *values) {
    __nv_fp8x4_e5m2 packed;
    packed.__x = *reinterpret_cast<const __nv_fp8x4_storage_t *>(values);
    const auto result = static_cast<float4>(packed);
    return {static_cast<double>(result.x), static_cast<double>(result.y),
            static_cast<double>(result.z), static_cast<double>(result.w)};
  }
};

template <> struct packed_decoder<fp16_e5m10> {
  using storage_type = storage_type_t<fp16_e5m10>;

  __device__ __forceinline__ static double2 load2(const storage_type *values) {
    const auto result =
        __half22float2(*reinterpret_cast<const __half2 *>(values));
    return {static_cast<double>(result.x), static_cast<double>(result.y)};
  }

  __device__ __forceinline__ static double4_values
  load4(const storage_type *values) {
    const auto words = *reinterpret_cast<const uint2 *>(values);
    const auto low = __half22float2(detail::half2_from_word(words.x));
    const auto high = __half22float2(detail::half2_from_word(words.y));
    return {static_cast<double>(low.x), static_cast<double>(low.y),
            static_cast<double>(high.x), static_cast<double>(high.y)};
  }
};

template <> struct packed_decoder<bf16_e8m7> {
  using storage_type = storage_type_t<bf16_e8m7>;

  __device__ __forceinline__ static double2 load2(const storage_type *values) {
    const auto result =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(values));
    return {static_cast<double>(result.x), static_cast<double>(result.y)};
  }

  __device__ __forceinline__ static double4_values
  load4(const storage_type *values) {
    const auto words = *reinterpret_cast<const uint2 *>(values);
    const auto low = __bfloat1622float2(detail::bfloat162_from_word(words.x));
    const auto high = __bfloat1622float2(detail::bfloat162_from_word(words.y));
    return {static_cast<double>(low.x), static_cast<double>(low.y),
            static_cast<double>(high.x), static_cast<double>(high.y)};
  }
};

template <> struct packed_decoder<fp32_e8m23> {
  using storage_type = float;

  __device__ __forceinline__ static double2 load2(const storage_type *values) {
    const auto packed = *reinterpret_cast<const float2 *>(values);
    return {static_cast<double>(packed.x), static_cast<double>(packed.y)};
  }

  __device__ __forceinline__ static double4_values
  load4(const storage_type *values) {
    const auto packed = *reinterpret_cast<const float4 *>(values);
    return {static_cast<double>(packed.x), static_cast<double>(packed.y),
            static_cast<double>(packed.z), static_cast<double>(packed.w)};
  }
};

template <> struct packed_decoder<fp64_e11m52> {
  using storage_type = double;

  __device__ __forceinline__ static double2 load2(const storage_type *values) {
    return *reinterpret_cast<const double2 *>(values);
  }

  __device__ __forceinline__ static double4_values
  load4(const storage_type *values) {
    const auto low = *reinterpret_cast<const double2 *>(values);
    const auto high = *reinterpret_cast<const double2 *>(values + 2);
    return {low.x, low.y, high.x, high.y};
  }
};

static_assert(sizeof(storage_type_t<fp8_e4m3>) == 1);
static_assert(sizeof(storage_type_t<fp8_e5m2>) == 1);
static_assert(sizeof(storage_type_t<fp16_e5m10>) == 2);
static_assert(sizeof(storage_type_t<bf16_e8m7>) == 2);
static_assert(sizeof(storage_type_t<fp32_e8m23>) == 4);
static_assert(sizeof(storage_type_t<fp64_e11m52>) == 8);

} // namespace aut::storage

#endif // ACCESSOR_UNIVERSAL_TEST_CUDA_STORAGE_FORMATS_CUH_
