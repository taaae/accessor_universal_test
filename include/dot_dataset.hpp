#ifndef ACCESSOR_UNIVERSAL_TEST_DOT_DATASET_HPP_
#define ACCESSOR_UNIVERSAL_TEST_DOT_DATASET_HPP_

#include <array>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define AUT_DATASET_HD __host__ __device__
#define AUT_DATASET_INLINE __forceinline__
#else
#define AUT_DATASET_HD
#define AUT_DATASET_INLINE inline
#endif

namespace aut::dataset {

enum class family : std::uint8_t {
  positive_uniform,
  signed_uniform,
  controlled_cancellation,
  dynamic_range
};

struct parameters {
  family kind{};
  std::uint64_t seed{};
  int parameter{};
};

struct case_spec {
  const char *id;
  parameters values;
  const char *purpose;
};

inline constexpr std::uint64_t base_seed = 0x123456789abcdef0ULL;

inline constexpr case_spec performance_case{
    "signed_uniform_seed0",
    {family::signed_uniform, base_seed, 0},
    "Finite normal FP32 values for performance and a typical accuracy case"};

inline constexpr std::array<case_spec, 9> accuracy_cases{{
    {"positive_uniform",
     {family::positive_uniform, base_seed, 0},
     "Well-conditioned positive products (dot condition number 1)"},
    {"signed_uniform_seed0",
     {family::signed_uniform, base_seed, 0},
     "Typical independent signed-uniform vectors, replicate 0"},
    {"signed_uniform_seed1",
     {family::signed_uniform, 0x8c3c010cb4754c9dULL, 0},
     "Typical independent signed-uniform vectors, replicate 1"},
    {"signed_uniform_seed2",
     {family::signed_uniform, 0xd2b74407b1ce6e93ULL, 0},
     "Typical independent signed-uniform vectors, replicate 2"},
    {"cancellation_1e2",
     {family::controlled_cancellation, base_seed, 6},
     "Constructed cancellation with expected condition near 2^7"},
    {"cancellation_1e4",
     {family::controlled_cancellation, base_seed, 13},
     "Constructed cancellation with expected condition near 2^14"},
    {"cancellation_1e6",
     {family::controlled_cancellation, base_seed, 19},
     "Constructed cancellation with expected condition near 2^20"},
    {"near_orthogonal",
     {family::controlled_cancellation, base_seed, 22},
     "Constructed nearly orthogonal vectors with condition near 2^23"},
    {"dynamic_range_40",
     {family::dynamic_range, 0xa4093822299f31d0ULL, 20},
     "Signed values with log-uniform exponents from -20 through 20"},
}};

AUT_DATASET_HD AUT_DATASET_INLINE std::uint64_t
splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

AUT_DATASET_HD AUT_DATASET_INLINE float positive_mantissa(std::uint64_t bits) {
  const auto fraction = static_cast<std::uint32_t>(bits >> 41);
  return static_cast<float>((1u << 23) + fraction) * 0x1p-24f;
}

AUT_DATASET_HD AUT_DATASET_INLINE float signed_unit(std::uint64_t bits) {
  const auto fraction = static_cast<std::uint32_t>(bits >> 41);
  const auto centered = static_cast<std::int32_t>(fraction) - (1 << 22);
  return static_cast<float>(centered) * 0x1p-22f;
}

AUT_DATASET_HD AUT_DATASET_INLINE float power_of_two(int exponent) {
  float result = 1.0f;
  if (exponent >= 0) {
    for (int i = 0; i < exponent; ++i) {
      result *= 2.0f;
    }
  } else {
    for (int i = 0; i > exponent; --i) {
      result *= 0.5f;
    }
  }
  return result;
}

AUT_DATASET_HD AUT_DATASET_INLINE float
independent_value(std::uint64_t index, std::uint64_t seed, int vector_id,
                  family kind, int parameter) {
  const auto lane_seed = seed ^ (0xd1b54a32d192ed03ULL *
                                 static_cast<std::uint64_t>(vector_id + 1));
  const auto bits = splitmix64(index ^ lane_seed);

  if (kind == family::positive_uniform) {
    return positive_mantissa(bits);
  }
  if (kind == family::signed_uniform) {
    return signed_unit(bits);
  }

  const auto span = parameter;
  const auto exponent =
      static_cast<int>((bits >> 1) % static_cast<std::uint64_t>(2 * span + 1)) -
      span;
  const auto sign = (bits & 1ULL) != 0 ? -1.0f : 1.0f;
  return sign * positive_mantissa(bits) * power_of_two(exponent);
}

AUT_DATASET_HD AUT_DATASET_INLINE float value(parameters spec, int vector_id,
                                              std::uint64_t index,
                                              std::uint64_t count) {
  if (spec.kind != family::controlled_cancellation) {
    return independent_value(index, spec.seed, vector_id, spec.kind,
                             spec.parameter);
  }

  const auto half = count / 2;
  const auto pair_index = half == 0 ? 0 : index % half;
  const auto a = positive_mantissa(
      splitmix64(pair_index ^ spec.seed ^ 0x243f6a8885a308d3ULL));
  if (vector_id == 0) {
    return a;
  }

  const auto b = positive_mantissa(
      splitmix64(pair_index ^ spec.seed ^ 0x13198a2e03707344ULL));
  if (index < half) {
    return b;
  }
  const auto residual = power_of_two(-spec.parameter);
  return -b * (1.0f - residual);
}

} // namespace aut::dataset

#undef AUT_DATASET_HD
#undef AUT_DATASET_INLINE

#endif
