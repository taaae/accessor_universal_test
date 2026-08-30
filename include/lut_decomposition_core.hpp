#ifndef ACCESSOR_UNIVERSAL_TEST_LUT_DECOMPOSITION_CORE_HPP_
#define ACCESSOR_UNIVERSAL_TEST_LUT_DECOMPOSITION_CORE_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "lut_distribution_core.hpp"

#if defined(__CUDACC__)
#define AUT_LUT_DECOMP_HD __host__ __device__
#define AUT_LUT_DECOMP_INLINE __forceinline__
#else
#define AUT_LUT_DECOMP_HD
#define AUT_LUT_DECOMP_INLINE inline
#endif

namespace aut::lut_decomposition {

inline constexpr int warp_width = 32;

inline double uniform_unique_indices(std::size_t entries) {
  if (entries == 0) {
    throw std::invalid_argument("entries must be positive");
  }
  const auto k = static_cast<double>(entries);
  return k * (1.0 - std::pow(1.0 - 1.0 / k,
                             static_cast<double>(warp_width)));
}

inline double expected_unique_indices(std::size_t entries, double q) {
  if (entries == 0 || q < 0.0 || q > 1.0) {
    throw std::invalid_argument("invalid entry count or mixture probability");
  }
  const auto k = static_cast<double>(entries);
  const auto p_hot = 1.0 - q + q / k;
  const auto hot_seen = 1.0 - std::pow(1.0 - p_hot, warp_width);
  const auto cold_seen =
      (k - 1.0) * (1.0 - std::pow(1.0 - q / k, warp_width));
  return hot_seen + cold_seen;
}

inline double normalized_lookup_dispersion(std::size_t entries,
                                           double mean_unique) {
  return (mean_unique - 1.0) / (uniform_unique_indices(entries) - 1.0);
}

inline double q_for_normalized_dispersion(std::size_t entries,
                                          double target) {
  if (entries < 2 || target < 0.0 || target > 1.0) {
    throw std::invalid_argument("invalid entry count or target dispersion");
  }
  if (target == 0.0) {
    return 0.0;
  }
  if (target == 1.0) {
    return 1.0;
  }
  double low = 0.0;
  double high = 1.0;
  for (int iteration = 0; iteration < 100; ++iteration) {
    const auto middle = 0.5 * (low + high);
    const auto x = normalized_lookup_dispersion(
        entries, expected_unique_indices(entries, middle));
    if (x < target) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return 0.5 * (low + high);
}

AUT_LUT_DECOMP_HD AUT_LUT_DECOMP_INLINE std::uint32_t
sample_index(std::size_t index, std::uint64_t seed, int field_bits,
             double q) {
  const auto selector = lut_distribution::splitmix64(
      seed ^ static_cast<std::uint64_t>(index));
  const auto payload = lut_distribution::splitmix64(selector);
  constexpr double inverse_53 = 1.0 / 9007199254740992.0;
  const auto uniform01 = static_cast<double>(selector >> 11) * inverse_53;
  if (!(uniform01 < q)) {
    return 0;
  }
  const auto mask = field_bits == 32
                        ? std::uint32_t{0xffffffffu}
                        : ((std::uint32_t{1} << field_bits) - 1u);
  return static_cast<std::uint32_t>(payload) & mask;
}

} // namespace aut::lut_decomposition

#undef AUT_LUT_DECOMP_HD
#undef AUT_LUT_DECOMP_INLINE

#endif // ACCESSOR_UNIVERSAL_TEST_LUT_DECOMPOSITION_CORE_HPP_
