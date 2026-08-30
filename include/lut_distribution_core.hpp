#ifndef ACCESSOR_UNIVERSAL_TEST_LUT_DISTRIBUTION_CORE_HPP_
#define ACCESSOR_UNIVERSAL_TEST_LUT_DISTRIBUTION_CORE_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define AUT_LUT_HD __host__ __device__
#define AUT_LUT_INLINE __forceinline__
#else
#define AUT_LUT_HD
#define AUT_LUT_INLINE inline
#endif

namespace aut::lut_distribution {

inline constexpr std::size_t code_count = std::size_t{1} << 16;
inline constexpr std::size_t entries_per_sector = 8;
inline constexpr std::size_t sector_count = code_count / entries_per_sector;
inline constexpr int warp_width = 32;
inline constexpr std::uint16_t default_hot_sector_base = 0;

AUT_LUT_HD AUT_LUT_INLINE std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

AUT_LUT_HD AUT_LUT_INLINE std::uint16_t
sample_code(std::size_t index, std::uint64_t seed, int q_eighths,
            std::uint16_t hot_sector_base = default_hot_sector_base) {
  const auto selector = splitmix64(seed ^ static_cast<std::uint64_t>(index));
  const auto payload = splitmix64(selector);
  const bool scattered = static_cast<int>(selector & 7u) < q_eighths;
  if (scattered) {
    return static_cast<std::uint16_t>(payload);
  }
  return static_cast<std::uint16_t>(hot_sector_base | (payload & 7u));
}

inline double uniform_unique_sectors() {
  return static_cast<double>(sector_count) *
         (1.0 - std::pow(1.0 - 1.0 / static_cast<double>(sector_count),
                         static_cast<double>(warp_width)));
}

inline double normalized_dispersion(double mean_unique_sectors) {
  return (mean_unique_sectors - 1.0) / (uniform_unique_sectors() - 1.0);
}

} // namespace aut::lut_distribution

#undef AUT_LUT_HD
#undef AUT_LUT_INLINE

#endif // ACCESSOR_UNIVERSAL_TEST_LUT_DISTRIBUTION_CORE_HPP_
