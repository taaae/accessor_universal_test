#ifndef ACCESSOR_UNIVERSAL_TEST_T16_CODEBOOK_HPP_
#define ACCESSOR_UNIVERSAL_TEST_T16_CODEBOOK_HPP_

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aut::t16 {

inline constexpr std::size_t code_count = 1u << 16;
inline constexpr double maximum_finite = 65504.0;
inline constexpr double normal_sigma = maximum_finite / 4.0;
inline constexpr double normal_cutoff_sigma = 4.0;

struct codebook {
  std::vector<float> values;
  std::vector<float> thresholds;
};

inline double standard_normal_pdf(double value) {
  constexpr double inverse_sqrt_two_pi = 0.398942280401432677939946059934381868;
  return inverse_sqrt_two_pi * std::exp(-0.5 * value * value);
}

inline double standard_normal_cdf(double value) {
  return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

inline double inverse_standard_normal_cdf(double probability, double lower,
                                          double upper) {
  if (!(probability > 0.0 && probability < 1.0) || !(lower < upper)) {
    throw std::invalid_argument("invalid inverse-normal interval");
  }
  for (int iteration = 0; iteration < 64; ++iteration) {
    const auto middle = 0.5 * (lower + upper);
    if (standard_normal_cdf(middle) < probability) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return 0.5 * (lower + upper);
}

inline double truncated_normal_centroid(double lower, double upper,
                                        double sigma) {
  const auto lower_z = lower / sigma;
  const auto upper_z = upper / sigma;
  const auto mass = standard_normal_cdf(upper_z) - standard_normal_cdf(lower_z);
  if (mass <= std::numeric_limits<double>::epsilon()) {
    return 0.5 * (lower + upper);
  }
  return sigma * (standard_normal_pdf(lower_z) - standard_normal_pdf(upper_z)) /
         mass;
}

inline codebook build_normal_codebook(std::size_t levels = code_count,
                                      double sigma = normal_sigma,
                                      double cutoff_sigma = normal_cutoff_sigma,
                                      int lloyd_iterations = 24) {
  if (levels < 2 || (levels & 1u) != 0) {
    throw std::invalid_argument("T16 codebook needs an even number of levels");
  }
  if (!(sigma > 0.0) || !(cutoff_sigma > 0.0) || lloyd_iterations < 0) {
    throw std::invalid_argument("invalid T16 normal-codebook parameters");
  }

  const auto lower_z = -cutoff_sigma;
  const auto upper_z = cutoff_sigma;
  const auto lower = lower_z * sigma;
  const auto upper = upper_z * sigma;
  const auto cdf_lower = standard_normal_cdf(lower_z);
  const auto retained_mass = standard_normal_cdf(upper_z) - cdf_lower;

  std::vector<double> boundaries(levels + 1);
  boundaries.front() = lower;
  boundaries.back() = upper;
  for (std::size_t index = 1; index < levels; ++index) {
    const auto probability = cdf_lower + retained_mass *
                                             static_cast<double>(index) /
                                             static_cast<double>(levels);
    boundaries[index] =
        sigma * inverse_standard_normal_cdf(probability, lower_z, upper_z);
  }

  std::vector<double> representatives(levels);
  const auto update_representatives = [&] {
    for (std::size_t index = 0; index < levels; ++index) {
      representatives[index] = truncated_normal_centroid(
          boundaries[index], boundaries[index + 1], sigma);
    }
  };

  for (int iteration = 0; iteration < lloyd_iterations; ++iteration) {
    update_representatives();
    for (std::size_t index = 1; index < levels; ++index) {
      boundaries[index] =
          0.5 * (representatives[index - 1] + representatives[index]);
    }
  }
  update_representatives();

  const auto half = levels / 2;
  for (std::size_t index = 0; index < half; ++index) {
    const auto mirror = levels - 1 - index;
    const auto magnitude =
        0.5 * (representatives[mirror] - representatives[index]);
    representatives[index] = -magnitude;
    representatives[mirror] = magnitude;
  }

  codebook result;
  result.values.resize(levels);
  result.thresholds.resize(levels - 1);
  for (std::size_t index = 0; index < levels; ++index) {
    result.values[index] = static_cast<float>(representatives[index]);
  }
  for (std::size_t index = 0; index + 1 < levels; ++index) {
    result.thresholds[index] = static_cast<float>(
        0.5 * (representatives[index] + representatives[index + 1]));
  }
  result.thresholds[half - 1] = 0.0f;
  return result;
}

} // namespace aut::t16

#endif // ACCESSOR_UNIVERSAL_TEST_T16_CODEBOOK_HPP_
