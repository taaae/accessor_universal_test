#include "dyadic_normal32_core.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::uint64_t bits(double value) {
  std::uint64_t result{};
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

} // namespace

int main() {
  namespace dn = aut::dyadic_normal32;
  const auto coefficients = dn::make_coefficients();
  const auto bitcast_coefficients = dn::make_bitcast_coefficients();

  double previous_boundary = -1.0;
  for (std::uint32_t boundary = 0; boundary <= dn::segment_count; ++boundary) {
    const auto value = dn::half_normal_density_boundary(boundary);
    require(std::isfinite(value), "density boundary is not finite");
    require(value > previous_boundary, "density boundaries are not monotonic");
    previous_boundary = value;
  }

  double previous_decoded = -1.0;
  for (std::uint32_t segment = 0; segment < dn::segment_count; ++segment) {
    const auto payload_bits = segment < 30 ? 30u - segment : 0u;
    const auto maximum_payload =
        payload_bits == 0 ? 0u : ((std::uint32_t{1} << payload_bits) - 1u);
    const auto minimum_rank = dn::rank_from_segment_payload(segment, 0);
    const auto maximum_rank =
        dn::rank_from_segment_payload(segment, maximum_payload);
    require(dn::segment_from_rank(minimum_rank) == segment,
            "minimum delimiter boundary decoded to wrong segment");
    require(dn::segment_from_rank(maximum_rank) == segment,
            "maximum delimiter boundary decoded to wrong segment");
    require(dn::payload_for_segment(minimum_rank, segment) == 0,
            "minimum payload is not zero");
    require(dn::payload_for_segment(maximum_rank, segment) == maximum_payload,
            "maximum payload was not preserved");
    const auto minimum_value = dn::decode(minimum_rank, coefficients);
    const auto maximum_value = dn::decode(maximum_rank, coefficients);
    const auto minimum_bitcast =
        dn::decode_bitcast(minimum_rank, bitcast_coefficients);
    const auto maximum_bitcast =
        dn::decode_bitcast(maximum_rank, bitcast_coefficients);
    require(std::isfinite(minimum_value) && std::isfinite(maximum_value),
            "decoded segment endpoint is not finite");
    require(minimum_value > previous_decoded,
            "decoded segment boundary is not monotonic");
    require(maximum_value >= minimum_value,
            "decoded values decrease inside a segment");
    require(std::abs(minimum_bitcast - minimum_value) <=
                8.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::abs(minimum_value)),
            "bitcast minimum differs from the linear decoder");
    require(std::abs(maximum_bitcast - maximum_value) <=
                8.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::abs(maximum_value)),
            "bitcast maximum differs from the linear decoder");
    const auto levels = payload_bits == 0
                            ? 1.0
                            : std::ldexp(1.0, static_cast<int>(payload_bits));
    require(dn::bitcast_coordinate(minimum_rank, segment) == 1.0,
            "bitcast minimum coordinate is not exactly one");
    require(dn::bitcast_coordinate(maximum_rank, segment) ==
                1.0 + static_cast<double>(maximum_payload) / levels,
            "bitcast maximum coordinate is not exact");
    previous_decoded = maximum_value;

    const auto negative = maximum_rank | 0x80000000u;
    const auto negative_value = dn::decode(negative, coefficients);
    require(bits(negative_value) ==
                (bits(maximum_value) | (std::uint64_t{1} << 63)),
            "sign bit was not applied exactly");
  }

  require(dn::rank_from_segment_payload(31, 0) == dn::magnitude_mask,
          "terminal segment is not the all-ones magnitude rank");
  require(dn::segment_from_rank(dn::magnitude_mask) == 31,
          "all-ones magnitude rank does not select terminal segment");
  require(coefficients[31].step == 0.0,
          "terminal coefficient must ignore its payload");
  require(std::abs(coefficients[31].start -
                   dn::half_normal_density_boundary(32)) < 1e-14,
          "terminal value does not equal the documented finite boundary");

  std::mt19937_64 generator(0x243f6a8885a308d3ULL);
  double previous_random_value = -1.0;
  std::uint32_t previous_rank = 0;
  for (std::size_t index = 0; index < 100000; ++index) {
    const auto rank =
        static_cast<std::uint32_t>(generator()) & dn::magnitude_mask;
    const auto segment = dn::segment_from_rank(rank);
    require(segment < dn::segment_count,
            "random rank selected invalid segment");
    const auto value = dn::decode(rank, coefficients);
    const auto bitcast_value = dn::decode_bitcast(rank, bitcast_coefficients);
    require(std::isfinite(value) && value > 0.0,
            "random positive code did not decode to a finite magnitude");
    require(std::abs(bitcast_value - value) <=
                8.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::abs(value)),
            "random bitcast decode differs from the linear decoder");
    if (index != 0 && rank > previous_rank) {
      const auto previous_value = dn::decode(previous_rank, coefficients);
      require(value > previous_value, "rank ordering is not monotonic");
    }
    previous_rank = rank;
    previous_random_value = value;
  }
  require(previous_random_value > 0.0, "random decode loop did not run");

  const auto probabilities =
      dn::genuine_standard_normal_segment_probabilities();
  double probability_sum{};
  for (const auto probability : probabilities) {
    require(probability >= 0.0 && probability <= 1.0,
            "genuine-source segment probability is invalid");
    probability_sum += probability;
  }
  require(std::abs(probability_sum - 1.0) < 1e-12,
          "genuine-source segment probabilities do not sum to one");
  const auto genuine_x = dn::genuine_standard_normal_dispersion();
  require(genuine_x > 0.0 && genuine_x < 1.0,
          "genuine-source dispersion is outside (0,1)");

  bool rejected_segment = false;
  try {
    (void)dn::rank_from_segment_payload(32, 0);
  } catch (const std::invalid_argument &) {
    rejected_segment = true;
  }
  require(rejected_segment, "invalid segment was not rejected");

  bool rejected_payload = false;
  try {
    (void)dn::rank_from_segment_payload(30, 1);
  } catch (const std::invalid_argument &) {
    rejected_payload = true;
  }
  require(rejected_payload, "oversized payload was not rejected");
  return 0;
}
