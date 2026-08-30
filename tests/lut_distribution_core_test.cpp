#include "lut_distribution_core.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  namespace lut = aut::lut_distribution;
  static_assert(lut::code_count == 65536);
  static_assert(lut::sector_count == 8192);

  constexpr auto seed = std::uint64_t{0x243f6a8885a308d3ULL};
  for (std::size_t index = 0; index < 10000; ++index) {
    const auto first = lut::sample_code(index, seed, 0);
    const auto second = lut::sample_code(index, seed, 0);
    require(first == second, "sampling is not deterministic");
    require((first >> 3) == 0, "q=0 escaped the hot sector");
  }

  std::set<std::uint16_t> full_range_codes;
  for (std::size_t index = 0; index < 10000; ++index) {
    full_range_codes.insert(lut::sample_code(index, seed, 8));
  }
  require(full_range_codes.size() > 9000, "q=1 did not cover the code range");
  require(lut::uniform_unique_sectors() > 31.9,
          "uniform sector expectation is too small");
  require(lut::uniform_unique_sectors() < 32.0,
          "uniform sector expectation is too large");
  require(std::abs(lut::normalized_dispersion(1.0)) < 1e-15,
          "dispersion origin is incorrect");
  require(std::abs(lut::normalized_dispersion(
                       lut::uniform_unique_sectors()) -
                   1.0) < 1e-15,
          "uniform dispersion is not one");
  return 0;
}
