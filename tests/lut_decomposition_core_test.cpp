#include "lut_decomposition_core.hpp"

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
  namespace lut = aut::lut_decomposition;
  for (const auto entries : {std::size_t{16}, std::size_t{256},
                             std::size_t{65536}}) {
    require(std::abs(lut::expected_unique_indices(entries, 0.0) - 1.0) <
                1e-15,
            "q=0 expectation is not one");
    require(std::abs(lut::expected_unique_indices(entries, 1.0) -
                     lut::uniform_unique_indices(entries)) < 1e-12,
            "q=1 expectation is not uniform");
    double previous_q = -1.0;
    for (int eighth = 0; eighth <= 8; ++eighth) {
      const auto target = static_cast<double>(eighth) / 8.0;
      const auto q = lut::q_for_normalized_dispersion(entries, target);
      const auto achieved = lut::normalized_lookup_dispersion(
          entries, lut::expected_unique_indices(entries, q));
      require(q >= previous_q, "q solution is not monotonic");
      require(std::abs(achieved - target) < 1e-10,
              "q solution misses the target");
      previous_q = q;
    }
  }

  constexpr auto seed = std::uint64_t{0x243f6a8885a308d3ULL};
  for (std::size_t index = 0; index < 10000; ++index) {
    require(lut::sample_index(index, seed, 8, 0.0) == 0,
            "q=0 escaped the hot index");
  }
  std::set<std::uint32_t> values;
  for (std::size_t index = 0; index < 10000; ++index) {
    values.insert(lut::sample_index(index, seed, 8, 1.0));
  }
  require(values.size() == 256, "q=1 did not cover the 8-bit table");
  return 0;
}
