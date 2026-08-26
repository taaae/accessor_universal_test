#include "t16_codebook.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  try {
    constexpr std::size_t levels = 256;
    const auto codebook = aut::t16::build_normal_codebook(levels);
    require(codebook.values.size() == levels, "wrong codebook size");
    require(codebook.thresholds.size() == levels - 1, "wrong threshold count");
    require(codebook.thresholds[levels / 2 - 1] == 0.0f,
            "central threshold is not exactly zero");

    for (std::size_t index = 1; index < levels; ++index) {
      require(codebook.values[index - 1] < codebook.values[index],
              "codebook values are not strictly increasing");
      require(codebook.thresholds[index - 1] > codebook.values[index - 1] &&
                  codebook.thresholds[index - 1] < codebook.values[index],
              "threshold does not separate adjacent values");
    }
    for (std::size_t index = 0; index < levels / 2; ++index) {
      const auto mirror = levels - 1 - index;
      require(codebook.values[index] == -codebook.values[mirror],
              "codebook is not exactly symmetric");
    }
    require(std::abs(codebook.values.front()) < aut::t16::maximum_finite,
            "first centroid lies outside the truncated range");
    require(std::abs(codebook.values.back()) < aut::t16::maximum_finite,
            "last centroid lies outside the truncated range");
  } catch (const std::exception &error) {
    std::cerr << "t16_codebook_test: " << error.what() << '\n';
    return 1;
  }
  std::cout << "t16_codebook_test passed\n";
  return 0;
}
