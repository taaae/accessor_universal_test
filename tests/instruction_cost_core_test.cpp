#include "instruction_cost_core.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

namespace ic = aut::instruction_cost;

void require(bool condition) {
  if (!condition) throw std::runtime_error("instruction-cost host test failed");
}

int main() {
  constexpr std::array<std::uint32_t, 7> edges{0, 1, 2, 0x7fffffffu,
      0x80000000u, 0xfffffffeu, 0xffffffffu};
  for (auto f : ic::families) for (int k : {0,1,2,4,8,12,16,24,32,48,64}) {
    for (auto u : edges) require(std::isfinite(ic::decode_reference(u, f, k)));
  }
  require(ic::decode_reference(7, ic::family::rot32, 32) == 7.0);
  require(ic::decode_reference(7, ic::family::add32f, 1) == static_cast<double>(7.0f + ic::fp_addend));
  require(ic::decode_reference(7, ic::family::mul32f, 1) == static_cast<double>(7.0f * ic::fp_multiplier));
  std::mt19937_64 random(0x123456789abcdef0ULL);
  for (int i = 0; i < 4096; ++i) {
    auto u = static_cast<std::uint32_t>(random());
    for (auto f : ic::families)
      require(std::isfinite(ic::decode_reference(u, f, 64)));
  }
  std::set<std::uint32_t> left, right;
  for (std::uint64_t i = 0; i < 4096; ++i) {
    left.insert(ic::code_at(i, ic::left_seed));
    right.insert(ic::code_at(i, ic::right_seed));
    require(ic::code_at(i, ic::left_seed) == ic::code_at(i, ic::left_seed));
  }
  require(left.size() > 4000 && right.size() > 4000 && left != right);
  std::cout << "instruction_cost_core_test passed\n";
}
