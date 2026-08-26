#include "posit_takum_core.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace pt = aut::pt;

int main() {
  assert((pt::decode_posit<8, 0, float>(0x00) == 0.0f));
  assert((pt::decode_posit<8, 0, float>(0x40) == 1.0f));
  assert((pt::decode_posit<8, 0, float>(0x60) == 2.0f));
  assert((pt::decode_posit<8, 0, float>(0xc0) == -1.0f));
  assert((std::isnan(pt::decode_posit<8, 0, float>(0x80))));

  assert((pt::decode_linear_takum<8, double>(0x00) == 0.0));
  assert((pt::decode_linear_takum<8, double>(0x40) == 1.0));
  assert((pt::decode_linear_takum<8, double>(0xc0) == -1.0));
  assert((std::isnan(pt::decode_linear_takum<8, double>(0x80))));

  assert((pt::decode_log_takum<8, double>(0x40) == 1.0));
  assert((pt::decode_log_takum<8, double>(0xc0) == -1.0));

  for (const long double q : {-5.0L, -1.0L, 0.0L, 1.0L, 5.0L}) {
    const auto raw = pt::encode_positive_log2<pt::family::posit, 8, 0>(q);
    const auto decoded = pt::decode_long_double<pt::family::posit, 8, 0>(raw);
    assert(decoded > 0.0L);
    assert(std::abs(std::log2(decoded) - q) < 1.1L);
  }

  std::cout << "posit/takum core tests passed\n";
  return 0;
}
