#include "decoder_strategy_core.hpp"
#include "storage_formats.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

std::uint64_t bits(double value) {
  std::uint64_t result{};
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

void test_e1m6() {
  using layout = aut::decoder::e1m6_layout;
  for (std::uint32_t raw = 0; raw < 256; ++raw) {
    const auto expected = aut::storage::decode<aut::storage::e1m6>(
        static_cast<std::uint8_t>(raw));
    const auto branchy = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<layout>(raw));
    const auto masked = aut::decoder::words_to_double(
        aut::decoder::decode_words_masked<layout>(raw));
    const auto fp32 = aut::decoder::decode_via_fp32<layout>(raw);
    const auto integer = aut::decoder::decode_e1_integer<layout>(raw);
    assert(bits(branchy) == bits(expected));
    assert(bits(masked) == bits(expected));
    assert(bits(fp32) == bits(expected));
    assert(bits(integer) == bits(expected));
  }
}

} // namespace

int main() {
  test_e1m6();
  std::cout << "decoder strategy core tests passed\n";
}
