#include "decoder_strategy_core.hpp"
#include "storage_formats.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

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

double reference_e4m3(std::uint32_t raw) {
  const auto sign = raw >> 7;
  const auto exponent = (raw >> 3) & 0xfu;
  const auto fraction = raw & 0x7u;
  if (exponent == 0xfu && fraction == 0x7u) {
    return std::nan("");
  }
  double magnitude{};
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<double>(fraction), -9);
  } else {
    magnitude = std::ldexp(1.0 + static_cast<double>(fraction) / 8.0,
                           static_cast<int>(exponent) - 7);
  }
  return sign != 0 ? -magnitude : magnitude;
}

void test_fp8_e4m3() {
  using layout = aut::decoder::fp8_e4m3_layout;
  for (std::uint32_t raw = 0; raw < 256; ++raw) {
    const auto expected = reference_e4m3(raw);
    const auto branchy = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<layout>(raw));
    const auto masked = aut::decoder::words_to_double(
        aut::decoder::decode_words_masked<layout>(raw));
    const auto fp32 = aut::decoder::decode_via_fp32<layout>(raw);
    if (std::isnan(expected)) {
      assert(std::isnan(branchy));
      assert(std::isnan(masked));
      assert(std::isnan(fp32));
    } else {
      assert(bits(branchy) == bits(expected));
      assert(bits(masked) == bits(expected));
      assert(bits(fp32) == bits(expected));
    }
  }
}

double reference_e5m2(std::uint32_t raw) {
  const auto sign = raw >> 7;
  const auto exponent = (raw >> 2) & 0x1fu;
  const auto fraction = raw & 0x3u;
  if (exponent == 0x1fu) {
    if (fraction == 0) {
      return sign != 0 ? -std::numeric_limits<double>::infinity()
                       : std::numeric_limits<double>::infinity();
    }
    return std::nan("");
  }
  double magnitude{};
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<double>(fraction), -16);
  } else {
    magnitude = std::ldexp(1.0 + static_cast<double>(fraction) / 4.0,
                           static_cast<int>(exponent) - 15);
  }
  return sign != 0 ? -magnitude : magnitude;
}

void test_fp8_e5m2() {
  using layout = aut::decoder::fp8_e5m2_layout;
  for (std::uint32_t raw = 0; raw < 256; ++raw) {
    const auto expected = reference_e5m2(raw);
    const auto branchy = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<layout>(raw));
    const auto masked = aut::decoder::words_to_double(
        aut::decoder::decode_words_masked<layout>(raw));
    const auto fp32 = aut::decoder::decode_via_fp32<layout>(raw);
    if (std::isnan(expected)) {
      assert(std::isnan(branchy));
      assert(std::isnan(masked));
      assert(std::isnan(fp32));
    } else {
      assert(bits(branchy) == bits(expected));
      assert(bits(masked) == bits(expected));
      assert(bits(fp32) == bits(expected));
    }
  }
}

} // namespace

int main() {
  test_e1m6();
  test_fp8_e4m3();
  test_fp8_e5m2();
  std::cout << "decoder strategy core tests passed\n";
}
