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

void test_e1m14() {
  using layout = aut::decoder::e1m14_layout;
  for (std::uint32_t raw = 0; raw < 65536; ++raw) {
    const auto expected = aut::storage::decode<aut::storage::e1m14>(
        static_cast<std::uint16_t>(raw));
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

void test_e2m13() {
  using layout = aut::decoder::e2m13_layout;
  for (std::uint32_t raw = 0; raw < 65536; ++raw) {
    const auto expected = aut::storage::decode<aut::storage::e2m13>(
        static_cast<std::uint16_t>(raw));
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

void test_e3m12() {
  using layout = aut::decoder::e3m12_layout;
  for (std::uint32_t raw = 0; raw < 65536; ++raw) {
    const auto expected = aut::storage::decode<aut::storage::e3m12>(
        static_cast<std::uint16_t>(raw));
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

double reference_fp16(std::uint32_t raw) {
  const auto sign = raw >> 15;
  const auto exponent = (raw >> 10) & 0x1fu;
  const auto fraction = raw & 0x3ffu;
  if (exponent == 0x1fu) {
    if (fraction == 0) {
      return sign != 0 ? -std::numeric_limits<double>::infinity()
                       : std::numeric_limits<double>::infinity();
    }
    return std::nan("");
  }
  double magnitude{};
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<double>(fraction), -24);
  } else {
    magnitude = std::ldexp(1.0 + static_cast<double>(fraction) / 1024.0,
                           static_cast<int>(exponent) - 15);
  }
  return sign != 0 ? -magnitude : magnitude;
}

void test_fp16() {
  using layout = aut::decoder::fp16_e5m10_layout;
  for (std::uint32_t raw = 0; raw < 65536; ++raw) {
    const auto expected = reference_fp16(raw);
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

void test_bf16() {
  using layout = aut::decoder::bf16_e8m7_layout;
  for (std::uint32_t raw = 0; raw < 65536; ++raw) {
    const auto float_value = aut::decoder::bits_to_float(raw << 16);
    const auto expected = static_cast<double>(float_value);
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

void test_e11m4() {
  for (std::uint32_t raw = 0; raw < 65536; ++raw) {
    const auto expected = aut::storage::decode<aut::storage::e11m4>(
        static_cast<std::uint16_t>(raw));
    const auto direct = aut::decoder::words_to_double(
        aut::decoder::decode_prefix_words<4>(raw));
    assert(bits(direct) == bits(expected));
  }
}

void test_e1m30() {
  using layout = aut::decoder::e1m30_layout;
  std::uint32_t raw = 0x243f6a88u;
  for (std::size_t sample = 0; sample < 200000; ++sample) {
    raw = raw * 1664525u + 1013904223u;
    const auto expected =
        aut::storage::decode<aut::storage::e1m30>(raw);
    const auto branchy = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<layout>(raw));
    const auto masked = aut::decoder::words_to_double(
        aut::decoder::decode_words_masked<layout>(raw));
    const auto integer = aut::decoder::decode_e1_integer<layout>(raw);
    assert(bits(branchy) == bits(expected));
    assert(bits(masked) == bits(expected));
    assert(bits(integer) == bits(expected));
  }
  for (const auto edge : {0u, 1u, 0x3fffffffu, 0x40000000u,
                          0x7fffffffu, 0x80000000u, 0xffffffffu}) {
    const auto expected = aut::storage::decode<aut::storage::e1m30>(edge);
    assert(bits(aut::decoder::words_to_double(
               aut::decoder::decode_words_branchy<layout>(edge))) ==
           bits(expected));
    assert(bits(aut::decoder::decode_e1_integer<layout>(edge)) ==
           bits(expected));
  }
}

void test_e2m29() {
  using layout = aut::decoder::e2m29_layout;
  std::uint32_t raw = 0x13198a2eu;
  for (std::size_t sample = 0; sample < 200000; ++sample) {
    raw = raw * 1664525u + 1013904223u;
    const auto expected = aut::storage::decode<aut::storage::e2m29>(raw);
    const auto branchy = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<layout>(raw));
    const auto masked = aut::decoder::words_to_double(
        aut::decoder::decode_words_masked<layout>(raw));
    if (std::isnan(expected)) {
      assert(std::isnan(branchy));
      assert(std::isnan(masked));
    } else {
      assert(bits(branchy) == bits(expected));
      assert(bits(masked) == bits(expected));
    }
  }
  for (const auto edge : {0u, 1u, 0x1fffffffu, 0x20000000u,
                          0x5fffffffu, 0x60000000u, 0x7fffffffu,
                          0x80000000u, 0xffffffffu}) {
    const auto expected = aut::storage::decode<aut::storage::e2m29>(edge);
    const auto actual = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<layout>(edge));
    assert(std::isnan(expected) ? std::isnan(actual)
                                : bits(actual) == bits(expected));
  }
}

void test_e3m28() {
  using layout = aut::decoder::e3m28_layout;
  std::uint32_t raw = 0xa4093822u;
  for (std::size_t sample = 0; sample < 200000; ++sample) {
    raw = raw * 1664525u + 1013904223u;
    const auto expected = aut::storage::decode<aut::storage::e3m28>(raw);
    const auto branchy = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<layout>(raw));
    const auto masked = aut::decoder::words_to_double(
        aut::decoder::decode_words_masked<layout>(raw));
    if (std::isnan(expected)) {
      assert(std::isnan(branchy));
      assert(std::isnan(masked));
    } else {
      assert(bits(branchy) == bits(expected));
      assert(bits(masked) == bits(expected));
    }
  }
  for (const auto edge : {0u, 1u, 0x0fffffffu, 0x10000000u,
                          0x6fffffffu, 0x70000000u, 0x7fffffffu,
                          0x80000000u, 0xffffffffu}) {
    const auto expected = aut::storage::decode<aut::storage::e3m28>(edge);
    const auto actual = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<layout>(edge));
    assert(std::isnan(expected) ? std::isnan(actual)
                                : bits(actual) == bits(expected));
  }
}

} // namespace

int main() {
  test_e1m6();
  test_fp8_e4m3();
  test_fp8_e5m2();
  test_e1m14();
  test_e2m13();
  test_e3m12();
  test_fp16();
  test_bf16();
  test_e11m4();
  test_e1m30();
  test_e2m29();
  test_e3m28();
  std::cout << "decoder strategy core tests passed\n";
}
