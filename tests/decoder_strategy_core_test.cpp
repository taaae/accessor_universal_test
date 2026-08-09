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

void test_e2m5() {
  using layout = aut::decoder::e2m5_layout;
  for (std::uint32_t raw = 0; raw < 256; ++raw) {
    const auto expected = aut::storage::decode<aut::storage::e2m5>(
        static_cast<std::uint8_t>(raw));
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

void test_e3m4() {
  using layout = aut::decoder::e3m4_layout;
  for (std::uint32_t raw = 0; raw < 256; ++raw) {
    const auto expected = aut::storage::decode<aut::storage::e3m4>(
        static_cast<std::uint8_t>(raw));
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

void test_fp32() {
  using layout = aut::decoder::fp32_e8m23_layout;
  std::uint32_t raw = 0x299f31d0u;
  for (std::size_t sample = 0; sample < 200000; ++sample) {
    raw = raw * 1664525u + 1013904223u;
    const auto expected =
        static_cast<double>(aut::decoder::bits_to_float(raw));
    const auto branchy = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<layout>(raw));
    const auto masked = aut::decoder::words_to_double(
        aut::decoder::decode_words_masked<layout>(raw));
    const auto native = aut::decoder::decode_via_fp32<layout>(raw);
    if (std::isnan(expected)) {
      assert(std::isnan(branchy));
      assert(std::isnan(masked));
      assert(std::isnan(native));
    } else {
      assert(bits(branchy) == bits(expected));
      assert(bits(masked) == bits(expected));
      assert(bits(native) == bits(expected));
    }
  }
}

void test_e11m20() {
  std::uint32_t raw = 0x082efa98u;
  for (std::size_t sample = 0; sample < 200000; ++sample) {
    raw = raw * 1664525u + 1013904223u;
    const auto expected = aut::storage::decode<aut::storage::e11m20>(raw);
    const auto direct = aut::decoder::words_to_double(
        aut::decoder::decode_prefix_words<20>(raw));
    assert(bits(direct) == bits(expected));
  }
}

template <typename Format, typename Layout, bool ViaFp32 = false,
          bool FixedInteger = false, bool ExponentOnly = false>
void test_added_format() {
  constexpr auto exhaustive = Format::total_bits <= 16;
  const std::uint64_t count =
      exhaustive ? (std::uint64_t{1} << Format::total_bits) : 200000;
  std::uint32_t raw = 0x6a09e667u;
  for (std::uint64_t sample = 0; sample < count; ++sample) {
    if constexpr (exhaustive) {
      raw = static_cast<std::uint32_t>(sample);
    } else {
      raw = raw * 1664525u + 1013904223u;
    }
    const auto expected = aut::storage::decode<Format>(
        static_cast<aut::storage::storage_type_t<Format>>(raw));
    const auto branchy = aut::decoder::words_to_double(
        aut::decoder::decode_words_branchy<Layout>(raw));
    const auto masked = aut::decoder::words_to_double(
        aut::decoder::decode_words_masked<Layout>(raw));
    const auto match = [expected](double actual) {
      return std::isnan(expected) ? std::isnan(actual)
                                  : bits(actual) == bits(expected);
    };
    assert(match(branchy));
    assert(match(masked));
    if constexpr (ViaFp32) {
      assert(match(aut::decoder::decode_via_fp32<Layout>(raw)));
    }
    if constexpr (FixedInteger) {
      assert(match(aut::decoder::decode_fixed_integer<Layout>(raw)));
    }
    if constexpr (ExponentOnly) {
      assert(match(aut::decoder::decode_exponent_only<Layout>(raw)));
    }
  }
}

} // namespace

int main() {
  test_added_format<aut::storage::e0m1, aut::decoder::e0m1_layout, true,
                    true>();
  test_added_format<aut::storage::e1m0, aut::decoder::e1m0_layout, true,
                    false, true>();
  test_added_format<aut::storage::e0m3, aut::decoder::e0m3_layout, true,
                    true>();
  test_added_format<aut::storage::e1m2, aut::decoder::e1m2_layout, true>();
  test_added_format<aut::storage::fp4_e2m1,
                    aut::decoder::fp4_e2m1_layout, true>();
  test_added_format<aut::storage::e3m0, aut::decoder::e3m0_layout, true,
                    false, true>();
  test_added_format<aut::storage::e0m7, aut::decoder::e0m7_layout, true,
                    true>();
  test_e1m6();
  test_e2m5();
  test_e3m4();
  test_fp8_e4m3();
  test_fp8_e5m2();
  test_added_format<aut::storage::e6m1, aut::decoder::e6m1_layout, true>();
  test_added_format<aut::storage::e7m0, aut::decoder::e7m0_layout, true,
                    false, true>();
  test_added_format<aut::storage::e0m15, aut::decoder::e0m15_layout, true,
                    true>();
  test_e1m14();
  test_e2m13();
  test_e3m12();
  test_added_format<aut::storage::e4m11, aut::decoder::e4m11_layout, true>();
  test_fp16();
  test_added_format<aut::storage::e6m9, aut::decoder::e6m9_layout, true>();
  test_added_format<aut::storage::e7m8, aut::decoder::e7m8_layout, true>();
  test_bf16();
  test_added_format<aut::storage::e9m6, aut::decoder::e9m6_layout>();
  test_added_format<aut::storage::e10m5, aut::decoder::e10m5_layout>();
  test_e11m4();
  test_added_format<aut::storage::e0m31, aut::decoder::e0m31_layout, false,
                    true>();
  test_e1m30();
  test_e2m29();
  test_e3m28();
  test_added_format<aut::storage::e4m27, aut::decoder::e4m27_layout>();
  test_added_format<aut::storage::e5m26, aut::decoder::e5m26_layout>();
  test_added_format<aut::storage::e6m25, aut::decoder::e6m25_layout>();
  test_added_format<aut::storage::e7m24, aut::decoder::e7m24_layout>();
  test_fp32();
  test_added_format<aut::storage::e9m22, aut::decoder::e9m22_layout>();
  test_added_format<aut::storage::e10m21, aut::decoder::e10m21_layout>();
  test_e11m20();
  std::cout << "decoder strategy core tests passed\n";
}
