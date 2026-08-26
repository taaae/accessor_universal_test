#include "posit_takum_core.hpp"

#include <universal/number/posit/posit.hpp>
#include <universal/number/takum/takum.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace pt = aut::pt;

template <typename Float> std::uint64_t bits(Float value) {
  if constexpr (sizeof(Float) == 4) {
    std::uint32_t raw{};
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
  } else {
    std::uint64_t raw{};
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
  }
}

template <typename Float>
void compare_value(const std::string &label, std::uint32_t raw, Float actual,
                   Float expected) {
  if (std::isnan(actual) && std::isnan(expected)) {
    return;
  }
  if (bits(actual) != bits(expected)) {
    throw std::runtime_error(label + " raw=" + std::to_string(raw) +
                             " expected_bits=" + std::to_string(bits(expected)) +
                             " actual_bits=" + std::to_string(bits(actual)));
  }
}

void compare_float_with_reference_rounding(const std::string &label,
                                           std::uint32_t raw, float actual,
                                           float expected) {
  if (std::isnan(actual) && std::isnan(expected)) {
    return;
  }
  const auto actual_bits = static_cast<std::uint32_t>(bits(actual));
  const auto expected_bits = static_cast<std::uint32_t>(bits(expected));
  const auto distance = actual_bits > expected_bits
                            ? actual_bits - expected_bits
                            : expected_bits - actual_bits;
  if (distance > 1u) {
    throw std::runtime_error(label + " raw=" + std::to_string(raw) +
                             " expected_bits=" + std::to_string(expected_bits) +
                             " actual_bits=" + std::to_string(actual_bits));
  }
}

template <int Bits, int Es> void check_posit_code(std::uint32_t raw) {
  sw::universal::posit<Bits, Es> reference;
  reference.setbits(raw);
  compare_value("posit fp32", raw, pt::decode_posit<Bits, Es, float>(raw),
                static_cast<float>(reference));
  compare_value("posit fp64", raw, pt::decode_posit<Bits, Es, double>(raw),
                static_cast<double>(reference));
}

template <int Bits> void check_takum_code(std::uint32_t raw) {
  sw::universal::takum<Bits> reference;
  reference.setbits(raw);
  const auto fields = pt::split_takum<Bits>(raw);
  // Universal's current FP32 conversion first casts 2^c to float. For
  // c < -149 that intermediate rounds to zero before the significand is
  // applied, so it is not an exact reference for boundary subnormals.
  if (raw == 0u || raw == pt::sign_mask<Bits>() ||
      fields.characteristic >= -149) {
    compare_float_with_reference_rounding(
        "linear takum fp32", raw,
        pt::decode_linear_takum<Bits, float>(raw),
        static_cast<float>(reference));
  }
  compare_value("linear takum fp64", raw,
                pt::decode_linear_takum<Bits, double>(raw),
                static_cast<double>(reference));
}

template <int Bits, int Es> void exhaustive() {
  const std::uint32_t end = std::uint32_t{1} << Bits;
  for (std::uint32_t raw = 0; raw < end; ++raw) {
    check_posit_code<Bits, Es>(raw);
    check_takum_code<Bits>(raw);
  }
}

int main() {
  exhaustive<8, 0>();
  exhaustive<14, 1>();
  exhaustive<16, 1>();

  std::mt19937_64 rng(0x6bd87c012a53f9e1ULL);
  for (std::size_t index = 0; index < 1'000'000; ++index) {
    const auto raw = static_cast<std::uint32_t>(rng());
    check_posit_code<32, 2>(raw);
    check_takum_code<32>(raw);
  }
  for (const auto raw : {0u, 1u, 2u, 0x3fffffffu, 0x40000000u,
                         0x7ffffffeu, 0x7fffffffu, 0x80000000u,
                         0x80000001u, 0xfffffffeu, 0xffffffffu}) {
    check_posit_code<32, 2>(raw);
    check_takum_code<32>(raw);
  }

  std::cout << "Universal cross-validation passed\n";
  return 0;
}
