#include "posit_takum_core.hpp"

#include <universal/number/posit/posit.hpp>
#include <universal/number/takum/takum.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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

template <typename Float>
void compare_ulp(const std::string &label, std::uint32_t raw, Float actual,
                 Float expected, std::uint64_t allowed) {
  if (std::isnan(actual) && std::isnan(expected)) return;
  const auto actual_bits = bits(actual);
  const auto expected_bits = bits(expected);
  const auto distance = actual_bits > expected_bits
                            ? actual_bits - expected_bits
                            : expected_bits - actual_bits;
  if (distance > allowed)
    throw std::runtime_error(label + " raw=" + std::to_string(raw) +
                             " expected_bits=" + std::to_string(expected_bits) +
                             " actual_bits=" + std::to_string(actual_bits));
}

template <int Bits, typename Float>
Float log_takum_paper_reference(std::uint32_t raw) {
  const auto value_mask = Bits == 32 ? 0xffffffffu
                                     : (std::uint32_t{1} << Bits) - 1u;
  const auto sign_mask = std::uint32_t{1} << (Bits - 1);
  raw &= value_mask;
  if (raw == 0u) return Float{0};
  if (raw == sign_mask) return std::numeric_limits<Float>::quiet_NaN();
  const bool sign = (raw & sign_mask) != 0u;
  const auto magnitude = sign ? ((~raw + 1u) & value_mask) : raw;
  const auto dr = (magnitude >> (Bits - 5)) & 0xfu;
  const auto direction = dr >> 3;
  const auto regime_code = dr & 7u;
  const auto characteristic_bits = direction ? regime_code : 7u - regime_code;
  constexpr unsigned available = Bits - 5;
  const auto stored_characteristic =
      std::min(characteristic_bits, available);
  const auto tail_bits = available - stored_characteristic;
  const auto stored =
      stored_characteristic == 0
          ? 0u
          : (magnitude >> tail_bits) &
                ((std::uint32_t{1} << stored_characteristic) - 1u);
  const auto characteristic_payload =
      stored << (characteristic_bits - stored_characteristic);
  const int characteristic =
      direction ? (static_cast<int>(std::uint32_t{1} << characteristic_bits) -
                   1 + static_cast<int>(characteristic_payload))
                : (1 - static_cast<int>(std::uint32_t{1}
                                        << (characteristic_bits + 1)) +
                   static_cast<int>(characteristic_payload));
  const auto tail = tail_bits == 0
                        ? 0u
                        : magnitude & ((std::uint32_t{1} << tail_bits) - 1u);
  const auto fraction =
      tail_bits == 0
          ? 0.0L
          : static_cast<long double>(tail) /
                static_cast<long double>(std::uint64_t{1} << tail_bits);
  constexpr long double inv_two_ln2 =
      0.721347520444481703679962340500946L;
  const auto decoded =
      std::exp2((static_cast<long double>(characteristic) + fraction) *
                inv_two_ln2);
  const auto result = static_cast<Float>(decoded);
  return sign ? -result : result;
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
  // Universal's direct FP32 conversion underflows some values before applying
  // their significand. Its FP64 value is exact at these widths, so round that
  // independent reference to FP32 after conversion.
  const auto reference_double = static_cast<double>(reference);
  compare_value("linear takum fp32", raw,
                pt::decode_linear_takum<Bits, float>(raw),
                static_cast<float>(reference_double));
  compare_value("linear takum fp64", raw,
                pt::decode_linear_takum<Bits, double>(raw),
                reference_double);
}

template <int Bits> void check_log_takum_code(std::uint32_t raw) {
  compare_ulp("log takum fp32", raw,
              pt::decode_log_takum<Bits, float>(raw),
              log_takum_paper_reference<Bits, float>(raw), 2);
  compare_ulp("log takum fp64", raw,
              pt::decode_log_takum<Bits, double>(raw),
              log_takum_paper_reference<Bits, double>(raw), 1);
}

template <int Bits, int Es> void exhaustive() {
  const std::uint32_t end = std::uint32_t{1} << Bits;
  for (std::uint32_t raw = 0; raw < end; ++raw) {
    check_posit_code<Bits, Es>(raw);
    check_takum_code<Bits>(raw);
    check_log_takum_code<Bits>(raw);
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
    check_log_takum_code<32>(raw);
  }
  for (const auto raw : {0u, 1u, 2u, 0x3fffffffu, 0x40000000u,
                         0x7ffffffeu, 0x7fffffffu, 0x80000000u,
                         0x80000001u, 0xfffffffeu, 0xffffffffu}) {
    check_posit_code<32, 2>(raw);
    check_takum_code<32>(raw);
    check_log_takum_code<32>(raw);
  }

  std::cout << "Universal and log-takum paper-formula cross-validation passed\n";
  return 0;
}
