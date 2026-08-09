#include "storage_formats.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>

namespace {

using namespace aut::storage;

template <typename Format> bool same_value_or_nan(storage_type_t<Format> raw) {
  const auto value = decode<Format>(raw);
  const auto encoded = encode<Format>(value);
  if (std::isnan(value)) {
    return std::isnan(decode<Format>(encoded));
  }
  return encoded == raw;
}

template <typename Format> bool exhaustive_round_trip() {
  using storage_type = storage_type_t<Format>;
  static_assert(sizeof(storage_type) <= 2);
  const std::uint64_t count = std::uint64_t{1} << Format::total_bits;
  for (std::uint64_t bits = 0; bits < count; ++bits) {
    if (!same_value_or_nan<Format>(static_cast<storage_type>(bits))) {
      std::cerr << Format::name << " failed round-trip at raw bits " << bits
                << '\n';
      return false;
    }
  }
  return true;
}

template <typename Format> bool sampled_round_trip() {
  using storage_type = storage_type_t<Format>;
  std::uint32_t state = 0x9e3779b9u;
  for (int i = 0; i < 100000; ++i) {
    state = state * 1664525u + 1013904223u;
    if (!same_value_or_nan<Format>(static_cast<storage_type>(state))) {
      std::cerr << Format::name << " failed sampled round-trip at raw bits "
                << state << '\n';
      return false;
    }
  }
  return true;
}

template <typename Format> bool check_exact(double value) {
  const auto decoded = decode<Format>(encode<Format>(value));
  if (decoded != value || std::signbit(decoded) != std::signbit(value)) {
    std::cerr << Format::name << " did not preserve " << value << '\n';
    return false;
  }
  return true;
}

} // namespace

int main() {
  static_assert(std::is_same_v<storage_type_t<e1m6>, std::uint8_t>);
  static_assert(std::is_same_v<storage_type_t<e2m13>, std::uint16_t>);
  static_assert(std::is_same_v<storage_type_t<e3m28>, std::uint32_t>);

  bool ok = true;
  ok = ok && check_exact<e1m6>(1.0);
  ok = ok && check_exact<e0m1>(0.5);
  ok = ok && check_exact<e0m3>(0.375);
  ok = ok && check_exact<fp4_e2m1>(6.0);
  ok = ok && check_exact<e3m0>(1.0);
  ok = ok && check_exact<e1m6>(-1.0);
  ok = ok && check_exact<e2m5>(0.5);
  ok = ok && check_exact<e3m4>(0.25);
  ok = ok && check_exact<e11m4>(1.0);
  ok = ok && check_exact<e11m20>(-0.0);

  if (decode<e2m5>(encode<e2m5>(1.0 + std::ldexp(1.0, -6))) != 1.0) {
    std::cerr << "e2m5 failed an even-lower RNE midpoint\n";
    ok = false;
  }
  if (decode<e2m5>(encode<e2m5>(1.0 + 3.0 * std::ldexp(1.0, -6))) !=
      1.0 + std::ldexp(1.0, -4)) {
    std::cerr << "e2m5 failed an even-upper RNE midpoint\n";
    ok = false;
  }
  if (decode<e11m4>(encode<e11m4>(1.0 + std::ldexp(1.0, -5))) != 1.0) {
    std::cerr << "e11m4 failed an RNE midpoint\n";
    ok = false;
  }

  ok = ok && exhaustive_round_trip<e0m1>();
  ok = ok && exhaustive_round_trip<e1m0>();
  ok = ok && exhaustive_round_trip<e0m3>();
  ok = ok && exhaustive_round_trip<e1m2>();
  ok = ok && exhaustive_round_trip<fp4_e2m1>();
  ok = ok && exhaustive_round_trip<e3m0>();
  ok = ok && exhaustive_round_trip<e0m7>();
  ok = ok && exhaustive_round_trip<e1m6>();
  ok = ok && exhaustive_round_trip<e2m5>();
  ok = ok && exhaustive_round_trip<e3m4>();
  ok = ok && exhaustive_round_trip<e6m1>();
  ok = ok && exhaustive_round_trip<e7m0>();
  ok = ok && exhaustive_round_trip<e0m15>();
  ok = ok && exhaustive_round_trip<e1m14>();
  ok = ok && exhaustive_round_trip<e2m13>();
  ok = ok && exhaustive_round_trip<e3m12>();
  ok = ok && exhaustive_round_trip<e4m11>();
  ok = ok && exhaustive_round_trip<e6m9>();
  ok = ok && exhaustive_round_trip<e7m8>();
  ok = ok && exhaustive_round_trip<e9m6>();
  ok = ok && exhaustive_round_trip<e10m5>();
  ok = ok && exhaustive_round_trip<e11m4>();

  ok = ok && sampled_round_trip<e0m31>();
  ok = ok && sampled_round_trip<e1m30>();
  ok = ok && sampled_round_trip<e2m29>();
  ok = ok && sampled_round_trip<e3m28>();
  ok = ok && sampled_round_trip<e4m27>();
  ok = ok && sampled_round_trip<e5m26>();
  ok = ok && sampled_round_trip<e6m25>();
  ok = ok && sampled_round_trip<e7m24>();
  ok = ok && sampled_round_trip<e9m22>();
  ok = ok && sampled_round_trip<e10m21>();
  ok = ok && sampled_round_trip<e11m20>();

  const auto e2_inf =
      decode<e2m5>(encode<e2m5>(std::numeric_limits<double>::infinity()));
  if (!std::isinf(e2_inf)) {
    std::cerr << "e2m5 did not preserve infinity\n";
    ok = false;
  }

  const auto e1_saturated = decode<e1m6>(encode<e1m6>(100.0));
  if (!std::isfinite(e1_saturated) || e1_saturated >= 4.0) {
    std::cerr << "e1m6 did not saturate to a finite value\n";
    ok = false;
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
