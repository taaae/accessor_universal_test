#include "lns_benchmark_core.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>

namespace {

template <typename Format> void test_raw_domain() {
  const auto count = std::uint64_t{1} << Format::total_bits;
  std::size_t zeros{};
  std::size_t nans{};
  const auto stride = count <= 65536 ? std::uint64_t{1} : count / 65535;
  for (std::uint64_t wide = 0; wide < count; wide += stride) {
    const auto raw = static_cast<std::uint32_t>(wide);
    const auto value = aut::lns::decode<Format, double>(raw);
    if (aut::lns::is_zero<Format>(raw)) {
      ++zeros;
      assert(value == 0.0);
    } else if (aut::lns::is_nan<Format>(raw)) {
      ++nans;
      assert(std::isnan(value));
    } else {
      assert(std::isfinite(value));
      assert(value != 0.0);
      assert(std::signbit(value) == aut::lns::sign<Format>(raw));
    }
  }
  if constexpr (Format::total_bits <= 16) {
    assert(zeros == 1);
    assert(nans == 1);
  } else {
    assert((aut::lns::decode<Format, double>(
                aut::lns::zero_raw<Format>()) == 0.0));
    assert(std::isnan(
        aut::lns::decode<Format, double>(aut::lns::nan_raw<Format>())));
  }
}

template <typename Format> void test_round_trip_representable() {
  const auto count = std::uint64_t{1} << Format::total_bits;
  for (std::uint64_t wide = 0; wide < count; ++wide) {
    const auto raw = static_cast<std::uint32_t>(wide);
    if (aut::lns::is_special<Format>(raw)) {
      continue;
    }
    const auto value = aut::lns::decode<Format, double>(raw);
    assert(aut::lns::encode<Format>(value) == raw);
  }
}

template <typename Format> void test_fused_products() {
  const auto count = std::uint64_t{1} << Format::total_bits;
  const auto stride = count <= 256 ? std::uint64_t{1} : count / 127u;
  for (std::uint64_t left_wide = 0; left_wide < count;
       left_wide += stride) {
    for (std::uint64_t right_wide = 0; right_wide < count;
         right_wide += stride) {
      const auto left = static_cast<std::uint32_t>(left_wide);
      const auto right = static_cast<std::uint32_t>(right_wide);
      const auto fused = aut::lns::multiply_fused<Format, double>(left, right);
      const auto ordinary = aut::lns::decode<Format, double>(left) *
                            aut::lns::decode<Format, double>(right);
      if (std::isnan(ordinary)) {
        assert(std::isnan(fused));
      } else if (std::isinf(ordinary)) {
        assert(std::isinf(fused));
        assert(std::signbit(fused) == std::signbit(ordinary));
      } else {
        const auto tolerance = std::abs(ordinary) * 4e-15;
        assert(std::abs(fused - ordinary) <= tolerance);
      }
    }
  }
}

template <typename Format> void test_format() {
  test_raw_domain<Format>();
  if constexpr (Format::total_bits <= 16) {
    test_round_trip_representable<Format>();
  }
  test_fused_products<Format>();

  constexpr auto scale = std::uint64_t{1} << Format::log_fraction_bits;
  const auto minimum = aut::lns::minimum_finite_log_code<Format>();
  const auto below_threshold =
      std::exp2((static_cast<double>(minimum) - 0.5001) / scale);
  const auto above_threshold =
      std::exp2((static_cast<double>(minimum) - 0.4999) / scale);
  assert(aut::lns::is_zero<Format>(
      aut::lns::encode<Format>(below_threshold)));
  assert(aut::lns::encode<Format>(above_threshold) ==
         aut::lns::make_raw<Format>(false, minimum));
  assert(aut::lns::log_code<Format>(aut::lns::encode<Format>(
             std::numeric_limits<double>::infinity())) ==
         aut::lns::maximum_finite_log_code<Format>());
}

} // namespace

int main() {
  test_format<aut::lns::lns4_r0>();
  test_format<aut::lns::lns4_r1>();
  test_format<aut::lns::lns6_r2>();
  test_format<aut::lns::lns8_r2>();
  test_format<aut::lns::lns8_r3>();
  test_format<aut::lns::lns8_r4>();
  test_format<aut::lns::lns8_r5>();
  test_format<aut::lns::lns10_r4>();
  test_format<aut::lns::lns12_r6>();
  test_format<aut::lns::lns16_r4>();
  test_format<aut::lns::lns16_r7>();
  test_format<aut::lns::lns16_r10>();
  test_format<aut::lns::lns16_r11>();
  test_format<aut::lns::lns16_r12>();
  test_format<aut::lns::lns16_r13>();
  test_format<aut::lns::lns32_r20>();
  test_format<aut::lns::lns32_r23>();
  test_format<aut::lns::lns32_r28>();
  std::cout << "lns_benchmark_core_test passed\n";
}
