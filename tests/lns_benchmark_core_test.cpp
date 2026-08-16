#include "lns_benchmark_core.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

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

// The fused-sum lookup indexes one table by the biased sum of two log codes.
// Build that table exactly as the benchmark does, then check across the whole
// (or a strided) operand domain that every finite product lands inside the
// table and reads back the value multiply_fused would have computed.  An
// indexing slip here would be an out-of-bounds read on the GPU.
template <typename Format> void test_product_log_table() {
  constexpr auto entries = aut::lns::product_log_entries<Format>();
  constexpr auto bias = aut::lns::product_log_bias<Format>();
  constexpr auto scale =
      static_cast<double>(std::uint64_t{1} << Format::log_fraction_bits);

  std::vector<double> table(entries);
  for (std::size_t index = 0; index < entries; ++index) {
    const auto raw_log = static_cast<std::int64_t>(index) - bias;
    const auto clamped = raw_log < -bias ? -bias : raw_log > bias ? bias : raw_log;
    table[index] = std::exp2(static_cast<double>(clamped) / scale);
  }

  const auto count = std::uint64_t{1} << Format::total_bits;
  const auto stride = count <= 256 ? std::uint64_t{1} : count / 255;
  std::size_t checked{};
  for (std::uint64_t a = 0; a < count; a += stride) {
    for (std::uint64_t b = 0; b < count; b += stride) {
      const auto left = static_cast<std::uint32_t>(a);
      const auto right = static_cast<std::uint32_t>(b);
      const auto product = aut::lns::multiply_codes<Format>(left, right);
      if (product.nan || product.zero) {
        continue;
      }
      const auto index = product.log + static_cast<std::int64_t>(bias);
      assert(index >= 0);
      assert(static_cast<std::size_t>(index) < entries);
      // The slot must hold the true magnitude, not a clamped stand-in.
      assert(product.log >= -bias && product.log <= bias);
      const auto expected =
          aut::lns::multiply_fused<Format, double>(left, right);
      const auto magnitude = table[static_cast<std::size_t>(index)];
      const auto actual = product.negative ? -magnitude : magnitude;
      // A wide-range format overflows the compute type long before it runs
      // out of log codes, so the table and the reference must agree on
      // infinity too rather than be differenced into a NaN.
      if (std::isinf(expected)) {
        assert(std::isinf(actual));
        assert(std::signbit(actual) == std::signbit(expected));
      } else {
        assert(std::isfinite(actual));
        const auto denominator =
            std::abs(expected) > 0.0 ? std::abs(expected) : 1.0;
        assert(std::abs(actual - expected) / denominator < 1e-12);
      }
      ++checked;
    }
  }
  assert(checked > 0);
}

template <typename Format> void test_format() {
  test_raw_domain<Format>();
  if constexpr (Format::total_bits <= 16) {
    test_round_trip_representable<Format>();
    test_product_log_table<Format>();
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
