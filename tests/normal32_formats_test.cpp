#include "normal32_formats.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <int Bits> void test_dense_loader() {
  constexpr std::array<std::uint64_t, 6> values{
      0,
      1,
      aut::normal32::raw_mask<Bits>() >> 1,
      aut::normal32::raw_mask<Bits>(),
      0x123456789abull & aut::normal32::raw_mask<Bits>(),
      0x2aaaaaaaaaaull & aut::normal32::raw_mask<Bits>(),
  };
  std::vector<std::uint32_t> words(
      aut::normal32::dense_word_count<Bits>(values.size()), 0u);
  const auto output_words =
      (values.size() * static_cast<std::size_t>(Bits) + 31u) / 32u;
  for (std::size_t output = 0; output < output_words; ++output) {
    const auto bit = output * 32u;
    const auto index = bit / Bits;
    const auto offset = static_cast<unsigned>(bit % Bits);
    auto value = values[index] >> offset;
    const auto available = Bits - static_cast<int>(offset);
    if (available < 32 && index + 1 < values.size()) {
      value |= values[index + 1] << available;
    }
    words[output] = static_cast<std::uint32_t>(value);
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    require(aut::normal32::load_dense<Bits>(words.data(), index) ==
                values[index],
            "dense scalar loader changed a code");
  }
}

} // namespace

int main() {
  try {
    using namespace aut::normal32;
    const auto tables = build_codebook_tables();
    require(tables.pwl.size() == pwl_segment_count,
            "wrong PWLNormal32 table size");
    require(tables.pwq.size() == pwq_segment_count,
            "wrong PWQNormal32 table size");
    require(sizeof(pwl_coeff) == 16, "PWL coefficients must occupy 16 bytes");
    require(sizeof(pwq_coeff) == 24, "PWQ coefficients must occupy 24 bytes");

    for (const auto &coefficient : tables.pwl) {
      require(std::isfinite(coefficient.a) && std::isfinite(coefficient.b) &&
                  coefficient.b > 0.0,
              "invalid PWLNormal32 coefficient");
    }
    for (const auto &coefficient : tables.pwq) {
      require(std::isfinite(coefficient.a) && std::isfinite(coefficient.b) &&
                  std::isfinite(coefficient.c) && coefficient.b > 0.0,
              "invalid PWQNormal32 coefficient");
    }

    constexpr std::array<double, 9> z_values{-4.0, -3.0, -1.0, -0.25, 0.0,
                                             0.25, 1.0,  3.0,  4.0};
    double previous_pwl = -std::numeric_limits<double>::infinity();
    double previous_pwq = -std::numeric_limits<double>::infinity();
    for (const auto z : z_values) {
      const auto value = z * normal_sigma;
      const auto pwl = decode_pwl(encode_pwl(value), tables.pwl);
      const auto pwq = decode_pwq(encode_pwq(value), tables.pwq);
      const auto qn = decode_qn32(encode_qn32(value));
      require(std::isfinite(pwl) && std::isfinite(pwq) && std::isfinite(qn),
              "Normal32 round trip produced a non-finite value");
      require(pwl >= previous_pwl && pwq >= previous_pwq,
              "piecewise codebook is not monotone on the source samples");
      require(std::abs(pwl - value) < 1.0e-6 * normal_sigma,
              "PWLNormal32 round-trip error is unexpectedly large");
      // The 256-segment quadratic deliberately spends approximation error in
      // the far tails. At 4 sigma its pointwise error is about 4.4e-4 sigma,
      // while the distribution-weighted error remains near the ideal table.
      require(std::abs(pwq - value) < 1.0e-3 * normal_sigma,
              "PWQNormal32 round-trip error is unexpectedly large");
      require(std::abs(qn - value) < 1.0e-8 * normal_sigma,
              "QN32 round-trip error is unexpectedly large");
      previous_pwl = pwl;
      previous_pwq = pwq;
    }

    require(decode_pwl(encode_pwl(0.0), tables.pwl) == 0.0,
            "PWLNormal32 must represent zero exactly");
    require(decode_pwq(encode_pwq(0.0), tables.pwq) == 0.0,
            "PWQNormal32 must represent zero exactly");
    require(decode_qn32(encode_qn32(0.0)) == 0.0,
            "QN32 must represent zero exactly");

    for (const auto value : {0.0, -0.0, 1.0, -1.0, normal_sigma, -normal_sigma,
                             4.0 * normal_sigma}) {
      const auto e8m29 = decode_ieee<8, 29>(encode_ieee<8, 29>(value));
      const auto e8m30 = decode_ieee<8, 30>(encode_ieee<8, 30>(value));
      const auto e11m36 = decode_ieee<11, 36>(encode_ieee<11, 36>(value));
      require(std::isfinite(e8m29) && std::isfinite(e8m30) &&
                  std::isfinite(e11m36),
              "extended IEEE round trip produced a non-finite value");
      require(std::signbit(e8m29) == std::signbit(value),
              "extended IEEE round trip changed the sign");
    }

    test_dense_loader<38>();
    test_dense_loader<39>();
    test_dense_loader<48>();
  } catch (const std::exception &error) {
    std::cerr << "normal32_formats_test: " << error.what() << '\n';
    return 1;
  }
  std::cout << "normal32_formats_test passed\n";
  return 0;
}
