#include "compander32_core.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace c = aut::compander32;

namespace {

void require_close(double actual, double expected, double tolerance) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error("compander round-trip check failed");
  }
}

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() try {
  static_assert(c::minimum_code != std::numeric_limits<std::int32_t>::min());
  static_assert(c::maximum_code == std::numeric_limits<std::int32_t>::max());

  constexpr std::array values{-8.0, -3.0, -1.0, -0.125, 0.0,
                              0.125, 1.0, 3.0, 8.0};
  for (const auto value : values) {
    const auto integer = c::encode_integer(value);
    require_close(c::decode_integer(integer) * c::int_value_scale, value,
                  2.0 * c::int_value_scale);

    const auto quadratic = c::encode_quadratic(value);
    require_close(c::decode_quadratic(quadratic) * c::quadratic_value_scale,
                  value, 2.0e-8);

    const auto blended_quadratic = c::encode_blended_quadratic(value);
    require_close(c::decode_blended_quadratic(blended_quadratic) *
                      c::blended_value_scale,
                  value, 2.0e-8);

    const auto blended_cubic = c::encode_blended_cubic(value);
    require_close(c::decode_blended_cubic(blended_cubic) *
                      c::blended_value_scale,
                  value, 2.0e-8);

    require_close(c::decode_pwl2(c::encode_pwl2(value)), value, 1.0e-8);
    require_close(c::decode_pwl4(c::encode_pwl4(value)), value, 1.0e-8);
  }

  require(c::encode_integer(-9.0) == c::minimum_code,
          "negative saturation failed");
  require(c::encode_integer(9.0) == c::maximum_code,
          "positive saturation failed");
  require(c::decode_pwl2(c::encode_pwl2(-1.0)) < 0.0,
          "PWL2 negative sign failed");
  require(c::decode_pwl4(c::encode_pwl4(-1.0)) < 0.0,
          "PWL4 negative sign failed");
  require(c::decode_pwl2(c::encode_pwl2(1.0)) > 0.0,
          "PWL2 positive sign failed");
  require(c::decode_pwl4(c::encode_pwl4(1.0)) > 0.0,
          "PWL4 positive sign failed");
  return 0;
} catch (const std::exception &error) {
  std::cerr << "compander32_core_test: " << error.what() << '\n';
  return 1;
}
