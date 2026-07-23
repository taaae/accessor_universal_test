#include "memory_accessor.hpp"

#include <cstdlib>
#include <type_traits>

int main() {
  const float values[]{1.25f, -2.5f, 4.0f};
  const aut::memory_accessor<float, float> fp32{values};
  const aut::memory_accessor<double, float> fp64{values};

  static_assert(std::is_same_v<decltype(fp32[0]), float>);
  static_assert(std::is_same_v<decltype(fp64[0]), double>);

  if (fp32.data() != values || fp64.data() != values) {
    return EXIT_FAILURE;
  }
  if (fp32[1] != -2.5f || fp64[1] != -2.5) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
