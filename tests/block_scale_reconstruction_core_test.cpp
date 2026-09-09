#include "block_scale_reconstruction_core.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace aut::block_scale;
namespace reconstruction = aut::block_scale_reconstruction;

static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  require(high32(value_left_seed, 0) == 0xfe73d3b6u,
          "generator fixture changed");
  require(reconstruction::scaled_id("fp32", 16, false) ==
              "reconstruct32_b16_s32",
          "FP32 id mismatch");
  bool rejected = false;
  try {
    (void)reconstruction::scaled_id("fp32", 16, true);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "accepted unsupported FP32/S64 combination");

  for (std::size_t n : {0u, 1u, 15u, 16u, 17u, 127u, 128u, 129u, 257u}) {
    for (int block : block_sizes) {
      const auto a = reconstruction::make_fixture(n, block, 0);
      require(a.ql.size() == n && a.sl32.size() == ceil_div(n, block),
              "generated size mismatch");
      for (std::size_t i = 0; i < a.sl32.size(); ++i) {
        require(a.sl64[i] == double(a.sl32[i]) &&
                    a.sr64[i] == double(a.sr32[i]),
                "scale widening changed value");
      }
    }
  }

  for (int block : block_sizes) {
    const auto a = reconstruction::make_fixture(257, block, 3);
    const long double r32 = reconstruction::reference_reconstruct32(a, block);
    const long double r64 = reconstruction::reference_reconstruct64(a, block);
    require(r32 != r64, "rounding fixture failed to distinguish contracts");
  }

  const float one_up = std::nextafter(1.0f, 2.0f);
  const float left_rounded = one_up * one_up;
  const float right_rounded = one_up * one_up;
  std::uint32_t left_bits = 0, right_bits = 0;
  std::memcpy(&left_bits, &left_rounded, sizeof(left_rounded));
  std::memcpy(&right_bits, &right_rounded, sizeof(right_rounded));
  require(left_bits == 0x3f800002u && right_bits == 0x3f800002u,
          "rounding-sensitive FP32 operand product bits changed");
  require(static_cast<double>(left_rounded) == 0x1.000004p0 &&
              static_cast<double>(right_rounded) == 0x1.000004p0,
          "rounded FP32 operand products did not widen exactly");

  const auto ragged = reconstruction::make_fixture(257, 128, 3);
  require(ragged.sl32.size() == 3, "ragged B128 block count mismatch");
  require(std::isfinite(double(reconstruction::reference_deferred_local(ragged))),
          "deferred reference is nonfinite");
  require(reconstruction::sum_abs_deferred_elements(ragged) > 0,
          "deferred absolute contribution sum is empty");
  std::cout << "block_scale_reconstruction_core_test passed\n";
}
