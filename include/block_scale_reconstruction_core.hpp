#pragma once

#include "block_scale_core.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace aut::block_scale_reconstruction {

inline constexpr std::string_view experiment_id =
    "034_block_scale_reconstruction";
inline constexpr std::string_view dataset_id = "block-scale-v1";

inline std::string scaled_id(std::string_view reconstruction, int block,
                             bool scale64) {
  if (block != 16 && block != 32 && block != 128) {
    throw std::runtime_error("invalid block size");
  }
  if (reconstruction == "fp32") {
    if (scale64) throw std::runtime_error("FP32 reconstruction requires S32");
    return "reconstruct32_b" + std::to_string(block) + "_s32";
  }
  if (reconstruction == "fp64") {
    return "reconstruct64_b" + std::to_string(block) +
           (scale64 ? "_s64" : "_s32");
  }
  throw std::runtime_error("invalid reconstruction");
}

inline block_scale::host_arrays make_fixture(std::size_t n, int block,
                                              int fixture) {
  if (fixture == 0) return block_scale::make_arrays(n, block);
  block_scale::host_arrays a;
  a.ql.resize(n);
  a.qr.resize(n);
  const std::size_t blocks = block_scale::ceil_div(n, std::size_t(block));
  a.sl32.resize(blocks);
  a.sr32.resize(blocks);
  a.sl64.resize(blocks);
  a.sr64.resize(blocks);
  const float one_up = std::nextafter(1.0f, 2.0f);
  const float two_up = std::nextafter(one_up, 2.0f);
  static constexpr float edge[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
                                   0.25f, -0.25f};
  for (std::size_t b = 0; b < blocks; ++b) {
    if (fixture == 1) {
      a.sl32[b] = b % 3 == 0 ? 0.50000006f
                  : b % 3 == 1 ? 1.00000012f
                               : 1.99999988f;
      a.sr32[b] = b % 2 == 0 ? 1.99999988f : 0.50000006f;
    } else if (fixture == 2) {
      a.sl32[b] = 1.25f;
      a.sr32[b] = 0.75f;
    } else if (fixture == 3) {
      a.sl32[b] = b % 2 ? two_up : one_up;
      a.sr32[b] = b % 3 ? one_up : two_up;
    } else {
      throw std::runtime_error("unknown fixture");
    }
    a.sl64[b] = double(a.sl32[b]);
    a.sr64[b] = double(a.sr32[b]);
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (fixture == 1) {
      a.ql[i] = edge[i % 7];
      a.qr[i] = edge[(i * 5 + 3) % 7];
    } else if (fixture == 2) {
      a.ql[i] = 0.25f;
      a.qr[i] = 0.5f;
    } else {
      a.ql[i] = i % 5 == 0 ? two_up : one_up;
      a.qr[i] = i % 7 == 0 ? two_up : one_up;
    }
  }
  return a;
}

inline long double reference_reconstruct64(
    const block_scale::host_arrays& a, int block) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < a.ql.size(); ++i) {
    const std::size_t b = i / block;
    const long double left =
        static_cast<long double>(a.ql[i]) * a.sl32[b];
    const long double right =
        static_cast<long double>(a.qr[i]) * a.sr32[b];
    sum += left * right;
  }
  return sum;
}

inline long double reference_reconstruct32(
    const block_scale::host_arrays& a, int block) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < a.ql.size(); ++i) {
    const std::size_t b = i / block;
    const float left = a.ql[i] * a.sl32[b];
    const float right = a.qr[i] * a.sr32[b];
    sum += static_cast<long double>(left) * right;
  }
  return sum;
}

inline long double reference_deferred_local(
    const block_scale::host_arrays& a) {
  long double sum = 0.0L;
  for (std::size_t block = 0; block < a.sl32.size(); ++block) {
    for (int lane = 0; lane < 32; ++lane) {
      long double subtotal = 0.0L;
      for (int j = 0; j < 4; ++j) {
        const std::size_t i = block * 128 + std::size_t(j) * 32 + lane;
        if (i < a.ql.size()) {
          subtotal += static_cast<long double>(a.ql[i]) * a.qr[i];
        }
      }
      const long double scale =
          static_cast<long double>(a.sl32[block]) * a.sr32[block];
      sum += scale * subtotal;
    }
  }
  return sum;
}

inline long double sum_abs_reconstruct64(
    const block_scale::host_arrays& a, int block) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < a.ql.size(); ++i) {
    const std::size_t b = i / block;
    sum += std::abs((static_cast<long double>(a.ql[i]) * a.sl32[b]) *
                    (static_cast<long double>(a.qr[i]) * a.sr32[b]));
  }
  return sum;
}

inline long double sum_abs_reconstruct32(
    const block_scale::host_arrays& a, int block) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < a.ql.size(); ++i) {
    const std::size_t b = i / block;
    const float left = a.ql[i] * a.sl32[b];
    const float right = a.qr[i] * a.sr32[b];
    sum += std::abs(static_cast<long double>(left) * right);
  }
  return sum;
}

inline long double sum_abs_deferred_elements(
    const block_scale::host_arrays& a) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < a.ql.size(); ++i) {
    const std::size_t b = i / 128;
    sum += std::abs((static_cast<long double>(a.ql[i]) * a.qr[i]) *
                    (static_cast<long double>(a.sl32[b]) * a.sr32[b]));
  }
  return sum;
}

}  // namespace aut::block_scale_reconstruction
