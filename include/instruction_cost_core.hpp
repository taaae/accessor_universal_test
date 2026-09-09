#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace aut::instruction_cost {

inline constexpr std::uint64_t left_seed = 0x6bd87c012a53f9e1ULL;
inline constexpr std::uint64_t right_seed = left_seed ^ 0x9e3779b97f4a7c15ULL;
inline constexpr std::uint32_t add_operand = 0x9e3779b9u;
inline constexpr std::uint32_t xor_operand = 0xa5a5c3c3u;
inline constexpr std::uint32_t rotate_operand = 7u;
inline constexpr std::uint32_t multiply_operand = 0x0019660du;
inline constexpr double fma_multiplier = 0x1.001p0;
inline constexpr double fma_addend = 0x1p-12;
inline constexpr std::array<int, 8> initial_k{1, 2, 4, 8, 12, 16, 24, 32};
inline constexpr std::array<int, 2> extension_k{48, 64};

enum class family { add32, xor32, rot32, mul32, fma64 };
inline constexpr std::array<family, 5> families{family::add32, family::xor32,
                                                family::rot32, family::mul32,
                                                family::fma64};

constexpr std::string_view name(family f) {
  switch (f) {
  case family::add32: return "add32";
  case family::xor32: return "xor32";
  case family::rot32: return "rot32";
  case family::mul32: return "mul32";
  case family::fma64: return "fma64";
  }
  return "invalid";
}

constexpr std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
constexpr std::uint32_t code_at(std::uint64_t index, std::uint64_t seed) {
  return static_cast<std::uint32_t>(splitmix64(seed ^ index) >> 32);
}
constexpr std::uint32_t rotl32(std::uint32_t x, unsigned s) {
  s &= 31u;
  return s == 0 ? x : static_cast<std::uint32_t>((x << s) | (x >> (32u - s)));
}
inline double decode_reference(std::uint32_t u, family f, int k) {
  if (k < 0 || k > 64) throw "invalid instruction count";
  if (f == family::fma64) {
    double v = static_cast<double>(u);
    for (int i = 0; i < k; ++i) v = std::fma(v, fma_multiplier, fma_addend);
    return v;
  }
  for (int i = 0; i < k; ++i) {
    switch (f) {
    case family::add32: u += add_operand; break;
    case family::xor32: u ^= xor_operand; break;
    case family::rot32: u = rotl32(u, rotate_operand); break;
    case family::mul32: u *= multiply_operand; break;
    case family::fma64: break;
    }
  }
  return static_cast<double>(u);
}

} // namespace aut::instruction_cost
