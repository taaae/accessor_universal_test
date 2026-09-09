#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace aut::instruction_cost {

inline constexpr std::uint64_t left_seed = 0x6bd87c012a53f9e1ULL;
inline constexpr std::uint64_t right_seed = left_seed ^ 0x9e3779b97f4a7c15ULL;
inline constexpr std::uint32_t rotate_operand = 7u;
inline constexpr float fp_multiplier = 1.0001f;
inline constexpr float fp_addend = 1024.1f;
inline constexpr std::array<int, 8> initial_k{1, 2, 4, 8, 12, 16, 24, 32};
inline constexpr std::array<int, 2> extension_k{48, 64};

enum class family { add32f, mul32f, fma32f, rot32 };
inline constexpr std::array<family, 4> families{family::add32f, family::mul32f,
                                                family::fma32f, family::rot32};

constexpr std::string_view name(family f) {
  switch (f) {
  case family::add32f: return "add32f";
  case family::mul32f: return "mul32f";
  case family::fma32f: return "fma32f";
  case family::rot32: return "rot32";
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
  if (f == family::rot32) for (int i=0;i<k;++i) u=rotl32(u,rotate_operand);
  float v=static_cast<float>(u);
  for(int i=0;i<k;++i) {
    if(f==family::add32f) v=v+fp_addend;
    else if(f==family::mul32f) v=v*fp_multiplier;
    else if(f==family::fma32f) v=std::fma(v,fp_multiplier,fp_addend);
  }
  return static_cast<double>(v);
}

} // namespace aut::instruction_cost
