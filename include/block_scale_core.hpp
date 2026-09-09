#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace aut::block_scale {

inline constexpr std::uint64_t value_left_seed = 0x6bd87c012a53f9e1ULL;
inline constexpr std::uint64_t value_right_seed = 0xf5ef05b8551985f4ULL;
inline constexpr std::uint64_t scale_left_seed = 0x243f6a8885a308d3ULL;
inline constexpr std::uint64_t scale_right_seed = 0x13198a2e03707344ULL;
inline constexpr std::string_view seed_set_id = "block-scale-v1";
inline constexpr std::array<std::size_t, 5> full_sizes{1ULL<<16,1ULL<<20,1ULL<<24,1ULL<<26,1ULL<<28};
inline constexpr std::array<int,3> block_sizes{16,32,128};

constexpr std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
constexpr std::uint32_t high32(std::uint64_t seed,std::uint64_t i) {
  return static_cast<std::uint32_t>(splitmix64(seed^i)>>32);
}
inline float source_value(std::uint64_t seed,std::uint64_t i) {
  const std::uint32_t r=high32(seed,i)>>8;
  return static_cast<float>((static_cast<std::int32_t>(r)-8388608)*0x1p-23);
}
inline float scale_value(std::uint64_t seed,std::uint64_t block) {
  const std::uint32_t r=high32(seed,block)>>9;
  return static_cast<float>(8388608u+3u*r)*0x1p-24f;
}
inline float encode(float x,float scale) {
  return static_cast<float>(static_cast<double>(x)/static_cast<double>(scale));
}
constexpr std::size_t ceil_div(std::size_t n,std::size_t d) { return n/d+(n%d!=0); }
inline std::string scaled_id(std::string_view path,int b,bool scale64) {
  if(path!="per_element"&&path!="deferred") throw std::runtime_error("invalid path");
  if(b!=16&&b!=32&&b!=128) throw std::runtime_error("invalid block size");
  return std::string(path)+"_b"+std::to_string(b)+(scale64?"_s64":"_s32");
}

struct host_arrays {
  std::vector<float> ql,qr,sl32,sr32;
  std::vector<double> sl64,sr64;
};
inline host_arrays make_arrays(std::size_t n,int b) {
  host_arrays a; a.ql.resize(n);a.qr.resize(n);
  const auto nb=ceil_div(n,static_cast<std::size_t>(b));
  a.sl32.resize(nb);a.sr32.resize(nb);a.sl64.resize(nb);a.sr64.resize(nb);
  for(std::size_t j=0;j<nb;++j){a.sl32[j]=scale_value(scale_left_seed,j);a.sr32[j]=scale_value(scale_right_seed,j);a.sl64[j]=a.sl32[j];a.sr64[j]=a.sr32[j];}
  for(std::size_t i=0;i<n;++i){a.ql[i]=encode(source_value(value_left_seed,i),a.sl32[i/b]);a.qr[i]=encode(source_value(value_right_seed,i),a.sr32[i/b]);}
  return a;
}
inline long double reference_per_element(const host_arrays&a,int b) {
  long double s=0;for(std::size_t i=0;i<a.ql.size();++i)s+=(static_cast<long double>(a.ql[i])*a.sl32[i/b])*(static_cast<long double>(a.qr[i])*a.sr32[i/b]);return s;
}
inline long double reference_deferred(const host_arrays&a,int b) {
  long double out=0;for(std::size_t base=0; base<a.ql.size();base+=b){long double sub=0;for(std::size_t i=base;i<std::min(base+static_cast<std::size_t>(b),a.ql.size());++i)sub+=static_cast<long double>(a.ql[i])*a.qr[i];out+=(static_cast<long double>(a.sl32[base/b])*a.sr32[base/b])*sub;}return out;
}
}
