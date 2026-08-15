#include "bitwidth_benchmark_core.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

std::uint64_t double_bits(double value) {
  std::uint64_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename Format> void test_format() {
  namespace bw = aut::bitwidth;
  const auto codes = std::uint64_t{1} << Format::total_bits;
  for (std::uint64_t code = 0; code < codes; ++code) {
    const auto raw = static_cast<std::uint32_t>(code);
    const auto reference64 = bw::decode_reference<Format, bw::compute_kind::fp64>(raw);
    const auto direct64 =
        bw::decode_raw<Format, bw::compute_kind::fp64,
                       bw::decoder_kind::direct_branchy>(raw);
    if (std::isnan(reference64)) {
      assert(std::isnan(direct64));
    } else {
      assert(double_bits(reference64) == double_bits(direct64));
    }

    if constexpr (Format::exponent_bits <= 8 && Format::fraction_bits <= 23) {
      const auto reference32 = static_cast<float>(reference64);
      const auto direct32 =
          bw::decode_raw<Format, bw::compute_kind::fp32,
                         bw::decoder_kind::direct_branchy>(raw);
      if (std::isnan(reference32)) {
        assert(std::isnan(direct32));
      } else {
        assert(float_bits(reference32) == float_bits(direct32));
      }
    }
  }
}

template <int Bits> void test_dense_storage() {
  namespace bw = aut::bitwidth;
  constexpr std::size_t count = 257;
  std::vector<std::uint32_t> words(bw::dense_word_count<Bits>(count), 0);
  for (std::size_t index = 0; index < count; ++index) {
    const auto raw = static_cast<std::uint32_t>((index * 29 + 7) &
                                                bw::raw_mask<Bits>());
    bw::or_dense_scalar<Bits>(words.data(), index, raw);
  }
  for (std::size_t index = 0; index < count; ++index) {
    const auto expected = static_cast<std::uint32_t>((index * 29 + 7) &
                                                     bw::raw_mask<Bits>());
    assert(bw::load_dense_scalar<Bits>(words.data(), index) == expected);
  }
}

} // namespace

int main() {
  test_dense_storage<2>();
  test_dense_storage<3>();
  test_dense_storage<4>();
  test_dense_storage<5>();
  test_dense_storage<6>();
  test_dense_storage<7>();
  test_dense_storage<9>();
  test_dense_storage<14>();
  test_dense_storage<17>();
  test_dense_storage<28>();
  test_dense_storage<32>();

  test_format<aut::storage::e0m1>();
  test_format<aut::storage::e1m0>();
  test_format<aut::storage::e0m2>();
  test_format<aut::storage::e1m1>();
  test_format<aut::storage::e2m0>();
  test_format<aut::storage::e0m3>();
  test_format<aut::storage::e1m2>();
  test_format<aut::storage::fp4_e2m1>();
  test_format<aut::storage::e3m0>();
  test_format<aut::storage::e0m4>();
  test_format<aut::storage::e1m3>();
  test_format<aut::storage::e2m2>();
  test_format<aut::storage::e3m1>();
  test_format<aut::storage::e4m0>();
  test_format<aut::storage::e0m5>();
  test_format<aut::storage::e1m4>();
  test_format<aut::storage::e2m3>();
  test_format<aut::storage::e3m2>();
  test_format<aut::storage::e4m1>();
  test_format<aut::storage::e5m0>();
  test_format<aut::storage::e0m6>();
  test_format<aut::storage::e2m4>();
  test_format<aut::storage::e3m3>();
  test_format<aut::storage::e5m1>();
  test_format<aut::storage::e6m0>();
  test_format<aut::storage::e0m8>();
  test_format<aut::storage::e2m6>();
  test_format<aut::storage::e4m4>();
  test_format<aut::storage::e5m3>();
  test_format<aut::storage::e8m0>();
  test_format<aut::storage::e2m7>();
  test_format<aut::storage::e5m4>();
  test_format<aut::storage::e8m1>();
  test_format<aut::storage::e0m11>();
  test_format<aut::storage::e2m9>();
  test_format<aut::storage::e5m6>();
  test_format<aut::storage::e8m3>();
  test_format<aut::storage::e11m0>();

  static_assert(aut::bitwidth::dense_geometry<6>::values_per_aligned_chunk ==
                16);
  static_assert(aut::bitwidth::dense_geometry<6>::words_per_aligned_chunk ==
                3);
  static_assert(aut::bitwidth::dense_geometry<24>::values_per_aligned_chunk ==
                4);
  static_assert(aut::bitwidth::dense_geometry<24>::words_per_aligned_chunk ==
                3);
  std::cout << "bitwidth benchmark core tests passed\n";
}
