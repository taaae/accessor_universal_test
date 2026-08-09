#include "format_decoder_strategies.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = aut::format_strategies;
namespace storage = aut::storage;

namespace {

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::uint64_t bits(double value) {
  std::uint64_t result{};
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

template <typename T> struct device_buffer {
  T *data{};
  std::size_t count{};

  explicit device_buffer(std::size_t count_) : count(count_) {
    if (count != 0) {
      check_cuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
    }
  }
  ~device_buffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
};

template <typename Format>
storage::storage_type_t<Format> host_storage_from_raw(std::uint32_t raw) {
  static_assert(std::is_integral_v<storage::storage_type_t<Format>>);
  return static_cast<storage::storage_type_t<Format>>(raw);
}

template <>
storage::storage_type_t<storage::fp8_e4m3>
host_storage_from_raw<storage::fp8_e4m3>(std::uint32_t raw) {
  storage::storage_type_t<storage::fp8_e4m3> value;
  value.__x = static_cast<__nv_fp8_storage_t>(raw);
  return value;
}

template <>
storage::storage_type_t<storage::fp8_e5m2>
host_storage_from_raw<storage::fp8_e5m2>(std::uint32_t raw) {
  storage::storage_type_t<storage::fp8_e5m2> value;
  value.__x = static_cast<__nv_fp8_storage_t>(raw);
  return value;
}

template <>
storage::storage_type_t<storage::fp16_e5m10>
host_storage_from_raw<storage::fp16_e5m10>(std::uint32_t raw) {
  return __half{__half_raw{static_cast<unsigned short>(raw)}};
}

template <typename Format> struct smoke_context {
  using storage_type = storage::storage_type_t<Format>;
  using layout = fs::format_layout_t<Format>;
  static constexpr auto code_count = std::size_t{1} << layout::total_bits;
  static constexpr auto subnormal_count =
      std::size_t{1} << layout::fraction_bits;
  static constexpr auto pair_count =
      layout::total_bits == 8 ? (std::size_t{1} << 16) : std::size_t{0};
  static constexpr auto prefix_count =
      std::size_t{1} << (layout::exponent_bits + 1);

  std::vector<storage_type> codes;
  std::vector<double> expected;
  std::vector<std::uint32_t> full_high;
  std::vector<std::uint32_t> subnormal_high;
  std::vector<std::uint32_t> prefix_high;
  std::vector<uint2> pair_high;
  device_buffer<storage_type> device_codes;
  device_buffer<double> device_output;
  device_buffer<std::uint32_t> device_full_high;
  device_buffer<std::uint32_t> device_subnormal_high;
  device_buffer<std::uint32_t> device_prefix_high;
  device_buffer<uint2> device_pair_high;

  smoke_context()
      : codes(code_count), expected(code_count), full_high(code_count),
        subnormal_high(subnormal_count), prefix_high(prefix_count),
        pair_high(pair_count),
        device_codes(codes.size()), device_output(codes.size()),
        device_full_high(full_high.size()),
        device_subnormal_high(subnormal_high.size()),
        device_prefix_high(prefix_high.size()),
        device_pair_high(pair_high.size()) {
    static_assert(layout::total_bits == 8 || layout::total_bits == 16,
                  "the exhaustive smoke context covers 8- and 16-bit codes");
    for (std::uint32_t raw = 0; raw < code_count; ++raw) {
      codes[raw] = host_storage_from_raw<Format>(raw);
      expected[raw] = storage::decode<Format>(codes[raw]);
      full_high[raw] = static_cast<std::uint32_t>(bits(expected[raw]) >> 32);
    }
    for (std::uint32_t fraction = 0; fraction < subnormal_count; ++fraction) {
      const auto value =
          storage::decode<Format>(host_storage_from_raw<Format>(fraction));
      subnormal_high[fraction] =
          static_cast<std::uint32_t>(bits(value) >> 32);
    }
    for (std::uint32_t prefix = 0; prefix < prefix_count; ++prefix) {
      const auto raw = prefix << layout::fraction_bits;
      const auto value =
          storage::decode<Format>(host_storage_from_raw<Format>(raw));
      prefix_high[prefix] = static_cast<std::uint32_t>(bits(value) >> 32);
    }
    if constexpr (layout::total_bits == 8) {
      for (std::uint32_t high = 0; high < 256; ++high) {
        for (std::uint32_t low = 0; low < 256; ++low) {
          pair_high[low + (high << 8)] = {full_high[low], full_high[high]};
        }
      }
    }
    check_cuda(cudaMemcpy(device_codes.data, codes.data(),
                          codes.size() * sizeof(codes[0]),
                          cudaMemcpyHostToDevice),
               "copy codes");
    check_cuda(cudaMemcpy(device_full_high.data, full_high.data(),
                          full_high.size() * sizeof(full_high[0]),
                          cudaMemcpyHostToDevice),
               "copy full table");
    check_cuda(cudaMemcpy(device_subnormal_high.data, subnormal_high.data(),
                          subnormal_high.size() * sizeof(subnormal_high[0]),
                          cudaMemcpyHostToDevice),
               "copy subnormal table");
    check_cuda(cudaMemcpy(device_prefix_high.data, prefix_high.data(),
                          prefix_high.size() * sizeof(prefix_high[0]),
                          cudaMemcpyHostToDevice),
               "copy prefix table");
    if constexpr (layout::total_bits == 8) {
      check_cuda(cudaMemcpy(device_pair_high.data, pair_high.data(),
                            pair_high.size() * sizeof(pair_high[0]),
                            cudaMemcpyHostToDevice),
                 "copy pair table");
    }
  }

  fs::table_bundle tables() const {
    return {device_full_high.data, device_subnormal_high.data,
            device_prefix_high.data,
            device_pair_high.data};
  }
};

template <typename Format, typename Strategy>
void run_strategy(smoke_context<Format> &context, const char *format_name,
                  const char *strategy_name, std::ofstream *csv) {
  constexpr auto lanes = Strategy::lanes;
  const auto packs = (context.codes.size() + lanes - 1) / lanes;
  const auto blocks = static_cast<unsigned>((packs + 255) / 256);
  constexpr auto shared_bytes = fs::shared_table_bytes_v<Format, Strategy>;
  if constexpr (shared_bytes > 48 * 1024) {
    check_cuda(cudaFuncSetAttribute(
                   fs::decode_codes<Format, Strategy>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(shared_bytes)),
               "set dynamic shared-memory size");
  }
  fs::decode_codes<Format, Strategy>
      <<<blocks, 256, shared_bytes>>>(
          context.device_codes.data, context.codes.size(), context.tables(),
          context.device_output.data);
  check_cuda(cudaGetLastError(), strategy_name);
  check_cuda(cudaDeviceSynchronize(), "decode synchronization");

  std::vector<double> actual(context.expected.size());
  check_cuda(cudaMemcpy(actual.data(), context.device_output.data,
                        actual.size() * sizeof(actual[0]),
                        cudaMemcpyDeviceToHost),
             "copy decoder output");
  std::size_t mismatches{};
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto matches = std::isnan(context.expected[i])
                             ? std::isnan(actual[i])
                             : bits(actual[i]) == bits(context.expected[i]);
    mismatches += !matches;
  }
  if (csv != nullptr) {
    *csv << format_name << ',' << strategy_name << ',' << lanes << ','
         << mismatches << '\n';
  }
  if (mismatches != 0) {
    throw std::runtime_error(std::string(format_name) + '/' + strategy_name +
                             " mismatches=" + std::to_string(mismatches));
  }
  std::cout << "  " << strategy_name << " passed\n";
}

template <fs::decode_kind Kind, int Lanes,
          fs::table_location Location = fs::table_location::global_read_only>
using s = fs::strategy<Kind, Lanes, Location>;

#define RUN(format, context, kind, lanes, location, name)                     \
  run_strategy<storage::format,                                               \
               s<fs::decode_kind::kind, lanes, fs::table_location::location>>( \
      context, #format, name, csv)

void run_e1m6_suite(std::ofstream *csv) {
  std::cout << "E1M6 exhaustive strategy validation\n";
  smoke_context<storage::e1m6> context;
  RUN(e1m6, context, generic, 1, global_read_only, "generic_x1");
  RUN(e1m6, context, generic, 4, global_read_only, "generic_x4");
  RUN(e1m6, context, generic, 8, global_read_only, "generic_x8");
  RUN(e1m6, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(e1m6, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(e1m6, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(e1m6, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(e1m6, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(e1m6, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(e1m6, context, fp32_bits, 1, global_read_only, "fp32_bits_x1");
  RUN(e1m6, context, fp32_bits, 4, global_read_only, "fp32_bits_x4");
  RUN(e1m6, context, fp32_bits, 8, global_read_only, "fp32_bits_x8");
  RUN(e1m6, context, e1_integer, 1, global_read_only, "e1_integer_x1");
  RUN(e1m6, context, e1_integer, 4, global_read_only, "e1_integer_x4");
  RUN(e1m6, context, e1_integer, 8, global_read_only, "e1_integer_x8");
  RUN(e1m6, context, full_high_lut, 1, global_read_only,
      "full_high_global_x1");
  RUN(e1m6, context, full_high_lut, 4, global_read_only,
      "full_high_global_x4");
  RUN(e1m6, context, full_high_lut, 8, global_read_only,
      "full_high_global_x8");
  RUN(e1m6, context, full_high_lut, 4, shared, "full_high_shared_x4");
  RUN(e1m6, context, full_high_lut, 8, shared, "full_high_shared_x8");
  RUN(e1m6, context, subnormal_high_lut, 4, global_read_only,
      "subnormal_global_x4");
  RUN(e1m6, context, subnormal_high_lut, 8, shared,
      "subnormal_shared_x8");
  RUN(e1m6, context, pair_high_lut, 4, global_read_only, "pair_l2_x4");
  RUN(e1m6, context, pair_high_lut, 8, global_read_only, "pair_l2_x8");
}

void run_fp8_e4m3_suite(std::ofstream *csv) {
  std::cout << "FP8 E4M3 exhaustive strategy validation\n";
  smoke_context<storage::fp8_e4m3> context;
  RUN(fp8_e4m3, context, generic, 1, global_read_only, "generic_x1");
  RUN(fp8_e4m3, context, native_direct, 1, global_read_only,
      "native_direct_x1");
  RUN(fp8_e4m3, context, native_fp32, 1, global_read_only,
      "native_fp32_x1");
  RUN(fp8_e4m3, context, native_packed, 2, global_read_only,
      "native_float2_x2");
  RUN(fp8_e4m3, context, native_packed, 4, global_read_only,
      "native_float4_x4");
  RUN(fp8_e4m3, context, native_packed, 8, global_read_only,
      "native_float4_x8");
  RUN(fp8_e4m3, context, native_half2, 2, global_read_only,
      "native_half2_x2");
  RUN(fp8_e4m3, context, native_half2, 4, global_read_only,
      "native_half2_x4");
  RUN(fp8_e4m3, context, native_half2, 8, global_read_only,
      "native_half2_x8");
  RUN(fp8_e4m3, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(fp8_e4m3, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(fp8_e4m3, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(fp8_e4m3, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(fp8_e4m3, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(fp8_e4m3, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(fp8_e4m3, context, fp32_bits, 1, global_read_only, "fp32_bits_x1");
  RUN(fp8_e4m3, context, fp32_bits, 4, global_read_only, "fp32_bits_x4");
  RUN(fp8_e4m3, context, fp32_bits, 8, global_read_only, "fp32_bits_x8");
  RUN(fp8_e4m3, context, full_high_lut, 1, global_read_only,
      "full_high_global_x1");
  RUN(fp8_e4m3, context, full_high_lut, 4, global_read_only,
      "full_high_global_x4");
  RUN(fp8_e4m3, context, full_high_lut, 8, global_read_only,
      "full_high_global_x8");
  RUN(fp8_e4m3, context, full_high_lut, 4, shared,
      "full_high_shared_x4");
  RUN(fp8_e4m3, context, full_high_lut, 8, shared,
      "full_high_shared_x8");
  RUN(fp8_e4m3, context, subnormal_high_lut, 8, shared,
      "subnormal_shared_x8");
  RUN(fp8_e4m3, context, pair_high_lut, 4, global_read_only, "pair_l2_x4");
  RUN(fp8_e4m3, context, pair_high_lut, 8, global_read_only, "pair_l2_x8");
}

void run_fp8_e5m2_suite(std::ofstream *csv) {
  std::cout << "FP8 E5M2 exhaustive strategy validation\n";
  smoke_context<storage::fp8_e5m2> context;
  RUN(fp8_e5m2, context, generic, 1, global_read_only, "generic_x1");
  RUN(fp8_e5m2, context, native_direct, 1, global_read_only,
      "native_direct_x1");
  RUN(fp8_e5m2, context, native_fp32, 1, global_read_only,
      "native_fp32_x1");
  RUN(fp8_e5m2, context, native_packed, 2, global_read_only,
      "native_float2_x2");
  RUN(fp8_e5m2, context, native_packed, 4, global_read_only,
      "native_float4_x4");
  RUN(fp8_e5m2, context, native_packed, 8, global_read_only,
      "native_float4_x8");
  RUN(fp8_e5m2, context, native_half2, 2, global_read_only,
      "native_half2_x2");
  RUN(fp8_e5m2, context, native_half2, 4, global_read_only,
      "native_half2_x4");
  RUN(fp8_e5m2, context, native_half2, 8, global_read_only,
      "native_half2_x8");
  RUN(fp8_e5m2, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(fp8_e5m2, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(fp8_e5m2, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(fp8_e5m2, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(fp8_e5m2, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(fp8_e5m2, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(fp8_e5m2, context, fp32_bits, 1, global_read_only, "fp32_bits_x1");
  RUN(fp8_e5m2, context, fp32_bits, 4, global_read_only, "fp32_bits_x4");
  RUN(fp8_e5m2, context, fp32_bits, 8, global_read_only, "fp32_bits_x8");
  RUN(fp8_e5m2, context, full_high_lut, 1, global_read_only,
      "full_high_global_x1");
  RUN(fp8_e5m2, context, full_high_lut, 4, global_read_only,
      "full_high_global_x4");
  RUN(fp8_e5m2, context, full_high_lut, 8, global_read_only,
      "full_high_global_x8");
  RUN(fp8_e5m2, context, full_high_lut, 4, shared,
      "full_high_shared_x4");
  RUN(fp8_e5m2, context, full_high_lut, 8, shared,
      "full_high_shared_x8");
  RUN(fp8_e5m2, context, subnormal_high_lut, 8, shared,
      "subnormal_shared_x8");
  RUN(fp8_e5m2, context, pair_high_lut, 4, global_read_only, "pair_l2_x4");
  RUN(fp8_e5m2, context, pair_high_lut, 8, global_read_only, "pair_l2_x8");
}

void run_e1m14_suite(std::ofstream *csv) {
  std::cout << "E1M14 exhaustive strategy validation\n";
  smoke_context<storage::e1m14> context;
  RUN(e1m14, context, generic, 1, global_read_only, "generic_x1");
  RUN(e1m14, context, generic, 4, global_read_only, "generic_x4");
  RUN(e1m14, context, generic, 8, global_read_only, "generic_x8");
  RUN(e1m14, context, e1_integer, 1, global_read_only, "e1_integer_x1");
  RUN(e1m14, context, e1_integer, 4, global_read_only, "e1_integer_x4");
  RUN(e1m14, context, e1_integer, 8, global_read_only, "e1_integer_x8");
  RUN(e1m14, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(e1m14, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(e1m14, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(e1m14, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(e1m14, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(e1m14, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(e1m14, context, fp32_bits, 1, global_read_only, "fp32_bits_x1");
  RUN(e1m14, context, fp32_bits, 4, global_read_only, "fp32_bits_x4");
  RUN(e1m14, context, fp32_bits, 8, global_read_only, "fp32_bits_x8");
  RUN(e1m14, context, full_high_lut, 1, global_read_only,
      "full_high_l2_x1");
  RUN(e1m14, context, full_high_lut, 4, global_read_only,
      "full_high_l2_x4");
  RUN(e1m14, context, full_high_lut, 8, global_read_only,
      "full_high_l2_x8");
  RUN(e1m14, context, subnormal_high_lut, 4, global_read_only,
      "subnormal_global_x4");
  RUN(e1m14, context, subnormal_high_lut, 8, global_read_only,
      "subnormal_global_x8");
  RUN(e1m14, context, subnormal_high_lut, 4, shared,
      "subnormal_shared_x4");
  RUN(e1m14, context, subnormal_high_lut, 8, shared,
      "subnormal_shared_x8");
}

void run_e2m13_suite(std::ofstream *csv) {
  std::cout << "E2M13 exhaustive strategy validation\n";
  smoke_context<storage::e2m13> context;
  RUN(e2m13, context, generic, 1, global_read_only, "generic_x1");
  RUN(e2m13, context, generic, 4, global_read_only, "generic_x4");
  RUN(e2m13, context, generic, 8, global_read_only, "generic_x8");
  RUN(e2m13, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(e2m13, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(e2m13, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(e2m13, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(e2m13, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(e2m13, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(e2m13, context, fp32_bits, 1, global_read_only, "fp32_bits_x1");
  RUN(e2m13, context, fp32_bits, 4, global_read_only, "fp32_bits_x4");
  RUN(e2m13, context, fp32_bits, 8, global_read_only, "fp32_bits_x8");
  RUN(e2m13, context, full_high_lut, 1, global_read_only,
      "full_high_l2_x1");
  RUN(e2m13, context, full_high_lut, 4, global_read_only,
      "full_high_l2_x4");
  RUN(e2m13, context, full_high_lut, 8, global_read_only,
      "full_high_l2_x8");
  RUN(e2m13, context, subnormal_high_lut, 4, global_read_only,
      "subnormal_global_x4");
  RUN(e2m13, context, subnormal_high_lut, 8, global_read_only,
      "subnormal_global_x8");
  RUN(e2m13, context, subnormal_high_lut, 4, shared,
      "subnormal_shared_x4");
  RUN(e2m13, context, subnormal_high_lut, 8, shared,
      "subnormal_shared_x8");
  RUN(e2m13, context, prefix_high_lut, 4, global_read_only,
      "prefix_global_x4");
  RUN(e2m13, context, prefix_high_lut, 8, global_read_only,
      "prefix_global_x8");
  RUN(e2m13, context, prefix_high_lut, 4, shared, "prefix_shared_x4");
  RUN(e2m13, context, prefix_high_lut, 8, shared, "prefix_shared_x8");
}

void run_e3m12_suite(std::ofstream *csv) {
  std::cout << "E3M12 exhaustive strategy validation\n";
  smoke_context<storage::e3m12> context;
  RUN(e3m12, context, generic, 1, global_read_only, "generic_x1");
  RUN(e3m12, context, generic, 4, global_read_only, "generic_x4");
  RUN(e3m12, context, generic, 8, global_read_only, "generic_x8");
  RUN(e3m12, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(e3m12, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(e3m12, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(e3m12, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(e3m12, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(e3m12, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(e3m12, context, fp32_bits, 1, global_read_only, "fp32_bits_x1");
  RUN(e3m12, context, fp32_bits, 4, global_read_only, "fp32_bits_x4");
  RUN(e3m12, context, fp32_bits, 8, global_read_only, "fp32_bits_x8");
  RUN(e3m12, context, full_high_lut, 1, global_read_only,
      "full_high_l2_x1");
  RUN(e3m12, context, full_high_lut, 4, global_read_only,
      "full_high_l2_x4");
  RUN(e3m12, context, full_high_lut, 8, global_read_only,
      "full_high_l2_x8");
  RUN(e3m12, context, subnormal_high_lut, 4, global_read_only,
      "subnormal_global_x4");
  RUN(e3m12, context, subnormal_high_lut, 8, global_read_only,
      "subnormal_global_x8");
  RUN(e3m12, context, subnormal_high_lut, 4, shared,
      "subnormal_shared_x4");
  RUN(e3m12, context, subnormal_high_lut, 8, shared,
      "subnormal_shared_x8");
  RUN(e3m12, context, prefix_high_lut, 4, global_read_only,
      "prefix_global_x4");
  RUN(e3m12, context, prefix_high_lut, 8, global_read_only,
      "prefix_global_x8");
  RUN(e3m12, context, prefix_high_lut, 4, shared, "prefix_shared_x4");
  RUN(e3m12, context, prefix_high_lut, 8, shared, "prefix_shared_x8");
}

void run_fp16_suite(std::ofstream *csv) {
  std::cout << "FP16 E5M10 exhaustive strategy validation\n";
  smoke_context<storage::fp16_e5m10> context;
  RUN(fp16_e5m10, context, generic, 1, global_read_only, "generic_x1");
  RUN(fp16_e5m10, context, native_direct, 1, global_read_only,
      "native_direct_x1");
  RUN(fp16_e5m10, context, native_fp32, 1, global_read_only,
      "native_fp32_x1");
  RUN(fp16_e5m10, context, native_packed, 2, global_read_only,
      "native_half2_x2");
  RUN(fp16_e5m10, context, native_packed, 4, global_read_only,
      "native_half2_x4");
  RUN(fp16_e5m10, context, native_packed, 8, global_read_only,
      "native_half2_x8");
  RUN(fp16_e5m10, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(fp16_e5m10, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(fp16_e5m10, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(fp16_e5m10, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(fp16_e5m10, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(fp16_e5m10, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(fp16_e5m10, context, fp32_bits, 1, global_read_only,
      "fp32_bits_x1");
  RUN(fp16_e5m10, context, fp32_bits, 4, global_read_only,
      "fp32_bits_x4");
  RUN(fp16_e5m10, context, fp32_bits, 8, global_read_only,
      "fp32_bits_x8");
  RUN(fp16_e5m10, context, full_high_lut, 1, global_read_only,
      "full_high_l2_x1");
  RUN(fp16_e5m10, context, full_high_lut, 4, global_read_only,
      "full_high_l2_x4");
  RUN(fp16_e5m10, context, full_high_lut, 8, global_read_only,
      "full_high_l2_x8");
  RUN(fp16_e5m10, context, subnormal_high_lut, 8, global_read_only,
      "subnormal_global_x8");
  RUN(fp16_e5m10, context, subnormal_high_lut, 8, shared,
      "subnormal_shared_x8");
}

#undef RUN

} // namespace

int main(int argc, char **argv) {
  try {
    std::string output;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--output" && i + 1 < argc) {
        output = argv[++i];
      } else {
        throw std::runtime_error(
            "usage: all_format_strategy_smoke [--output FILE]");
      }
    }
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, 0),
               "cudaGetDeviceProperties");
    std::cout << "All-format decoder strategy smoke test\nGPU: "
              << properties.name << " (sm_" << properties.major
              << properties.minor << ")\n";

    std::ofstream csv;
    if (!output.empty()) {
      csv.open(output);
      if (!csv) {
        throw std::runtime_error("cannot open " + output);
      }
      csv << "format,strategy,lanes,mismatches\n";
    }
    auto *csv_ptr = output.empty() ? nullptr : &csv;
    run_e1m6_suite(csv_ptr);
    run_fp8_e4m3_suite(csv_ptr);
    run_fp8_e5m2_suite(csv_ptr);
    run_e1m14_suite(csv_ptr);
    run_e2m13_suite(csv_ptr);
    run_e3m12_suite(csv_ptr);
    run_fp16_suite(csv_ptr);
    std::cout << "All registered strategies passed.\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
