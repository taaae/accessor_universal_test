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
namespace decoder = aut::decoder;

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

template <>
storage::storage_type_t<storage::bf16_e8m7>
host_storage_from_raw<storage::bf16_e8m7>(std::uint32_t raw) {
  return __nv_bfloat16{
      __nv_bfloat16_raw{static_cast<unsigned short>(raw)}};
}

template <>
storage::storage_type_t<storage::fp32_e8m23>
host_storage_from_raw<storage::fp32_e8m23>(std::uint32_t raw) {
  return decoder::bits_to_float(raw);
}

template <typename Format> struct smoke_context {
  using storage_type = storage::storage_type_t<Format>;
  using device_storage_type = fs::device_storage_t<Format>;
  using layout = fs::format_layout_t<Format>;
  static constexpr auto code_count = std::size_t{1} << layout::total_bits;
  static constexpr auto validation_count =
      layout::total_bits < 8 ? std::size_t{256} : code_count;
  static constexpr auto subnormal_count =
      std::size_t{1} << layout::fraction_bits;
  static constexpr auto pair_count = layout::total_bits <= 8
                                         ? (std::size_t{1}
                                            << (2 * layout::total_bits))
                                         : std::size_t{0};
  static constexpr auto quad_count =
      layout::total_bits == 2 ? std::size_t{256} : std::size_t{0};
  static constexpr auto prefix_count =
      std::size_t{1} << (layout::exponent_bits + 1);

  std::vector<storage_type> codes;
  std::vector<device_storage_type> packed_codes;
  std::vector<double> expected;
  std::vector<std::uint32_t> full_high;
  std::vector<std::uint32_t> subnormal_high;
  std::vector<std::uint32_t> prefix_high;
  std::vector<uint2> pair_high;
  std::vector<uint4> quad_high;
  device_buffer<device_storage_type> device_codes;
  device_buffer<double> device_output;
  device_buffer<std::uint32_t> device_full_high;
  device_buffer<std::uint32_t> device_subnormal_high;
  device_buffer<std::uint32_t> device_prefix_high;
  device_buffer<uint2> device_pair_high;
  device_buffer<uint4> device_quad_high;

  smoke_context()
      : codes(validation_count),
        packed_codes(fs::packed_storage_count<Format>(validation_count)),
        expected(validation_count), full_high(code_count),
        subnormal_high(subnormal_count), prefix_high(prefix_count),
        pair_high(pair_count), quad_high(quad_count),
        device_codes(packed_codes.size()), device_output(codes.size()),
        device_full_high(full_high.size()),
        device_subnormal_high(subnormal_high.size()),
        device_prefix_high(prefix_high.size()),
        device_pair_high(pair_high.size()),
        device_quad_high(quad_high.size()) {
    static_assert(layout::total_bits == 2 || layout::total_bits == 4 ||
                      layout::total_bits == 8 || layout::total_bits == 16,
                  "the exhaustive smoke context covers 2/4/8/16-bit codes");
    for (std::size_t index = 0; index < codes.size(); ++index) {
      const auto raw = static_cast<std::uint32_t>(
          layout::total_bits < 8 && index < 64 ? 0 : index % code_count);
      codes[index] = host_storage_from_raw<Format>(raw);
      expected[index] = storage::decode<Format>(codes[index]);
      if constexpr (layout::total_bits < 8) {
        const auto bit_offset = index * layout::total_bits;
        packed_codes[bit_offset / 8] |= static_cast<device_storage_type>(
            raw << (bit_offset % 8));
      } else {
        packed_codes[index] = codes[index];
      }
    }
    for (std::uint32_t raw = 0; raw < code_count; ++raw) {
      const auto value =
          storage::decode<Format>(host_storage_from_raw<Format>(raw));
      full_high[raw] = static_cast<std::uint32_t>(bits(value) >> 32);
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
    if constexpr (layout::total_bits <= 8) {
      for (std::uint32_t high = 0; high < code_count; ++high) {
        for (std::uint32_t low = 0; low < code_count; ++low) {
          pair_high[low + (high << layout::total_bits)] = {
              full_high[low], full_high[high]};
        }
      }
    }
    if constexpr (layout::total_bits == 2) {
      for (std::uint32_t packed = 0; packed < 256; ++packed) {
        quad_high[packed] = {full_high[packed & 3u],
                             full_high[(packed >> 2) & 3u],
                             full_high[(packed >> 4) & 3u],
                             full_high[(packed >> 6) & 3u]};
      }
    }
    check_cuda(cudaMemcpy(device_codes.data, packed_codes.data(),
                          packed_codes.size() * sizeof(packed_codes[0]),
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
    if constexpr (layout::total_bits <= 8) {
      check_cuda(cudaMemcpy(device_pair_high.data, pair_high.data(),
                            pair_high.size() * sizeof(pair_high[0]),
                            cudaMemcpyHostToDevice),
                 "copy pair table");
    }
    if constexpr (layout::total_bits == 2) {
      check_cuda(cudaMemcpy(device_quad_high.data, quad_high.data(),
                            quad_high.size() * sizeof(quad_high[0]),
                            cudaMemcpyHostToDevice),
                 "copy quad table");
    }
  }

  fs::table_bundle tables() const {
    return {device_full_high.data, device_subnormal_high.data,
            device_prefix_high.data,
            device_pair_high.data, device_quad_high.data};
  }
};

template <typename Format> struct sampled_smoke_context {
  using storage_type = storage::storage_type_t<Format>;
  using layout = fs::format_layout_t<Format>;
  static constexpr std::size_t sample_count = 65536;
  static constexpr auto prefix_count =
      std::size_t{1} << (layout::exponent_bits + 1);

  std::vector<storage_type> codes;
  std::vector<double> expected;
  std::vector<std::uint32_t> prefix_high;
  device_buffer<storage_type> device_codes;
  device_buffer<double> device_output;
  device_buffer<std::uint32_t> device_prefix_high;

  sampled_smoke_context()
      : codes(sample_count), expected(sample_count), prefix_high(prefix_count),
        device_codes(codes.size()), device_output(codes.size()),
        device_prefix_high(prefix_high.size()) {
    static_assert(layout::total_bits == 32);
    const std::uint32_t edges[] = {
        0u,          1u,          decoder::fraction_mask<layout>(),
        1u << layout::fraction_bits,
        decoder::raw_mask<layout>() >> 1,
        1u << 31,    0xffffffffu, 0x7fffffffu,
    };
    std::size_t index{};
    for (std::uint32_t raw = 0; raw < 64; ++raw) {
      codes[index++] = host_storage_from_raw<Format>(raw);
    }
    for (const auto raw : edges) {
      codes[index++] = host_storage_from_raw<Format>(raw);
    }
    std::uint32_t raw = 0x9e3779b9u;
    while (index < codes.size()) {
      raw = raw * 1664525u + 1013904223u;
      codes[index++] = host_storage_from_raw<Format>(raw);
    }
    for (std::size_t i = 0; i < codes.size(); ++i) {
      expected[i] = storage::decode<Format>(codes[i]);
    }
    for (std::uint32_t prefix = 0; prefix < prefix_count; ++prefix) {
      const auto prefix_raw = prefix << layout::fraction_bits;
      const auto value =
          storage::decode<Format>(host_storage_from_raw<Format>(prefix_raw));
      prefix_high[prefix] = static_cast<std::uint32_t>(bits(value) >> 32);
    }
    check_cuda(cudaMemcpy(device_codes.data, codes.data(),
                          codes.size() * sizeof(codes[0]),
                          cudaMemcpyHostToDevice),
               "copy sampled codes");
    check_cuda(cudaMemcpy(device_prefix_high.data, prefix_high.data(),
                          prefix_high.size() * sizeof(prefix_high[0]),
                          cudaMemcpyHostToDevice),
               "copy sampled prefix table");
  }

  fs::table_bundle tables() const {
    return {nullptr, nullptr, device_prefix_high.data, nullptr, nullptr};
  }
};

template <typename Format, typename Strategy, typename Context>
void run_strategy(Context &context, const char *format_name,
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
    check_cuda(cudaFuncSetAttribute(
                   fs::dot_map_reduce<Format, Strategy>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(shared_bytes)),
               "set DOT dynamic shared-memory size");
    check_cuda(cudaFuncSetAttribute(
                   fs::gemv<Format, Strategy>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(shared_bytes)),
               "set GEMV dynamic shared-memory size");
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


  constexpr std::size_t kernel_count = 64;
  fs::dot_map_reduce<Format, Strategy><<<1, 256, shared_bytes>>>(
      context.device_codes.data, context.device_codes.data, kernel_count,
      context.tables(), context.device_output.data);
  check_cuda(cudaGetLastError(), "DOT strategy smoke launch");
  double dot_result{};
  check_cuda(cudaMemcpy(&dot_result, context.device_output.data, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "copy DOT smoke result");
  if (!std::isfinite(dot_result)) {
    throw std::runtime_error(std::string(format_name) + '/' + strategy_name +
                             " produced non-finite DOT smoke result");
  }

  fs::gemv<Format, Strategy><<<1, 256, shared_bytes>>>(
      context.device_codes.data, context.device_codes.data, 1, kernel_count,
      kernel_count, context.tables(), context.device_output.data);
  check_cuda(cudaGetLastError(), "GEMV strategy smoke launch");
  double gemv_result{};
  check_cuda(cudaMemcpy(&gemv_result, context.device_output.data,
                        sizeof(double), cudaMemcpyDeviceToHost),
             "copy GEMV smoke result");
  if (!std::isfinite(gemv_result)) {
    throw std::runtime_error(std::string(format_name) + '/' + strategy_name +
                             " produced non-finite GEMV smoke result");
  }
  std::cout << "  " << strategy_name << " passed\n";
}

template <fs::decode_kind Kind, int Lanes,
          fs::table_location Location = fs::table_location::global_read_only>
using s = fs::strategy<Kind, Lanes, Location>;

template <fs::decode_kind Kind, int Lanes, fs::table_location Location,
          fs::unpack_kind Unpack>
using sp = fs::strategy<Kind, Lanes, Location, Unpack>;

#define RUN(format, context, kind, lanes, location, name)                     \
  run_strategy<storage::format,                                               \
               s<fs::decode_kind::kind, lanes, fs::table_location::location>>( \
      context, #format, name, csv)
#define RUNP(format, context, kind, lanes, location, unpack, name)             \
  run_strategy<storage::format,                                                \
               sp<fs::decode_kind::kind, lanes, fs::table_location::location, \
                  fs::unpack_kind::unpack>>(context, #format, name, csv)

template <typename Format, fs::decode_kind Kind,
          fs::table_location Location = fs::table_location::global_read_only,
          typename Context>
void run_all_widths(Context &context, const char *base, std::ofstream *csv) {
  const auto run = [&](auto width) {
    constexpr auto lanes = decltype(width)::value;
    const auto name = std::string(base) + "_x" + std::to_string(lanes);
    run_strategy<Format, s<Kind, lanes, Location>>(
        context, Format::name, name.c_str(), csv);
  };
  run(std::integral_constant<int, 1>{});
  run(std::integral_constant<int, 2>{});
  run(std::integral_constant<int, 4>{});
  run(std::integral_constant<int, 8>{});
}

template <typename Format, fs::decode_kind Kind,
          fs::table_location Location = fs::table_location::global_read_only,
          typename Context>
void run_packed_widths(Context &context, const char *base,
                       std::ofstream *csv) {
  const auto run = [&](auto width) {
    constexpr auto lanes = decltype(width)::value;
    const auto name = std::string(base) + "_x" + std::to_string(lanes);
    run_strategy<Format, s<Kind, lanes, Location>>(
        context, Format::name, name.c_str(), csv);
  };
  run(std::integral_constant<int, 2>{});
  run(std::integral_constant<int, 4>{});
  run(std::integral_constant<int, 8>{});
}

template <typename Format>
void run_subbyte_suite(std::ofstream *csv) {
  std::cout << Format::name << " dense sub-byte strategy validation\n";
  smoke_context<Format> context;
  run_all_widths<Format, fs::decode_kind::generic>(context, "generic", csv);
  run_all_widths<Format, fs::decode_kind::direct_words_branchy>(
      context, "word_branchy", csv);
  run_all_widths<Format, fs::decode_kind::direct_words_masked>(
      context, "word_masked", csv);
  run_all_widths<Format, fs::decode_kind::fp32_bits>(context, "fp32_bits",
                                                     csv);
  if constexpr (Format::exponent_bits == 0) {
    run_all_widths<Format, fs::decode_kind::fixed_integer>(
        context, "fixed_integer", csv);
  }
  if constexpr (Format::exponent_bits == 1 && Format::finite) {
    run_all_widths<Format, fs::decode_kind::e1_integer>(
        context, "e1_integer", csv);
  }
  if constexpr (Format::fraction_bits == 0) {
    run_all_widths<Format, fs::decode_kind::exponent_only>(
        context, "exponent_only", csv);
  }
  run_all_widths<Format, fs::decode_kind::full_high_lut>(
      context, "full_high_global", csv);
  run_all_widths<Format, fs::decode_kind::full_high_lut,
                 fs::table_location::shared>(context, "full_high_shared",
                                             csv);
  run_all_widths<Format, fs::decode_kind::warp_high_lut>(
      context, "full_high_warp", csv);
  run_packed_widths<Format, fs::decode_kind::pair_high_lut>(
      context, "pair_l2", csv);
  run_packed_widths<Format, fs::decode_kind::pair_high_lut,
                    fs::table_location::shared>(context, "pair_shared", csv);
  if constexpr (Format::total_bits == 2) {
    run_strategy<Format, s<fs::decode_kind::quad_high_lut, 4>>(
        context, Format::name, "byte_quad_l2_x4", csv);
    run_strategy<Format, s<fs::decode_kind::quad_high_lut, 8>>(
        context, Format::name, "byte_quad_l2_x8", csv);
    run_strategy<Format,
                 s<fs::decode_kind::quad_high_lut, 4,
                   fs::table_location::shared>>(
        context, Format::name, "byte_quad_shared_x4", csv);
    run_strategy<Format,
                 s<fs::decode_kind::quad_high_lut, 8,
                   fs::table_location::shared>>(
        context, Format::name, "byte_quad_shared_x8", csv);
  }
  if constexpr (std::is_same_v<Format, storage::fp4_e2m1>) {
    run_strategy<Format, s<fs::decode_kind::native_direct, 1>>(
        context, Format::name, "native_half_x1", csv);
    run_packed_widths<Format, fs::decode_kind::native_packed>(
        context, "native_half2", csv);
  }
}

template <typename Format>
void run_added_8bit_suite(std::ofstream *csv) {
  std::cout << Format::name << " exhaustive strategy validation\n";
  smoke_context<Format> context;
  run_all_widths<Format, fs::decode_kind::generic>(context, "generic", csv);
  run_all_widths<Format, fs::decode_kind::direct_words_branchy>(
      context, "word_branchy", csv);
  run_all_widths<Format, fs::decode_kind::direct_words_masked>(
      context, "word_masked", csv);
  run_all_widths<Format, fs::decode_kind::fp32_bits>(context, "fp32_bits",
                                                     csv);
  if constexpr (Format::exponent_bits == 0) {
    run_all_widths<Format, fs::decode_kind::fixed_integer>(
        context, "fixed_integer", csv);
  }
  if constexpr (Format::fraction_bits == 0) {
    run_all_widths<Format, fs::decode_kind::exponent_only>(
        context, "exponent_only", csv);
  }
  run_all_widths<Format, fs::decode_kind::full_high_lut>(
      context, "full_high_global", csv);
  run_all_widths<Format, fs::decode_kind::full_high_lut,
                 fs::table_location::shared>(context, "full_high_shared",
                                             csv);
  run_packed_widths<Format, fs::decode_kind::pair_high_lut>(
      context, "pair_l2", csv);
  if constexpr (Format::exponent_bits > 0 && Format::fraction_bits > 0) {
    run_all_widths<Format, fs::decode_kind::subnormal_high_lut>(
        context, "subnormal_global", csv);
    run_all_widths<Format, fs::decode_kind::subnormal_high_lut,
                   fs::table_location::shared>(context, "subnormal_shared",
                                               csv);
    run_all_widths<Format, fs::decode_kind::prefix_high_lut>(
        context, "prefix_global", csv);
    run_all_widths<Format, fs::decode_kind::prefix_high_lut,
                   fs::table_location::shared>(context, "prefix_shared",
                                               csv);
  }
  run_strategy<Format,
               s<fs::decode_kind::full_high_lut_swizzled, 4,
                 fs::table_location::shared>>(
      context, Format::name, "full_high_swizzled_x4", csv);
  run_strategy<Format,
               s<fs::decode_kind::full_high_lut_swizzled, 8,
                 fs::table_location::shared>>(
      context, Format::name, "full_high_swizzled_x8", csv);
}

template <typename Format>
void run_added_16bit_suite(std::ofstream *csv) {
  std::cout << Format::name << " exhaustive strategy validation\n";
  smoke_context<Format> context;
  run_all_widths<Format, fs::decode_kind::generic>(context, "generic", csv);
  run_all_widths<Format, fs::decode_kind::direct_words_branchy>(
      context, "word_branchy", csv);
  run_all_widths<Format, fs::decode_kind::direct_words_masked>(
      context, "word_masked", csv);
  if constexpr (Format::exponent_bits <= 8) {
    run_all_widths<Format, fs::decode_kind::fp32_bits>(context, "fp32_bits",
                                                       csv);
  }
  if constexpr (Format::exponent_bits == 0) {
    run_all_widths<Format, fs::decode_kind::fixed_integer>(
        context, "fixed_integer", csv);
  }
  run_all_widths<Format, fs::decode_kind::full_high_lut>(
      context, "full_high_l2", csv);
  if constexpr (Format::exponent_bits > 0) {
    run_all_widths<Format, fs::decode_kind::subnormal_high_lut>(
        context, "subnormal_global", csv);
    run_all_widths<Format, fs::decode_kind::subnormal_high_lut,
                   fs::table_location::shared>(context, "subnormal_shared",
                                               csv);
    run_all_widths<Format, fs::decode_kind::prefix_high_lut>(
        context, "prefix_global", csv);
    run_all_widths<Format, fs::decode_kind::prefix_high_lut,
                   fs::table_location::shared>(context, "prefix_shared",
                                               csv);
  }
}

template <typename Format>
void run_added_32bit_suite(std::ofstream *csv) {
  std::cout << Format::name << " sampled strategy validation\n";
  sampled_smoke_context<Format> context;
  run_all_widths<Format, fs::decode_kind::generic>(context, "generic", csv);
  run_all_widths<Format, fs::decode_kind::direct_words_branchy>(
      context, "word_branchy", csv);
  run_all_widths<Format, fs::decode_kind::direct_words_masked>(
      context, "word_masked", csv);
  if constexpr (Format::exponent_bits == 0) {
    run_all_widths<Format, fs::decode_kind::fixed_integer>(
        context, "fixed_integer", csv);
  } else {
    run_all_widths<Format, fs::decode_kind::prefix_high_lut>(
        context, "prefix_global", csv);
    run_all_widths<Format, fs::decode_kind::prefix_high_lut,
                   fs::table_location::shared>(context, "prefix_shared",
                                               csv);
  }
}

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
  RUNP(e1m6, context, direct_words_branchy, 4, global_read_only,
       byte_permute, "word_branchy_prmt_x4");
  RUNP(e1m6, context, direct_words_branchy, 8, global_read_only,
       byte_permute, "word_branchy_prmt_x8");
  RUN(e1m6, context, full_high_lut_swizzled, 4, shared,
      "full_high_swizzled_x4");
  RUN(e1m6, context, full_high_lut_swizzled, 8, shared,
      "full_high_swizzled_x8");
}

void run_e2m5_additions(std::ofstream *csv) {
  std::cout << "E2M5 additional packet strategy validation\n";
  smoke_context<storage::e2m5> context;
  RUN(e2m5, context, pair_high_lut, 4, global_read_only, "pair_l2_x4");
  RUN(e2m5, context, pair_high_lut, 8, global_read_only, "pair_l2_x8");
  RUN(e2m5, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_shift_x4");
  RUN(e2m5, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_shift_x8");
  RUNP(e2m5, context, direct_words_branchy, 4, global_read_only,
       byte_permute, "word_branchy_prmt_x4");
  RUNP(e2m5, context, direct_words_branchy, 8, global_read_only,
       byte_permute, "word_branchy_prmt_x8");
}

void run_e3m4_additions(std::ofstream *csv) {
  std::cout << "E3M4 additional packet strategy validation\n";
  smoke_context<storage::e3m4> context;
  RUN(e3m4, context, pair_high_lut, 4, global_read_only, "pair_l2_x4");
  RUN(e3m4, context, pair_high_lut, 8, global_read_only, "pair_l2_x8");
  RUN(e3m4, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_shift_x4");
  RUN(e3m4, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_shift_x8");
  RUNP(e3m4, context, direct_words_branchy, 4, global_read_only,
       byte_permute, "word_branchy_prmt_x4");
  RUNP(e3m4, context, direct_words_branchy, 8, global_read_only,
       byte_permute, "word_branchy_prmt_x8");
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
  RUNP(fp8_e4m3, context, direct_words_branchy, 4, global_read_only,
       byte_permute, "word_branchy_prmt_x4");
  RUNP(fp8_e4m3, context, direct_words_branchy, 8, global_read_only,
       byte_permute, "word_branchy_prmt_x8");
  RUN(fp8_e4m3, context, full_high_lut_swizzled, 4, shared,
      "full_high_swizzled_x4");
  RUN(fp8_e4m3, context, full_high_lut_swizzled, 8, shared,
      "full_high_swizzled_x8");
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
  RUNP(fp8_e5m2, context, direct_words_branchy, 4, global_read_only,
       byte_permute, "word_branchy_prmt_x4");
  RUNP(fp8_e5m2, context, direct_words_branchy, 8, global_read_only,
       byte_permute, "word_branchy_prmt_x8");
  RUN(fp8_e5m2, context, full_high_lut_swizzled, 4, shared,
      "full_high_swizzled_x4");
  RUN(fp8_e5m2, context, full_high_lut_swizzled, 8, shared,
      "full_high_swizzled_x8");
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

void run_bf16_suite(std::ofstream *csv) {
  std::cout << "BF16 E8M7 exhaustive strategy validation\n";
  smoke_context<storage::bf16_e8m7> context;
  RUN(bf16_e8m7, context, generic, 1, global_read_only, "generic_x1");
  RUN(bf16_e8m7, context, native_direct, 1, global_read_only,
      "native_direct_x1");
  RUN(bf16_e8m7, context, native_fp32, 1, global_read_only,
      "native_fp32_x1");
  RUN(bf16_e8m7, context, native_packed, 2, global_read_only,
      "native_bfloat162_x2");
  RUN(bf16_e8m7, context, native_packed, 4, global_read_only,
      "native_bfloat162_x4");
  RUN(bf16_e8m7, context, native_packed, 8, global_read_only,
      "native_bfloat162_x8");
  RUN(bf16_e8m7, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(bf16_e8m7, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(bf16_e8m7, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(bf16_e8m7, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(bf16_e8m7, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(bf16_e8m7, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(bf16_e8m7, context, fp32_bits, 1, global_read_only,
      "fp32_bit_lift_x1");
  RUN(bf16_e8m7, context, fp32_bits, 4, global_read_only,
      "fp32_bit_lift_x4");
  RUN(bf16_e8m7, context, fp32_bits, 8, global_read_only,
      "fp32_bit_lift_x8");
  RUN(bf16_e8m7, context, full_high_lut, 1, global_read_only,
      "full_high_l2_x1");
  RUN(bf16_e8m7, context, full_high_lut, 4, global_read_only,
      "full_high_l2_x4");
  RUN(bf16_e8m7, context, full_high_lut, 8, global_read_only,
      "full_high_l2_x8");
  RUN(bf16_e8m7, context, subnormal_high_lut, 8, global_read_only,
      "subnormal_global_x8");
  RUN(bf16_e8m7, context, subnormal_high_lut, 8, shared,
      "subnormal_shared_x8");
}

void run_e11m4_suite(std::ofstream *csv) {
  std::cout << "E11M4 exhaustive strategy validation\n";
  smoke_context<storage::e11m4> context;
  RUN(e11m4, context, generic, 1, global_read_only, "generic_x1");
  RUN(e11m4, context, generic, 2, global_read_only, "generic_x2");
  RUN(e11m4, context, generic, 4, global_read_only, "generic_x4");
  RUN(e11m4, context, generic, 8, global_read_only, "generic_x8");
  RUN(e11m4, context, prefix_word, 1, global_read_only,
      "prefix_word_x1");
  RUN(e11m4, context, prefix_word, 2, global_read_only,
      "prefix_word_x2");
  RUN(e11m4, context, prefix_word, 4, global_read_only,
      "prefix_word_x4");
  RUN(e11m4, context, prefix_word, 8, global_read_only,
      "prefix_word_x8");
  RUN(e11m4, context, full_high_lut, 1, global_read_only,
      "full_high_l2_x1");
  RUN(e11m4, context, full_high_lut, 4, global_read_only,
      "full_high_l2_x4");
  RUN(e11m4, context, full_high_lut, 8, global_read_only,
      "full_high_l2_x8");
}

void run_e1m30_suite(std::ofstream *csv) {
  std::cout << "E1M30 sampled strategy validation\n";
  sampled_smoke_context<storage::e1m30> context;
  RUN(e1m30, context, generic, 1, global_read_only, "generic_x1");
  RUN(e1m30, context, generic, 2, global_read_only, "generic_x2");
  RUN(e1m30, context, generic, 4, global_read_only, "generic_x4");
  RUN(e1m30, context, generic, 8, global_read_only, "generic_x8");
  RUN(e1m30, context, e1_integer, 1, global_read_only, "e1_integer_x1");
  RUN(e1m30, context, e1_integer, 2, global_read_only, "e1_integer_x2");
  RUN(e1m30, context, e1_integer, 4, global_read_only, "e1_integer_x4");
  RUN(e1m30, context, e1_integer, 8, global_read_only, "e1_integer_x8");
  RUN(e1m30, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(e1m30, context, direct_words_branchy, 2, global_read_only,
      "word_branchy_x2");
  RUN(e1m30, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(e1m30, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(e1m30, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(e1m30, context, direct_words_masked, 2, global_read_only,
      "word_masked_x2");
  RUN(e1m30, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(e1m30, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(e1m30, context, prefix_high_lut, 4, global_read_only,
      "prefix_global_x4");
  RUN(e1m30, context, prefix_high_lut, 8, global_read_only,
      "prefix_global_x8");
  RUN(e1m30, context, prefix_high_lut, 4, shared, "prefix_shared_x4");
  RUN(e1m30, context, prefix_high_lut, 8, shared, "prefix_shared_x8");
}

void run_e2m29_suite(std::ofstream *csv) {
  std::cout << "E2M29 sampled strategy validation\n";
  sampled_smoke_context<storage::e2m29> context;
  RUN(e2m29, context, generic, 1, global_read_only, "generic_x1");
  RUN(e2m29, context, generic, 2, global_read_only, "generic_x2");
  RUN(e2m29, context, generic, 4, global_read_only, "generic_x4");
  RUN(e2m29, context, generic, 8, global_read_only, "generic_x8");
  RUN(e2m29, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(e2m29, context, direct_words_branchy, 2, global_read_only,
      "word_branchy_x2");
  RUN(e2m29, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(e2m29, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(e2m29, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(e2m29, context, direct_words_masked, 2, global_read_only,
      "word_masked_x2");
  RUN(e2m29, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(e2m29, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(e2m29, context, prefix_high_lut, 4, global_read_only,
      "prefix_global_x4");
  RUN(e2m29, context, prefix_high_lut, 8, global_read_only,
      "prefix_global_x8");
  RUN(e2m29, context, prefix_high_lut, 4, shared, "prefix_shared_x4");
  RUN(e2m29, context, prefix_high_lut, 8, shared, "prefix_shared_x8");
}

void run_e3m28_suite(std::ofstream *csv) {
  std::cout << "E3M28 sampled strategy validation\n";
  sampled_smoke_context<storage::e3m28> context;
  RUN(e3m28, context, generic, 1, global_read_only, "generic_x1");
  RUN(e3m28, context, generic, 2, global_read_only, "generic_x2");
  RUN(e3m28, context, generic, 4, global_read_only, "generic_x4");
  RUN(e3m28, context, generic, 8, global_read_only, "generic_x8");
  RUN(e3m28, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(e3m28, context, direct_words_branchy, 2, global_read_only,
      "word_branchy_x2");
  RUN(e3m28, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(e3m28, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(e3m28, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(e3m28, context, direct_words_masked, 2, global_read_only,
      "word_masked_x2");
  RUN(e3m28, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(e3m28, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(e3m28, context, prefix_high_lut, 4, global_read_only,
      "prefix_global_x4");
  RUN(e3m28, context, prefix_high_lut, 8, global_read_only,
      "prefix_global_x8");
  RUN(e3m28, context, prefix_high_lut, 4, shared, "prefix_shared_x4");
  RUN(e3m28, context, prefix_high_lut, 8, shared, "prefix_shared_x8");
}

void run_fp32_suite(std::ofstream *csv) {
  std::cout << "FP32 E8M23 sampled strategy validation\n";
  sampled_smoke_context<storage::fp32_e8m23> context;
  RUN(fp32_e8m23, context, generic, 1, global_read_only, "generic_x1");
  RUN(fp32_e8m23, context, native_direct, 1, global_read_only,
      "native_f64_x1");
  RUN(fp32_e8m23, context, native_packed, 2, global_read_only,
      "native_float2_x2");
  RUN(fp32_e8m23, context, native_packed, 4, global_read_only,
      "native_float4_x4");
  RUN(fp32_e8m23, context, native_packed, 8, global_read_only,
      "native_float4_x8");
  RUN(fp32_e8m23, context, direct_words_branchy, 1, global_read_only,
      "word_branchy_x1");
  RUN(fp32_e8m23, context, direct_words_branchy, 2, global_read_only,
      "word_branchy_x2");
  RUN(fp32_e8m23, context, direct_words_branchy, 4, global_read_only,
      "word_branchy_x4");
  RUN(fp32_e8m23, context, direct_words_branchy, 8, global_read_only,
      "word_branchy_x8");
  RUN(fp32_e8m23, context, direct_words_masked, 1, global_read_only,
      "word_masked_x1");
  RUN(fp32_e8m23, context, direct_words_masked, 2, global_read_only,
      "word_masked_x2");
  RUN(fp32_e8m23, context, direct_words_masked, 4, global_read_only,
      "word_masked_x4");
  RUN(fp32_e8m23, context, direct_words_masked, 8, global_read_only,
      "word_masked_x8");
  RUN(fp32_e8m23, context, prefix_high_lut, 4, global_read_only,
      "prefix_global_x4");
  RUN(fp32_e8m23, context, prefix_high_lut, 8, global_read_only,
      "prefix_global_x8");
  RUN(fp32_e8m23, context, prefix_high_lut, 4, shared,
      "prefix_shared_x4");
  RUN(fp32_e8m23, context, prefix_high_lut, 8, shared,
      "prefix_shared_x8");
}

void run_e11m20_suite(std::ofstream *csv) {
  std::cout << "E11M20 sampled strategy validation\n";
  sampled_smoke_context<storage::e11m20> context;
  RUN(e11m20, context, generic, 1, global_read_only, "generic_x1");
  RUN(e11m20, context, generic, 2, global_read_only, "generic_x2");
  RUN(e11m20, context, generic, 4, global_read_only, "generic_x4");
  RUN(e11m20, context, generic, 8, global_read_only, "generic_x8");
  RUN(e11m20, context, prefix_word, 1, global_read_only,
      "prefix_word_x1");
  RUN(e11m20, context, prefix_word, 2, global_read_only,
      "prefix_word_x2");
  RUN(e11m20, context, prefix_word, 4, global_read_only,
      "prefix_word_x4");
  RUN(e11m20, context, prefix_word, 8, global_read_only,
      "prefix_word_x8");
}

#undef RUN
#undef RUNP

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
#if defined(AUT_EXPANDED_FORMAT_SMOKE)
#if AUT_EXPANDED_SMOKE_BITS == 2
    run_subbyte_suite<storage::e0m1>(csv_ptr);
    run_subbyte_suite<storage::e1m0>(csv_ptr);
#elif AUT_EXPANDED_SMOKE_BITS == 4
    run_subbyte_suite<storage::e0m3>(csv_ptr);
    run_subbyte_suite<storage::e1m2>(csv_ptr);
    run_subbyte_suite<storage::fp4_e2m1>(csv_ptr);
    run_subbyte_suite<storage::e3m0>(csv_ptr);
#elif AUT_EXPANDED_SMOKE_BITS == 8
    run_added_8bit_suite<storage::e0m7>(csv_ptr);
    run_added_8bit_suite<storage::e6m1>(csv_ptr);
    run_added_8bit_suite<storage::e7m0>(csv_ptr);
#elif AUT_EXPANDED_SMOKE_BITS == 16
    run_added_16bit_suite<storage::e0m15>(csv_ptr);
    run_added_16bit_suite<storage::e4m11>(csv_ptr);
    run_added_16bit_suite<storage::e6m9>(csv_ptr);
    run_added_16bit_suite<storage::e7m8>(csv_ptr);
    run_added_16bit_suite<storage::e9m6>(csv_ptr);
    run_added_16bit_suite<storage::e10m5>(csv_ptr);
#elif AUT_EXPANDED_SMOKE_BITS == 32
    run_added_32bit_suite<storage::e0m31>(csv_ptr);
    run_added_32bit_suite<storage::e4m27>(csv_ptr);
    run_added_32bit_suite<storage::e5m26>(csv_ptr);
    run_added_32bit_suite<storage::e6m25>(csv_ptr);
    run_added_32bit_suite<storage::e7m24>(csv_ptr);
    run_added_32bit_suite<storage::e9m22>(csv_ptr);
    run_added_32bit_suite<storage::e10m21>(csv_ptr);
#else
#error "AUT_EXPANDED_SMOKE_BITS must be 2, 4, 8, 16, or 32"
#endif
#else
    run_e1m6_suite(csv_ptr);
    run_e2m5_additions(csv_ptr);
    run_e3m4_additions(csv_ptr);
    run_fp8_e4m3_suite(csv_ptr);
    run_fp8_e5m2_suite(csv_ptr);
    run_e1m14_suite(csv_ptr);
    run_e2m13_suite(csv_ptr);
    run_e3m12_suite(csv_ptr);
    run_fp16_suite(csv_ptr);
    run_bf16_suite(csv_ptr);
    run_e11m4_suite(csv_ptr);
    run_e1m30_suite(csv_ptr);
    run_e2m29_suite(csv_ptr);
    run_e3m28_suite(csv_ptr);
    run_fp32_suite(csv_ptr);
    run_e11m20_suite(csv_ptr);
#endif
    std::cout << "All registered strategies passed.\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
