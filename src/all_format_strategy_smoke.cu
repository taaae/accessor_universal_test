#include "format_decoder_strategies.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
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
    check_cuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
  }
  ~device_buffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
};

struct smoke_context {
  std::vector<std::uint8_t> codes;
  std::vector<double> expected;
  std::vector<std::uint32_t> full_high;
  std::vector<std::uint32_t> subnormal_high;
  device_buffer<std::uint8_t> device_codes;
  device_buffer<double> device_output;
  device_buffer<std::uint32_t> device_full_high;
  device_buffer<std::uint32_t> device_subnormal_high;

  smoke_context()
      : codes(256), expected(256), full_high(256), subnormal_high(64),
        device_codes(codes.size()), device_output(codes.size()),
        device_full_high(full_high.size()),
        device_subnormal_high(subnormal_high.size()) {
    for (std::uint32_t raw = 0; raw < 256; ++raw) {
      codes[raw] = static_cast<std::uint8_t>(raw);
      expected[raw] = storage::decode<storage::e1m6>(codes[raw]);
      full_high[raw] = static_cast<std::uint32_t>(bits(expected[raw]) >> 32);
    }
    for (std::uint32_t fraction = 0; fraction < 64; ++fraction) {
      const auto value = storage::decode<storage::e1m6>(
          static_cast<std::uint8_t>(fraction));
      subnormal_high[fraction] =
          static_cast<std::uint32_t>(bits(value) >> 32);
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
  }
};

template <typename Strategy>
void run_e1m6(smoke_context &context, const char *name, std::ofstream *csv) {
  fs::table_bundle tables{context.device_full_high.data,
                          context.device_subnormal_high.data, nullptr};
  constexpr auto lanes = Strategy::lanes;
  const auto packs = (context.codes.size() + lanes - 1) / lanes;
  const auto blocks = static_cast<unsigned>((packs + 255) / 256);
  fs::decode_codes<storage::e1m6, Strategy>
      <<<blocks, 256, fs::shared_table_bytes_v<storage::e1m6, Strategy>>>>(
          context.device_codes.data, context.codes.size(), tables,
          context.device_output.data);
  check_cuda(cudaGetLastError(), name);
  check_cuda(cudaDeviceSynchronize(), "decode synchronization");

  std::vector<double> actual(context.expected.size());
  check_cuda(cudaMemcpy(actual.data(), context.device_output.data,
                        actual.size() * sizeof(actual[0]),
                        cudaMemcpyDeviceToHost),
             "copy decoder output");
  std::size_t mismatches{};
  for (std::size_t i = 0; i < actual.size(); ++i) {
    mismatches += bits(actual[i]) != bits(context.expected[i]);
  }
  if (csv != nullptr) {
    *csv << "e1m6," << name << ',' << lanes << ',' << mismatches << '\n';
  }
  if (mismatches != 0) {
    throw std::runtime_error(std::string(name) + " mismatches=" +
                             std::to_string(mismatches));
  }
  std::cout << "  " << name << " passed\n";
}

template <fs::decode_kind Kind, int Lanes,
          fs::table_location Location = fs::table_location::global_read_only>
using s = fs::strategy<Kind, Lanes, Location>;

void run_e1m6_suite(std::ofstream *csv) {
  std::cout << "E1M6 exhaustive strategy validation\n";
  smoke_context context;
#define RUN(kind, lanes, location, name)                                      \
  run_e1m6<s<fs::decode_kind::kind, lanes, fs::table_location::location>>(    \
      context, name, csv)
  RUN(generic, 1, global_read_only, "generic_x1");
  RUN(generic, 4, global_read_only, "generic_x4");
  RUN(generic, 8, global_read_only, "generic_x8");
  RUN(direct_words_branchy, 1, global_read_only, "word_branchy_x1");
  RUN(direct_words_branchy, 4, global_read_only, "word_branchy_x4");
  RUN(direct_words_branchy, 8, global_read_only, "word_branchy_x8");
  RUN(direct_words_masked, 1, global_read_only, "word_masked_x1");
  RUN(direct_words_masked, 4, global_read_only, "word_masked_x4");
  RUN(direct_words_masked, 8, global_read_only, "word_masked_x8");
  RUN(fp32_bits, 1, global_read_only, "fp32_bits_x1");
  RUN(fp32_bits, 4, global_read_only, "fp32_bits_x4");
  RUN(fp32_bits, 8, global_read_only, "fp32_bits_x8");
  RUN(e1_integer, 1, global_read_only, "e1_integer_x1");
  RUN(e1_integer, 4, global_read_only, "e1_integer_x4");
  RUN(e1_integer, 8, global_read_only, "e1_integer_x8");
  RUN(full_high_lut, 1, global_read_only, "full_high_global_x1");
  RUN(full_high_lut, 4, global_read_only, "full_high_global_x4");
  RUN(full_high_lut, 8, global_read_only, "full_high_global_x8");
  RUN(full_high_lut, 4, shared, "full_high_shared_x4");
  RUN(full_high_lut, 8, shared, "full_high_shared_x8");
  RUN(subnormal_high_lut, 4, global_read_only, "subnormal_global_x4");
  RUN(subnormal_high_lut, 8, shared, "subnormal_shared_x8");
#undef RUN
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string output;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--output" && i + 1 < argc) {
        output = argv[++i];
      } else {
        throw std::runtime_error("usage: all_format_strategy_smoke [--output FILE]");
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
    run_e1m6_suite(output.empty() ? nullptr : &csv);
    std::cout << "All registered strategies passed.\n";
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
