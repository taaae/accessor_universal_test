#include "e2e3_decoder_strategies.cuh"
#include "storage_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace decoder = aut::e2e3_strategies;
namespace kernels = aut::kernels;
namespace storage = aut::storage;

class smoke_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void check_cuda(cudaError_t status, const char *expression, const char *file,
                int line) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << ": "
            << cudaGetErrorName(status) << " (" << cudaGetErrorString(status)
            << ')';
    throw smoke_error(message.str());
  }
}

#define CUDA_CHECK(expression)                                                 \
  check_cuda((expression), #expression, __FILE__, __LINE__)

template <typename T> class device_buffer {
public:
  explicit device_buffer(std::size_t count) : count_{count} {
    if (count != 0) {
      CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }
  }
  ~device_buffer() {
    if (data_) {
      cudaFree(data_);
    }
  }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
  T *get() { return data_; }
  const T *get() const { return data_; }
  std::size_t size() const { return count_; }

private:
  T *data_{};
  std::size_t count_{};
};

class event_timer {
public:
  event_timer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~event_timer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  float measure(const std::function<void()> &launch, std::size_t iterations) {
    CUDA_CHECK(cudaEventRecord(start_));
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      launch();
    }
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float elapsed{};
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
    return elapsed;
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

enum class distribution { uniform_0_1, normal_0_1 };

const char *name(distribution value) {
  return value == distribution::uniform_0_1 ? "uniform_0_1" : "normal_0_1";
}

struct options {
  std::vector<int> dot_powers{16, 22};
  std::vector<int> gemv_powers{10, 12};
  std::size_t gemv_rows{256};
  int warmup{3};
  int samples{3};
  double target_sample_ms{2.0};
  std::uint64_t base_seed{0x243f6a8885a308d3ULL};
  std::string output{"strategy_smoke_samples.csv"};
  std::string decoder_validation_output{"decoder_validation.csv"};
  std::string kernel_validation_output{"kernel_validation.csv"};
};

std::size_t parse_positive_size(const std::string &text,
                                const std::string &option) {
  std::size_t consumed{};
  const auto value = std::stoull(text, &consumed, 0);
  if (consumed != text.size() || value == 0) {
    throw smoke_error(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(value);
}

int parse_positive_int(const std::string &text, const std::string &option) {
  const auto value = parse_positive_size(text, option);
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw smoke_error(option + " is too large");
  }
  return static_cast<int>(value);
}

std::vector<int> parse_powers(const std::string &text,
                              const std::string &option) {
  std::vector<int> result;
  std::stringstream input{text};
  std::string token;
  while (std::getline(input, token, ',')) {
    const auto power = parse_positive_int(token, option);
    if (power > 28) {
      throw smoke_error(option + " powers must be in [1,28]");
    }
    result.push_back(power);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  if (result.empty()) {
    throw smoke_error(option + " must not be empty");
  }
  return result;
}

options parse_options(int argc, char **argv) {
  options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument{argv[index]};
    auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw smoke_error("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--dot-powers") {
      result.dot_powers = parse_powers(value(), argument);
    } else if (argument == "--gemv-powers") {
      result.gemv_powers = parse_powers(value(), argument);
    } else if (argument == "--gemv-rows") {
      result.gemv_rows = parse_positive_size(value(), argument);
    } else if (argument == "--warmup") {
      result.warmup = parse_positive_int(value(), argument);
    } else if (argument == "--samples") {
      result.samples = parse_positive_int(value(), argument);
    } else if (argument == "--target-sample-ms") {
      result.target_sample_ms = std::stod(value());
      if (!(result.target_sample_ms > 0.0)) {
        throw smoke_error(argument + " must be positive");
      }
    } else if (argument == "--base-seed") {
      result.base_seed = parse_positive_size(value(), argument);
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--decoder-validation-output") {
      result.decoder_validation_output = value();
    } else if (argument == "--kernel-validation-output") {
      result.kernel_validation_output = value();
    } else if (argument == "--help") {
      std::cout
          << "Usage: e2e3_strategy_smoke [options]\n"
          << "  --dot-powers P,... --gemv-powers P,... --gemv-rows M\n"
          << "  --warmup N --samples N --target-sample-ms X --base-seed N\n"
          << "  --output FILE --decoder-validation-output FILE\n"
          << "  --kernel-validation-output FILE\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw smoke_error("unknown argument: " + argument);
    }
  }
  return result;
}

std::size_t size_from_power(int power) { return std::size_t{1} << power; }

std::uint64_t mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct device_info {
  std::string name;
  std::string capability;
  int multiprocessors{};
};

device_info query_device() {
  int device{};
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::ostringstream capability;
  capability << "sm_" << properties.major << properties.minor;
  return {properties.name, capability.str(), properties.multiProcessorCount};
}

int work_blocks(std::size_t count, int lanes, int multiprocessors) {
  const auto packs = (count + static_cast<std::size_t>(lanes) - 1) / lanes;
  const auto wanted =
      (packs + decoder::block_threads - 1) / decoder::block_threads;
  return static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
}

std::uint64_t double_bits(double value) {
  std::uint64_t result{};
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

template <typename Format> struct host_tables {
  std::array<float, 256> fp32{};
  std::array<double, 256> fp64{};
  std::array<std::uint16_t, 256> prefix16{};
  std::array<std::uint32_t, 256> prefix32{};
  std::array<std::uint64_t, 16> exponent_prefix{};
  std::array<std::uint32_t, 256> high_word{};
  std::array<std::uint32_t, 32> subnormal_high_word{};

  host_tables() {
    static_assert(decoder::supported_format_v<Format>);
    for (std::size_t code = 0; code < 256; ++code) {
      const auto value =
          storage::decode<Format>(static_cast<std::uint8_t>(code));
      const auto bits = double_bits(value);
      fp32[code] = static_cast<float>(value);
      fp64[code] = value;
      high_word[code] = static_cast<std::uint32_t>(bits >> 32);
      if constexpr (Format::fraction_bits == 4) {
        prefix16[code] = static_cast<std::uint16_t>(bits >> 48);
      } else {
        prefix32[code] = static_cast<std::uint32_t>(bits >> 47);
      }
    }
    constexpr auto subnormal_count = std::size_t{1} << Format::fraction_bits;
    for (std::size_t fraction = 0; fraction < subnormal_count; ++fraction) {
      const auto value =
          storage::decode<Format>(static_cast<std::uint8_t>(fraction));
      subnormal_high_word[fraction] =
          static_cast<std::uint32_t>(double_bits(value) >> 32);
    }
    constexpr auto upper_count = std::size_t{1} << (1 + Format::exponent_bits);
    for (std::size_t upper = 0; upper < upper_count; ++upper) {
      const auto raw =
          static_cast<std::uint8_t>(upper << Format::fraction_bits);
      exponent_prefix[upper] = double_bits(storage::decode<Format>(raw));
    }
  }
};

template <typename T, std::size_t Size>
void upload_array(device_buffer<T> &target, const std::array<T, Size> &source) {
  CUDA_CHECK(cudaMemcpy(target.get(), source.data(), sizeof(source),
                        cudaMemcpyHostToDevice));
}

template <typename Format> class device_tables {
public:
  explicit device_tables(const host_tables<Format> &host)
      : fp32_{256}, fp64_{256}, prefix16_{256}, prefix32_{256},
        exponent_prefix_{16}, high_word_{256}, subnormal_high_word_{32} {
    upload_array(fp32_, host.fp32);
    upload_array(fp64_, host.fp64);
    upload_array(prefix16_, host.prefix16);
    upload_array(prefix32_, host.prefix32);
    upload_array(exponent_prefix_, host.exponent_prefix);
    upload_array(high_word_, host.high_word);
    upload_array(subnormal_high_word_, host.subnormal_high_word);
  }

  decoder::table_bundle bundle() const {
    return {fp32_.get(),
            fp64_.get(),
            prefix16_.get(),
            prefix32_.get(),
            exponent_prefix_.get(),
            high_word_.get(),
            subnormal_high_word_.get()};
  }

private:
  device_buffer<float> fp32_;
  device_buffer<double> fp64_;
  device_buffer<std::uint16_t> prefix16_;
  device_buffer<std::uint32_t> prefix32_;
  device_buffer<std::uint64_t> exponent_prefix_;
  device_buffer<std::uint32_t> high_word_;
  device_buffer<std::uint32_t> subnormal_high_word_;
};

const char *kind_name(decoder::decode_kind kind) {
  switch (kind) {
  case decoder::decode_kind::generic_fp64:
    return "generic_fp64";
  case decoder::decode_kind::branchless_fp32:
    return "branchless_fp32";
  case decoder::decode_kind::lut_fp32:
    return "lut_fp32";
  case decoder::decode_kind::lut_fp64:
    return "lut_fp64";
  case decoder::decode_kind::lut_prefix:
    return "lut_prefix";
  case decoder::decode_kind::direct_fp64_bits:
    return "direct_fp64_bits";
  case decoder::decode_kind::decomposed_bits:
    return "decomposed_bits";
  case decoder::decode_kind::direct_fp64_words_branchy:
    return "direct_fp64_words_branchy";
  case decoder::decode_kind::direct_fp64_words_masked:
    return "direct_fp64_words_masked";
  case decoder::decode_kind::lut_subnormal:
    return "lut_subnormal";
  case decoder::decode_kind::lut_high_word:
    return "lut_high_word";
  case decoder::decode_kind::lut_high_word_swizzled:
    return "lut_high_word_swizzled";
  }
  return "unknown";
}

template <typename Strategy> std::string strategy_name() {
  std::string result{kind_name(Strategy::kind)};
  if constexpr (Strategy::kind == decoder::decode_kind::lut_fp32 ||
                Strategy::kind == decoder::decode_kind::lut_fp64 ||
                Strategy::kind == decoder::decode_kind::lut_prefix ||
                Strategy::kind == decoder::decode_kind::lut_subnormal ||
                Strategy::kind == decoder::decode_kind::lut_high_word ||
                Strategy::kind ==
                    decoder::decode_kind::lut_high_word_swizzled) {
    result += Strategy::location == decoder::table_location::shared ? "_shared"
                                                                    : "_global";
  }
  if constexpr (Strategy::pipelined) {
    result += "_pipelined";
  }
  result += "_x" + std::to_string(Strategy::lanes);
  return result;
}

template <int Lanes, typename Callback>
void for_core_width(Callback &&callback) {
  callback(decoder::strategy<decoder::decode_kind::generic_fp64, Lanes>{});
  callback(decoder::strategy<decoder::decode_kind::branchless_fp32, Lanes>{});
  callback(decoder::strategy<decoder::decode_kind::lut_fp32, Lanes>{});
  callback(decoder::strategy<decoder::decode_kind::lut_fp64, Lanes>{});
  callback(decoder::strategy<decoder::decode_kind::lut_prefix, Lanes>{});
}

template <int Lanes, typename Callback>
void for_new_width(Callback &&callback) {
  callback(decoder::strategy<decoder::decode_kind::direct_fp64_words_branchy,
                             Lanes>{});
  callback(decoder::strategy<decoder::decode_kind::direct_fp64_words_masked,
                             Lanes>{});
  callback(decoder::strategy<decoder::decode_kind::lut_subnormal, Lanes>{});
  callback(decoder::strategy<decoder::decode_kind::lut_subnormal, Lanes,
                             decoder::table_location::shared>{});
  callback(decoder::strategy<decoder::decode_kind::lut_high_word, Lanes>{});
  callback(decoder::strategy<decoder::decode_kind::lut_high_word, Lanes,
                             decoder::table_location::shared>{});
  callback(decoder::strategy<decoder::decode_kind::lut_high_word_swizzled,
                             Lanes, decoder::table_location::shared>{});
}

template <typename Format, typename Callback>
void for_each_strategy(Callback &&callback) {
  static_assert(decoder::supported_format_v<Format>);
  for_core_width<1>(callback);
  for_core_width<2>(callback);
  for_core_width<4>(callback);
  for_core_width<8>(callback);
  callback(decoder::strategy<decoder::decode_kind::direct_fp64_bits, 4>{});
  callback(decoder::strategy<decoder::decode_kind::decomposed_bits, 4>{});
  callback(decoder::strategy<decoder::decode_kind::lut_fp32, 4,
                             decoder::table_location::shared>{});
  callback(decoder::strategy<decoder::decode_kind::lut_fp64, 4,
                             decoder::table_location::shared>{});
  callback(decoder::strategy<decoder::decode_kind::lut_prefix, 4,
                             decoder::table_location::shared>{});
  callback(decoder::strategy<decoder::decode_kind::lut_prefix, 8,
                             decoder::table_location::shared>{});
  callback(
      decoder::strategy<decoder::decode_kind::lut_fp32, 4,
                        decoder::table_location::global_read_only, true>{});
  callback(
      decoder::strategy<decoder::decode_kind::lut_prefix, 4,
                        decoder::table_location::global_read_only, true>{});
  for_new_width<4>(callback);
  for_new_width<8>(callback);
}

template <typename Format, typename Strategy>
void validate_decoder(const std::uint8_t *codes, decoder::table_bundle tables,
                      double *device_output, std::ofstream &output) {
  constexpr auto shared_bytes = decoder::shared_table_bytes_v<Format, Strategy>;
  decoder::decode_all_codes<Format, Strategy>
      <<<1, decoder::block_threads, shared_bytes>>>(codes, tables,
                                                    device_output);
  CUDA_CHECK(cudaGetLastError());
  std::array<double, 256> actual{};
  CUDA_CHECK(cudaMemcpy(actual.data(), device_output, sizeof(actual),
                        cudaMemcpyDeviceToHost));
  int finite_bit_mismatches{};
  int classification_mismatches{};
  double maximum_absolute_error{};
  for (std::size_t code = 0; code < actual.size(); ++code) {
    const auto expected =
        storage::decode<Format>(static_cast<std::uint8_t>(code));
    if (std::isnan(expected)) {
      classification_mismatches += !std::isnan(actual[code]);
    } else if (std::isinf(expected)) {
      classification_mismatches +=
          !std::isinf(actual[code]) ||
          std::signbit(expected) != std::signbit(actual[code]);
    } else {
      finite_bit_mismatches +=
          double_bits(expected) != double_bits(actual[code]);
      maximum_absolute_error =
          std::max(maximum_absolute_error, std::abs(expected - actual[code]));
    }
  }
  output << Format::name << ',' << strategy_name<Strategy>() << ','
         << Strategy::lanes << ','
         << decoder::lookup_entry_bytes_v<Format, Strategy> << ','
         << decoder::shared_table_bytes_v<Format, Strategy> << ','
         << Strategy::pipelined << ',' << finite_bit_mismatches << ','
         << classification_mismatches << ',' << maximum_absolute_error << '\n';
  if (finite_bit_mismatches != 0 || classification_mismatches != 0) {
    throw smoke_error(std::string{"decoder validation failed for "} +
                      Format::name + '/' + strategy_name<Strategy>());
  }
}

template <typename Format>
void validate_format(decoder::table_bundle tables, std::ofstream &output) {
  std::array<std::uint8_t, 256> host_codes{};
  for (std::size_t code = 0; code < host_codes.size(); ++code) {
    host_codes[code] = static_cast<std::uint8_t>(code);
  }
  device_buffer<std::uint8_t> codes{host_codes.size()};
  device_buffer<double> decoded{host_codes.size()};
  CUDA_CHECK(cudaMemcpy(codes.get(), host_codes.data(), sizeof(host_codes),
                        cudaMemcpyHostToDevice));
  for_each_strategy<Format>([&](auto strategy) {
    using strategy_type = decltype(strategy);
    validate_decoder<Format, strategy_type>(codes.get(), tables, decoded.get(),
                                            output);
  });
}

struct source_dataset {
  std::vector<double> dot_left;
  std::vector<double> dot_right;
  std::vector<double> matrix;
  std::vector<double> vector;
};

source_dataset make_source_dataset(std::size_t dot_count,
                                   std::size_t matrix_count,
                                   std::size_t vector_count, distribution kind,
                                   std::uint64_t seed) {
  source_dataset result{{}, {}, {}, {}};
  result.dot_left.resize(dot_count);
  result.dot_right.resize(dot_count);
  result.matrix.resize(matrix_count);
  result.vector.resize(vector_count);
  std::mt19937_64 engine{seed};
  std::uniform_real_distribution<double> uniform{0.0, 1.0};
  std::normal_distribution<double> normal{0.0, 1.0};
  auto fill = [&](std::vector<double> &values) {
    for (auto &value : values) {
      value =
          kind == distribution::uniform_0_1 ? uniform(engine) : normal(engine);
    }
  };
  fill(result.dot_left);
  fill(result.dot_right);
  fill(result.matrix);
  fill(result.vector);
  return result;
}

template <typename Format> struct encoded_dataset {
  explicit encoded_dataset(const source_dataset &source)
      : dot_left_host(source.dot_left.size()),
        dot_right_host(source.dot_right.size()),
        matrix_host(source.matrix.size()), vector_host(source.vector.size()),
        dot_left(dot_left_host.size()), dot_right(dot_right_host.size()),
        matrix(matrix_host.size()), vector(vector_host.size()) {
    encode(source.dot_left, dot_left_host);
    encode(source.dot_right, dot_right_host);
    encode(source.matrix, matrix_host);
    encode(source.vector, vector_host);
    upload(dot_left, dot_left_host);
    upload(dot_right, dot_right_host);
    upload(matrix, matrix_host);
    upload(vector, vector_host);
  }

  static void encode(const std::vector<double> &source,
                     std::vector<std::uint8_t> &target) {
    for (std::size_t index = 0; index < source.size(); ++index) {
      target[index] = storage::encode<Format>(source[index]);
    }
  }

  static void upload(device_buffer<std::uint8_t> &target,
                     const std::vector<std::uint8_t> &source) {
    CUDA_CHECK(cudaMemcpy(target.get(), source.data(), source.size(),
                          cudaMemcpyHostToDevice));
  }

  std::vector<std::uint8_t> dot_left_host;
  std::vector<std::uint8_t> dot_right_host;
  std::vector<std::uint8_t> matrix_host;
  std::vector<std::uint8_t> vector_host;
  device_buffer<std::uint8_t> dot_left;
  device_buffer<std::uint8_t> dot_right;
  device_buffer<std::uint8_t> matrix;
  device_buffer<std::uint8_t> vector;
};

struct scalar_reference {
  long double value{};
  long double sum_abs{};
};

template <typename Format>
scalar_reference dot_reference(const std::vector<std::uint8_t> &left,
                               const std::vector<std::uint8_t> &right,
                               std::size_t count) {
  scalar_reference result{};
  for (std::size_t index = 0; index < count; ++index) {
    const auto term =
        static_cast<long double>(storage::decode<Format>(left[index])) *
        static_cast<long double>(storage::decode<Format>(right[index]));
    result.value += term;
    result.sum_abs += std::abs(term);
  }
  return result;
}

template <typename Format>
std::vector<scalar_reference>
gemv_reference(const std::vector<std::uint8_t> &matrix,
               const std::vector<std::uint8_t> &vector, std::size_t rows,
               std::size_t columns) {
  std::vector<scalar_reference> result(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      const auto term =
          static_cast<long double>(
              storage::decode<Format>(matrix[row * columns + column])) *
          static_cast<long double>(storage::decode<Format>(vector[column]));
      result[row].value += term;
      result[row].sum_abs += std::abs(term);
    }
  }
  return result;
}

struct comparison {
  std::size_t finite_results{};
  std::size_t nonfinite_results{};
  std::size_t classification_mismatches{};
  long double maximum_normalized_error{};
};

void compare_one(double actual, const scalar_reference &expected,
                 comparison &result) {
  if (std::isnan(expected.value)) {
    ++result.nonfinite_results;
    result.classification_mismatches += !std::isnan(actual);
  } else if (std::isinf(expected.value)) {
    ++result.nonfinite_results;
    result.classification_mismatches +=
        !std::isinf(actual) ||
        std::signbit(actual) != std::signbit(expected.value);
  } else {
    ++result.finite_results;
    if (!std::isfinite(actual)) {
      ++result.classification_mismatches;
      return;
    }
    const auto error =
        std::abs(static_cast<long double>(actual) - expected.value);
    const auto denominator =
        std::max(expected.sum_abs, std::numeric_limits<long double>::min());
    result.maximum_normalized_error =
        std::max(result.maximum_normalized_error, error / denominator);
  }
}

class csv_outputs {
public:
  csv_outputs(const options &settings, const device_info &device)
      : samples_{settings.output},
        validation_{settings.kernel_validation_output}, device_{device} {
    if (!samples_ || !validation_) {
      throw smoke_error("could not create smoke CSV outputs");
    }
    samples_ << "gpu,compute_capability,distribution,format,strategy,component,"
                "lanes,n,m,lookup_entry_bytes,shared_table_bytes,pipelined,"
                "sample,iterations,total_time_ms,time_ms\n";
    validation_
        << "distribution,format,strategy,component,lanes,n,m,finite_results,"
           "nonfinite_results,classification_mismatches,"
           "max_normalized_error,pass\n";
    samples_ << std::setprecision(17);
    validation_ << std::setprecision(21);
  }

  template <typename Format, typename Strategy>
  void write_sample(distribution kind, const char *component, std::size_t n,
                    std::size_t m, int sample, std::size_t iterations,
                    double total_ms) {
    samples_ << device_.name << ',' << device_.capability << ',' << name(kind)
             << ',' << Format::name << ',' << strategy_name<Strategy>() << ','
             << component << ',' << Strategy::lanes << ',' << n << ',' << m
             << ',' << decoder::lookup_entry_bytes_v<Format, Strategy> << ','
             << decoder::shared_table_bytes_v<Format, Strategy> << ','
             << Strategy::pipelined << ',' << sample << ',' << iterations << ','
             << total_ms << ',' << total_ms / iterations << '\n';
  }

  template <typename Format, typename Strategy>
  void write_validation(distribution kind, const char *component, std::size_t n,
                        std::size_t m, const comparison &value) {
    const auto passed = value.classification_mismatches == 0 &&
                        value.maximum_normalized_error <= 1.0e-12L;
    validation_ << name(kind) << ',' << Format::name << ','
                << strategy_name<Strategy>() << ',' << component << ','
                << Strategy::lanes << ',' << n << ',' << m << ','
                << value.finite_results << ',' << value.nonfinite_results << ','
                << value.classification_mismatches << ','
                << value.maximum_normalized_error << ',' << passed << '\n';
    if (!passed) {
      throw smoke_error(std::string{"kernel validation failed for "} +
                        Format::name + '/' + strategy_name<Strategy>() + '/' +
                        component);
    }
  }

private:
  std::ofstream samples_;
  std::ofstream validation_;
  device_info device_;
};

template <typename Format, typename Strategy>
void time_and_write(const options &settings, distribution kind,
                    const char *component, std::size_t n, std::size_t m,
                    const std::function<void()> &launch, csv_outputs &output,
                    event_timer &timer) {
  for (int warmup = 0; warmup < settings.warmup; ++warmup) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto probe = std::max(0.0001f, timer.measure(launch, 1));
  const auto wanted =
      static_cast<std::size_t>(std::ceil(settings.target_sample_ms / probe));
  const auto iterations =
      std::max<std::size_t>(1, std::min<std::size_t>(wanted, 100000));
  for (int sample = 0; sample < settings.samples; ++sample) {
    const auto elapsed = timer.measure(launch, iterations);
    output.write_sample<Format, Strategy>(kind, component, n, m, sample,
                                          iterations, elapsed);
  }
}

template <typename Format, typename Strategy>
void run_dot_strategy(const options &settings, const device_info &device,
                      distribution kind, const encoded_dataset<Format> &data,
                      decoder::table_bundle tables, std::size_t count,
                      const scalar_reference &reference, double *partials,
                      double *result, csv_outputs &output, event_timer &timer) {
  const auto blocks =
      work_blocks(count, Strategy::lanes, device.multiprocessors);
  constexpr auto shared_bytes = decoder::shared_table_bytes_v<Format, Strategy>;
  const auto *left = data.dot_left.get();
  const auto *right = data.dot_right.get();
  const auto launch = [=] {
    decoder::dot_map_reduce<Format, Strategy>
        <<<blocks, decoder::block_threads, shared_bytes>>>(left, right, count,
                                                           tables, partials);
    kernels::storage_dot_finalize<<<1, kernels::reduction_block_threads>>>(
        partials, static_cast<std::size_t>(blocks), result);
  };
  launch();
  CUDA_CHECK(cudaGetLastError());
  double actual{};
  CUDA_CHECK(
      cudaMemcpy(&actual, result, sizeof(actual), cudaMemcpyDeviceToHost));
  comparison checked{};
  compare_one(actual, reference, checked);
  output.write_validation<Format, Strategy>(kind, "dot", count, 1, checked);
  time_and_write<Format, Strategy>(settings, kind, "dot", count, 1, launch,
                                   output, timer);
}

template <typename Format, typename Strategy>
void run_gemv_strategy(const options &settings, distribution kind,
                       const encoded_dataset<Format> &data,
                       decoder::table_bundle tables, std::size_t rows,
                       std::size_t columns,
                       const std::vector<scalar_reference> &reference,
                       double *result, csv_outputs &output,
                       event_timer &timer) {
  constexpr auto shared_bytes = decoder::shared_table_bytes_v<Format, Strategy>;
  const auto *matrix = data.matrix.get();
  const auto *vector = data.vector.get();
  const auto launch = [=] {
    decoder::gemv<Format, Strategy>
        <<<static_cast<unsigned>(rows), decoder::block_threads, shared_bytes>>>(
            matrix, vector, rows, columns, columns, tables, result);
  };
  launch();
  CUDA_CHECK(cudaGetLastError());
  std::vector<double> actual(rows);
  CUDA_CHECK(cudaMemcpy(actual.data(), result, rows * sizeof(double),
                        cudaMemcpyDeviceToHost));
  comparison checked{};
  for (std::size_t row = 0; row < rows; ++row) {
    compare_one(actual[row], reference[row], checked);
  }
  output.write_validation<Format, Strategy>(kind, "gemv", columns, rows,
                                            checked);
  time_and_write<Format, Strategy>(settings, kind, "gemv", columns, rows,
                                   launch, output, timer);
}

template <typename Format>
void run_format(const options &settings, const device_info &device,
                distribution kind, const source_dataset &source,
                decoder::table_bundle tables, csv_outputs &output,
                event_timer &timer) {
  encoded_dataset<Format> data{source};
  const auto max_blocks = device.multiprocessors * 16;
  device_buffer<double> partials{static_cast<std::size_t>(max_blocks)};
  device_buffer<double> result{std::max<std::size_t>(1, settings.gemv_rows)};

  for (const auto power : settings.dot_powers) {
    const auto count = size_from_power(power);
    const auto reference =
        dot_reference<Format>(data.dot_left_host, data.dot_right_host, count);
    for_each_strategy<Format>([&](auto strategy) {
      using strategy_type = decltype(strategy);
      run_dot_strategy<Format, strategy_type>(
          settings, device, kind, data, tables, count, reference,
          partials.get(), result.get(), output, timer);
    });
    std::cout << "  " << Format::name << " DOT N=2^" << power << " complete\n";
  }

  for (const auto power : settings.gemv_powers) {
    const auto columns = size_from_power(power);
    const auto reference = gemv_reference<Format>(
        data.matrix_host, data.vector_host, settings.gemv_rows, columns);
    for_each_strategy<Format>([&](auto strategy) {
      using strategy_type = decltype(strategy);
      run_gemv_strategy<Format, strategy_type>(
          settings, kind, data, tables, settings.gemv_rows, columns, reference,
          result.get(), output, timer);
    });
    std::cout << "  " << Format::name << " GEMV N=2^" << power << " complete\n";
  }
}

void create_parent(const std::string &path) {
  const auto parent = std::filesystem::path{path}.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

} // namespace

int main(int argc, char **argv) try {
  const auto settings = parse_options(argc, argv);
  const auto device = query_device();
  if (device.capability != "sm_90") {
    throw smoke_error("this preliminary experiment is calibrated for sm_90");
  }
  create_parent(settings.output);
  create_parent(settings.decoder_validation_output);
  create_parent(settings.kernel_validation_output);
  std::cout << "E2M5/E3M4 decoder strategy smoke test\n"
            << "GPU: " << device.name << " (" << device.capability << ")\n";

  const host_tables<storage::e2m5> e2_host_tables;
  const host_tables<storage::e3m4> e3_host_tables;
  const device_tables<storage::e2m5> e2_tables{e2_host_tables};
  const device_tables<storage::e3m4> e3_tables{e3_host_tables};

  std::ofstream decoder_validation{settings.decoder_validation_output};
  if (!decoder_validation) {
    throw smoke_error("could not create " + settings.decoder_validation_output);
  }
  decoder_validation
      << "format,strategy,lanes,lookup_entry_bytes,shared_table_bytes,"
         "pipelined,finite_bit_mismatches,classification_mismatches,"
         "max_finite_abs_error\n";
  decoder_validation << std::setprecision(17);
  validate_format<storage::e2m5>(e2_tables.bundle(), decoder_validation);
  validate_format<storage::e3m4>(e3_tables.bundle(), decoder_validation);
  decoder_validation.close();
  std::cout << "All 256 encodings passed every decoder strategy.\n";

  const auto max_dot = size_from_power(settings.dot_powers.back());
  const auto max_columns = size_from_power(settings.gemv_powers.back());
  const auto max_matrix = settings.gemv_rows * max_columns;
  csv_outputs output{settings, device};
  event_timer timer;
  for (const auto kind :
       {distribution::uniform_0_1, distribution::normal_0_1}) {
    std::cout << name(kind) << '\n';
    const auto source = make_source_dataset(
        max_dot, max_matrix, max_columns, kind,
        mix(settings.base_seed ^ static_cast<std::uint64_t>(kind)));
    run_format<storage::e2m5>(settings, device, kind, source,
                              e2_tables.bundle(), output, timer);
    run_format<storage::e3m4>(settings, device, kind, source,
                              e3_tables.bundle(), output, timer);
  }

  std::cout << "Wrote " << settings.decoder_validation_output << '\n'
            << "Wrote " << settings.kernel_validation_output << '\n'
            << "Wrote " << settings.output << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return EXIT_FAILURE;
}
