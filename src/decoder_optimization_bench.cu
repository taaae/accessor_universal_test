#include "decoder_optimization_kernels.cuh"

#include <cuda_runtime.h>
#include <curand.h>

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace decoder = aut::decoder_optimization;
namespace kernels = aut::kernels;
namespace performance = aut::performance;
namespace storage = aut::storage;

class benchmark_error : public std::runtime_error {
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
    throw benchmark_error(message.str());
  }
}

void check_curand(curandStatus_t status, const char *expression,
                  const char *file, int line) {
  if (status != CURAND_STATUS_SUCCESS) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line
            << " with cuRAND status " << static_cast<int>(status);
    throw benchmark_error(message.str());
  }
}

#define CUDA_CHECK(expression)                                                 \
  check_cuda((expression), #expression, __FILE__, __LINE__)
#define CURAND_CHECK(expression)                                               \
  check_curand((expression), #expression, __FILE__, __LINE__)

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
  std::vector<int> dot_powers{16, 20, 24, 27};
  std::vector<int> gemv_powers{10, 12, 14, 16};
  std::size_t gemv_rows{1024};
  int warmup{10};
  int rounds{3};
  int samples{5};
  double target_sample_ms{20.0};
  int decode_repeats{256};
  std::uint64_t base_seed{0x243f6a8885a308d3ULL};
  std::string output{"decoder_timing_samples.csv"};
  std::string validation_output{"decoder_validation.csv"};
};

std::size_t parse_positive_size(const std::string &text,
                                const std::string &option) {
  std::size_t consumed{};
  const auto value = std::stoull(text, &consumed, 0);
  if (consumed != text.size() || value == 0) {
    throw benchmark_error(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(value);
}

int parse_positive_int(const std::string &text, const std::string &option) {
  const auto value = parse_positive_size(text, option);
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw benchmark_error(option + " is too large");
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
    if (power > 30) {
      throw benchmark_error(option + " powers must be in [1,30]");
    }
    result.push_back(power);
  }
  if (result.empty()) {
    throw benchmark_error(option + " must not be empty");
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

options parse_options(int argc, char **argv) {
  options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument{argv[index]};
    auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw benchmark_error("missing value after " + argument);
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
    } else if (argument == "--rounds") {
      result.rounds = parse_positive_int(value(), argument);
    } else if (argument == "--samples") {
      result.samples = parse_positive_int(value(), argument);
    } else if (argument == "--target-sample-ms") {
      result.target_sample_ms = std::stod(value());
      if (!(result.target_sample_ms > 0.0)) {
        throw benchmark_error(argument + " must be positive");
      }
    } else if (argument == "--decode-repeats") {
      result.decode_repeats = parse_positive_int(value(), argument);
    } else if (argument == "--base-seed") {
      result.base_seed = parse_positive_size(value(), argument);
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--validation-output") {
      result.validation_output = value();
    } else if (argument == "--help") {
      std::cout << "Usage: decoder_optimization_bench [options]\n"
                << "  --dot-powers P,... --gemv-powers P,... --gemv-rows M\n"
                << "  --warmup N --rounds N --samples N --target-sample-ms X\n"
                << "  --decode-repeats N --base-seed N --output FILE\n"
                << "  --validation-output FILE\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw benchmark_error("unknown argument: " + argument);
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

class random_generator {
public:
  random_generator() {
    CURAND_CHECK(
        curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  }
  ~random_generator() {
    if (generator_) {
      curandDestroyGenerator(generator_);
    }
  }
  void fill(double *values, std::size_t count, distribution kind,
            std::uint64_t seed) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, seed));
    CURAND_CHECK(curandSetGeneratorOffset(generator_, 0));
    if (kind == distribution::uniform_0_1) {
      CURAND_CHECK(curandGenerateUniformDouble(generator_, values, count));
    } else {
      if (count % 2 != 0) {
        throw benchmark_error("normal generation requires an even count");
      }
      CURAND_CHECK(
          curandGenerateNormalDouble(generator_, values, count, 0.0, 1.0));
    }
  }

private:
  curandGenerator_t generator_{};
};

template <typename Format>
__global__ void encode_values_kernel(const double *source,
                                     storage::storage_type_t<Format> *encoded,
                                     std::size_t count) {
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto index = first; index < count; index += stride) {
    encoded[index] = storage::encode<Format>(source[index]);
  }
}

template <typename Format>
void encode_values(const double *source,
                   storage::storage_type_t<Format> *target, std::size_t count,
                   int multiprocessors) {
  const auto wanted = (count + 255) / 256;
  const auto blocks = static_cast<unsigned>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
  encode_values_kernel<Format><<<blocks, 256>>>(source, target, count);
  CUDA_CHECK(cudaGetLastError());
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
      (packs + performance::block_threads - 1) / performance::block_threads;
  return static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
}

template <typename Format> std::array<float, 256> make_lut() {
  static_assert(sizeof(storage::storage_type_t<Format>) == 1);
  std::array<float, 256> result{};
  for (std::size_t code = 0; code < result.size(); ++code) {
    result[code] = static_cast<float>(storage::decode<Format>(
        static_cast<storage::storage_type_t<Format>>(code)));
  }
  return result;
}

template <typename Format, typename Callback>
void for_each_decoder(Callback &&callback) {
  callback(decoder::current_x1{});
  callback(decoder::current_x4{});
  if constexpr (decoder::is_optimized_format_v<Format>) {
    callback(decoder::branchless_x4{});
    callback(decoder::lut_x1{});
    callback(decoder::lut_x4{});
  }
}

struct work_model {
  std::string component;
  int lanes{};
  std::size_t n{};
  std::size_t m{};
  int blocks{};
  int threads{performance::block_threads};
  int decode_repeats{};
  double decoded_values{};
  double source_bytes{};
  double lookup_bytes_requested{};
  double useful_flops{};
};

struct timed_variant {
  std::string format;
  std::string strategy;
  int storage_bits{};
  work_model model;
  std::function<void()> launch;
  std::size_t iterations{1};
};

class sample_output {
public:
  sample_output(const std::string &path, const device_info &device)
      : stream_{path}, device_{device} {
    if (!stream_) {
      throw benchmark_error("could not create " + path);
    }
    stream_ << "gpu,compute_capability,distribution,format,strategy,"
               "storage_bits,component,lanes,n,m,blocks,threads,"
               "decode_repeats,warmup,round,sample,order_position,iterations,"
               "total_time_ms,time_ms,decoded_values,source_bytes,"
               "lookup_bytes_requested,useful_flops,decoded_gvalues_per_s,"
               "source_gb_per_s,lookup_gb_per_s,useful_gflop_per_s\n";
    stream_ << std::setprecision(17);
  }

  void write(distribution kind, const timed_variant &variant, int warmup,
             int round, int sample, int order_position, double total_ms) {
    const auto time_ms = total_ms / variant.iterations;
    const auto seconds = time_ms * 1.0e-3;
    const auto rate = [&](double amount, double scale) {
      return seconds > 0.0 ? amount / seconds / scale : 0.0;
    };
    const auto &model = variant.model;
    stream_ << device_.name << ',' << device_.capability << ',' << name(kind)
            << ',' << variant.format << ',' << variant.strategy << ','
            << variant.storage_bits << ',' << model.component << ','
            << model.lanes << ',' << model.n << ',' << model.m << ','
            << model.blocks << ',' << model.threads << ','
            << model.decode_repeats << ',' << warmup << ',' << round << ','
            << sample << ',' << order_position << ',' << variant.iterations
            << ',' << total_ms << ',' << time_ms << ',' << model.decoded_values
            << ',' << model.source_bytes << ',' << model.lookup_bytes_requested
            << ',' << model.useful_flops << ','
            << rate(model.decoded_values, 1.0e9) << ','
            << rate(model.source_bytes, 1.0e9) << ','
            << rate(model.lookup_bytes_requested, 1.0e9) << ','
            << rate(model.useful_flops, 1.0e9) << '\n';
    stream_.flush();
  }

private:
  std::ofstream stream_;
  device_info device_;
};

void measure_variants(std::vector<timed_variant> &variants,
                      const options &settings, distribution kind,
                      sample_output &output, event_timer &timer) {
  for (auto &variant : variants) {
    for (int warmup = 0; warmup < settings.warmup; ++warmup) {
      variant.launch();
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  for (auto &variant : variants) {
    const auto probe = std::max(0.0001f, timer.measure(variant.launch, 1));
    const auto wanted =
        static_cast<std::size_t>(std::ceil(settings.target_sample_ms / probe));
    variant.iterations =
        std::max<std::size_t>(1, std::min<std::size_t>(wanted, 1000000));
  }
  const auto count = static_cast<int>(variants.size());
  for (int round = 0; round < settings.rounds; ++round) {
    for (int sample = 0; sample < settings.samples; ++sample) {
      const auto offset = (round + sample) % count;
      for (int position = 0; position < count; ++position) {
        auto &variant = variants[(offset + position) % count];
        const auto elapsed = timer.measure(variant.launch, variant.iterations);
        output.write(kind, variant, settings.warmup, round, sample, position,
                     elapsed);
      }
    }
  }
}

template <typename Format, typename Strategy>
timed_variant
make_register_variant(const storage::storage_type_t<Format> *values,
                      std::size_t count, const float *lut, double *sink,
                      int blocks, int repeats) {
  constexpr auto lanes = Strategy::lanes;
  const auto threads = static_cast<double>(blocks) * performance::block_threads;
  const auto decoded_values = threads * lanes * repeats;
  work_model model{"register_decode",
                   lanes,
                   count,
                   1,
                   blocks,
                   performance::block_threads,
                   repeats,
                   decoded_values,
                   threads * lanes * sizeof(storage::storage_type_t<Format>),
                   decoder::uses_lut_v<Strategy> ? 4.0 * decoded_values : 0.0,
                   decoded_values};
  return {Format::name, Strategy::name, Format::total_bits, model, [=] {
            decoder::register_decode<Format, Strategy>
                <<<blocks, performance::block_threads>>>(values, count, repeats,
                                                         lut, sink);
          }};
}

template <typename Format, int Lanes>
timed_variant make_load_variant(const storage::storage_type_t<Format> *values,
                                std::size_t count, unsigned long long *sink,
                                int multiprocessors) {
  const auto blocks = work_blocks(count, Lanes, multiprocessors);
  work_model model{
      "stream_load",
      Lanes,
      count,
      1,
      blocks,
      performance::block_threads,
      0,
      0.0,
      static_cast<double>(count * sizeof(storage::storage_type_t<Format>)),
      0.0,
      0.0};
  const auto strategy = Lanes == 1 ? "load_only_x1" : "load_only_x4";
  return {Format::name, strategy, Format::total_bits, model, [=] {
            performance::stream_load<Format, Lanes>
                <<<blocks, performance::block_threads>>>(values, count, sink);
          }};
}

template <typename Format, typename Strategy>
timed_variant make_stream_variant(const storage::storage_type_t<Format> *values,
                                  std::size_t count, const float *lut,
                                  double *sink, int multiprocessors) {
  constexpr auto lanes = Strategy::lanes;
  const auto blocks = work_blocks(count, lanes, multiprocessors);
  const auto bytes =
      static_cast<double>(count * sizeof(storage::storage_type_t<Format>));
  work_model model{"stream_decode",
                   lanes,
                   count,
                   1,
                   blocks,
                   performance::block_threads,
                   0,
                   static_cast<double>(count),
                   bytes,
                   decoder::uses_lut_v<Strategy> ? 4.0 * count : 0.0,
                   static_cast<double>(count)};
  return {Format::name, Strategy::name, Format::total_bits, model, [=] {
            decoder::stream_load_decode<Format, Strategy>
                <<<blocks, performance::block_threads>>>(values, count, lut,
                                                         sink);
          }};
}

template <typename Format, typename Strategy>
timed_variant make_dot_variant(const storage::storage_type_t<Format> *left,
                               const storage::storage_type_t<Format> *right,
                               std::size_t count, const float *lut,
                               double *partials, double *result,
                               int multiprocessors) {
  constexpr auto lanes = Strategy::lanes;
  const auto blocks = work_blocks(count, lanes, multiprocessors);
  const auto bytes = static_cast<double>(
      2ULL * count * sizeof(storage::storage_type_t<Format>) +
      2ULL * blocks * sizeof(double) + sizeof(double));
  const auto decoded_values = static_cast<double>(2ULL * count);
  work_model model{"dot",
                   lanes,
                   count,
                   1,
                   blocks,
                   kernels::reduction_block_threads,
                   0,
                   decoded_values,
                   bytes,
                   decoder::uses_lut_v<Strategy> ? 4.0 * decoded_values : 0.0,
                   decoded_values};
  return {
      Format::name, Strategy::name, Format::total_bits, model, [=] {
        decoder::dot_map_reduce<Format, Strategy>
            <<<blocks, kernels::reduction_block_threads>>>(left, right, count,
                                                           lut, partials);
        kernels::storage_dot_finalize<<<1, kernels::reduction_block_threads>>>(
            partials, static_cast<std::size_t>(blocks), result);
      }};
}

template <typename Format, typename Strategy>
timed_variant make_gemv_variant(const storage::storage_type_t<Format> *matrix,
                                const storage::storage_type_t<Format> *vector,
                                std::size_t rows, std::size_t columns,
                                const float *lut, double *result) {
  constexpr auto lanes = Strategy::lanes;
  const auto bytes = static_cast<double>(
      (rows * columns + columns) * sizeof(storage::storage_type_t<Format>) +
      rows * sizeof(double));
  const auto decoded_values = static_cast<double>(2ULL * rows * columns);
  work_model model{"gemv",
                   lanes,
                   columns,
                   rows,
                   static_cast<int>(rows),
                   kernels::reduction_block_threads,
                   0,
                   decoded_values,
                   bytes,
                   decoder::uses_lut_v<Strategy> ? 4.0 * decoded_values : 0.0,
                   decoded_values};
  return {
      Format::name, Strategy::name, Format::total_bits, model, [=] {
        decoder::gemv<Format, Strategy>
            <<<static_cast<unsigned>(rows), kernels::reduction_block_threads>>>(
                matrix, vector, rows, columns, columns, lut, result);
      }};
}

template <typename Format> struct encoded_data {
  encoded_data(std::size_t dot_count, std::size_t matrix_count,
               std::size_t vector_count)
      : dot_left{dot_count}, dot_right{dot_count}, matrix{matrix_count},
        vector{vector_count} {}

  device_buffer<storage::storage_type_t<Format>> dot_left;
  device_buffer<storage::storage_type_t<Format>> dot_right;
  device_buffer<storage::storage_type_t<Format>> matrix;
  device_buffer<storage::storage_type_t<Format>> vector;
};

template <typename Format>
void encode_dataset(const double *source_dot_left,
                    const double *source_dot_right, const double *source_matrix,
                    const double *source_vector, std::size_t dot_count,
                    std::size_t matrix_count, std::size_t vector_count,
                    int multiprocessors, encoded_data<Format> &target) {
  encode_values<Format>(source_dot_left, target.dot_left.get(), dot_count,
                        multiprocessors);
  encode_values<Format>(source_dot_right, target.dot_right.get(), dot_count,
                        multiprocessors);
  encode_values<Format>(source_matrix, target.matrix.get(), matrix_count,
                        multiprocessors);
  encode_values<Format>(source_vector, target.vector.get(), vector_count,
                        multiprocessors);
}

template <typename Format, typename Strategy>
void validate_strategy(const storage::storage_type_t<Format> *codes,
                       const float *lut, double *device_output,
                       std::ofstream &output) {
  constexpr auto lanes = Strategy::lanes;
  constexpr auto packs = 256 / lanes;
  decoder::decode_all_codes<Format, Strategy>
      <<<1, packs>>>(codes, lut, device_output);
  CUDA_CHECK(cudaGetLastError());
  std::array<double, 256> actual{};
  CUDA_CHECK(cudaMemcpy(actual.data(), device_output, sizeof(actual),
                        cudaMemcpyDeviceToHost));
  int finite_bit_mismatches{};
  int classification_mismatches{};
  double maximum_absolute_error{};
  for (std::size_t code = 0; code < actual.size(); ++code) {
    const auto expected = storage::decode<Format>(
        static_cast<storage::storage_type_t<Format>>(code));
    if (std::isnan(expected)) {
      classification_mismatches += !std::isnan(actual[code]);
    } else if (std::isinf(expected)) {
      classification_mismatches +=
          !std::isinf(actual[code]) ||
          std::signbit(expected) != std::signbit(actual[code]);
    } else {
      std::uint64_t expected_bits{};
      std::uint64_t actual_bits{};
      std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
      std::memcpy(&actual_bits, &actual[code], sizeof(actual_bits));
      finite_bit_mismatches += expected_bits != actual_bits;
      maximum_absolute_error =
          std::max(maximum_absolute_error, std::abs(expected - actual[code]));
    }
  }
  output << Format::name << ',' << Strategy::name << ',' << lanes << ','
         << finite_bit_mismatches << ',' << classification_mismatches << ','
         << maximum_absolute_error << '\n';
  if (finite_bit_mismatches != 0 || classification_mismatches != 0) {
    throw benchmark_error(std::string{"decoder validation failed for "} +
                          Format::name + '/' + Strategy::name);
  }
}

template <typename Format>
void validate_format(const float *lut, std::ofstream &output) {
  std::array<storage::storage_type_t<Format>, 256> host_codes{};
  for (std::size_t code = 0; code < host_codes.size(); ++code) {
    host_codes[code] = static_cast<storage::storage_type_t<Format>>(code);
  }
  device_buffer<storage::storage_type_t<Format>> codes{host_codes.size()};
  device_buffer<double> decoded{host_codes.size()};
  CUDA_CHECK(cudaMemcpy(codes.get(), host_codes.data(), sizeof(host_codes),
                        cudaMemcpyHostToDevice));
  for_each_decoder<Format>([&](auto strategy) {
    using strategy_type = decltype(strategy);
    validate_strategy<Format, strategy_type>(codes.get(), lut, decoded.get(),
                                             output);
  });
}

void run_validation(const std::string &path, const float *e2_lut,
                    const float *e3_lut) {
  std::ofstream output{path};
  if (!output) {
    throw benchmark_error("could not create " + path);
  }
  output << "format,strategy,lanes,finite_bit_mismatches,"
            "classification_mismatches,max_finite_abs_error\n";
  output << std::setprecision(17);
  validate_format<storage::e2m5>(e2_lut, output);
  validate_format<storage::e3m4>(e3_lut, output);
}

template <typename Format>
void append_register_variants(std::vector<timed_variant> &variants,
                              const storage::storage_type_t<Format> *values,
                              std::size_t count, const float *lut, double *sink,
                              int blocks, int repeats) {
  for_each_decoder<Format>([&](auto strategy) {
    using strategy_type = decltype(strategy);
    variants.push_back(make_register_variant<Format, strategy_type>(
        values, count, lut, sink, blocks, repeats));
  });
}

template <typename Format>
void append_stream_variants(std::vector<timed_variant> &variants,
                            const storage::storage_type_t<Format> *values,
                            std::size_t count, const float *lut, double *sink,
                            int multiprocessors) {
  for_each_decoder<Format>([&](auto strategy) {
    using strategy_type = decltype(strategy);
    variants.push_back(make_stream_variant<Format, strategy_type>(
        values, count, lut, sink, multiprocessors));
  });
}

template <typename Format>
void append_dot_variants(std::vector<timed_variant> &variants,
                         const storage::storage_type_t<Format> *left,
                         const storage::storage_type_t<Format> *right,
                         std::size_t count, const float *lut, double *partials,
                         double *result, int multiprocessors) {
  for_each_decoder<Format>([&](auto strategy) {
    using strategy_type = decltype(strategy);
    variants.push_back(make_dot_variant<Format, strategy_type>(
        left, right, count, lut, partials, result, multiprocessors));
  });
}

template <typename Format>
void append_gemv_variants(std::vector<timed_variant> &variants,
                          const storage::storage_type_t<Format> *matrix,
                          const storage::storage_type_t<Format> *vector,
                          std::size_t rows, std::size_t columns,
                          const float *lut, double *result) {
  for_each_decoder<Format>([&](auto strategy) {
    using strategy_type = decltype(strategy);
    variants.push_back(make_gemv_variant<Format, strategy_type>(
        matrix, vector, rows, columns, lut, result));
  });
}

void run_distribution(const options &settings, const device_info &device,
                      distribution kind, double *source_dot_left,
                      double *source_dot_right, double *source_matrix,
                      double *source_vector, encoded_data<storage::e2m5> &e2,
                      encoded_data<storage::e3m4> &e3, const float *e2_lut,
                      const float *e3_lut, sample_output &output,
                      event_timer &timer, random_generator &generator) {
  const auto max_dot = size_from_power(settings.dot_powers.back());
  const auto max_columns = size_from_power(settings.gemv_powers.back());
  const auto max_matrix = settings.gemv_rows * max_columns;
  const auto tag = static_cast<std::uint64_t>(kind);
  generator.fill(source_dot_left, max_dot, kind,
                 mix(settings.base_seed ^ tag ^ 0x444f544cULL));
  generator.fill(source_dot_right, max_dot, kind,
                 mix(settings.base_seed ^ tag ^ 0x444f5452ULL));
  generator.fill(source_matrix, max_matrix, kind,
                 mix(settings.base_seed ^ tag ^ 0x47454d41ULL));
  generator.fill(source_vector, max_columns, kind,
                 mix(settings.base_seed ^ tag ^ 0x47454d58ULL));
  encode_dataset<storage::e2m5>(
      source_dot_left, source_dot_right, source_matrix, source_vector, max_dot,
      max_matrix, max_columns, device.multiprocessors, e2);
  encode_dataset<storage::e3m4>(
      source_dot_left, source_dot_right, source_matrix, source_vector, max_dot,
      max_matrix, max_columns, device.multiprocessors, e3);
  CUDA_CHECK(cudaDeviceSynchronize());

  const auto max_blocks = device.multiprocessors * 16;
  const auto register_blocks = device.multiprocessors * 4;
  device_buffer<double> double_sink{std::max<std::size_t>(
      settings.gemv_rows,
      static_cast<std::size_t>(register_blocks) * performance::block_threads)};
  device_buffer<unsigned long long> integer_sink{max_blocks};
  device_buffer<double> partials{max_blocks};
  device_buffer<double> dot_result{1};

  std::cout << name(kind) << '\n';
  {
    std::vector<timed_variant> variants;
    append_register_variants<storage::fp64_e11m52>(
        variants, source_dot_left, max_dot, nullptr, double_sink.get(),
        register_blocks, settings.decode_repeats);
    append_register_variants<storage::e2m5>(
        variants, e2.dot_left.get(), max_dot, e2_lut, double_sink.get(),
        register_blocks, settings.decode_repeats);
    append_register_variants<storage::e3m4>(
        variants, e3.dot_left.get(), max_dot, e3_lut, double_sink.get(),
        register_blocks, settings.decode_repeats);
    measure_variants(variants, settings, kind, output, timer);
  }

  for (const auto power : settings.dot_powers) {
    const auto count = size_from_power(power);
    {
      std::vector<timed_variant> variants;
      variants.push_back(make_load_variant<storage::fp64_e11m52, 1>(
          source_dot_left, count, integer_sink.get(), device.multiprocessors));
      variants.push_back(make_load_variant<storage::fp64_e11m52, 4>(
          source_dot_left, count, integer_sink.get(), device.multiprocessors));
      variants.push_back(make_load_variant<storage::e2m5, 1>(
          e2.dot_left.get(), count, integer_sink.get(),
          device.multiprocessors));
      variants.push_back(make_load_variant<storage::e2m5, 4>(
          e2.dot_left.get(), count, integer_sink.get(),
          device.multiprocessors));
      variants.push_back(make_load_variant<storage::e3m4, 1>(
          e3.dot_left.get(), count, integer_sink.get(),
          device.multiprocessors));
      variants.push_back(make_load_variant<storage::e3m4, 4>(
          e3.dot_left.get(), count, integer_sink.get(),
          device.multiprocessors));
      measure_variants(variants, settings, kind, output, timer);
    }
    {
      std::vector<timed_variant> variants;
      append_stream_variants<storage::fp64_e11m52>(
          variants, source_dot_left, count, nullptr, double_sink.get(),
          device.multiprocessors);
      append_stream_variants<storage::e2m5>(variants, e2.dot_left.get(), count,
                                            e2_lut, double_sink.get(),
                                            device.multiprocessors);
      append_stream_variants<storage::e3m4>(variants, e3.dot_left.get(), count,
                                            e3_lut, double_sink.get(),
                                            device.multiprocessors);
      measure_variants(variants, settings, kind, output, timer);
    }
    {
      std::vector<timed_variant> variants;
      append_dot_variants<storage::fp64_e11m52>(
          variants, source_dot_left, source_dot_right, count, nullptr,
          partials.get(), dot_result.get(), device.multiprocessors);
      append_dot_variants<storage::e2m5>(
          variants, e2.dot_left.get(), e2.dot_right.get(), count, e2_lut,
          partials.get(), dot_result.get(), device.multiprocessors);
      append_dot_variants<storage::e3m4>(
          variants, e3.dot_left.get(), e3.dot_right.get(), count, e3_lut,
          partials.get(), dot_result.get(), device.multiprocessors);
      measure_variants(variants, settings, kind, output, timer);
    }
    std::cout << "  DOT N=2^" << power << " complete\n";
  }

  for (const auto power : settings.gemv_powers) {
    const auto columns = size_from_power(power);
    std::vector<timed_variant> variants;
    append_gemv_variants<storage::fp64_e11m52>(
        variants, source_matrix, source_vector, settings.gemv_rows, columns,
        nullptr, double_sink.get());
    append_gemv_variants<storage::e2m5>(variants, e2.matrix.get(),
                                        e2.vector.get(), settings.gemv_rows,
                                        columns, e2_lut, double_sink.get());
    append_gemv_variants<storage::e3m4>(variants, e3.matrix.get(),
                                        e3.vector.get(), settings.gemv_rows,
                                        columns, e3_lut, double_sink.get());
    measure_variants(variants, settings, kind, output, timer);
    std::cout << "  GEMV N=2^" << power << " complete\n";
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
    throw benchmark_error("this experiment is calibrated for an sm_90 H200");
  }
  create_parent(settings.output);
  create_parent(settings.validation_output);
  std::cout << "E2/E3 decoder optimization benchmark\n"
            << "GPU: " << device.name << " (" << device.capability << ")\n";

  const auto e2_host_lut = make_lut<storage::e2m5>();
  const auto e3_host_lut = make_lut<storage::e3m4>();
  device_buffer<float> e2_lut{e2_host_lut.size()};
  device_buffer<float> e3_lut{e3_host_lut.size()};
  CUDA_CHECK(cudaMemcpy(e2_lut.get(), e2_host_lut.data(), sizeof(e2_host_lut),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(e3_lut.get(), e3_host_lut.data(), sizeof(e3_host_lut),
                        cudaMemcpyHostToDevice));
  run_validation(settings.validation_output, e2_lut.get(), e3_lut.get());
  std::cout << "All 256 encodings validated for every E2/E3 strategy.\n";

  const auto max_dot = size_from_power(settings.dot_powers.back());
  const auto max_columns = size_from_power(settings.gemv_powers.back());
  const auto max_matrix = settings.gemv_rows * max_columns;
  device_buffer<double> source_dot_left{max_dot};
  device_buffer<double> source_dot_right{max_dot};
  device_buffer<double> source_matrix{max_matrix};
  device_buffer<double> source_vector{max_columns};
  encoded_data<storage::e2m5> e2{max_dot, max_matrix, max_columns};
  encoded_data<storage::e3m4> e3{max_dot, max_matrix, max_columns};
  sample_output output{settings.output, device};
  event_timer timer;
  random_generator generator;
  for (const auto kind :
       {distribution::uniform_0_1, distribution::normal_0_1}) {
    run_distribution(settings, device, kind, source_dot_left.get(),
                     source_dot_right.get(), source_matrix.get(),
                     source_vector.get(), e2, e3, e2_lut.get(), e3_lut.get(),
                     output, timer, generator);
  }
  std::cout << "Wrote " << settings.output << '\n'
            << "Wrote " << settings.validation_output << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return EXIT_FAILURE;
}
