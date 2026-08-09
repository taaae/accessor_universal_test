#include "e2e3_decoder_strategies.cuh"
#include "storage_kernels.cuh"

#include <cub/block/block_reduce.cuh>
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
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace decoder = aut::e2e3_strategies;
namespace kernels = aut::kernels;
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
  std::vector<int> dot_powers{12, 16, 20, 24, 27};
  std::vector<int> gemv_powers{8, 10, 12, 14, 16};
  std::size_t gemv_rows{1024};
  int warmup{10};
  int rounds{3};
  int samples{5};
  double target_sample_ms{15.0};
  std::uint64_t base_seed{0x243f6a8885a308d3ULL};
  std::string output{"e2e3_strategy_samples.csv"};
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
    if (power > 28) {
      throw benchmark_error(option + " powers must be in [1,28]");
    }
    result.push_back(power);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  if (result.empty()) {
    throw benchmark_error(option + " must not be empty");
  }
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
    } else if (argument == "--base-seed") {
      result.base_seed = parse_positive_size(value(), argument);
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--help") {
      std::cout << "Usage: e2e3_strategy_bench [options]\n"
                << "  --dot-powers P,... --gemv-powers P,... --gemv-rows M\n"
                << "  --warmup N --rounds N --samples N --target-sample-ms X\n"
                << "  --base-seed N --output FILE\n";
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
                                     std::uint8_t *encoded, std::size_t count) {
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto index = first; index < count; index += stride) {
    encoded[index] = storage::encode<Format>(source[index]);
  }
}

template <typename Format>
void encode_values(const double *source, std::uint8_t *encoded,
                   std::size_t count, int multiprocessors) {
  const auto wanted = (count + 255) / 256;
  const auto blocks = static_cast<unsigned>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
  encode_values_kernel<Format><<<blocks, 256>>>(source, encoded, count);
  CUDA_CHECK(cudaGetLastError());
}

int work_blocks(std::size_t count, int lanes, int multiprocessors) {
  const auto packs = (count + static_cast<std::size_t>(lanes) - 1) / lanes;
  const auto wanted =
      (packs + decoder::block_threads - 1) / decoder::block_threads;
  return static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
}

__global__ void raw_fp64_dot_map_reduce(const double *left, const double *right,
                                        std::size_t count, double *partials) {
  using block_reduce = cub::BlockReduce<double, decoder::block_threads>;
  __shared__ typename block_reduce::TempStorage reduction_storage;
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  double sum{};
  for (auto index = first; index < count; index += stride) {
    sum = fma(left[index], right[index], sum);
  }
  const auto reduced = block_reduce(reduction_storage).Sum(sum);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

__global__ void raw_fp64_gemv(const double *matrix, const double *vector,
                              std::size_t rows, std::size_t columns,
                              std::size_t leading_dimension, double *result) {
  using block_reduce = cub::BlockReduce<double, decoder::block_threads>;
  __shared__ typename block_reduce::TempStorage reduction_storage;
  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  double sum{};
  for (auto column = static_cast<std::size_t>(threadIdx.x); column < columns;
       column += blockDim.x) {
    sum = fma(matrix[row * leading_dimension + column], vector[column], sum);
  }
  const auto reduced = block_reduce(reduction_storage).Sum(sum);
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
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
    for (std::size_t code = 0; code < 256; ++code) {
      const auto value =
          storage::decode<Format>(static_cast<std::uint8_t>(code));
      std::uint64_t bits{};
      std::memcpy(&bits, &value, sizeof(bits));
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
      std::uint64_t bits{};
      std::memcpy(&bits, &value, sizeof(bits));
      subnormal_high_word[fraction] = static_cast<std::uint32_t>(bits >> 32);
    }
    constexpr auto count = std::size_t{1} << (1 + Format::exponent_bits);
    for (std::size_t upper = 0; upper < count; ++upper) {
      const auto raw =
          static_cast<std::uint8_t>(upper << Format::fraction_bits);
      const auto value = storage::decode<Format>(raw);
      std::memcpy(&exponent_prefix[upper], &value, sizeof(value));
    }
  }
};

template <typename T, std::size_t Size>
void upload(device_buffer<T> &target, const std::array<T, Size> &source) {
  CUDA_CHECK(cudaMemcpy(target.get(), source.data(), sizeof(source),
                        cudaMemcpyHostToDevice));
}

template <typename Format> class device_tables {
public:
  explicit device_tables(const host_tables<Format> &host)
      : fp32_{256}, fp64_{256}, prefix16_{256}, prefix32_{256},
        exponent_prefix_{16}, high_word_{256}, subnormal_high_word_{32} {
    upload(fp32_, host.fp32);
    upload(fp64_, host.fp64);
    upload(prefix16_, host.prefix16);
    upload(prefix32_, host.prefix32);
    upload(exponent_prefix_, host.exponent_prefix);
    upload(high_word_, host.high_word);
    upload(subnormal_high_word_, host.subnormal_high_word);
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

template <typename Strategy> const char *table_location_name() {
  if constexpr (Strategy::kind == decoder::decode_kind::lut_fp32 ||
                Strategy::kind == decoder::decode_kind::lut_fp64 ||
                Strategy::kind == decoder::decode_kind::lut_prefix ||
                Strategy::kind == decoder::decode_kind::lut_subnormal ||
                Strategy::kind == decoder::decode_kind::lut_high_word ||
                Strategy::kind ==
                    decoder::decode_kind::lut_high_word_swizzled) {
    return Strategy::location == decoder::table_location::shared ? "shared"
                                                                 : "global";
  }
  return "none";
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

struct work_model {
  std::string format;
  int storage_bits{};
  std::string strategy;
  std::string decode_kind;
  std::string table_location;
  int lanes{};
  std::size_t lookup_entry_bytes{};
  std::size_t shared_table_bytes{};
  bool pipelined{};
  std::string component;
  std::size_t n{};
  std::size_t m{};
  int blocks{};
  double main_array_unique_bytes{};
  double main_array_requested_bytes{};
  double useful_flops{};
};

struct timed_variant {
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
    stream_ << "gpu,compute_capability,distribution,format,storage_bits,"
               "strategy,decode_kind,table_location,lanes,lookup_entry_bytes,"
               "shared_table_bytes,pipelined,component,n,m,blocks,threads,"
               "warmup,round,sample,order_position,iterations,total_time_ms,"
               "time_ms,main_array_unique_bytes,main_array_requested_bytes,"
               "useful_flops,main_array_unique_gb_per_s,"
               "main_array_requested_gb_per_s,useful_gflop_per_s\n";
    stream_ << std::setprecision(17);
  }

  void write(distribution kind, const work_model &model, int warmup, int round,
             int sample, int order_position, std::size_t iterations,
             double total_ms) {
    const auto time_ms = total_ms / iterations;
    const auto seconds = time_ms * 1.0e-3;
    const auto rate = [&](double amount, double scale) {
      return seconds > 0.0 ? amount / seconds / scale : 0.0;
    };
    stream_ << device_.name << ',' << device_.capability << ',' << name(kind)
            << ',' << model.format << ',' << model.storage_bits << ','
            << model.strategy << ',' << model.decode_kind << ','
            << model.table_location << ',' << model.lanes << ','
            << model.lookup_entry_bytes << ',' << model.shared_table_bytes
            << ',' << model.pipelined << ',' << model.component << ','
            << model.n << ',' << model.m << ',' << model.blocks << ','
            << decoder::block_threads << ',' << warmup << ',' << round << ','
            << sample << ',' << order_position << ',' << iterations << ','
            << total_ms << ',' << time_ms << ','
            << model.main_array_unique_bytes << ','
            << model.main_array_requested_bytes << ',' << model.useful_flops
            << ',' << rate(model.main_array_unique_bytes, 1.0e9) << ','
            << rate(model.main_array_requested_bytes, 1.0e9) << ','
            << rate(model.useful_flops, 1.0e9) << '\n';
  }

  void flush() { stream_.flush(); }

private:
  std::ofstream stream_;
  device_info device_;
};

template <typename Format, typename Strategy>
work_model strategy_model(const char *component, std::size_t n, std::size_t m,
                          int blocks) {
  const auto storage_bytes = sizeof(std::uint8_t);
  const auto unique = component == std::string{"dot"}
                          ? 2.0 * n * storage_bytes +
                                2.0 * blocks * sizeof(double) + sizeof(double)
                          : (m * n + n) * storage_bytes + m * sizeof(double);
  const auto requested = component == std::string{"dot"}
                             ? unique
                             : 2.0 * m * n * storage_bytes + m * sizeof(double);
  return {Format::name,
          Format::total_bits,
          strategy_name<Strategy>(),
          kind_name(Strategy::kind),
          table_location_name<Strategy>(),
          Strategy::lanes,
          decoder::lookup_entry_bytes_v<Format, Strategy>,
          decoder::shared_table_bytes_v<Format, Strategy>,
          Strategy::pipelined,
          component,
          n,
          m,
          blocks,
          unique,
          requested,
          2.0 * n * m};
}

work_model raw_model(const char *component, std::size_t n, std::size_t m,
                     int blocks) {
  const auto unique = component == std::string{"dot"}
                          ? 2.0 * n * sizeof(double) +
                                2.0 * blocks * sizeof(double) + sizeof(double)
                          : (m * n + n) * sizeof(double) + m * sizeof(double);
  const auto requested =
      component == std::string{"dot"}
          ? unique
          : 2.0 * m * n * sizeof(double) + m * sizeof(double);
  return {
      "fp64",    64,         "raw_pointer_x1", "none", "none", 1,      0,
      0,         false,      component,        n,      m,      blocks, unique,
      requested, 2.0 * n * m};
}

template <typename Format, typename Strategy>
timed_variant make_dot_variant(const std::uint8_t *left,
                               const std::uint8_t *right, std::size_t count,
                               decoder::table_bundle tables, double *partials,
                               double *result, int multiprocessors) {
  const auto blocks = work_blocks(count, Strategy::lanes, multiprocessors);
  constexpr auto shared_bytes = decoder::shared_table_bytes_v<Format, Strategy>;
  return {strategy_model<Format, Strategy>("dot", count, 1, blocks), [=] {
            decoder::dot_map_reduce<Format, Strategy>
                <<<blocks, decoder::block_threads, shared_bytes>>>(
                    left, right, count, tables, partials);
            kernels::storage_dot_finalize<<<1, decoder::block_threads>>>(
                partials, blocks, result);
          }};
}

timed_variant make_raw_dot_variant(const double *left, const double *right,
                                   std::size_t count, double *partials,
                                   double *result, int multiprocessors) {
  const auto blocks = work_blocks(count, 1, multiprocessors);
  return {raw_model("dot", count, 1, blocks), [=] {
            raw_fp64_dot_map_reduce<<<blocks, decoder::block_threads>>>(
                left, right, count, partials);
            kernels::storage_dot_finalize<<<1, decoder::block_threads>>>(
                partials, blocks, result);
          }};
}

template <typename Format, typename Strategy>
timed_variant make_gemv_variant(const std::uint8_t *matrix,
                                const std::uint8_t *vector, std::size_t rows,
                                std::size_t columns,
                                decoder::table_bundle tables, double *result) {
  constexpr auto shared_bytes = decoder::shared_table_bytes_v<Format, Strategy>;
  return {strategy_model<Format, Strategy>("gemv", columns, rows,
                                           static_cast<int>(rows)),
          [=] {
            decoder::gemv<Format, Strategy>
                <<<static_cast<unsigned>(rows), decoder::block_threads,
                   shared_bytes>>>(matrix, vector, rows, columns, columns,
                                   tables, result);
          }};
}

timed_variant make_raw_gemv_variant(const double *matrix, const double *vector,
                                    std::size_t rows, std::size_t columns,
                                    double *result) {
  return {
      raw_model("gemv", columns, rows, static_cast<int>(rows)), [=] {
        raw_fp64_gemv<<<static_cast<unsigned>(rows), decoder::block_threads>>>(
            matrix, vector, rows, columns, columns, result);
      }};
}

void measure_variants(std::vector<timed_variant> &variants,
                      const options &settings, distribution kind,
                      std::uint64_t order_seed, sample_output &output,
                      event_timer &timer) {
  if (variants.size() != 85) {
    throw benchmark_error("internal error: expected 85 timed variants");
  }
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

  std::vector<std::size_t> order(variants.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 engine{order_seed};
  for (int round = 0; round < settings.rounds; ++round) {
    for (int sample = 0; sample < settings.samples; ++sample) {
      std::shuffle(order.begin(), order.end(), engine);
      for (std::size_t position = 0; position < order.size(); ++position) {
        auto &variant = variants[order[position]];
        const auto elapsed = timer.measure(variant.launch, variant.iterations);
        output.write(kind, variant.model, settings.warmup, round, sample,
                     static_cast<int>(position), variant.iterations, elapsed);
      }
    }
  }
  output.flush();
}

template <typename Format>
void append_dot_strategies(std::vector<timed_variant> &variants,
                           const std::uint8_t *left, const std::uint8_t *right,
                           std::size_t count, decoder::table_bundle tables,
                           double *partials, double *result,
                           int multiprocessors) {
  for_each_strategy<Format>([&](auto strategy) {
    using strategy_type = decltype(strategy);
    variants.push_back(make_dot_variant<Format, strategy_type>(
        left, right, count, tables, partials, result, multiprocessors));
  });
}

template <typename Format>
void append_gemv_strategies(std::vector<timed_variant> &variants,
                            const std::uint8_t *matrix,
                            const std::uint8_t *vector, std::size_t rows,
                            std::size_t columns, decoder::table_bundle tables,
                            double *result) {
  for_each_strategy<Format>([&](auto strategy) {
    using strategy_type = decltype(strategy);
    variants.push_back(make_gemv_variant<Format, strategy_type>(
        matrix, vector, rows, columns, tables, result));
  });
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
    throw benchmark_error("this benchmark is calibrated for sm_90");
  }
  create_parent(settings.output);
  std::cout << "E2M5/E3M4 full strategy benchmark\n"
            << "GPU: " << device.name << " (" << device.capability << ")\n";

  const auto max_dot = size_from_power(settings.dot_powers.back());
  const auto max_columns = size_from_power(settings.gemv_powers.back());
  const auto max_matrix = settings.gemv_rows * max_columns;
  device_buffer<double> source_dot_left{max_dot};
  device_buffer<double> source_dot_right{max_dot};
  device_buffer<double> source_matrix{max_matrix};
  device_buffer<double> source_vector{max_columns};
  device_buffer<std::uint8_t> e2_dot_left{max_dot};
  device_buffer<std::uint8_t> e2_dot_right{max_dot};
  device_buffer<std::uint8_t> e2_matrix{max_matrix};
  device_buffer<std::uint8_t> e2_vector{max_columns};
  device_buffer<std::uint8_t> e3_dot_left{max_dot};
  device_buffer<std::uint8_t> e3_dot_right{max_dot};
  device_buffer<std::uint8_t> e3_matrix{max_matrix};
  device_buffer<std::uint8_t> e3_vector{max_columns};
  const auto max_blocks = device.multiprocessors * 16;
  device_buffer<double> partials{static_cast<std::size_t>(max_blocks)};
  device_buffer<double> result{std::max<std::size_t>(1, settings.gemv_rows)};

  const host_tables<storage::e2m5> e2_host_tables;
  const host_tables<storage::e3m4> e3_host_tables;
  const device_tables<storage::e2m5> e2_tables{e2_host_tables};
  const device_tables<storage::e3m4> e3_tables{e3_host_tables};
  random_generator random;
  sample_output output{settings.output, device};
  event_timer timer;

  for (const auto kind :
       {distribution::uniform_0_1, distribution::normal_0_1}) {
    const auto distribution_seed =
        mix(settings.base_seed ^ static_cast<std::uint64_t>(kind));
    random.fill(source_dot_left.get(), max_dot, kind,
                mix(distribution_seed ^ 0x01));
    random.fill(source_dot_right.get(), max_dot, kind,
                mix(distribution_seed ^ 0x02));
    random.fill(source_matrix.get(), max_matrix, kind,
                mix(distribution_seed ^ 0x03));
    random.fill(source_vector.get(), max_columns, kind,
                mix(distribution_seed ^ 0x04));

    encode_values<storage::e2m5>(source_dot_left.get(), e2_dot_left.get(),
                                 max_dot, device.multiprocessors);
    encode_values<storage::e2m5>(source_dot_right.get(), e2_dot_right.get(),
                                 max_dot, device.multiprocessors);
    encode_values<storage::e2m5>(source_matrix.get(), e2_matrix.get(),
                                 max_matrix, device.multiprocessors);
    encode_values<storage::e2m5>(source_vector.get(), e2_vector.get(),
                                 max_columns, device.multiprocessors);
    encode_values<storage::e3m4>(source_dot_left.get(), e3_dot_left.get(),
                                 max_dot, device.multiprocessors);
    encode_values<storage::e3m4>(source_dot_right.get(), e3_dot_right.get(),
                                 max_dot, device.multiprocessors);
    encode_values<storage::e3m4>(source_matrix.get(), e3_matrix.get(),
                                 max_matrix, device.multiprocessors);
    encode_values<storage::e3m4>(source_vector.get(), e3_vector.get(),
                                 max_columns, device.multiprocessors);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << name(kind) << '\n';
    for (const auto power : settings.dot_powers) {
      const auto count = size_from_power(power);
      std::vector<timed_variant> variants;
      variants.reserve(85);
      variants.push_back(make_raw_dot_variant(
          source_dot_left.get(), source_dot_right.get(), count, partials.get(),
          result.get(), device.multiprocessors));
      append_dot_strategies<storage::e2m5>(
          variants, e2_dot_left.get(), e2_dot_right.get(), count,
          e2_tables.bundle(), partials.get(), result.get(),
          device.multiprocessors);
      append_dot_strategies<storage::e3m4>(
          variants, e3_dot_left.get(), e3_dot_right.get(), count,
          e3_tables.bundle(), partials.get(), result.get(),
          device.multiprocessors);
      measure_variants(variants, settings, kind,
                       mix(distribution_seed ^ count ^ 0xd07), output, timer);
      std::cout << "  DOT N=2^" << power << " complete\n";
    }

    for (const auto power : settings.gemv_powers) {
      const auto columns = size_from_power(power);
      std::vector<timed_variant> variants;
      variants.reserve(85);
      variants.push_back(
          make_raw_gemv_variant(source_matrix.get(), source_vector.get(),
                                settings.gemv_rows, columns, result.get()));
      append_gemv_strategies<storage::e2m5>(
          variants, e2_matrix.get(), e2_vector.get(), settings.gemv_rows,
          columns, e2_tables.bundle(), result.get());
      append_gemv_strategies<storage::e3m4>(
          variants, e3_matrix.get(), e3_vector.get(), settings.gemv_rows,
          columns, e3_tables.bundle(), result.get());
      measure_variants(variants, settings, kind,
                       mix(distribution_seed ^ columns ^ 0x6e6d76), output,
                       timer);
      std::cout << "  GEMV M=" << settings.gemv_rows << ", N=2^" << power
                << " complete\n";
    }
  }

  std::cout << "Wrote " << settings.output << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return EXIT_FAILURE;
}
