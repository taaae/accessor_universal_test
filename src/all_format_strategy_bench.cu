#include "format_decoder_strategies.cuh"
#include "storage_kernels.cuh"

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fs = aut::format_strategies;
namespace kernels = aut::kernels;
namespace storage = aut::storage;
namespace decoder = aut::decoder;

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
  explicit device_buffer(std::size_t count) {
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

const std::vector<std::string> supported_formats{
    "e1m6",  "fp8_e4m3",   "fp8_e5m2",   "e1m14",  "e2m13",
    "e3m12", "fp16_e5m10", "bf16_e8m7",  "e11m4",  "e1m30",
    "e2m29", "e3m28",      "fp32_e8m23", "e11m20",
};

struct options {
  std::vector<int> dot_powers{12, 16, 20, 24, 27};
  std::vector<int> gemv_powers{8, 10, 12, 14, 16};
  std::vector<std::string> formats = supported_formats;
  std::size_t gemv_rows{1024};
  int warmup{10};
  int rounds{3};
  int samples{5};
  double target_sample_ms{15.0};
  std::uint64_t base_seed{0x243f6a8885a308d3ULL};
  std::string output{"all_format_strategy_samples.csv"};
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

std::vector<std::string> split_list(const std::string &text,
                                    const std::string &option) {
  std::vector<std::string> result;
  std::stringstream input{text};
  std::string token;
  while (std::getline(input, token, ',')) {
    if (token.empty()) {
      throw benchmark_error(option + " contains an empty entry");
    }
    result.push_back(token);
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  if (result.empty()) {
    throw benchmark_error(option + " must not be empty");
  }
  return result;
}

std::vector<int> parse_powers(const std::string &text,
                              const std::string &option) {
  std::vector<int> result;
  for (const auto &token : split_list(text, option)) {
    const auto power = parse_positive_int(token, option);
    if (power > 28) {
      throw benchmark_error(option + " powers must be in [1,28]");
    }
    result.push_back(power);
  }
  std::sort(result.begin(), result.end());
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
    } else if (argument == "--formats") {
      result.formats = split_list(value(), argument);
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
      std::cout << "Usage: all_format_strategy_bench [options]\n"
                << "  --formats NAME,...\n"
                << "  --dot-powers P,... --gemv-powers P,... --gemv-rows M\n"
                << "  --warmup N --rounds N --samples N --target-sample-ms X\n"
                << "  --base-seed N --output FILE\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw benchmark_error("unknown argument: " + argument);
    }
  }
  const std::set<std::string> supported(supported_formats.begin(),
                                        supported_formats.end());
  for (const auto &format : result.formats) {
    if (!supported.count(format)) {
      throw benchmark_error("unsupported format: " + format);
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
                   storage::storage_type_t<Format> *encoded, std::size_t count,
                   int multiprocessors) {
  const auto wanted = (count + 255) / 256;
  const auto blocks = static_cast<unsigned>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
  encode_values_kernel<Format><<<blocks, 256>>>(source, encoded, count);
  CUDA_CHECK(cudaGetLastError());
}

int work_blocks(std::size_t count, int lanes, int multiprocessors) {
  const auto packs = (count + static_cast<std::size_t>(lanes) - 1) / lanes;
  const auto wanted = (packs + fs::block_threads - 1) / fs::block_threads;
  return static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
}

__global__ void raw_fp64_dot_map_reduce(const double *left, const double *right,
                                        std::size_t count, double *partials) {
  using block_reduce = cub::BlockReduce<double, fs::block_threads>;
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
  using block_reduce = cub::BlockReduce<double, fs::block_threads>;
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

std::uint64_t bits(double value) {
  std::uint64_t result{};
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

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
  return __nv_bfloat16{__nv_bfloat16_raw{static_cast<unsigned short>(raw)}};
}

template <>
storage::storage_type_t<storage::fp32_e8m23>
host_storage_from_raw<storage::fp32_e8m23>(std::uint32_t raw) {
  return decoder::bits_to_float(raw);
}

template <typename Format> class format_tables {
public:
  using layout = fs::format_layout_t<Format>;
  static constexpr auto full_count =
      layout::total_bits <= 16 ? (std::size_t{1} << layout::total_bits) : 0;
  static constexpr auto subnormal_count =
      layout::total_bits <= 16 ? (std::size_t{1} << layout::fraction_bits) : 0;
  static constexpr auto prefix_count = std::size_t{1}
                                       << (layout::exponent_bits + 1);
  static constexpr auto pair_count =
      layout::total_bits == 8 ? (std::size_t{1} << 16) : 0;

  format_tables()
      : full_(full_count), subnormal_(subnormal_count), prefix_(prefix_count),
        pair_(pair_count), device_full_(full_count),
        device_subnormal_(subnormal_count), device_prefix_(prefix_count),
        device_pair_(pair_count) {
    for (std::uint32_t raw = 0; raw < full_count; ++raw) {
      const auto value =
          storage::decode<Format>(host_storage_from_raw<Format>(raw));
      full_[raw] = static_cast<std::uint32_t>(bits(value) >> 32);
    }
    for (std::uint32_t fraction = 0; fraction < subnormal_count; ++fraction) {
      const auto value =
          storage::decode<Format>(host_storage_from_raw<Format>(fraction));
      subnormal_[fraction] = static_cast<std::uint32_t>(bits(value) >> 32);
    }
    for (std::uint32_t prefix = 0; prefix < prefix_count; ++prefix) {
      const auto raw = prefix << layout::fraction_bits;
      const auto value =
          storage::decode<Format>(host_storage_from_raw<Format>(raw));
      prefix_[prefix] = static_cast<std::uint32_t>(bits(value) >> 32);
    }
    if constexpr (layout::total_bits == 8) {
      for (std::uint32_t high = 0; high < 256; ++high) {
        for (std::uint32_t low = 0; low < 256; ++low) {
          pair_[low + (high << 8)] = {full_[low], full_[high]};
        }
      }
    }
    upload(device_full_, full_);
    upload(device_subnormal_, subnormal_);
    upload(device_prefix_, prefix_);
    upload(device_pair_, pair_);
  }

  fs::table_bundle bundle() const {
    return {device_full_.get(), device_subnormal_.get(), device_prefix_.get(),
            device_pair_.get()};
  }

private:
  template <typename T>
  static void upload(device_buffer<T> &target, const std::vector<T> &source) {
    if (!source.empty()) {
      CUDA_CHECK(cudaMemcpy(target.get(), source.data(),
                            source.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  std::vector<std::uint32_t> full_;
  std::vector<std::uint32_t> subnormal_;
  std::vector<std::uint32_t> prefix_;
  std::vector<uint2> pair_;
  device_buffer<std::uint32_t> device_full_;
  device_buffer<std::uint32_t> device_subnormal_;
  device_buffer<std::uint32_t> device_prefix_;
  device_buffer<uint2> device_pair_;
};

const char *kind_name(fs::decode_kind kind) {
  switch (kind) {
  case fs::decode_kind::generic:
    return "generic";
  case fs::decode_kind::direct_words_branchy:
    return "direct_words_branchy";
  case fs::decode_kind::direct_words_masked:
    return "direct_words_masked";
  case fs::decode_kind::fp32_bits:
    return "fp32_bits";
  case fs::decode_kind::e1_integer:
    return "e1_integer";
  case fs::decode_kind::prefix_word:
    return "prefix_word";
  case fs::decode_kind::full_high_lut:
    return "full_high_lut";
  case fs::decode_kind::subnormal_high_lut:
    return "subnormal_high_lut";
  case fs::decode_kind::prefix_high_lut:
    return "prefix_high_lut";
  case fs::decode_kind::native_direct:
    return "native_direct";
  case fs::decode_kind::native_fp32:
    return "native_fp32";
  case fs::decode_kind::native_packed:
    return "native_packed";
  case fs::decode_kind::native_half2:
    return "native_half2";
  case fs::decode_kind::pair_high_lut:
    return "pair_high_lut";
  case fs::decode_kind::full_high_lut_swizzled:
    return "full_high_lut_swizzled";
  }
  return "unknown";
}

template <typename Strategy> const char *table_location_name() {
  if constexpr (fs::uses_table_v<Strategy>) {
    return Strategy::location == fs::table_location::shared ? "shared"
                                                            : "global";
  }
  return "none";
}

template <typename Strategy> const char *unpack_name() {
  return Strategy::unpack == fs::unpack_kind::byte_permute ? "byte_permute"
                                                           : "shift_mask";
}

template <typename Format, typename Strategy>
constexpr std::size_t lookup_entry_bytes() {
  if constexpr (Strategy::kind == fs::decode_kind::pair_high_lut) {
    return sizeof(uint2);
  } else if constexpr (fs::uses_table_v<Strategy>) {
    return sizeof(std::uint32_t);
  }
  return 0;
}

template <fs::decode_kind Kind, int Lanes,
          fs::table_location Location = fs::table_location::global_read_only,
          fs::unpack_kind Unpack = fs::unpack_kind::shift_mask>
using strategy = fs::strategy<Kind, Lanes, Location, Unpack>;

template <fs::decode_kind Kind, int Lanes,
          fs::table_location Location = fs::table_location::global_read_only,
          fs::unpack_kind Unpack = fs::unpack_kind::shift_mask,
          typename Callback>
void add(Callback &callback, const char *name) {
  callback(strategy<Kind, Lanes, Location, Unpack>{}, name);
}

template <typename Callback> void add_generic_148(Callback &callback) {
  add<fs::decode_kind::generic, 1>(callback, "generic_x1");
  add<fs::decode_kind::generic, 4>(callback, "generic_x4");
  add<fs::decode_kind::generic, 8>(callback, "generic_x8");
}

template <typename Callback> void add_generic_1248(Callback &callback) {
  add<fs::decode_kind::generic, 1>(callback, "generic_x1");
  add<fs::decode_kind::generic, 2>(callback, "generic_x2");
  add<fs::decode_kind::generic, 4>(callback, "generic_x4");
  add<fs::decode_kind::generic, 8>(callback, "generic_x8");
}

template <fs::decode_kind Kind, typename Callback>
void add_widths_148(Callback &callback, const char *x1, const char *x4,
                    const char *x8) {
  add<Kind, 1>(callback, x1);
  add<Kind, 4>(callback, x4);
  add<Kind, 8>(callback, x8);
}

template <fs::decode_kind Kind, typename Callback>
void add_widths_1248(Callback &callback, const char *x1, const char *x2,
                     const char *x4, const char *x8) {
  add<Kind, 1>(callback, x1);
  add<Kind, 2>(callback, x2);
  add<Kind, 4>(callback, x4);
  add<Kind, 8>(callback, x8);
}

template <typename Callback> void add_words_148(Callback &callback) {
  add_widths_148<fs::decode_kind::direct_words_branchy>(
      callback, "word_branchy_x1", "word_branchy_x4", "word_branchy_x8");
  add_widths_148<fs::decode_kind::direct_words_masked>(
      callback, "word_masked_x1", "word_masked_x4", "word_masked_x8");
}

template <typename Callback> void add_words_1248(Callback &callback) {
  add_widths_1248<fs::decode_kind::direct_words_branchy>(
      callback, "word_branchy_x1", "word_branchy_x2", "word_branchy_x4",
      "word_branchy_x8");
  add_widths_1248<fs::decode_kind::direct_words_masked>(
      callback, "word_masked_x1", "word_masked_x2", "word_masked_x4",
      "word_masked_x8");
}

template <typename Callback> void add_fp32_148(Callback &callback) {
  add_widths_148<fs::decode_kind::fp32_bits>(callback, "fp32_bits_x1",
                                             "fp32_bits_x4", "fp32_bits_x8");
}

template <typename Callback> void add_full_high_148(Callback &callback) {
  add_widths_148<fs::decode_kind::full_high_lut>(
      callback, "full_high_l2_x1", "full_high_l2_x4", "full_high_l2_x8");
}

template <typename Callback> void add_prefix_tables(Callback &callback) {
  add<fs::decode_kind::prefix_high_lut, 4>(callback, "prefix_global_x4");
  add<fs::decode_kind::prefix_high_lut, 8>(callback, "prefix_global_x8");
  add<fs::decode_kind::prefix_high_lut, 4, fs::table_location::shared>(
      callback, "prefix_shared_x4");
  add<fs::decode_kind::prefix_high_lut, 8, fs::table_location::shared>(
      callback, "prefix_shared_x8");
}

template <typename Callback> void add_pair_and_prmt(Callback &callback) {
  add<fs::decode_kind::pair_high_lut, 4>(callback, "pair_l2_x4");
  add<fs::decode_kind::pair_high_lut, 8>(callback, "pair_l2_x8");
  add<fs::decode_kind::direct_words_branchy, 4,
      fs::table_location::global_read_only, fs::unpack_kind::byte_permute>(
      callback, "word_branchy_prmt_x4");
  add<fs::decode_kind::direct_words_branchy, 8,
      fs::table_location::global_read_only, fs::unpack_kind::byte_permute>(
      callback, "word_branchy_prmt_x8");
}

template <typename Callback> void strategies_e1m6(Callback &callback) {
  add_generic_148(callback);
  add_words_148(callback);
  add_fp32_148(callback);
  add_widths_148<fs::decode_kind::e1_integer>(callback, "e1_integer_x1",
                                              "e1_integer_x4", "e1_integer_x8");
  add<fs::decode_kind::full_high_lut, 1>(callback, "full_high_global_x1");
  add<fs::decode_kind::full_high_lut, 4>(callback, "full_high_global_x4");
  add<fs::decode_kind::full_high_lut, 8>(callback, "full_high_global_x8");
  add<fs::decode_kind::full_high_lut, 4, fs::table_location::shared>(
      callback, "full_high_shared_x4");
  add<fs::decode_kind::full_high_lut, 8, fs::table_location::shared>(
      callback, "full_high_shared_x8");
  add<fs::decode_kind::subnormal_high_lut, 4>(callback, "subnormal_global_x4");
  add<fs::decode_kind::subnormal_high_lut, 8, fs::table_location::shared>(
      callback, "subnormal_shared_x8");
  add_pair_and_prmt(callback);
  add<fs::decode_kind::full_high_lut_swizzled, 4, fs::table_location::shared>(
      callback, "full_high_swizzled_x4");
  add<fs::decode_kind::full_high_lut_swizzled, 8, fs::table_location::shared>(
      callback, "full_high_swizzled_x8");
}

template <typename Callback> void strategies_fp8(Callback &callback) {
  add<fs::decode_kind::generic, 1>(callback, "generic_x1");
  add<fs::decode_kind::native_direct, 1>(callback, "native_direct_x1");
  add<fs::decode_kind::native_fp32, 1>(callback, "native_fp32_x1");
  add<fs::decode_kind::native_packed, 2>(callback, "native_float2_x2");
  add<fs::decode_kind::native_packed, 4>(callback, "native_float4_x4");
  add<fs::decode_kind::native_packed, 8>(callback, "native_float4_x8");
  add<fs::decode_kind::native_half2, 2>(callback, "native_half2_x2");
  add<fs::decode_kind::native_half2, 4>(callback, "native_half2_x4");
  add<fs::decode_kind::native_half2, 8>(callback, "native_half2_x8");
  add_words_148(callback);
  add_fp32_148(callback);
  add<fs::decode_kind::full_high_lut, 1>(callback, "full_high_global_x1");
  add<fs::decode_kind::full_high_lut, 4>(callback, "full_high_global_x4");
  add<fs::decode_kind::full_high_lut, 8>(callback, "full_high_global_x8");
  add<fs::decode_kind::full_high_lut, 4, fs::table_location::shared>(
      callback, "full_high_shared_x4");
  add<fs::decode_kind::full_high_lut, 8, fs::table_location::shared>(
      callback, "full_high_shared_x8");
  add<fs::decode_kind::subnormal_high_lut, 8, fs::table_location::shared>(
      callback, "subnormal_shared_x8");
  add_pair_and_prmt(callback);
  add<fs::decode_kind::full_high_lut_swizzled, 4, fs::table_location::shared>(
      callback, "full_high_swizzled_x4");
  add<fs::decode_kind::full_high_lut_swizzled, 8, fs::table_location::shared>(
      callback, "full_high_swizzled_x8");
}

template <typename Callback> void strategies_e1m14(Callback &callback) {
  add_generic_148(callback);
  add_widths_148<fs::decode_kind::e1_integer>(callback, "e1_integer_x1",
                                              "e1_integer_x4", "e1_integer_x8");
  add_words_148(callback);
  add_fp32_148(callback);
  add_full_high_148(callback);
  add<fs::decode_kind::subnormal_high_lut, 4>(callback, "subnormal_global_x4");
  add<fs::decode_kind::subnormal_high_lut, 8>(callback, "subnormal_global_x8");
  add<fs::decode_kind::subnormal_high_lut, 4, fs::table_location::shared>(
      callback, "subnormal_shared_x4");
  add<fs::decode_kind::subnormal_high_lut, 8, fs::table_location::shared>(
      callback, "subnormal_shared_x8");
}

template <typename Callback> void strategies_e2e3_16(Callback &callback) {
  add_generic_148(callback);
  add_words_148(callback);
  add_fp32_148(callback);
  add_full_high_148(callback);
  add<fs::decode_kind::subnormal_high_lut, 4>(callback, "subnormal_global_x4");
  add<fs::decode_kind::subnormal_high_lut, 8>(callback, "subnormal_global_x8");
  add<fs::decode_kind::subnormal_high_lut, 4, fs::table_location::shared>(
      callback, "subnormal_shared_x4");
  add<fs::decode_kind::subnormal_high_lut, 8, fs::table_location::shared>(
      callback, "subnormal_shared_x8");
  add_prefix_tables(callback);
}

template <typename Callback>
void strategies_native_16(Callback &callback, bool bf16) {
  add<fs::decode_kind::generic, 1>(callback, "generic_x1");
  add<fs::decode_kind::native_direct, 1>(callback, "native_direct_x1");
  add<fs::decode_kind::native_fp32, 1>(callback, "native_fp32_x1");
  add<fs::decode_kind::native_packed, 2>(callback, bf16 ? "native_bfloat162_x2"
                                                        : "native_half2_x2");
  add<fs::decode_kind::native_packed, 4>(callback, bf16 ? "native_bfloat162_x4"
                                                        : "native_half2_x4");
  add<fs::decode_kind::native_packed, 8>(callback, bf16 ? "native_bfloat162_x8"
                                                        : "native_half2_x8");
  add_words_148(callback);
  add_widths_148<fs::decode_kind::fp32_bits>(
      callback, bf16 ? "fp32_bit_lift_x1" : "fp32_bits_x1",
      bf16 ? "fp32_bit_lift_x4" : "fp32_bits_x4",
      bf16 ? "fp32_bit_lift_x8" : "fp32_bits_x8");
  add_full_high_148(callback);
  add<fs::decode_kind::subnormal_high_lut, 8>(callback, "subnormal_global_x8");
  add<fs::decode_kind::subnormal_high_lut, 8, fs::table_location::shared>(
      callback, "subnormal_shared_x8");
}

template <typename Callback> void strategies_e11m4(Callback &callback) {
  add_generic_1248(callback);
  add_widths_1248<fs::decode_kind::prefix_word>(
      callback, "prefix_word_x1", "prefix_word_x2", "prefix_word_x4",
      "prefix_word_x8");
  add_full_high_148(callback);
}

template <typename Callback> void strategies_e1m30(Callback &callback) {
  add_generic_1248(callback);
  add_widths_1248<fs::decode_kind::e1_integer>(callback, "e1_integer_x1",
                                               "e1_integer_x2", "e1_integer_x4",
                                               "e1_integer_x8");
  add_words_1248(callback);
  add_prefix_tables(callback);
}

template <typename Callback> void strategies_e2e3_32(Callback &callback) {
  add_generic_1248(callback);
  add_words_1248(callback);
  add_prefix_tables(callback);
}

template <typename Callback> void strategies_fp32(Callback &callback) {
  add<fs::decode_kind::generic, 1>(callback, "generic_x1");
  add<fs::decode_kind::native_direct, 1>(callback, "native_f64_x1");
  add<fs::decode_kind::native_packed, 2>(callback, "native_float2_x2");
  add<fs::decode_kind::native_packed, 4>(callback, "native_float4_x4");
  add<fs::decode_kind::native_packed, 8>(callback, "native_float4_x8");
  add_words_1248(callback);
  add_prefix_tables(callback);
}

template <typename Callback> void strategies_e11m20(Callback &callback) {
  add_generic_1248(callback);
  add_widths_1248<fs::decode_kind::prefix_word>(
      callback, "prefix_word_x1", "prefix_word_x2", "prefix_word_x4",
      "prefix_word_x8");
}

template <typename Format, typename Callback>
void for_each_strategy(Callback &callback) {
  if constexpr (std::is_same_v<Format, storage::e1m6>) {
    strategies_e1m6(callback);
  } else if constexpr (std::is_same_v<Format, storage::fp8_e4m3> ||
                       std::is_same_v<Format, storage::fp8_e5m2>) {
    strategies_fp8(callback);
  } else if constexpr (std::is_same_v<Format, storage::e1m14>) {
    strategies_e1m14(callback);
  } else if constexpr (std::is_same_v<Format, storage::e2m13> ||
                       std::is_same_v<Format, storage::e3m12>) {
    strategies_e2e3_16(callback);
  } else if constexpr (std::is_same_v<Format, storage::fp16_e5m10>) {
    strategies_native_16(callback, false);
  } else if constexpr (std::is_same_v<Format, storage::bf16_e8m7>) {
    strategies_native_16(callback, true);
  } else if constexpr (std::is_same_v<Format, storage::e11m4>) {
    strategies_e11m4(callback);
  } else if constexpr (std::is_same_v<Format, storage::e1m30>) {
    strategies_e1m30(callback);
  } else if constexpr (std::is_same_v<Format, storage::e2m29> ||
                       std::is_same_v<Format, storage::e3m28>) {
    strategies_e2e3_32(callback);
  } else if constexpr (std::is_same_v<Format, storage::fp32_e8m23>) {
    strategies_fp32(callback);
  } else if constexpr (std::is_same_v<Format, storage::e11m20>) {
    strategies_e11m20(callback);
  }
}

struct work_model {
  std::string benchmark_format;
  std::string format;
  int storage_bits{};
  std::string strategy;
  std::string decode_kind;
  std::string table_location;
  std::string unpack;
  int lanes{};
  std::size_t lookup_entry_bytes{};
  std::size_t shared_table_bytes{};
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
    stream_ << "gpu,compute_capability,distribution,benchmark_format,format,"
               "storage_bits,strategy,decode_kind,table_location,unpack,lanes,"
               "lookup_entry_bytes,shared_table_bytes,component,n,m,blocks,"
               "threads,warmup,round,sample,order_position,iterations,"
               "total_time_ms,time_ms,main_array_unique_bytes,"
               "main_array_requested_bytes,useful_flops,"
               "main_array_unique_gb_per_s,main_array_requested_gb_per_s,"
               "useful_gflop_per_s\n";
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
            << ',' << model.benchmark_format << ',' << model.format << ','
            << model.storage_bits << ',' << model.strategy << ','
            << model.decode_kind << ',' << model.table_location << ','
            << model.unpack << ',' << model.lanes << ','
            << model.lookup_entry_bytes << ',' << model.shared_table_bytes
            << ',' << model.component << ',' << model.n << ',' << model.m << ','
            << model.blocks << ',' << fs::block_threads << ',' << warmup << ','
            << round << ',' << sample << ',' << order_position << ','
            << iterations << ',' << total_ms << ',' << time_ms << ','
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
work_model strategy_model(const char *strategy_name, const char *component,
                          std::size_t n, std::size_t m, int blocks) {
  const auto storage_bytes = sizeof(storage::storage_type_t<Format>);
  const auto unique = component == std::string{"dot"}
                          ? 2.0 * n * storage_bytes +
                                2.0 * blocks * sizeof(double) + sizeof(double)
                          : (m * n + n) * storage_bytes + m * sizeof(double);
  const auto requested = component == std::string{"dot"}
                             ? unique
                             : 2.0 * m * n * storage_bytes + m * sizeof(double);
  return {Format::name,
          Format::name,
          Format::total_bits,
          strategy_name,
          kind_name(Strategy::kind),
          table_location_name<Strategy>(),
          unpack_name<Strategy>(),
          Strategy::lanes,
          lookup_entry_bytes<Format, Strategy>(),
          fs::shared_table_bytes_v<Format, Strategy>,
          component,
          n,
          m,
          blocks,
          unique,
          requested,
          2.0 * n * m};
}

work_model raw_model(const char *benchmark_format, const char *component,
                     std::size_t n, std::size_t m, int blocks) {
  const auto unique = component == std::string{"dot"}
                          ? 2.0 * n * sizeof(double) +
                                2.0 * blocks * sizeof(double) + sizeof(double)
                          : (m * n + n) * sizeof(double) + m * sizeof(double);
  const auto requested =
      component == std::string{"dot"}
          ? unique
          : 2.0 * m * n * sizeof(double) + m * sizeof(double);
  return {benchmark_format,
          "fp64_e11m52",
          64,
          "raw_pointer_x1",
          "none",
          "none",
          "scalar",
          1,
          0,
          0,
          component,
          n,
          m,
          blocks,
          unique,
          requested,
          2.0 * n * m};
}

template <typename Format, typename Strategy>
void configure_dynamic_shared_memory() {
  constexpr auto bytes = fs::shared_table_bytes_v<Format, Strategy>;
  if constexpr (bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(fs::dot_map_reduce<Format, Strategy>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(bytes)));
    CUDA_CHECK(cudaFuncSetAttribute(fs::gemv<Format, Strategy>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(bytes)));
  }
}

template <typename Format, typename Strategy>
timed_variant make_dot_variant(const char *strategy_name,
                               const storage::storage_type_t<Format> *left,
                               const storage::storage_type_t<Format> *right,
                               std::size_t count, fs::table_bundle tables,
                               double *partials, double *result,
                               int multiprocessors) {
  configure_dynamic_shared_memory<Format, Strategy>();
  const auto blocks = work_blocks(count, Strategy::lanes, multiprocessors);
  constexpr auto shared_bytes = fs::shared_table_bytes_v<Format, Strategy>;
  return {
      strategy_model<Format, Strategy>(strategy_name, "dot", count, 1, blocks),
      [=] {
        fs::dot_map_reduce<Format, Strategy>
            <<<blocks, fs::block_threads, shared_bytes>>>(left, right, count,
                                                          tables, partials);
        kernels::storage_dot_finalize<<<1, fs::block_threads>>>(partials,
                                                                blocks, result);
      }};
}

timed_variant make_raw_dot_variant(const char *benchmark_format,
                                   const double *left, const double *right,
                                   std::size_t count, double *partials,
                                   double *result, int multiprocessors) {
  const auto blocks = work_blocks(count, 1, multiprocessors);
  return {raw_model(benchmark_format, "dot", count, 1, blocks), [=] {
            raw_fp64_dot_map_reduce<<<blocks, fs::block_threads>>>(
                left, right, count, partials);
            kernels::storage_dot_finalize<<<1, fs::block_threads>>>(
                partials, blocks, result);
          }};
}

template <typename Format, typename Strategy>
timed_variant make_gemv_variant(const char *strategy_name,
                                const storage::storage_type_t<Format> *matrix,
                                const storage::storage_type_t<Format> *vector,
                                std::size_t rows, std::size_t columns,
                                fs::table_bundle tables, double *result) {
  configure_dynamic_shared_memory<Format, Strategy>();
  constexpr auto shared_bytes = fs::shared_table_bytes_v<Format, Strategy>;
  return {
      strategy_model<Format, Strategy>(strategy_name, "gemv", columns, rows,
                                       static_cast<int>(rows)),
      [=] {
        fs::gemv<Format, Strategy>
            <<<static_cast<unsigned>(rows), fs::block_threads, shared_bytes>>>(
                matrix, vector, rows, columns, columns, tables, result);
      }};
}

timed_variant make_raw_gemv_variant(const char *benchmark_format,
                                    const double *matrix, const double *vector,
                                    std::size_t rows, std::size_t columns,
                                    double *result) {
  return {raw_model(benchmark_format, "gemv", columns, rows,
                    static_cast<int>(rows)),
          [=] {
            raw_fp64_gemv<<<static_cast<unsigned>(rows), fs::block_threads>>>(
                matrix, vector, rows, columns, columns, result);
          }};
}

void measure_variants(std::vector<timed_variant> &variants,
                      const options &settings, distribution kind,
                      std::uint64_t order_seed, sample_output &output,
                      event_timer &timer) {
  if (variants.size() < 2) {
    throw benchmark_error("internal error: no decoder strategies registered");
  }
  std::set<std::string> names;
  for (const auto &variant : variants) {
    if (!names.insert(variant.model.strategy).second) {
      throw benchmark_error("duplicate strategy name: " +
                            variant.model.strategy);
    }
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

struct source_buffers {
  device_buffer<double> dot_left;
  device_buffer<double> dot_right;
  device_buffer<double> matrix;
  device_buffer<double> vector;
  device_buffer<double> partials;
  device_buffer<double> result;

  source_buffers(std::size_t dot_count, std::size_t matrix_count,
                 std::size_t vector_count, std::size_t partial_count,
                 std::size_t result_count)
      : dot_left(dot_count), dot_right(dot_count), matrix(matrix_count),
        vector(vector_count), partials(partial_count), result(result_count) {}
};

template <typename Format>
void run_format(const options &settings, const device_info &device,
                source_buffers &source, random_generator &random,
                sample_output &output, event_timer &timer, std::size_t max_dot,
                std::size_t max_columns, std::size_t max_matrix) {
  using storage_type = storage::storage_type_t<Format>;
  device_buffer<storage_type> dot_left{max_dot};
  device_buffer<storage_type> dot_right{max_dot};
  device_buffer<storage_type> matrix{max_matrix};
  device_buffer<storage_type> vector{max_columns};
  format_tables<Format> tables;

  std::size_t strategy_count{};
  auto count_strategy = [&](auto, const char *) { ++strategy_count; };
  for_each_strategy<Format>(count_strategy);
  std::cout << Format::name << " (" << Format::total_bits << " bit, "
            << strategy_count << " strategies)\n";

  for (const auto kind :
       {distribution::uniform_0_1, distribution::normal_0_1}) {
    const auto distribution_seed =
        mix(settings.base_seed ^ static_cast<std::uint64_t>(kind));
    random.fill(source.dot_left.get(), max_dot, kind,
                mix(distribution_seed ^ 0x01));
    random.fill(source.dot_right.get(), max_dot, kind,
                mix(distribution_seed ^ 0x02));
    random.fill(source.matrix.get(), max_matrix, kind,
                mix(distribution_seed ^ 0x03));
    random.fill(source.vector.get(), max_columns, kind,
                mix(distribution_seed ^ 0x04));
    encode_values<Format>(source.dot_left.get(), dot_left.get(), max_dot,
                          device.multiprocessors);
    encode_values<Format>(source.dot_right.get(), dot_right.get(), max_dot,
                          device.multiprocessors);
    encode_values<Format>(source.matrix.get(), matrix.get(), max_matrix,
                          device.multiprocessors);
    encode_values<Format>(source.vector.get(), vector.get(), max_columns,
                          device.multiprocessors);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "  " << name(kind) << '\n';
    for (const auto power : settings.dot_powers) {
      const auto count = size_from_power(power);
      std::vector<timed_variant> variants;
      variants.reserve(strategy_count + 1);
      variants.push_back(make_raw_dot_variant(
          Format::name, source.dot_left.get(), source.dot_right.get(), count,
          source.partials.get(), source.result.get(), device.multiprocessors));
      auto append = [&](auto strategy_value, const char *strategy_name) {
        using strategy_type = decltype(strategy_value);
        variants.push_back(make_dot_variant<Format, strategy_type>(
            strategy_name, dot_left.get(), dot_right.get(), count,
            tables.bundle(), source.partials.get(), source.result.get(),
            device.multiprocessors));
      };
      for_each_strategy<Format>(append);
      measure_variants(variants, settings, kind,
                       mix(distribution_seed ^ count ^ 0xd07 ^
                           static_cast<std::uint64_t>(Format::total_bits)),
                       output, timer);
      std::cout << "    DOT N=2^" << power << " complete\n";
    }

    for (const auto power : settings.gemv_powers) {
      const auto columns = size_from_power(power);
      std::vector<timed_variant> variants;
      variants.reserve(strategy_count + 1);
      variants.push_back(make_raw_gemv_variant(
          Format::name, source.matrix.get(), source.vector.get(),
          settings.gemv_rows, columns, source.result.get()));
      auto append = [&](auto strategy_value, const char *strategy_name) {
        using strategy_type = decltype(strategy_value);
        variants.push_back(make_gemv_variant<Format, strategy_type>(
            strategy_name, matrix.get(), vector.get(), settings.gemv_rows,
            columns, tables.bundle(), source.result.get()));
      };
      for_each_strategy<Format>(append);
      measure_variants(variants, settings, kind,
                       mix(distribution_seed ^ columns ^ 0x6e6d76 ^
                           static_cast<std::uint64_t>(Format::total_bits)),
                       output, timer);
      std::cout << "    GEMV M=" << settings.gemv_rows << ", N=2^" << power
                << " complete\n";
    }
  }
}

template <typename Callback>
void dispatch_format(const std::string &name, Callback &callback) {
#define DISPATCH(format)                                                       \
  if (name == storage::format::name) {                                         \
    callback(storage::format{});                                               \
    return;                                                                    \
  }
  DISPATCH(e1m6)
  DISPATCH(fp8_e4m3)
  DISPATCH(fp8_e5m2)
  DISPATCH(e1m14)
  DISPATCH(e2m13)
  DISPATCH(e3m12)
  DISPATCH(fp16_e5m10)
  DISPATCH(bf16_e8m7)
  DISPATCH(e11m4)
  DISPATCH(e1m30)
  DISPATCH(e2m29)
  DISPATCH(e3m28)
  DISPATCH(fp32_e8m23)
  DISPATCH(e11m20)
#undef DISPATCH
  throw benchmark_error("internal format dispatch failure: " + name);
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
  std::cout << "All-format decoder strategy benchmark\nGPU: " << device.name
            << " (" << device.capability << ")\n";

  const auto max_dot = size_from_power(settings.dot_powers.back());
  const auto max_columns = size_from_power(settings.gemv_powers.back());
  const auto max_matrix = settings.gemv_rows * max_columns;
  const auto max_blocks = static_cast<std::size_t>(device.multiprocessors * 16);
  source_buffers source{max_dot, max_matrix, max_columns, max_blocks,
                        std::max<std::size_t>(1, settings.gemv_rows)};
  random_generator random;
  sample_output output{settings.output, device};
  event_timer timer;

  for (const auto &format_name : settings.formats) {
    auto run = [&](auto format_value) {
      using format_type = decltype(format_value);
      run_format<format_type>(settings, device, source, random, output, timer,
                              max_dot, max_columns, max_matrix);
    };
    dispatch_format(format_name, run);
  }
  std::cout << "Wrote " << settings.output << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return EXIT_FAILURE;
}
