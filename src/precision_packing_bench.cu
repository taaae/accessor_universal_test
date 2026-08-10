#include "precision_packing_kernels.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace pp = aut::precision_packing;
namespace storage = aut::storage;

#ifndef AUT_PRECISION_FORMAT
#error "AUT_PRECISION_FORMAT must select one storage format"
#endif

#if AUT_PRECISION_FORMAT == 1
using selected_format = storage::fp16_e5m10;
#elif AUT_PRECISION_FORMAT == 2
using selected_format = storage::bf16_e8m7;
#elif AUT_PRECISION_FORMAT == 3
using selected_format = storage::fp32_e8m23;
#elif AUT_PRECISION_FORMAT == 4
using selected_format = storage::fp8_e4m3;
#elif AUT_PRECISION_FORMAT == 5
using selected_format = storage::fp8_e5m2;
#elif AUT_PRECISION_FORMAT == 6
using selected_format = storage::fp4_e2m1;
#elif AUT_PRECISION_FORMAT == 7
using selected_format = storage::fp64_e11m52;
#else
#error "unsupported AUT_PRECISION_FORMAT"
#endif

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

  double measure(const std::function<void()> &launch, std::size_t iterations) {
    CUDA_CHECK(cudaEventRecord(start_));
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      launch();
    }
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float elapsed_ms{};
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
    return static_cast<double>(elapsed_ms);
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

class random_generator {
public:
  explicit random_generator(std::uint64_t seed) {
    CURAND_CHECK(
        curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, seed));
  }
  ~random_generator() { curandDestroyGenerator(generator_); }

  void uniform(double *values, std::size_t count) {
    CURAND_CHECK(curandGenerateUniformDouble(generator_, values, count));
  }
  void normal(double *values, std::size_t count) {
    CURAND_CHECK(
        curandGenerateNormalDouble(generator_, values, count, 0.0, 1.0));
  }

private:
  curandGenerator_t generator_{};
};

enum class run_mode { sweep, components, validate, profile };
enum class distribution { uniform_0_1, normal_0_1 };
enum class kernel_kind { dot, gemv };

const char *name(distribution value) {
  return value == distribution::uniform_0_1 ? "uniform_0_1" : "normal_0_1";
}
const char *name(kernel_kind value) {
  return value == kernel_kind::dot ? "dot" : "gemv";
}
const char *name(pp::arithmetic_kind value) {
  switch (value) {
  case pp::arithmetic_kind::fp16:
    return "fp16";
  case pp::arithmetic_kind::bf16:
    return "bf16";
  case pp::arithmetic_kind::fp32:
    return "fp32";
  case pp::arithmetic_kind::fp64:
    return "fp64";
  }
  return "unknown";
}
const char *name(pp::access_kind value) {
  switch (value) {
  case pp::access_kind::scalar_single:
    return "scalar_single";
  case pp::access_kind::scalar_unrolled:
    return "scalar_unrolled";
  case pp::access_kind::vector_packet:
    return "vector_packet";
  }
  return "unknown";
}

struct options {
  run_mode mode{run_mode::sweep};
  std::vector<int> dot_powers{12, 16, 20, 24, 27};
  std::vector<int> gemv_powers{8, 10, 12, 14, 16};
  std::size_t gemv_rows{1024};
  int warmup{10};
  int rounds{3};
  int samples{5};
  double target_sample_ms{15.0};
  std::uint64_t seed{0x243f6a8885a308d3ULL};
  std::string output{"timing_samples.csv"};
  std::size_t component_n{std::size_t{1} << 27};
  std::size_t register_n{std::size_t{1} << 20};
  int decode_repeats{256};
  int arithmetic_repeats{4096};
  kernel_kind profile_kernel{kernel_kind::dot};
  pp::arithmetic_kind profile_arithmetic{pp::arithmetic_kind::fp64};
  std::string profile_family{"vector_packet"};
  int profile_lanes{8};
  std::size_t profile_n{std::size_t{1} << 27};
  std::size_t profile_m{1024};
  distribution profile_distribution{distribution::normal_0_1};
};

std::size_t parse_size(const std::string &text, const std::string &option) {
  std::size_t consumed{};
  const auto result = std::stoull(text, &consumed, 0);
  if (consumed != text.size() || result == 0) {
    throw benchmark_error(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(result);
}

int parse_int(const std::string &text, const std::string &option) {
  const auto result = parse_size(text, option);
  if (result > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw benchmark_error(option + " is too large");
  }
  return static_cast<int>(result);
}

std::vector<int> parse_powers(const std::string &text,
                              const std::string &option) {
  std::vector<int> result;
  std::stringstream input{text};
  std::string token;
  while (std::getline(input, token, ',')) {
    const auto power = parse_int(token, option);
    if (power > 30) {
      throw benchmark_error(option + " values must be in [1,30]");
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

pp::arithmetic_kind parse_arithmetic(const std::string &text) {
  if (text == "fp16") {
    return pp::arithmetic_kind::fp16;
  }
  if (text == "bf16") {
    return pp::arithmetic_kind::bf16;
  }
  if (text == "fp32") {
    return pp::arithmetic_kind::fp32;
  }
  if (text == "fp64") {
    return pp::arithmetic_kind::fp64;
  }
  throw benchmark_error("unknown arithmetic type: " + text);
}

distribution parse_distribution(const std::string &text) {
  if (text == "uniform_0_1") {
    return distribution::uniform_0_1;
  }
  if (text == "normal_0_1") {
    return distribution::normal_0_1;
  }
  throw benchmark_error("unknown distribution: " + text);
}

options parse_options(int argc, char **argv) {
  options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument{argv[index]};
    auto value = [&]() {
      if (++index >= argc) {
        throw benchmark_error("missing value after " + argument);
      }
      return std::string{argv[index]};
    };
    if (argument == "--mode") {
      const auto selected = value();
      if (selected == "sweep") {
        result.mode = run_mode::sweep;
      } else if (selected == "components") {
        result.mode = run_mode::components;
      } else if (selected == "validate") {
        result.mode = run_mode::validate;
      } else if (selected == "profile") {
        result.mode = run_mode::profile;
      } else {
        throw benchmark_error("--mode must be sweep, validate, or profile");
      }
    } else if (argument == "--dot-powers") {
      result.dot_powers = parse_powers(value(), argument);
    } else if (argument == "--gemv-powers") {
      result.gemv_powers = parse_powers(value(), argument);
    } else if (argument == "--gemv-rows") {
      result.gemv_rows = parse_size(value(), argument);
    } else if (argument == "--warmup") {
      result.warmup = parse_int(value(), argument);
    } else if (argument == "--rounds") {
      result.rounds = parse_int(value(), argument);
    } else if (argument == "--samples") {
      result.samples = parse_int(value(), argument);
    } else if (argument == "--target-sample-ms") {
      result.target_sample_ms = std::stod(value());
    } else if (argument == "--base-seed") {
      result.seed = parse_size(value(), argument);
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--component-n") {
      result.component_n = parse_size(value(), argument);
    } else if (argument == "--register-n") {
      result.register_n = parse_size(value(), argument);
    } else if (argument == "--decode-repeats") {
      result.decode_repeats = parse_int(value(), argument);
    } else if (argument == "--arithmetic-repeats") {
      result.arithmetic_repeats = parse_int(value(), argument);
    } else if (argument == "--kernel") {
      const auto selected = value();
      if (selected == "dot") {
        result.profile_kernel = kernel_kind::dot;
      } else if (selected == "gemv") {
        result.profile_kernel = kernel_kind::gemv;
      } else {
        throw benchmark_error("--kernel must be dot or gemv");
      }
    } else if (argument == "--arithmetic") {
      result.profile_arithmetic = parse_arithmetic(value());
    } else if (argument == "--family") {
      result.profile_family = value();
    } else if (argument == "--lanes") {
      result.profile_lanes = parse_int(value(), argument);
    } else if (argument == "--n") {
      result.profile_n = parse_size(value(), argument);
    } else if (argument == "--m") {
      result.profile_m = parse_size(value(), argument);
    } else if (argument == "--distribution") {
      result.profile_distribution = parse_distribution(value());
    } else if (argument == "--help") {
      std::cout
          << "Sweep: --mode sweep --dot-powers P,... --gemv-powers P,...\n"
             "       --gemv-rows M --warmup N --rounds N --samples N\n"
             "       --target-sample-ms T --base-seed S --output FILE\n"
             "Components: --mode components --component-n N --register-n N\n"
             "            --decode-repeats N --arithmetic-repeats N\n"
             "Profile: --mode profile --kernel dot|gemv --arithmetic TYPE\n"
             "         --family scalar_single|scalar_unrolled|vector_packet|"
             "packed_arithmetic --lanes 1|2|4|8 --n N --m M\n";
      std::exit(0);
    } else {
      throw benchmark_error("unknown option: " + argument);
    }
  }
  if (result.warmup < 1 || result.rounds < 1 || result.samples < 1 ||
      result.decode_repeats < 1 || result.arithmetic_repeats < 1 ||
      !(result.target_sample_ms > 0.0)) {
    throw benchmark_error(
        "warmup, rounds, samples, and target time must be positive");
  }
  if (result.profile_lanes != 1 && result.profile_lanes != 2 &&
      result.profile_lanes != 4 && result.profile_lanes != 8) {
    throw benchmark_error("--lanes must be 1, 2, 4, or 8");
  }
  return result;
}

struct device_info {
  std::string name;
  std::string capability;
  int multiprocessors{};
};

device_info query_device() {
  int device{};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::ostringstream capability;
  capability << "sm_" << properties.major << properties.minor;
  return {properties.name, capability.str(), properties.multiProcessorCount};
}

template <typename Format>
std::size_t storage_elements(std::size_t logical_count) {
  if constexpr (Format::total_bits < 8) {
    return pp::storage_bytes<Format>(logical_count);
  } else {
    return logical_count;
  }
}

template <typename Format>
void encode(const double *source, pp::device_storage_t<Format> *destination,
            std::size_t count, int multiprocessors) {
  const auto blocks = std::max(1, multiprocessors * 4);
  pp::encode_values<Format>
      <<<blocks, pp::block_threads>>>(source, destination, count);
  CUDA_CHECK(cudaGetLastError());
}

void fill(random_generator &generator, double *values, std::size_t count,
          distribution kind) {
  if (kind == distribution::uniform_0_1) {
    generator.uniform(values, count);
  } else {
    generator.normal(values, count);
  }
}

template <typename Format, typename Callback>
void for_each_arithmetic(Callback &&callback) {
  if constexpr (std::is_same_v<Format, storage::fp16_e5m10>) {
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::fp16>{});
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::fp64>{});
  } else if constexpr (std::is_same_v<Format, storage::bf16_e8m7>) {
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::bf16>{});
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::fp64>{});
  } else if constexpr (std::is_same_v<Format, storage::fp32_e8m23>) {
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::fp32>{});
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::fp64>{});
  } else if constexpr (std::is_same_v<Format, storage::fp64_e11m52>) {
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::fp64>{});
  } else {
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::fp16>{});
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::fp32>{});
    callback(std::integral_constant<pp::arithmetic_kind,
                                    pp::arithmetic_kind::fp64>{});
  }
}

template <typename Format>
bool valid_arithmetic(pp::arithmetic_kind requested) {
  bool found{};
  for_each_arithmetic<Format>([&](auto tag) {
    if (decltype(tag)::value == requested) {
      found = true;
    }
  });
  return found;
}

struct work_model {
  kernel_kind kernel{};
  pp::arithmetic_kind arithmetic{};
  std::string family;
  int lanes{};
  std::size_t rows{};
  std::size_t columns{};
  double useful_flops{};
  double logical_storage_bytes{};
  double modeled_load_instructions{};
};

struct result_metrics {
  double value{};
  double absolute_error{};
  double relative_error{};
};

struct timed_variant {
  work_model model;
  std::function<void()> launch;
  std::function<std::vector<double>()> read_result;
  result_metrics result;
};

double load_instructions_per_element(int storage_bits, int lanes,
                                     const std::string &family) {
  if (family == "scalar_single" || family == "scalar_unrolled") {
    return 2.0;
  }
  const auto packet_bytes = std::max(1, (storage_bits * lanes + 7) / 8);
  const auto loads_per_input = (packet_bytes + 15) / 16;
  return 2.0 * static_cast<double>(loads_per_input) / lanes;
}

template <typename Format, pp::arithmetic_kind Arithmetic,
          pp::access_kind Access, int Lanes>
timed_variant make_dot_variant(const pp::device_storage_t<Format> *left,
                               const pp::device_storage_t<Format> *right,
                               std::size_t count, int blocks,
                               void *partial_storage, double *result) {
  auto *partials = static_cast<pp::arithmetic_t<Arithmetic> *>(partial_storage);
  const auto family = std::string{name(Access)};
  work_model model{
      kernel_kind::dot,
      Arithmetic,
      family,
      Lanes,
      1,
      count,
      2.0 * count,
      2.0 * pp::storage_bytes<Format>(count),
      count * load_instructions_per_element(Format::total_bits, Lanes, family)};
  return {model,
          [=] {
            pp::dot_map<Format, Arithmetic, Access, Lanes>
                <<<blocks, pp::block_threads>>>(left, right, count, partials);
            pp::dot_finalize<Arithmetic>
                <<<1, pp::block_threads>>>(partials, blocks, result);
          },
          [=] {
            double host{};
            CUDA_CHECK(cudaMemcpy(&host, result, sizeof(host),
                                  cudaMemcpyDeviceToHost));
            return std::vector<double>{host};
          },
          {}};
}

template <typename Format, pp::arithmetic_kind Arithmetic,
          pp::access_kind Access, int Lanes>
timed_variant make_gemv_variant(const pp::device_storage_t<Format> *matrix,
                                const pp::device_storage_t<Format> *vector,
                                std::size_t rows, std::size_t columns,
                                std::size_t stride, double *result) {
  const auto family = std::string{name(Access)};
  work_model model{
      kernel_kind::gemv,
      Arithmetic,
      family,
      Lanes,
      rows,
      columns,
      2.0 * rows * columns,
      static_cast<double>(pp::storage_bytes<Format>(rows * columns) +
                          pp::storage_bytes<Format>(columns)),
      rows * columns *
          load_instructions_per_element(Format::total_bits, Lanes, family)};
  return {model,
          [=] {
            pp::gemv<Format, Arithmetic, Access, Lanes>
                <<<static_cast<unsigned>(rows), pp::block_threads>>>(
                    matrix, vector, rows, columns, stride, result);
          },
          [=] {
            std::vector<double> host(rows);
            CUDA_CHECK(cudaMemcpy(host.data(), result, rows * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            return host;
          },
          {}};
}

template <typename Format, int Lanes>
timed_variant make_packed_dot_variant(const pp::device_storage_t<Format> *left,
                                      const pp::device_storage_t<Format> *right,
                                      std::size_t count, int blocks,
                                      void *partial_storage, double *result) {
  static_assert(pp::supports_packed_arithmetic_v<Format>);
  constexpr auto arithmetic = std::is_same_v<Format, storage::fp16_e5m10>
                                  ? pp::arithmetic_kind::fp16
                                  : pp::arithmetic_kind::bf16;
  auto *partials =
      static_cast<storage::storage_type_t<Format> *>(partial_storage);
  work_model model{kernel_kind::dot,
                   arithmetic,
                   "packed_arithmetic",
                   Lanes,
                   1,
                   count,
                   2.0 * count,
                   2.0 * pp::storage_bytes<Format>(count),
                   count * load_instructions_per_element(
                               Format::total_bits, Lanes, "packed_arithmetic")};
  return {model,
          [=] {
            pp::packed_arithmetic_dot_map<Format, Lanes>
                <<<blocks, pp::block_threads>>>(left, right, count, partials);
            pp::dot_finalize<arithmetic>
                <<<1, pp::block_threads>>>(partials, blocks, result);
          },
          [=] {
            double host{};
            CUDA_CHECK(cudaMemcpy(&host, result, sizeof(host),
                                  cudaMemcpyDeviceToHost));
            return std::vector<double>{host};
          },
          {}};
}

template <typename Format, int Lanes>
timed_variant
make_packed_gemv_variant(const pp::device_storage_t<Format> *matrix,
                         const pp::device_storage_t<Format> *vector,
                         std::size_t rows, std::size_t columns,
                         std::size_t stride, double *result) {
  static_assert(pp::supports_packed_arithmetic_v<Format>);
  constexpr auto arithmetic = std::is_same_v<Format, storage::fp16_e5m10>
                                  ? pp::arithmetic_kind::fp16
                                  : pp::arithmetic_kind::bf16;
  work_model model{
      kernel_kind::gemv,
      arithmetic,
      "packed_arithmetic",
      Lanes,
      rows,
      columns,
      2.0 * rows * columns,
      static_cast<double>(pp::storage_bytes<Format>(rows * columns) +
                          pp::storage_bytes<Format>(columns)),
      rows * columns *
          load_instructions_per_element(Format::total_bits, Lanes,
                                        "packed_arithmetic")};
  return {model,
          [=] {
            pp::packed_arithmetic_gemv<Format, Lanes>
                <<<static_cast<unsigned>(rows), pp::block_threads>>>(
                    matrix, vector, rows, columns, stride, result);
          },
          [=] {
            std::vector<double> host(rows);
            CUDA_CHECK(cudaMemcpy(host.data(), result, rows * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            return host;
          },
          {}};
}

template <typename Format, pp::arithmetic_kind Arithmetic>
void append_dot_variants(std::vector<timed_variant> &variants,
                         const pp::device_storage_t<Format> *left,
                         const pp::device_storage_t<Format> *right,
                         std::size_t count, int blocks, void *partial_storage,
                         double *result) {
  variants.push_back(
      make_dot_variant<Format, Arithmetic, pp::access_kind::scalar_single, 1>(
          left, right, count, blocks, partial_storage, result));
  variants.push_back(
      make_dot_variant<Format, Arithmetic, pp::access_kind::scalar_unrolled, 2>(
          left, right, count, blocks, partial_storage, result));
  variants.push_back(
      make_dot_variant<Format, Arithmetic, pp::access_kind::vector_packet, 2>(
          left, right, count, blocks, partial_storage, result));
  variants.push_back(
      make_dot_variant<Format, Arithmetic, pp::access_kind::scalar_unrolled, 4>(
          left, right, count, blocks, partial_storage, result));
  variants.push_back(
      make_dot_variant<Format, Arithmetic, pp::access_kind::vector_packet, 4>(
          left, right, count, blocks, partial_storage, result));
  variants.push_back(
      make_dot_variant<Format, Arithmetic, pp::access_kind::scalar_unrolled, 8>(
          left, right, count, blocks, partial_storage, result));
  variants.push_back(
      make_dot_variant<Format, Arithmetic, pp::access_kind::vector_packet, 8>(
          left, right, count, blocks, partial_storage, result));
}

template <typename Format, pp::arithmetic_kind Arithmetic>
void append_gemv_variants(std::vector<timed_variant> &variants,
                          const pp::device_storage_t<Format> *matrix,
                          const pp::device_storage_t<Format> *vector,
                          std::size_t rows, std::size_t columns,
                          std::size_t stride, double *result) {
  variants.push_back(
      make_gemv_variant<Format, Arithmetic, pp::access_kind::scalar_single, 1>(
          matrix, vector, rows, columns, stride, result));
  variants.push_back(
      make_gemv_variant<Format, Arithmetic, pp::access_kind::scalar_unrolled,
                        2>(matrix, vector, rows, columns, stride, result));
  variants.push_back(
      make_gemv_variant<Format, Arithmetic, pp::access_kind::vector_packet, 2>(
          matrix, vector, rows, columns, stride, result));
  variants.push_back(
      make_gemv_variant<Format, Arithmetic, pp::access_kind::scalar_unrolled,
                        4>(matrix, vector, rows, columns, stride, result));
  variants.push_back(
      make_gemv_variant<Format, Arithmetic, pp::access_kind::vector_packet, 4>(
          matrix, vector, rows, columns, stride, result));
  variants.push_back(
      make_gemv_variant<Format, Arithmetic, pp::access_kind::scalar_unrolled,
                        8>(matrix, vector, rows, columns, stride, result));
  variants.push_back(
      make_gemv_variant<Format, Arithmetic, pp::access_kind::vector_packet, 8>(
          matrix, vector, rows, columns, stride, result));
}

template <typename Format>
void add_packed_dot_variants(std::vector<timed_variant> &variants,
                             const pp::device_storage_t<Format> *left,
                             const pp::device_storage_t<Format> *right,
                             std::size_t count, int blocks,
                             void *partial_storage, double *result) {
  if constexpr (pp::supports_packed_arithmetic_v<Format>) {
    variants.push_back(make_packed_dot_variant<Format, 2>(
        left, right, count, blocks, partial_storage, result));
    variants.push_back(make_packed_dot_variant<Format, 4>(
        left, right, count, blocks, partial_storage, result));
    variants.push_back(make_packed_dot_variant<Format, 8>(
        left, right, count, blocks, partial_storage, result));
  }
}

template <typename Format>
void add_packed_gemv_variants(std::vector<timed_variant> &variants,
                              const pp::device_storage_t<Format> *matrix,
                              const pp::device_storage_t<Format> *vector,
                              std::size_t rows, std::size_t columns,
                              std::size_t stride, double *result) {
  if constexpr (pp::supports_packed_arithmetic_v<Format>) {
    variants.push_back(make_packed_gemv_variant<Format, 2>(
        matrix, vector, rows, columns, stride, result));
    variants.push_back(make_packed_gemv_variant<Format, 4>(
        matrix, vector, rows, columns, stride, result));
    variants.push_back(make_packed_gemv_variant<Format, 8>(
        matrix, vector, rows, columns, stride, result));
  }
}

result_metrics compare_results(const std::vector<double> &actual,
                               const std::vector<double> &reference) {
  if (actual.size() != reference.size()) {
    throw benchmark_error("result size mismatch");
  }
  double checksum{};
  double squared_error{};
  double squared_reference{};
  double maximum_absolute{};
  for (std::size_t index = 0; index < actual.size(); ++index) {
    checksum += actual[index];
    const auto difference = actual[index] - reference[index];
    squared_error += difference * difference;
    squared_reference += reference[index] * reference[index];
    maximum_absolute = std::max(maximum_absolute, std::abs(difference));
  }
  const auto relative = squared_reference > 0.0
                            ? std::sqrt(squared_error / squared_reference)
                            : std::sqrt(squared_error);
  return {checksum, maximum_absolute, relative};
}

class sample_writer {
public:
  sample_writer(const std::string &path, const device_info &device)
      : stream_{path}, device_{device} {
    if (!stream_) {
      throw benchmark_error("cannot open output: " + path);
    }
    stream_
        << "gpu,compute_capability,distribution,kernel,storage,storage_bits,"
           "arithmetic,family,lanes,m,n,round,sample,batch_iterations,"
           "time_ms,useful_flops,logical_storage_bytes,"
           "modeled_load_instructions,result_checksum,max_abs_error,"
           "relative_l2_error,useful_gflop_per_s,effective_gb_per_s\n";
    stream_ << std::setprecision(17);
  }

  void write(distribution kind, const work_model &model, int round, int sample,
             std::size_t batch_iterations, double time_ms,
             const result_metrics &result) {
    const auto seconds = time_ms * 1.0e-3;
    stream_ << device_.name << ',' << device_.capability << ',' << name(kind)
            << ',' << name(model.kernel) << ',' << selected_format::name << ','
            << selected_format::total_bits << ',' << name(model.arithmetic)
            << ',' << model.family << ',' << model.lanes << ',' << model.rows
            << ',' << model.columns << ',' << round << ',' << sample << ','
            << batch_iterations << ',' << time_ms << ',' << model.useful_flops
            << ',' << model.logical_storage_bytes << ','
            << model.modeled_load_instructions << ',' << result.value << ','
            << result.absolute_error << ',' << result.relative_error << ','
            << model.useful_flops / seconds / 1.0e9 << ','
            << model.logical_storage_bytes / seconds / 1.0e9 << '\n';
    stream_.flush();
  }

private:
  std::ofstream stream_;
  device_info device_;
};

void prepare_results(std::vector<timed_variant> &variants) {
  if (variants.empty()) {
    return;
  }
  std::vector<double> reference;
  // Always use the FP64 scalar variant as the storage-value reference.
  for (auto &variant : variants) {
    if (variant.model.arithmetic == pp::arithmetic_kind::fp64 &&
        variant.model.family == "scalar_single") {
      variant.launch();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      reference = variant.read_result();
      break;
    }
  }
  if (reference.empty()) {
    throw benchmark_error("missing FP64 scalar storage reference");
  }
  for (auto &variant : variants) {
    variant.launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    variant.result = compare_results(variant.read_result(), reference);
  }
}

void measure_variants(std::vector<timed_variant> &variants,
                      const options &settings, distribution kind,
                      sample_writer &writer, event_timer &timer,
                      std::mt19937_64 &shuffle) {
  prepare_results(variants);
  for (auto &variant : variants) {
    for (int warmup = 0; warmup < settings.warmup; ++warmup) {
      variant.launch();
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::size_t> iterations(variants.size(), 1);
  for (std::size_t index = 0; index < variants.size(); ++index) {
    const auto elapsed = timer.measure(variants[index].launch, 1);
    CUDA_CHECK(cudaGetLastError());
    const auto estimate = static_cast<std::size_t>(
        std::ceil(settings.target_sample_ms / std::max(elapsed, 1.0e-6)));
    iterations[index] =
        std::max<std::size_t>(1, std::min<std::size_t>(estimate, 100000));
  }

  std::vector<std::size_t> order(variants.size());
  std::iota(order.begin(), order.end(), 0);
  for (int round = 0; round < settings.rounds; ++round) {
    for (int sample = 0; sample < settings.samples; ++sample) {
      std::shuffle(order.begin(), order.end(), shuffle);
      for (const auto index : order) {
        const auto total =
            timer.measure(variants[index].launch, iterations[index]);
        CUDA_CHECK(cudaGetLastError());
        writer.write(kind, variants[index].model, round, sample,
                     iterations[index], total / iterations[index],
                     variants[index].result);
      }
    }
  }
}

template <typename Format>
void run_sweep(const options &settings, const device_info &device) {
  const auto max_dot = std::size_t{1} << settings.dot_powers.back();
  const auto max_columns = std::size_t{1} << settings.gemv_powers.back();
  const auto matrix_count = settings.gemv_rows * max_columns;
  const auto source_count = std::max(max_dot, matrix_count);
  const auto dot_blocks = std::max(1, device.multiprocessors * 4);

  device_buffer<double> source{source_count};
  device_buffer<pp::device_storage_t<Format>> dot_left{
      storage_elements<Format>(max_dot)};
  device_buffer<pp::device_storage_t<Format>> dot_right{
      storage_elements<Format>(max_dot)};
  device_buffer<pp::device_storage_t<Format>> matrix{
      storage_elements<Format>(matrix_count)};
  device_buffer<pp::device_storage_t<Format>> vector{
      storage_elements<Format>(max_columns)};
  device_buffer<std::uint64_t> partial_storage{
      static_cast<std::size_t>(dot_blocks)};
  device_buffer<double> dot_result{1};
  device_buffer<double> gemv_result{settings.gemv_rows};

  random_generator generator{settings.seed};
  event_timer timer;
  sample_writer writer{settings.output, device};
  std::mt19937_64 shuffle{settings.seed ^ 0x9e3779b97f4a7c15ULL};

  for (const auto kind :
       {distribution::uniform_0_1, distribution::normal_0_1}) {
    fill(generator, source.get(), max_dot, kind);
    encode<Format>(source.get(), dot_left.get(), max_dot,
                   device.multiprocessors);
    fill(generator, source.get(), max_dot, kind);
    encode<Format>(source.get(), dot_right.get(), max_dot,
                   device.multiprocessors);
    fill(generator, source.get(), matrix_count, kind);
    encode<Format>(source.get(), matrix.get(), matrix_count,
                   device.multiprocessors);
    fill(generator, source.get(), max_columns, kind);
    encode<Format>(source.get(), vector.get(), max_columns,
                   device.multiprocessors);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (const auto power : settings.dot_powers) {
      const auto count = std::size_t{1} << power;
      std::vector<timed_variant> variants;
      for_each_arithmetic<Format>([&](auto tag) {
        constexpr auto arithmetic = decltype(tag)::value;
        append_dot_variants<Format, arithmetic>(
            variants, dot_left.get(), dot_right.get(), count, dot_blocks,
            partial_storage.get(), dot_result.get());
      });
      add_packed_dot_variants<Format>(variants, dot_left.get(), dot_right.get(),
                                      count, dot_blocks, partial_storage.get(),
                                      dot_result.get());
      measure_variants(variants, settings, kind, writer, timer, shuffle);
      std::cout << name(kind) << " DOT " << Format::name << " N=2^" << power
                << " complete\n";
    }

    for (const auto power : settings.gemv_powers) {
      const auto columns = std::size_t{1} << power;
      std::vector<timed_variant> variants;
      for_each_arithmetic<Format>([&](auto tag) {
        constexpr auto arithmetic = decltype(tag)::value;
        append_gemv_variants<Format, arithmetic>(
            variants, matrix.get(), vector.get(), settings.gemv_rows, columns,
            max_columns, gemv_result.get());
      });
      add_packed_gemv_variants<Format>(variants, matrix.get(), vector.get(),
                                       settings.gemv_rows, columns, max_columns,
                                       gemv_result.get());
      measure_variants(variants, settings, kind, writer, timer, shuffle);
      std::cout << name(kind) << " GEMV " << Format::name
                << " M=" << settings.gemv_rows << " N=2^" << power
                << " complete\n";
    }
  }
}

struct component_model {
  std::string component;
  pp::arithmetic_kind arithmetic{};
  bool has_arithmetic{};
  std::string family;
  int lanes{};
  std::size_t logical_values{};
  int repeats{};
  double logical_storage_bytes{};
  double modeled_load_instructions{};
  double modeled_conversions{};
  double useful_flops{};
};

struct component_variant {
  component_model model;
  std::function<void()> launch;
};

template <typename Format, pp::access_kind Access, int Lanes>
component_variant
make_stream_load_variant(const pp::device_storage_t<Format> *values,
                         std::size_t count, int blocks,
                         std::uint64_t *checksums) {
  const auto family = std::string{name(Access)};
  const auto one_input_loads =
      0.5 * load_instructions_per_element(Format::total_bits, Lanes, family);
  return {{"stream_load", pp::arithmetic_kind::fp64, false, family, Lanes,
           count, 1, static_cast<double>(pp::storage_bytes<Format>(count)),
           count * one_input_loads, 0.0, 0.0},
          [=] {
            pp::stream_load<Format, Access, Lanes>
                <<<blocks, pp::block_threads>>>(values, count, checksums);
          }};
}

template <typename Format, pp::arithmetic_kind Arithmetic,
          pp::access_kind Access, int Lanes>
component_variant
make_stream_decode_variant(const pp::device_storage_t<Format> *values,
                           std::size_t count, int blocks, double *block_sums) {
  const auto family = std::string{name(Access)};
  const auto one_input_loads =
      0.5 * load_instructions_per_element(Format::total_bits, Lanes, family);
  return {{"stream_decode", Arithmetic, true, family, Lanes, count, 1,
           static_cast<double>(pp::storage_bytes<Format>(count)),
           count * one_input_loads, static_cast<double>(count), 0.0},
          [=] {
            pp::stream_decode<Format, Arithmetic, Access, Lanes>
                <<<blocks, pp::block_threads>>>(values, count, block_sums);
          }};
}

template <typename Format, pp::arithmetic_kind Arithmetic, int Lanes>
component_variant
make_register_decode_variant(const pp::device_storage_t<Format> *values,
                             std::size_t count, int blocks, int repeats,
                             double *thread_sums) {
  const auto threads = static_cast<std::size_t>(blocks) * pp::block_threads;
  return {{"register_decode", Arithmetic, true, "register_resident", Lanes,
           threads * Lanes * static_cast<std::size_t>(repeats), repeats,
           static_cast<double>(pp::storage_bytes<Format>(threads * Lanes)),
           static_cast<double>(threads * Lanes),
           static_cast<double>(threads * Lanes) * repeats, 0.0},
          [=] {
            pp::register_decode<Format, Arithmetic, Lanes>
                <<<blocks, pp::block_threads>>>(values, count, repeats,
                                                thread_sums);
          }};
}

template <pp::arithmetic_kind Arithmetic, int Chains>
component_variant make_arithmetic_chain_variant(int blocks, int repeats,
                                                double *thread_sums) {
  const auto threads = static_cast<std::size_t>(blocks) * pp::block_threads;
  return {{"arithmetic_chain", Arithmetic, true, "independent_chains", Chains,
           threads * Chains * static_cast<std::size_t>(repeats), repeats, 0.0,
           0.0, 0.0, 2.0 * threads * Chains * repeats},
          [=] {
            pp::arithmetic_chain<Arithmetic, Chains>
                <<<blocks, pp::block_threads>>>(repeats, thread_sums);
          }};
}

template <typename Format>
void append_stream_load_components(std::vector<component_variant> &variants,
                                   const pp::device_storage_t<Format> *values,
                                   std::size_t count, int blocks,
                                   std::uint64_t *checksums) {
  variants.push_back(
      make_stream_load_variant<Format, pp::access_kind::scalar_single, 1>(
          values, count, blocks, checksums));
  variants.push_back(
      make_stream_load_variant<Format, pp::access_kind::scalar_unrolled, 2>(
          values, count, blocks, checksums));
  variants.push_back(
      make_stream_load_variant<Format, pp::access_kind::vector_packet, 2>(
          values, count, blocks, checksums));
  variants.push_back(
      make_stream_load_variant<Format, pp::access_kind::scalar_unrolled, 4>(
          values, count, blocks, checksums));
  variants.push_back(
      make_stream_load_variant<Format, pp::access_kind::vector_packet, 4>(
          values, count, blocks, checksums));
  variants.push_back(
      make_stream_load_variant<Format, pp::access_kind::scalar_unrolled, 8>(
          values, count, blocks, checksums));
  variants.push_back(
      make_stream_load_variant<Format, pp::access_kind::vector_packet, 8>(
          values, count, blocks, checksums));
}

template <typename Format, pp::arithmetic_kind Arithmetic>
void append_arithmetic_components(std::vector<component_variant> &variants,
                                  const pp::device_storage_t<Format> *values,
                                  std::size_t stream_count,
                                  std::size_t register_count, int blocks,
                                  int decode_repeats, int arithmetic_repeats,
                                  double *block_sums, double *thread_sums) {
  variants.push_back(
      make_stream_decode_variant<Format, Arithmetic,
                                 pp::access_kind::scalar_single, 1>(
          values, stream_count, blocks, block_sums));
  variants.push_back(
      make_stream_decode_variant<Format, Arithmetic,
                                 pp::access_kind::scalar_unrolled, 2>(
          values, stream_count, blocks, block_sums));
  variants.push_back(
      make_stream_decode_variant<Format, Arithmetic,
                                 pp::access_kind::vector_packet, 2>(
          values, stream_count, blocks, block_sums));
  variants.push_back(
      make_stream_decode_variant<Format, Arithmetic,
                                 pp::access_kind::scalar_unrolled, 4>(
          values, stream_count, blocks, block_sums));
  variants.push_back(
      make_stream_decode_variant<Format, Arithmetic,
                                 pp::access_kind::vector_packet, 4>(
          values, stream_count, blocks, block_sums));
  variants.push_back(
      make_stream_decode_variant<Format, Arithmetic,
                                 pp::access_kind::scalar_unrolled, 8>(
          values, stream_count, blocks, block_sums));
  variants.push_back(
      make_stream_decode_variant<Format, Arithmetic,
                                 pp::access_kind::vector_packet, 8>(
          values, stream_count, blocks, block_sums));
  variants.push_back(make_register_decode_variant<Format, Arithmetic, 1>(
      values, register_count, blocks, decode_repeats, thread_sums));
  variants.push_back(make_register_decode_variant<Format, Arithmetic, 2>(
      values, register_count, blocks, decode_repeats, thread_sums));
  variants.push_back(make_register_decode_variant<Format, Arithmetic, 4>(
      values, register_count, blocks, decode_repeats, thread_sums));
  variants.push_back(make_register_decode_variant<Format, Arithmetic, 8>(
      values, register_count, blocks, decode_repeats, thread_sums));
  variants.push_back(make_arithmetic_chain_variant<Arithmetic, 1>(
      blocks, arithmetic_repeats, thread_sums));
  variants.push_back(make_arithmetic_chain_variant<Arithmetic, 2>(
      blocks, arithmetic_repeats, thread_sums));
  variants.push_back(make_arithmetic_chain_variant<Arithmetic, 4>(
      blocks, arithmetic_repeats, thread_sums));
  variants.push_back(make_arithmetic_chain_variant<Arithmetic, 8>(
      blocks, arithmetic_repeats, thread_sums));
}

class component_writer {
public:
  component_writer(const std::string &path, const device_info &device)
      : stream_{path}, device_{device} {
    if (!stream_) {
      throw benchmark_error("cannot open component output: " + path);
    }
    stream_ << "gpu,compute_capability,distribution,component,storage,"
               "storage_bits,arithmetic,family,lanes,logical_values,repeats,"
               "round,sample,batch_iterations,time_ms,logical_storage_bytes,"
               "modeled_load_instructions,modeled_conversions,useful_flops,"
               "values_per_s,effective_gb_per_s,useful_gflop_per_s\n";
    stream_ << std::setprecision(17);
  }

  void write(distribution kind, const component_model &model, int round,
             int sample, std::size_t batch_iterations, double time_ms) {
    const auto seconds = time_ms * 1.0e-3;
    stream_ << device_.name << ',' << device_.capability << ',' << name(kind)
            << ',' << model.component << ',' << selected_format::name << ','
            << selected_format::total_bits << ','
            << (model.has_arithmetic ? name(model.arithmetic) : "none") << ','
            << model.family << ',' << model.lanes << ',' << model.logical_values
            << ',' << model.repeats << ',' << round << ',' << sample << ','
            << batch_iterations << ',' << time_ms << ','
            << model.logical_storage_bytes << ','
            << model.modeled_load_instructions << ','
            << model.modeled_conversions << ',' << model.useful_flops << ','
            << model.logical_values / seconds << ','
            << model.logical_storage_bytes / seconds / 1.0e9 << ','
            << model.useful_flops / seconds / 1.0e9 << '\n';
    stream_.flush();
  }

private:
  std::ofstream stream_;
  device_info device_;
};

void measure_components(std::vector<component_variant> &variants,
                        const options &settings, distribution kind,
                        component_writer &writer, event_timer &timer,
                        std::mt19937_64 &shuffle) {
  for (auto &variant : variants) {
    for (int warmup = 0; warmup < settings.warmup; ++warmup) {
      variant.launch();
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::size_t> iterations(variants.size(), 1);
  for (std::size_t index = 0; index < variants.size(); ++index) {
    const auto elapsed = timer.measure(variants[index].launch, 1);
    const auto estimate = static_cast<std::size_t>(
        std::ceil(settings.target_sample_ms / std::max(elapsed, 1.0e-6)));
    iterations[index] =
        std::max<std::size_t>(1, std::min<std::size_t>(estimate, 100000));
  }
  std::vector<std::size_t> order(variants.size());
  std::iota(order.begin(), order.end(), 0);
  for (int round = 0; round < settings.rounds; ++round) {
    for (int sample = 0; sample < settings.samples; ++sample) {
      std::shuffle(order.begin(), order.end(), shuffle);
      for (const auto index : order) {
        const auto total =
            timer.measure(variants[index].launch, iterations[index]);
        writer.write(kind, variants[index].model, round, sample,
                     iterations[index], total / iterations[index]);
      }
    }
  }
}

template <typename Format>
void run_components(const options &settings, const device_info &device) {
  if (settings.component_n % 8 != 0 || settings.register_n < 8) {
    throw benchmark_error(
        "component N must be divisible by 8 and register N must be at least 8");
  }
  const auto blocks = std::max(1, device.multiprocessors * 4);
  const auto thread_count =
      static_cast<std::size_t>(blocks) * pp::block_threads;
  const auto input_count = std::max(settings.component_n, settings.register_n);
  device_buffer<double> source{input_count};
  device_buffer<pp::device_storage_t<Format>> input{
      storage_elements<Format>(input_count)};
  device_buffer<std::uint64_t> checksums{static_cast<std::size_t>(blocks)};
  device_buffer<double> block_sums{static_cast<std::size_t>(blocks)};
  device_buffer<double> thread_sums{thread_count};
  random_generator generator{settings.seed};
  event_timer timer;
  component_writer writer{settings.output, device};
  std::mt19937_64 shuffle{settings.seed ^ 0xd1b54a32d192ed03ULL};

  for (const auto kind :
       {distribution::uniform_0_1, distribution::normal_0_1}) {
    fill(generator, source.get(), input_count, kind);
    encode<Format>(source.get(), input.get(), input_count,
                   device.multiprocessors);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<component_variant> variants;
    append_stream_load_components<Format>(
        variants, input.get(), settings.component_n, blocks, checksums.get());
    for_each_arithmetic<Format>([&](auto tag) {
      constexpr auto arithmetic = decltype(tag)::value;
      append_arithmetic_components<Format, arithmetic>(
          variants, input.get(), settings.component_n, settings.register_n,
          blocks, settings.decode_repeats, settings.arithmetic_repeats,
          block_sums.get(), thread_sums.get());
    });
    measure_components(variants, settings, kind, writer, timer, shuffle);
    std::cout << name(kind) << " components " << Format::name << " complete\n";
  }
}

template <typename Format, pp::arithmetic_kind Arithmetic>
std::vector<timed_variant>
make_profile_variants(const options &settings,
                      const pp::device_storage_t<Format> *left_or_matrix,
                      const pp::device_storage_t<Format> *right_or_vector,
                      int dot_blocks, void *partials, double *result) {
  std::vector<timed_variant> variants;
  if (settings.profile_kernel == kernel_kind::dot) {
    append_dot_variants<Format, Arithmetic>(variants, left_or_matrix,
                                            right_or_vector, settings.profile_n,
                                            dot_blocks, partials, result);
    if constexpr (pp::supports_packed_arithmetic_v<Format>) {
      if constexpr (Arithmetic == pp::arithmetic_kind::fp16 ||
                    Arithmetic == pp::arithmetic_kind::bf16) {
        add_packed_dot_variants<Format>(variants, left_or_matrix,
                                        right_or_vector, settings.profile_n,
                                        dot_blocks, partials, result);
      }
    }
  } else {
    append_gemv_variants<Format, Arithmetic>(
        variants, left_or_matrix, right_or_vector, settings.profile_m,
        settings.profile_n, settings.profile_n, result);
    if constexpr (pp::supports_packed_arithmetic_v<Format>) {
      if constexpr (Arithmetic == pp::arithmetic_kind::fp16 ||
                    Arithmetic == pp::arithmetic_kind::bf16) {
        add_packed_gemv_variants<Format>(
            variants, left_or_matrix, right_or_vector, settings.profile_m,
            settings.profile_n, settings.profile_n, result);
      }
    }
  }
  return variants;
}

template <typename Format, pp::arithmetic_kind Arithmetic>
timed_variant
select_variant(const options &settings,
               const pp::device_storage_t<Format> *left_or_matrix,
               const pp::device_storage_t<Format> *right_or_vector,
               int dot_blocks, void *partials, double *result) {
  auto variants = make_profile_variants<Format, Arithmetic>(
      settings, left_or_matrix, right_or_vector, dot_blocks, partials, result);
  for (auto &variant : variants) {
    if (variant.model.family == settings.profile_family &&
        variant.model.lanes == settings.profile_lanes) {
      return std::move(variant);
    }
  }
  throw benchmark_error("requested profile variant is not valid");
}

template <typename Format, typename Callback>
void dispatch_arithmetic(pp::arithmetic_kind requested, Callback &&callback) {
  bool dispatched{};
  for_each_arithmetic<Format>([&](auto tag) {
    if (decltype(tag)::value == requested) {
      callback(tag);
      dispatched = true;
    }
  });
  if (!dispatched) {
    throw benchmark_error(
        "arithmetic type is not valid for this storage format");
  }
}

template <typename Format>
void run_profile(const options &settings, const device_info &device) {
  const auto logical_a = settings.profile_kernel == kernel_kind::dot
                             ? settings.profile_n
                             : settings.profile_m * settings.profile_n;
  const auto logical_b = settings.profile_n;
  const auto dot_blocks = std::max(1, device.multiprocessors * 4);
  device_buffer<double> source{std::max(logical_a, logical_b)};
  device_buffer<pp::device_storage_t<Format>> a{
      storage_elements<Format>(logical_a)};
  device_buffer<pp::device_storage_t<Format>> b{
      storage_elements<Format>(logical_b)};
  device_buffer<std::uint64_t> partials{static_cast<std::size_t>(dot_blocks)};
  device_buffer<double> result{
      settings.profile_kernel == kernel_kind::dot ? 1 : settings.profile_m};
  random_generator generator{settings.seed};
  fill(generator, source.get(), logical_a, settings.profile_distribution);
  encode<Format>(source.get(), a.get(), logical_a, device.multiprocessors);
  fill(generator, source.get(), logical_b, settings.profile_distribution);
  encode<Format>(source.get(), b.get(), logical_b, device.multiprocessors);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<timed_variant> selected;
  dispatch_arithmetic<Format>(settings.profile_arithmetic, [&](auto tag) {
    constexpr auto arithmetic = decltype(tag)::value;
    if (settings.profile_family == "all") {
      selected = make_profile_variants<Format, arithmetic>(
          settings, a.get(), b.get(), dot_blocks, partials.get(), result.get());
    } else {
      selected.push_back(select_variant<Format, arithmetic>(
          settings, a.get(), b.get(), dot_blocks, partials.get(),
          result.get()));
    }
  });
  for (int warmup = 0; warmup < settings.warmup; ++warmup) {
    for (auto &variant : selected) {
      variant.launch();
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::ofstream output{settings.output};
  if (!output) {
    throw benchmark_error("cannot open profile metadata output");
  }
  output << "profile_sequence,distribution,kernel,storage,storage_bits,"
            "arithmetic,family,lanes,m,n,useful_flops,logical_storage_bytes,"
            "modeled_load_instructions,timing_status\n";
  for (std::size_t sequence = 0; sequence < selected.size(); ++sequence) {
    const auto &model = selected[sequence].model;
    output << sequence << ',' << name(settings.profile_distribution) << ','
           << name(settings.profile_kernel) << ',' << Format::name << ','
           << Format::total_bits << ',' << name(model.arithmetic) << ','
           << model.family << ',' << model.lanes << ',' << model.rows << ','
           << model.columns << ',' << model.useful_flops << ','
           << model.logical_storage_bytes << ','
           << model.modeled_load_instructions << ",profiler_contaminated\n";
  }
  output.flush();

  CUDA_CHECK(cudaProfilerStart());
  for (auto &variant : selected) {
    variant.launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaProfilerStop());
}

template <typename Format>
void run_validation(const options &settings, const device_info &device) {
  options validation = settings;
  validation.dot_powers = {12};
  validation.gemv_powers = {8};
  validation.gemv_rows = 32;
  validation.warmup = 1;
  validation.rounds = 1;
  validation.samples = 1;
  validation.target_sample_ms = 0.1;
  run_sweep<Format>(validation, device);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto settings = parse_options(argc, argv);
    const auto device = query_device();
    std::cout << "Precision packing benchmark\n"
              << "GPU: " << device.name << " (" << device.capability << ")\n"
              << "Storage: " << selected_format::name << '\n';
    if (settings.mode == run_mode::sweep) {
      run_sweep<selected_format>(settings, device);
    } else if (settings.mode == run_mode::components) {
      run_components<selected_format>(settings, device);
    } else if (settings.mode == run_mode::validate) {
      run_validation<selected_format>(settings, device);
    } else {
      run_profile<selected_format>(settings, device);
    }
    std::cout << "Wrote " << settings.output << '\n';
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
