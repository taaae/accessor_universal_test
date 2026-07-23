#include "dot_dataset.hpp"
#include "memory_accessor.hpp"

#include <cub/block/block_reduce.cuh>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr int block_threads = 256;
constexpr int blocks_per_sm = 16;

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

#define CUDA_CHECK(expression)                                                 \
  check_cuda((expression), #expression, __FILE__, __LINE__)

class cuda_stream {
public:
  cuda_stream() {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  ~cuda_stream() {
    if (stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  cuda_stream(const cuda_stream &) = delete;
  cuda_stream &operator=(const cuda_stream &) = delete;

  operator cudaStream_t() const { return stream_; }

private:
  cudaStream_t stream_{};
};

class cuda_event {
public:
  cuda_event() { CUDA_CHECK(cudaEventCreate(&event_)); }

  ~cuda_event() {
    if (event_) {
      cudaEventDestroy(event_);
    }
  }

  cuda_event(const cuda_event &) = delete;
  cuda_event &operator=(const cuda_event &) = delete;

  operator cudaEvent_t() const { return event_; }

private:
  cudaEvent_t event_{};
};

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

enum class mode { performance, accuracy, profile };
enum class variant { raw_f32, raw_f64, accessor_f32_f32, accessor_f64_f32 };

inline constexpr std::array<variant, 4> variants{
    variant::raw_f32, variant::raw_f64, variant::accessor_f32_f32,
    variant::accessor_f64_f32};

const char *to_string(mode value) {
  switch (value) {
  case mode::performance:
    return "performance";
  case mode::accuracy:
    return "accuracy";
  case mode::profile:
    return "profile";
  }
  return "unknown";
}

const char *to_string(variant value) {
  switch (value) {
  case variant::raw_f32:
    return "raw_f32";
  case variant::raw_f64:
    return "raw_f64";
  case variant::accessor_f32_f32:
    return "accessor_f32_f32";
  case variant::accessor_f64_f32:
    return "accessor_f64_f32";
  }
  return "unknown";
}

const char *arithmetic_name(variant value) {
  return value == variant::raw_f32 || value == variant::accessor_f32_f32
             ? "float32"
             : "float64";
}

const char *access_name(variant value) {
  return value == variant::raw_f32 || value == variant::raw_f64 ? "raw"
                                                                : "accessor";
}

bool uses_fp64(variant value) {
  return value == variant::raw_f64 || value == variant::accessor_f64_f32;
}

struct options {
  mode run_mode{mode::performance};
  bool all_variants{true};
  variant selected_variant{variant::raw_f32};
  std::vector<int> powers;
  std::size_t exact_count{};
  int warmup{5};
  int rounds{3};
  int samples{20};
  int iterations{};
  double target_sample_ms{50.0};
  std::string dataset_id{aut::dataset::performance_case.id};
  std::string output{"-"};
  bool powers_set{false};
  bool help{false};
};

std::vector<int> parse_int_list(const std::string &value) {
  std::vector<int> result;
  std::stringstream stream{value};
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      throw benchmark_error("empty item in integer list: " + value);
    }
    result.push_back(std::stoi(token));
  }
  if (result.empty()) {
    throw benchmark_error("integer list must not be empty");
  }
  return result;
}

std::size_t parse_size(const std::string &value) {
  std::size_t consumed{};
  const auto parsed = std::stoull(value, &consumed, 0);
  if (consumed != value.size() || parsed == 0) {
    throw benchmark_error("expected a positive integer: " + value);
  }
  return static_cast<std::size_t>(parsed);
}

variant parse_variant(const std::string &value) {
  for (const auto candidate : variants) {
    if (value == to_string(candidate)) {
      return candidate;
    }
  }
  throw benchmark_error("unknown variant: " + value);
}

options parse_options(int argc, char **argv) {
  options result;
  auto require_value = [&](int &index, const std::string &option_name) {
    if (++index >= argc) {
      throw benchmark_error("missing value after " + option_name);
    }
    return std::string{argv[index]};
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg{argv[i]};
    if (arg == "--help" || arg == "-h") {
      result.help = true;
    } else if (arg == "--mode") {
      const auto value = require_value(i, arg);
      if (value == "performance") {
        result.run_mode = mode::performance;
      } else if (value == "accuracy") {
        result.run_mode = mode::accuracy;
      } else if (value == "profile") {
        result.run_mode = mode::profile;
      } else {
        throw benchmark_error("unknown mode: " + value);
      }
    } else if (arg == "--variant") {
      const auto value = require_value(i, arg);
      if (value == "all") {
        result.all_variants = true;
      } else {
        result.all_variants = false;
        result.selected_variant = parse_variant(value);
      }
    } else if (arg == "--powers") {
      result.powers = parse_int_list(require_value(i, arg));
      result.powers_set = true;
    } else if (arg == "--n") {
      result.exact_count = parse_size(require_value(i, arg));
    } else if (arg == "--warmup") {
      result.warmup = std::stoi(require_value(i, arg));
    } else if (arg == "--rounds") {
      result.rounds = std::stoi(require_value(i, arg));
    } else if (arg == "--samples") {
      result.samples = std::stoi(require_value(i, arg));
    } else if (arg == "--iterations") {
      result.iterations = std::stoi(require_value(i, arg));
    } else if (arg == "--target-sample-ms") {
      result.target_sample_ms = std::stod(require_value(i, arg));
    } else if (arg == "--dataset") {
      result.dataset_id = require_value(i, arg);
    } else if (arg == "--output") {
      result.output = require_value(i, arg);
    } else {
      throw benchmark_error("unknown argument: " + arg);
    }
  }

  if (result.warmup < 0 || result.rounds <= 0 || result.samples <= 0 ||
      result.iterations < 0 || result.target_sample_ms <= 0.0) {
    throw benchmark_error(
        "warmup must be nonnegative; rounds, samples, and target time must be "
        "positive; iterations must be nonnegative");
  }
  if (result.exact_count != 0 && result.powers_set) {
    throw benchmark_error("--n and --powers are mutually exclusive");
  }
  if (result.run_mode == mode::profile && result.all_variants) {
    throw benchmark_error("profile mode requires one explicit --variant");
  }
  if (!result.powers_set) {
    result.powers = result.run_mode == mode::accuracy
                        ? std::vector<int>{10, 16, 20}
                        : std::vector<int>{10, 16, 20, 23, 26, 27};
  }
  for (const auto power : result.powers) {
    if (power < 0 || power > 30) {
      throw benchmark_error("powers must be in [0, 30]");
    }
  }
  return result;
}

void print_help(const char *program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "  --mode performance|accuracy|profile\n"
            << "  --variant all|raw_f32|raw_f64|accessor_f32_f32|"
               "accessor_f64_f32\n"
            << "  --powers comma-separated-powers | --n element-count\n"
            << "  --dataset dataset-id       (profile mode)\n"
            << "  --warmup count\n"
            << "  --rounds count             (performance mode)\n"
            << "  --samples count            (performance mode)\n"
            << "  --iterations count         (0 calibrates automatically)\n"
            << "  --target-sample-ms ms\n"
            << "  --output path              ('-' writes CSV to stdout)\n";
}

const aut::dataset::case_spec &find_dataset(const std::string &id) {
  for (const auto &candidate : aut::dataset::accuracy_cases) {
    if (id == candidate.id) {
      return candidate;
    }
  }
  throw benchmark_error("unknown dataset id: " + id);
}

std::vector<std::size_t> selected_counts(const options &opts) {
  if (opts.exact_count != 0) {
    return {opts.exact_count};
  }
  std::vector<std::size_t> result;
  result.reserve(opts.powers.size());
  for (const auto power : opts.powers) {
    result.push_back(std::size_t{1} << power);
  }
  return result;
}

struct environment {
  std::string gpu_name;
  int compute_major{};
  int compute_minor{};
  int multiprocessors{};
  int cuda_runtime{};
  int cuda_driver{};
};

environment query_environment() {
  int device{};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  environment result;
  result.gpu_name = properties.name;
  result.compute_major = properties.major;
  result.compute_minor = properties.minor;
  result.multiprocessors = properties.multiProcessorCount;
  CUDA_CHECK(cudaRuntimeGetVersion(&result.cuda_runtime));
  CUDA_CHECK(cudaDriverGetVersion(&result.cuda_driver));
  return result;
}

std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string escaped{"\""};
  for (const auto character : value) {
    if (character == '"') {
      escaped += "\"\"";
    } else {
      escaped += character;
    }
  }
  escaped += '"';
  return escaped;
}

std::string timestamp_utc() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  std::ostringstream result;
  result << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return result.str();
}

__global__ void fill_inputs(float *x, float *y, std::size_t count,
                            aut::dataset::parameters spec) {
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto i = first; i < count; i += stride) {
    x[i] = aut::dataset::value(spec, 0, i, count);
    y[i] = aut::dataset::value(spec, 1, i, count);
  }
}

void initialize_inputs(float *x, float *y, std::size_t count,
                       aut::dataset::parameters spec, int multiprocessors,
                       cudaStream_t stream) {
  const auto wanted =
      (count + static_cast<std::size_t>(block_threads) - 1) / block_threads;
  const auto blocks = static_cast<int>(std::min<std::size_t>(
      wanted, static_cast<std::size_t>(multiprocessors * blocks_per_sm)));
  fill_inputs<<<blocks, block_threads, 0, stream>>>(x, y, count, spec);
  CUDA_CHECK(cudaGetLastError());
}

int first_stage_blocks(std::size_t count, int multiprocessors) {
  const auto wanted =
      (count + static_cast<std::size_t>(block_threads) - 1) / block_threads;
  return static_cast<int>(std::min<std::size_t>(
      wanted, static_cast<std::size_t>(multiprocessors * blocks_per_sm)));
}

template <typename ArithmeticType, typename Input>
__global__ void dot_map_reduce_kernel(Input x, Input y, std::size_t count,
                                      ArithmeticType *partials) {
  using block_reduce = cub::BlockReduce<ArithmeticType, block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;

  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  ArithmeticType sum{};
  for (auto i = first; i < count; i += stride) {
    const auto x_value = static_cast<ArithmeticType>(x[i]);
    const auto y_value = static_cast<ArithmeticType>(y[i]);
    sum += x_value * y_value;
  }
  const auto block_sum = block_reduce(temporary).Sum(sum);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = block_sum;
  }
}

template <typename ArithmeticType>
__global__ void dot_finalize_kernel(const ArithmeticType *partials,
                                    std::size_t partial_count,
                                    ArithmeticType *result) {
  using block_reduce = cub::BlockReduce<ArithmeticType, block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;

  ArithmeticType sum{};
  for (auto i = static_cast<std::size_t>(threadIdx.x); i < partial_count;
       i += blockDim.x) {
    sum += partials[i];
  }
  const auto final_sum = block_reduce(temporary).Sum(sum);
  if (threadIdx.x == 0) {
    *result = final_sum;
  }
}

template <typename ArithmeticType> struct dot_workspace {
  explicit dot_workspace(std::size_t partial_count)
      : partials{partial_count}, result{1} {}

  device_buffer<ArithmeticType> partials;
  device_buffer<ArithmeticType> result;
};

template <typename ArithmeticType, typename Input>
void launch_dot(Input x, Input y, std::size_t count, int blocks,
                dot_workspace<ArithmeticType> &workspace, cudaStream_t stream) {
  dot_map_reduce_kernel<ArithmeticType><<<blocks, block_threads, 0, stream>>>(
      x, y, count, workspace.partials.get());
  CUDA_CHECK(cudaGetLastError());
  dot_finalize_kernel<ArithmeticType><<<1, block_threads, 0, stream>>>(
      workspace.partials.get(), static_cast<std::size_t>(blocks),
      workspace.result.get());
  CUDA_CHECK(cudaGetLastError());
}

void launch_variant(variant selected, const float *x, const float *y,
                    std::size_t count, int blocks, dot_workspace<float> &fp32,
                    dot_workspace<double> &fp64, cudaStream_t stream) {
  switch (selected) {
  case variant::raw_f32:
    launch_dot<float>(x, y, count, blocks, fp32, stream);
    break;
  case variant::raw_f64:
    launch_dot<double>(x, y, count, blocks, fp64, stream);
    break;
  case variant::accessor_f32_f32:
    launch_dot<float>(aut::memory_accessor<float, float>{x},
                      aut::memory_accessor<float, float>{y}, count, blocks,
                      fp32, stream);
    break;
  case variant::accessor_f64_f32:
    launch_dot<double>(aut::memory_accessor<double, float>{x},
                       aut::memory_accessor<double, float>{y}, count, blocks,
                       fp64, stream);
    break;
  }
}

template <typename Launch>
float elapsed_ms(cudaStream_t stream, Launch &&launch, int iterations) {
  cuda_event start;
  cuda_event stop;
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds{};
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  return milliseconds;
}

template <typename Launch>
int calibrate_iterations(const options &opts, cudaStream_t stream,
                         const Launch &launch) {
  if (opts.iterations != 0) {
    return opts.iterations;
  }
  const auto calibration_ms = elapsed_ms(stream, launch, 1);
  const auto wanted = static_cast<int>(
      std::ceil(opts.target_sample_ms /
                std::max(0.001, static_cast<double>(calibration_ms))));
  return std::clamp(wanted, 1, 100000);
}

std::ostream *open_output(const std::string &path, std::ofstream &file) {
  if (path == "-") {
    return &std::cout;
  }
  file.open(path);
  if (!file) {
    throw benchmark_error("could not open output file: " + path);
  }
  return &file;
}

void write_common_environment(std::ostream &output, const environment &env) {
  output << csv_escape(env.gpu_name) << ',' << env.compute_major << '.'
         << env.compute_minor << ',' << env.multiprocessors << ','
         << env.cuda_runtime << ',' << env.cuda_driver;
}

void run_performance(const options &opts, const environment &env,
                     cudaStream_t stream, std::ostream &output) {
  output
      << "gpu,compute_capability,multiprocessors,cuda_runtime,cuda_driver,"
         "dataset,variant,storage_type,arithmetic_type,access_kind,n,"
         "algorithmic_bytes,algorithmic_flops,"
         "arithmetic_intensity_flop_per_byte,blocks,threads,round,"
         "order_in_round,iterations,sample,total_time_ms,giga_elements_per_s,"
         "algorithmic_gb_per_s,gflop_per_s\n";

  const auto counts = selected_counts(opts);
  for (std::size_t count_index = 0; count_index < counts.size();
       ++count_index) {
    const auto count = counts[count_index];
    const auto blocks = first_stage_blocks(count, env.multiprocessors);
    device_buffer<float> x{count};
    device_buffer<float> y{count};
    dot_workspace<float> fp32{static_cast<std::size_t>(blocks)};
    dot_workspace<double> fp64{static_cast<std::size_t>(blocks)};
    initialize_inputs(x.get(), y.get(), count,
                      aut::dataset::performance_case.values,
                      env.multiprocessors, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::array<int, variants.size()> iteration_counts{};
    for (std::size_t index = 0; index < variants.size(); ++index) {
      const auto selected = variants[index];
      const auto launch = [&] {
        launch_variant(selected, x.get(), y.get(), count, blocks, fp32, fp64,
                       stream);
      };
      for (int warmup = 0; warmup < opts.warmup; ++warmup) {
        launch();
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      iteration_counts[index] = calibrate_iterations(opts, stream, launch);
    }

    for (int round = 0; round < opts.rounds; ++round) {
      const auto rotation = (static_cast<int>(count_index) + round) %
                            static_cast<int>(variants.size());
      for (std::size_t order = 0; order < variants.size(); ++order) {
        const auto variant_index =
            (order + static_cast<std::size_t>(rotation)) % variants.size();
        const auto selected = variants[variant_index];
        const auto iterations = iteration_counts[variant_index];
        const auto launch = [&] {
          launch_variant(selected, x.get(), y.get(), count, blocks, fp32, fp64,
                         stream);
        };
        for (int sample = 0; sample < opts.samples; ++sample) {
          const auto total_ms = elapsed_ms(stream, launch, iterations);
          const auto operation_ms = static_cast<double>(total_ms) / iterations;
          const auto seconds = operation_ms * 1.0e-3;
          const auto algorithmic_bytes = 8.0 * static_cast<double>(count);
          const auto algorithmic_flops = 2.0 * static_cast<double>(count);

          write_common_environment(output, env);
          output << ',' << aut::dataset::performance_case.id << ','
                 << to_string(selected) << ",float32,"
                 << arithmetic_name(selected) << ',' << access_name(selected)
                 << ',' << count << ',' << std::setprecision(17)
                 << algorithmic_bytes << ',' << algorithmic_flops << ",0.25,"
                 << blocks << ',' << block_threads << ',' << round << ','
                 << order << ',' << iterations << ',' << sample << ','
                 << operation_ms << ','
                 << static_cast<double>(count) / seconds / 1.0e9 << ','
                 << algorithmic_bytes / seconds / 1.0e9 << ','
                 << algorithmic_flops / seconds / 1.0e9 << '\n';
        }
        output.flush();
      }
    }
    std::cerr << "Performance N=" << count << " complete\n";
  }
}

class compensated_sum {
public:
  void add(long double value) {
    const auto updated = sum_ + value;
    if (std::fabs(sum_) >= std::fabs(value)) {
      correction_ += (sum_ - updated) + value;
    } else {
      correction_ += (value - updated) + sum_;
    }
    sum_ = updated;
  }

  long double value() const { return sum_ + correction_; }

private:
  long double sum_{};
  long double correction_{};
};

struct reference_result {
  long double dot{};
  long double sum_abs{};
  std::uint64_t x_fingerprint{1469598103934665603ULL};
  std::uint64_t y_fingerprint{1469598103934665603ULL};
};

std::uint32_t float_bits(float value) {
  std::uint32_t result{};
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

void update_fingerprint(std::uint64_t &fingerprint, float value) {
  fingerprint ^= float_bits(value);
  fingerprint *= 1099511628211ULL;
}

reference_result compute_reference(aut::dataset::parameters spec,
                                   std::size_t count) {
  compensated_sum dot;
  compensated_sum sum_abs;
  reference_result result;
  for (std::size_t i = 0; i < count; ++i) {
    const auto x = aut::dataset::value(spec, 0, i, count);
    const auto y = aut::dataset::value(spec, 1, i, count);
    update_fingerprint(result.x_fingerprint, x);
    update_fingerprint(result.y_fingerprint, y);
    const auto product =
        static_cast<long double>(x) * static_cast<long double>(y);
    dot.add(product);
    sum_abs.add(std::fabs(product));
  }
  result.dot = dot.value();
  result.sum_abs = sum_abs.value();
  return result;
}

template <typename T> std::string value_bits(T value) {
  static_assert(sizeof(T) == sizeof(std::uint32_t) ||
                sizeof(T) == sizeof(std::uint64_t));
  std::ostringstream result;
  result << "0x" << std::hex << std::setfill('0');
  if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
    std::uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    result << std::setw(8) << bits;
  } else {
    std::uint64_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    result << std::setw(16) << bits;
  }
  return result.str();
}

struct computed_result {
  long double value{};
  std::string bits;
};

computed_result copy_result(variant selected, dot_workspace<float> &fp32,
                            dot_workspace<double> &fp64, cudaStream_t stream) {
  if (uses_fp64(selected)) {
    double value{};
    CUDA_CHECK(cudaMemcpyAsync(&value, fp64.result.get(), sizeof(value),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return {static_cast<long double>(value), value_bits(value)};
  }
  float value{};
  CUDA_CHECK(cudaMemcpyAsync(&value, fp32.result.get(), sizeof(value),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return {static_cast<long double>(value), value_bits(value)};
}

void run_accuracy(const options &opts, const environment &env,
                  cudaStream_t stream, std::ostream &output) {
  output
      << "gpu,compute_capability,multiprocessors,cuda_runtime,cuda_driver,"
         "dataset,variant,storage_type,arithmetic_type,access_kind,n,blocks,"
         "threads,result,result_bits,reference,absolute_error,relative_error,"
         "normalized_error,dot_condition_number,x_fingerprint,"
         "y_fingerprint,finite\n";

  for (const auto &dataset : aut::dataset::accuracy_cases) {
    for (const auto count : selected_counts(opts)) {
      const auto blocks = first_stage_blocks(count, env.multiprocessors);
      device_buffer<float> x{count};
      device_buffer<float> y{count};
      dot_workspace<float> fp32{static_cast<std::size_t>(blocks)};
      dot_workspace<double> fp64{static_cast<std::size_t>(blocks)};
      initialize_inputs(x.get(), y.get(), count, dataset.values,
                        env.multiprocessors, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const auto reference = compute_reference(dataset.values, count);

      for (const auto selected : variants) {
        for (int warmup = 0; warmup < opts.warmup; ++warmup) {
          launch_variant(selected, x.get(), y.get(), count, blocks, fp32, fp64,
                         stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        launch_variant(selected, x.get(), y.get(), count, blocks, fp32, fp64,
                       stream);
        const auto computed = copy_result(selected, fp32, fp64, stream);
        const auto absolute_error = std::fabs(computed.value - reference.dot);
        const auto relative_error =
            reference.dot == 0.0L
                ? std::numeric_limits<long double>::quiet_NaN()
                : absolute_error / std::fabs(reference.dot);
        const auto normalized_error =
            reference.sum_abs == 0.0L
                ? std::numeric_limits<long double>::quiet_NaN()
                : absolute_error / reference.sum_abs;
        const auto condition =
            reference.dot == 0.0L
                ? std::numeric_limits<long double>::infinity()
                : reference.sum_abs / std::fabs(reference.dot);

        write_common_environment(output, env);
        output << ',' << dataset.id << ',' << to_string(selected) << ",float32,"
               << arithmetic_name(selected) << ',' << access_name(selected)
               << ',' << count << ',' << blocks << ',' << block_threads << ','
               << std::setprecision(21) << computed.value << ','
               << computed.bits << ',' << reference.dot << ',' << absolute_error
               << ',' << relative_error << ',' << normalized_error << ','
               << condition << ",0x" << std::hex << reference.x_fingerprint
               << ",0x" << reference.y_fingerprint << std::dec << ','
               << (std::isfinite(computed.value) ? "true" : "false") << '\n';
      }
      output.flush();
      std::cerr << "Accuracy " << dataset.id << " N=" << count << " complete\n";
    }
  }
}

void run_profile(const options &opts, const environment &env,
                 cudaStream_t stream, std::ostream &output) {
  const auto &dataset = find_dataset(opts.dataset_id);
  const auto counts = selected_counts(opts);
  if (counts.size() != 1) {
    throw benchmark_error("profile mode requires exactly one size");
  }
  const auto count = counts.front();
  const auto blocks = first_stage_blocks(count, env.multiprocessors);
  device_buffer<float> x{count};
  device_buffer<float> y{count};
  dot_workspace<float> fp32{static_cast<std::size_t>(blocks)};
  dot_workspace<double> fp64{static_cast<std::size_t>(blocks)};
  initialize_inputs(x.get(), y.get(), count, dataset.values,
                    env.multiprocessors, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const auto launch = [&] {
    launch_variant(opts.selected_variant, x.get(), y.get(), count, blocks, fp32,
                   fp64, stream);
  };
  for (int warmup = 0; warmup < opts.warmup; ++warmup) {
    launch();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaProfilerStart());
  launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaProfilerStop());
  const auto computed = copy_result(opts.selected_variant, fp32, fp64, stream);

  output
      << "gpu,compute_capability,multiprocessors,cuda_runtime,cuda_driver,"
         "mode,dataset,variant,storage_type,arithmetic_type,access_kind,n,"
         "algorithmic_bytes,algorithmic_flops,"
         "arithmetic_intensity_flop_per_byte,blocks,threads,result,result_bits,"
         "timing_status\n";
  write_common_environment(output, env);
  output << ',' << to_string(opts.run_mode) << ',' << dataset.id << ','
         << to_string(opts.selected_variant) << ",float32,"
         << arithmetic_name(opts.selected_variant) << ','
         << access_name(opts.selected_variant) << ',' << count << ','
         << 8.0 * static_cast<double>(count) << ','
         << 2.0 * static_cast<double>(count) << ",0.25," << blocks << ','
         << block_threads << ',' << std::setprecision(21) << computed.value
         << ',' << computed.bits << ",profiler_contaminated\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto opts = parse_options(argc, argv);
    if (opts.help) {
      print_help(argv[0]);
      return EXIT_SUCCESS;
    }

    cuda_stream stream;
    const auto env = query_environment();
    std::ofstream output_file;
    auto *output = open_output(opts.output, output_file);

    std::cerr << "accessor_dot_bench " << timestamp_utc() << '\n'
              << "GPU: " << env.gpu_name << " (sm_" << env.compute_major
              << env.compute_minor << ", " << env.multiprocessors << " SMs)\n"
              << "CUDA runtime/driver: " << env.cuda_runtime << '/'
              << env.cuda_driver << "\nMode: " << to_string(opts.run_mode)
              << '\n';

    switch (opts.run_mode) {
    case mode::performance:
      run_performance(opts, env, stream, *output);
      break;
    case mode::accuracy:
      run_accuracy(opts, env, stream, *output);
      break;
    case mode::profile:
      run_profile(opts, env, stream, *output);
      break;
    }

    if (opts.output != "-") {
      std::cerr << "Wrote " << opts.output << '\n';
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
