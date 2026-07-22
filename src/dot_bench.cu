#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

class benchmark_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

const char *cublas_status_name(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "unknown cuBLAS status";
  }
}

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

void check_cublas(cublasStatus_t status, const char *expression,
                  const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << ": "
            << cublas_status_name(status) << " (" << static_cast<int>(status)
            << ')';
    throw benchmark_error(message.str());
  }
}

#define CUDA_CHECK(expression)                                                 \
  check_cuda((expression), #expression, __FILE__, __LINE__)
#define CUBLAS_CHECK(expression)                                               \
  check_cublas((expression), #expression, __FILE__, __LINE__)

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

class cublas_handle {
public:
  explicit cublas_handle(cudaStream_t stream) {
    CUBLAS_CHECK(cublasCreate(&handle_));
    CUBLAS_CHECK(cublasSetStream(handle_, stream));
    CUBLAS_CHECK(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE));
  }

  ~cublas_handle() {
    if (handle_) {
      cublasDestroy(handle_);
    }
  }

  cublas_handle(const cublas_handle &) = delete;
  cublas_handle &operator=(const cublas_handle &) = delete;

  operator cublasHandle_t() const { return handle_; }

private:
  cublasHandle_t handle_{};
};

template <typename T> class device_buffer {
public:
  explicit device_buffer(std::size_t count) : count_{count} {
    if (count_ != 0) {
      CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
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

enum class mode { performance, accuracy, profile };
enum class operation { sdot, ddot, both };
enum class dataset { positive, signed_uniform, cancellation, dynamic_range };

struct options {
  mode run_mode{mode::performance};
  operation op{operation::both};
  dataset data{dataset::signed_uniform};
  std::vector<int> powers{10, 16, 20, 23, 26, 27};
  int warmup{5};
  int samples{20};
  int iterations{0};
  double target_sample_ms{20.0};
  std::uint64_t seed{0x123456789abcdef0ULL};
  std::string output{"-"};
  bool help{false};
};

std::string to_string(mode value) {
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

std::string to_string(dataset value) {
  switch (value) {
  case dataset::positive:
    return "positive";
  case dataset::signed_uniform:
    return "signed";
  case dataset::cancellation:
    return "cancellation";
  case dataset::dynamic_range:
    return "dynamic";
  }
  return "unknown";
}

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

std::uint64_t parse_u64(const std::string &value) {
  std::size_t consumed{};
  const auto result = std::stoull(value, &consumed, 0);
  if (consumed != value.size()) {
    throw benchmark_error("invalid integer: " + value);
  }
  return result;
}

options parse_options(int argc, char **argv) {
  options result;
  bool powers_set = false;

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
    } else if (arg == "--operation") {
      const auto value = require_value(i, arg);
      if (value == "sdot") {
        result.op = operation::sdot;
      } else if (value == "ddot") {
        result.op = operation::ddot;
      } else if (value == "both") {
        result.op = operation::both;
      } else {
        throw benchmark_error("unknown operation: " + value);
      }
    } else if (arg == "--dataset") {
      const auto value = require_value(i, arg);
      if (value == "positive") {
        result.data = dataset::positive;
      } else if (value == "signed") {
        result.data = dataset::signed_uniform;
      } else if (value == "cancellation") {
        result.data = dataset::cancellation;
      } else if (value == "dynamic") {
        result.data = dataset::dynamic_range;
      } else {
        throw benchmark_error("unknown dataset: " + value);
      }
    } else if (arg == "--powers") {
      result.powers = parse_int_list(require_value(i, arg));
      powers_set = true;
    } else if (arg == "--n") {
      const auto n = parse_u64(require_value(i, arg));
      if (n == 0 || (n & (n - 1)) != 0) {
        throw benchmark_error("--n must be a nonzero power of two");
      }
      int power = 0;
      for (auto value = n; value > 1; value >>= 1) {
        ++power;
      }
      result.powers = {power};
      powers_set = true;
    } else if (arg == "--warmup") {
      result.warmup = std::stoi(require_value(i, arg));
    } else if (arg == "--samples") {
      result.samples = std::stoi(require_value(i, arg));
    } else if (arg == "--iterations") {
      result.iterations = std::stoi(require_value(i, arg));
    } else if (arg == "--target-sample-ms") {
      result.target_sample_ms = std::stod(require_value(i, arg));
    } else if (arg == "--seed") {
      result.seed = parse_u64(require_value(i, arg));
    } else if (arg == "--output") {
      result.output = require_value(i, arg);
    } else {
      throw benchmark_error("unknown argument: " + arg);
    }
  }

  if (!powers_set && result.run_mode == mode::accuracy) {
    result.powers = {10, 16, 20};
  }
  if (!powers_set && result.run_mode == mode::profile) {
    result.powers = {27};
  }
  if (result.warmup < 0 || result.samples <= 0 || result.iterations < 0 ||
      result.target_sample_ms <= 0.0) {
    throw benchmark_error("warmup, samples, iterations, and target time "
                          "must be nonnegative/positive as appropriate");
  }
  for (const auto power : result.powers) {
    if (power < 0 || power > 30) {
      throw benchmark_error("powers must be in [0, 30]");
    }
  }
  return result;
}

void print_help(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Options:\n"
      << "  --mode performance|accuracy|profile\n"
      << "  --operation sdot|ddot|both\n"
      << "  --dataset positive|signed|cancellation|dynamic\n"
      << "  --powers P1,P2,...       vector sizes are 2^P\n"
      << "  --n N                    one vector size; must be a power of two\n"
      << "  --warmup N               warm-up calls (default: 5)\n"
      << "  --samples N              timed samples (default: 20)\n"
      << "  --iterations N           calls per sample; 0 calibrates "
         "automatically\n"
      << "  --target-sample-ms X     calibration target (default: 20 ms)\n"
      << "  --seed N                 decimal or 0x-prefixed seed\n"
      << "  --output PATH            CSV path, or - for stdout\n"
      << "  --help\n";
}

__host__ __device__ std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

template <typename T> __host__ __device__ T positive_unit(std::uint64_t bits);

template <> __host__ __device__ float positive_unit<float>(std::uint64_t bits) {
  const auto fraction = static_cast<std::uint32_t>(bits >> 40) + 1u;
  return static_cast<float>(fraction) * 0x1p-24f;
}

template <>
__host__ __device__ double positive_unit<double>(std::uint64_t bits) {
  const auto fraction = (bits >> 11) + 1ULL;
  return static_cast<double>(fraction) * 0x1p-53;
}

template <typename T> __host__ __device__ T signed_unit(std::uint64_t bits);

template <> __host__ __device__ float signed_unit<float>(std::uint64_t bits) {
  const auto fraction = static_cast<std::uint32_t>(bits >> 41);
  const auto centered = static_cast<std::int32_t>(fraction) - (1 << 22);
  return static_cast<float>(centered) * 0x1p-22f;
}

template <> __host__ __device__ double signed_unit<double>(std::uint64_t bits) {
  const auto fraction = static_cast<std::int64_t>(bits >> 12);
  const auto centered = fraction - (1LL << 51);
  return static_cast<double>(centered) * 0x1p-51;
}

template <typename T>
__host__ __device__ T sample_value(std::uint64_t index, std::uint64_t seed,
                                   dataset data, int vector_id) {
  const auto lane_seed = seed ^ (0xd1b54a32d192ed03ULL *
                                 static_cast<std::uint64_t>(vector_id + 1));
  const auto bits = splitmix64(index ^ lane_seed);

  if (data == dataset::positive) {
    return positive_unit<T>(bits);
  }
  if (data == dataset::signed_uniform) {
    return signed_unit<T>(bits);
  }
  if (data == dataset::cancellation) {
    if (vector_id == 0) {
      return T{1};
    }
    const auto pair_bits = splitmix64((index / 2) ^ lane_seed);
    const auto magnitude = positive_unit<T>(pair_bits);
    if ((index & 1ULL) == 0) {
      return magnitude;
    }
    return -magnitude * (T{1} - static_cast<T>(0x1p-10));
  }

  const auto exponent = static_cast<int>((bits >> 1) % 41ULL) - 20;
  const auto magnitude = positive_unit<T>(bits);
  const auto sign = (bits & 1ULL) ? T{-1} : T{1};
  return sign * ::ldexp(magnitude, exponent);
}

template <typename T>
__global__ void fill_vector(T *values, std::size_t count, std::uint64_t seed,
                            dataset data, int vector_id) {
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto i = first; i < count; i += stride) {
    values[i] = sample_value<T>(i, seed, data, vector_id);
  }
}

template <typename T>
void initialize_inputs(T *x, T *y, std::size_t count, std::uint64_t seed,
                       dataset data, cudaStream_t stream) {
  constexpr int block_size = 256;
  const auto wanted_blocks =
      (count + static_cast<std::size_t>(block_size) - 1) / block_size;
  const auto blocks = static_cast<int>(
      std::min<std::size_t>(wanted_blocks, static_cast<std::size_t>(65535)));
  fill_vector<<<blocks, block_size, 0, stream>>>(x, count, seed, data, 0);
  fill_vector<<<blocks, block_size, 0, stream>>>(y, count, seed, data, 1);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
cublasStatus_t launch_dot(cublasHandle_t handle, int count, const T *x,
                          const T *y, T *result);

template <>
cublasStatus_t launch_dot<float>(cublasHandle_t handle, int count,
                                 const float *x, const float *y,
                                 float *result) {
  return cublasSdot(handle, count, x, 1, y, 1, result);
}

template <>
cublasStatus_t launch_dot<double>(cublasHandle_t handle, int count,
                                  const double *x, const double *y,
                                  double *result) {
  return cublasDdot(handle, count, x, 1, y, 1, result);
}

struct statistics {
  double minimum{};
  double median{};
  double p10{};
  double p90{};
  double mean{};
  double coefficient_of_variation{};
};

double percentile(const std::vector<double> &sorted, double p) {
  if (sorted.size() == 1) {
    return sorted.front();
  }
  const auto position = p * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const auto fraction = position - static_cast<double>(lower);
  return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

statistics summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  statistics result;
  result.minimum = samples.front();
  result.median = percentile(samples, 0.5);
  result.p10 = percentile(samples, 0.1);
  result.p90 = percentile(samples, 0.9);
  result.mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  double variance{};
  for (const auto sample : samples) {
    const auto difference = sample - result.mean;
    variance += difference * difference;
  }
  variance /= samples.size();
  result.coefficient_of_variation =
      result.mean == 0.0 ? 0.0 : std::sqrt(variance) / result.mean;
  return result;
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

struct accuracy_result {
  long double reference{std::numeric_limits<long double>::quiet_NaN()};
  long double sum_abs{std::numeric_limits<long double>::quiet_NaN()};
  long double absolute_error{std::numeric_limits<long double>::quiet_NaN()};
  long double relative_error{std::numeric_limits<long double>::quiet_NaN()};
  long double normalized_error{std::numeric_limits<long double>::quiet_NaN()};
  long double condition_number{std::numeric_limits<long double>::quiet_NaN()};
};

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

template <typename T>
accuracy_result compute_accuracy(std::size_t count, std::uint64_t seed,
                                 dataset data, T computed) {
  compensated_sum reference;
  compensated_sum sum_abs;
  for (std::size_t i = 0; i < count; ++i) {
    const auto x = static_cast<long double>(sample_value<T>(i, seed, data, 0));
    const auto y = static_cast<long double>(sample_value<T>(i, seed, data, 1));
    const auto product = x * y;
    reference.add(product);
    sum_abs.add(std::fabs(product));
  }

  accuracy_result result;
  result.reference = reference.value();
  result.sum_abs = sum_abs.value();
  result.absolute_error =
      std::fabs(static_cast<long double>(computed) - result.reference);
  if (result.reference != 0.0L) {
    result.relative_error = result.absolute_error / std::fabs(result.reference);
    result.condition_number = result.sum_abs / std::fabs(result.reference);
  }
  if (result.sum_abs != 0.0L) {
    result.normalized_error = result.absolute_error / result.sum_abs;
  }
  return result;
}

struct environment {
  std::string gpu_name;
  int compute_major{};
  int compute_minor{};
  int cuda_runtime{};
  int cuda_driver{};
  int cublas_version{};
};

environment query_environment(cublasHandle_t handle) {
  int device{};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  environment result;
  result.gpu_name = properties.name;
  result.compute_major = properties.major;
  result.compute_minor = properties.minor;
  CUDA_CHECK(cudaRuntimeGetVersion(&result.cuda_runtime));
  CUDA_CHECK(cudaDriverGetVersion(&result.cuda_driver));
  CUBLAS_CHECK(cublasGetVersion(handle, &result.cublas_version));
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

struct benchmark_result {
  std::string operation_name;
  std::string type_name;
  std::size_t count{};
  std::size_t algorithmic_bytes{};
  double algorithmic_flops{};
  int iterations{};
  int samples{};
  statistics timing_ms{};
  long double value{};
  accuracy_result accuracy{};
};

void write_csv_header(std::ostream &output) {
  output << "gpu,compute_capability,cuda_runtime,cuda_driver,cublas_version,"
            "mode,dataset,operation,input_type,n,algorithmic_bytes,"
            "algorithmic_flops,arithmetic_intensity_flop_per_byte,iterations,"
            "samples,min_ms,median_ms,p10_ms,p90_ms,coefficient_of_variation,"
            "giga_elements_per_s,algorithmic_gb_per_s,gflop_per_s,result,"
            "reference,absolute_error,relative_error,normalized_error,"
            "dot_condition_number\n";
}

void write_csv_row(std::ostream &output, const environment &env,
                   const options &opts, const benchmark_result &result) {
  const auto seconds = result.timing_ms.median * 1.0e-3;
  const auto intensity =
      result.algorithmic_flops / static_cast<double>(result.algorithmic_bytes);
  const auto giga_elements =
      static_cast<double>(result.count) / seconds / 1.0e9;
  const auto bandwidth =
      static_cast<double>(result.algorithmic_bytes) / seconds / 1.0e9;
  const auto gflops = result.algorithmic_flops / seconds / 1.0e9;

  output << csv_escape(env.gpu_name) << ',' << env.compute_major << '.'
         << env.compute_minor << ',' << env.cuda_runtime << ','
         << env.cuda_driver << ',' << env.cublas_version << ','
         << to_string(opts.run_mode) << ',' << to_string(opts.data) << ','
         << result.operation_name << ',' << result.type_name << ','
         << result.count << ',' << result.algorithmic_bytes << ','
         << std::setprecision(17) << result.algorithmic_flops << ','
         << intensity << ',' << result.iterations << ',' << result.samples
         << ',' << result.timing_ms.minimum << ',' << result.timing_ms.median
         << ',' << result.timing_ms.p10 << ',' << result.timing_ms.p90 << ','
         << result.timing_ms.coefficient_of_variation << ',' << giga_elements
         << ',' << bandwidth << ',' << gflops << ',' << std::setprecision(21)
         << result.value << ',' << result.accuracy.reference << ','
         << result.accuracy.absolute_error << ','
         << result.accuracy.relative_error << ','
         << result.accuracy.normalized_error << ','
         << result.accuracy.condition_number << '\n';
}

template <typename T>
benchmark_result run_one(cublasHandle_t handle, cudaStream_t stream,
                         const options &opts, std::size_t count) {
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw benchmark_error("vector size exceeds the cuBLAS 32-bit API");
  }

  device_buffer<T> x{count};
  device_buffer<T> y{count};
  device_buffer<T> device_result{1};
  initialize_inputs(x.get(), y.get(), count, opts.seed, opts.data, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto launch = [&] {
    CUBLAS_CHECK(launch_dot<T>(handle, static_cast<int>(count), x.get(),
                               y.get(), device_result.get()));
  };

  for (int i = 0; i < opts.warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  benchmark_result result;
  result.operation_name =
      std::is_same_v<T, float> ? "cublasSdot" : "cublasDdot";
  result.type_name = std::is_same_v<T, float> ? "float32" : "float64";
  result.count = count;
  result.algorithmic_bytes = 2 * count * sizeof(T);
  result.algorithmic_flops = 2.0 * static_cast<double>(count);

  if (opts.run_mode == mode::profile) {
    cuda_event start;
    cuda_event stop;
    CUDA_CHECK(cudaProfilerStart());
    CUDA_CHECK(cudaEventRecord(start, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaProfilerStop());
    float milliseconds{};
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    result.iterations = 1;
    result.samples = 1;
    result.timing_ms = summarize({static_cast<double>(milliseconds)});
  } else {
    int iterations = opts.iterations;
    if (iterations == 0) {
      const auto calibration_ms = elapsed_ms(stream, launch, 1);
      const auto wanted = static_cast<int>(
          std::ceil(opts.target_sample_ms /
                    std::max(0.001, static_cast<double>(calibration_ms))));
      iterations = std::clamp(wanted, 1, 100000);
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(opts.samples));
    for (int sample = 0; sample < opts.samples; ++sample) {
      const auto total_ms = elapsed_ms(stream, launch, iterations);
      timings.push_back(static_cast<double>(total_ms) / iterations);
    }
    result.iterations = iterations;
    result.samples = opts.samples;
    result.timing_ms = summarize(std::move(timings));
  }

  T host_result{};
  CUDA_CHECK(cudaMemcpyAsync(&host_result, device_result.get(), sizeof(T),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  result.value = static_cast<long double>(host_result);
  if (!std::isfinite(static_cast<double>(host_result))) {
    throw benchmark_error(result.operation_name +
                          " produced a nonfinite result");
  }
  if (opts.run_mode == mode::accuracy) {
    result.accuracy =
        compute_accuracy<T>(count, opts.seed, opts.data, host_result);
  }
  return result;
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

} // namespace

int main(int argc, char **argv) {
  try {
    const auto opts = parse_options(argc, argv);
    if (opts.help) {
      print_help(argv[0]);
      return EXIT_SUCCESS;
    }

    cuda_stream stream;
    cublas_handle handle{stream};
    const auto env = query_environment(handle);

    std::ofstream output_file;
    std::ostream *output = &std::cout;
    if (opts.output != "-") {
      output_file.open(opts.output);
      if (!output_file) {
        throw benchmark_error("could not open output file: " + opts.output);
      }
      output = &output_file;
    }
    write_csv_header(*output);

    std::cerr << "dot_bench " << timestamp_utc() << "\n"
              << "GPU: " << env.gpu_name << " (sm_" << env.compute_major
              << env.compute_minor << ")\n"
              << "CUDA runtime/driver: " << env.cuda_runtime << '/'
              << env.cuda_driver << ", cuBLAS: " << env.cublas_version
              << "\nMode: " << to_string(opts.run_mode)
              << ", dataset: " << to_string(opts.data) << '\n';

    for (const auto power : opts.powers) {
      const auto count = std::size_t{1} << power;
      if (opts.op == operation::sdot || opts.op == operation::both) {
        const auto result = run_one<float>(handle, stream, opts, count);
        write_csv_row(*output, env, opts, result);
        std::cerr << result.operation_name << " N=2^" << power
                  << " median=" << result.timing_ms.median << " ms\n";
      }
      if (opts.op == operation::ddot || opts.op == operation::both) {
        const auto result = run_one<double>(handle, stream, opts, count);
        write_csv_row(*output, env, opts, result);
        std::cerr << result.operation_name << " N=2^" << power
                  << " median=" << result.timing_ms.median << " ms\n";
      }
      output->flush();
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
