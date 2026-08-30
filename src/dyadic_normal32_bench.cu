#include "dyadic_normal32_core.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace dn = aut::dyadic_normal32;
namespace lut = aut::lut_decomposition;

constexpr int threads = 256;
constexpr int dot_blocks = 512;
constexpr int metric_blocks = 4096;
constexpr std::uint64_t left_seed = 0x6a09e667f3bcc909ULL;
constexpr std::uint64_t right_seed = 0xbb67ae8584caa73bULL;
constexpr std::uint64_t raw_left_seed = 0xa54ff53a5f1d36f1ULL;
constexpr std::uint64_t raw_right_seed = 0x510e527fade682d1ULL;
constexpr std::uint64_t payload_salt = 0x3c6ef372fe94f82bULL;
constexpr std::uint64_t sign_salt = 0x9b05688c2b3e6c1fULL;
constexpr std::size_t coefficient_bytes =
    dn::segment_count * sizeof(dn::segment_coefficients);
constexpr std::size_t timed_shared_bytes =
    coefficient_bytes + threads * sizeof(double);

static_assert(coefficient_bytes == 512);

class benchmark_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void cuda_check(cudaError_t status, const char *expression, const char *file,
                int line) {
  if (status != cudaSuccess) {
    throw benchmark_error(std::string(file) + ':' + std::to_string(line) + " " +
                          expression + ": " + cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expression)                                                 \
  cuda_check((expression), #expression, __FILE__, __LINE__)

template <typename T> class device_buffer {
public:
  device_buffer() = default;
  explicit device_buffer(std::size_t count) { reset(count); }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
  device_buffer(device_buffer &&other) noexcept
      : data_(std::exchange(other.data_, nullptr)) {}
  device_buffer &operator=(device_buffer &&other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
    }
    return *this;
  }
  ~device_buffer() { release(); }

  void reset(std::size_t count) {
    release();
    if (count != 0) {
      CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)));
    }
  }
  T *get() { return data_; }
  const T *get() const { return data_; }

private:
  void release() {
    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
    }
  }
  T *data_{};
};

struct settings {
  std::string mode{"full"};
  std::size_t n{std::size_t{1} << 26};
  int warmup{10};
  int samples{50};
  std::string output{"timing_samples.csv"};
  std::string metrics_output{"access_metrics.csv"};
  std::string correctness_output{"correctness_checks.txt"};
  std::string coefficients_output{"coefficient_table.csv"};
};

std::size_t parse_size(const std::string &text, const std::string &option) {
  std::size_t used{};
  const auto value = std::stoull(text, &used);
  if (used != text.size() || value == 0) {
    throw benchmark_error(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(value);
}

settings parse_arguments(int argc, char **argv) {
  settings result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw benchmark_error("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--mode") {
      result.mode = value();
    } else if (argument == "--n") {
      result.n = parse_size(value(), argument);
    } else if (argument == "--warmup") {
      result.warmup = static_cast<int>(parse_size(value(), argument));
    } else if (argument == "--samples") {
      result.samples = static_cast<int>(parse_size(value(), argument));
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--metrics-output") {
      result.metrics_output = value();
    } else if (argument == "--correctness-output") {
      result.correctness_output = value();
    } else if (argument == "--coefficients-output") {
      result.coefficients_output = value();
    } else {
      throw benchmark_error("unknown argument: " + argument);
    }
  }
  if (result.mode == "smoke") {
    result.n = std::min<std::size_t>(result.n, std::size_t{1} << 20);
    result.warmup = std::min(result.warmup, 1);
    result.samples = std::min(result.samples, 3);
  } else if (result.mode != "full") {
    throw benchmark_error("--mode must be smoke or full");
  }
  if ((result.n % lut::warp_width) != 0) {
    throw benchmark_error("N must be divisible by 32");
  }
  return result;
}

std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto epoch = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&epoch, &utc);
  std::ostringstream stream;
  stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return stream.str();
}

void ensure_parent(const std::string &filename) {
  const std::filesystem::path path(filename);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

__host__ __device__ __forceinline__ std::uint32_t
sample_code_for_segment(std::size_t index, std::uint64_t seed,
                        std::uint32_t segment) {
  const auto payload_bits = aut::lut_distribution::splitmix64(
      seed ^ payload_salt ^ static_cast<std::uint64_t>(index));
  const auto sign_bits = aut::lut_distribution::splitmix64(
      seed ^ sign_salt ^ static_cast<std::uint64_t>(index));
  const auto rank = dn::rank_from_segment_payload_unchecked(
      segment, static_cast<std::uint32_t>(payload_bits));
  const auto sign = static_cast<std::uint32_t>(sign_bits >> 32) & 0x80000000u;
  return rank | sign;
}

__host__ __device__ __forceinline__ std::uint32_t
sample_code(std::size_t index, std::uint64_t seed, double q) {
  const auto segment = lut::sample_index(index, seed, 5, q);
  return sample_code_for_segment(index, seed, segment);
}

__global__ void generate_codes_kernel(std::uint32_t *codes, std::size_t count,
                                      std::uint64_t seed, double q) {
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    codes[index] = sample_code(index, seed, q);
  }
}

__global__ void
generate_genuine_codes_kernel(std::uint32_t *codes, std::size_t count,
                              std::uint64_t seed,
                              const double *cumulative_probabilities) {
  constexpr double inverse_53 = 1.0 / 9007199254740992.0;
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto random_bits = aut::lut_distribution::splitmix64(
        seed ^ static_cast<std::uint64_t>(index) ^ 0x1f83d9abfb41bd6bULL);
    const auto uniform01 = static_cast<double>(random_bits >> 11) * inverse_53;
    std::uint32_t segment{};
#pragma unroll
    for (std::uint32_t candidate = 0; candidate < dn::segment_count - 1;
         ++candidate) {
      segment += uniform01 >= cumulative_probabilities[candidate];
    }
    codes[index] = sample_code_for_segment(index, seed, segment);
  }
}

__global__ void generate_raw_values_kernel(double *values, std::size_t count,
                                           std::uint64_t seed) {
  constexpr double scale = 1.0 / 9007199254740992.0;
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto bits = aut::lut_distribution::splitmix64(
                          seed ^ static_cast<std::uint64_t>(index)) >>
                      11;
    values[index] = static_cast<double>(bits) * scale - 0.5;
  }
}

__global__ void generate_raw_values_kernel(float *values, std::size_t count,
                                           std::uint64_t seed) {
  constexpr float scale = 1.0f / 16777216.0f;
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto bits = static_cast<std::uint32_t>(
        aut::lut_distribution::splitmix64(
            seed ^ static_cast<std::uint64_t>(index)) >>
        40);
    values[index] = static_cast<float>(bits) * scale - 0.5f;
  }
}

__device__ __forceinline__ double
decode_code(std::uint32_t code, const dn::segment_coefficients *coefficients) {
  const auto rank = code & dn::magnitude_mask;
  const auto segment = dn::segment_from_rank(rank);
  const auto payload = dn::payload_for_segment(rank, segment);
  const auto coefficient = coefficients[segment];
  const auto magnitude =
      fma(static_cast<double>(payload), coefficient.step, coefficient.start);
  const auto magnitude_bits =
      static_cast<unsigned long long>(__double_as_longlong(magnitude));
  const auto sign_bits = static_cast<unsigned long long>(code >> 31) << 63;
  return __longlong_as_double(
      static_cast<long long>(magnitude_bits | sign_bits));
}

__device__ __forceinline__ double block_sum(double value, double *shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  return shared[0];
}

__device__ __forceinline__ float block_sum(float value, float *shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  return shared[0];
}

__global__ void dot_dyadic_normal32_kernel(
    const std::uint32_t *left, const std::uint32_t *right, std::size_t count,
    const dn::segment_coefficients *global_coefficients, double *partials) {
  extern __shared__ __align__(16) unsigned char storage[];
  auto *coefficients = reinterpret_cast<dn::segment_coefficients *>(storage);
  auto *reduction = reinterpret_cast<double *>(storage + coefficient_bytes);
  for (std::size_t index = threadIdx.x; index < dn::segment_count;
       index += blockDim.x) {
    coefficients[index] = global_coefficients[index];
  }
  __syncthreads();

  double sum{};
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto a = decode_code(left[index], coefficients);
    const auto b = decode_code(right[index], coefficients);
    sum = fma(a, b, sum);
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

__global__ void dot_raw_fp64_kernel(const double *left, const double *right,
                                    std::size_t count, double *partials) {
  __shared__ double reduction[threads];
  double sum{};
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum = fma(left[index], right[index], sum);
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

__global__ void dot_raw_fp32_kernel(const float *left, const float *right,
                                    std::size_t count, float *partials) {
  __shared__ float reduction[threads];
  float sum{};
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum = fmaf(left[index], right[index], sum);
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

__global__ void dot_fp32_to_fp64_kernel(const float *left, const float *right,
                                        std::size_t count, double *partials) {
  __shared__ double reduction[threads];
  double sum{};
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum = fma(static_cast<double>(left[index]),
              static_cast<double>(right[index]), sum);
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

__global__ void finalize_dot_kernel(const double *partials, std::size_t count,
                                    double *result) {
  __shared__ double reduction[threads];
  double sum{};
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    sum += partials[index];
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    *result = reduced;
  }
}

__global__ void finalize_dot_kernel(const float *partials, std::size_t count,
                                    float *result) {
  __shared__ float reduction[threads];
  float sum{};
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    sum += partials[index];
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    *result = reduced;
  }
}

__global__ void segment_metrics_kernel(const std::uint32_t *codes,
                                       std::size_t count,
                                       unsigned long long *unique_total) {
  constexpr int warps_per_block = threads / lut::warp_width;
  const auto lane = threadIdx.x & (lut::warp_width - 1);
  const auto warp_in_block = threadIdx.x / lut::warp_width;
  const auto first_warp =
      static_cast<std::size_t>(blockIdx.x) * warps_per_block +
      static_cast<std::size_t>(warp_in_block);
  const auto stride = static_cast<std::size_t>(gridDim.x) * warps_per_block;
  const auto warp_count = count / lut::warp_width;
  unsigned long long local_unique{};
  for (auto warp = first_warp; warp < warp_count; warp += stride) {
    const auto rank = codes[warp * lut::warp_width + lane] & dn::magnitude_mask;
    const auto segment = dn::segment_from_rank(rank);
    const auto peers = __match_any_sync(0xffffffffu, segment);
    const auto leader = lane == (__ffs(static_cast<int>(peers)) - 1);
    const auto leaders = __ballot_sync(0xffffffffu, leader);
    if (lane == 0) {
      local_unique += __popc(leaders);
    }
  }
  if (lane == 0) {
    atomicAdd(unique_total, local_unique);
  }
}

__global__ void
decode_validation_kernel(const std::uint32_t *codes, std::size_t count,
                         const dn::segment_coefficients *global_coefficients,
                         double *values) {
  __shared__ dn::segment_coefficients coefficients[dn::segment_count];
  for (std::size_t index = threadIdx.x; index < dn::segment_count;
       index += blockDim.x) {
    coefficients[index] = global_coefficients[index];
  }
  __syncthreads();
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    values[index] = decode_code(codes[index], coefficients);
  }
}

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

  template <typename Launch> double measure(Launch &&launch) {
    CUDA_CHECK(cudaEventRecord(start_));
    launch();
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    CUDA_CHECK(cudaGetLastError());
    float milliseconds{};
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return static_cast<double>(milliseconds);
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

double measure_unique_segments(const std::uint32_t *codes, std::size_t count) {
  device_buffer<unsigned long long> total(1);
  CUDA_CHECK(cudaMemset(total.get(), 0, sizeof(unsigned long long)));
  segment_metrics_kernel<<<metric_blocks, threads>>>(codes, count, total.get());
  CUDA_CHECK(cudaGetLastError());
  unsigned long long host_total{};
  CUDA_CHECK(cudaMemcpy(&host_total, total.get(), sizeof(host_total),
                        cudaMemcpyDeviceToHost));
  return static_cast<double>(host_total) /
         static_cast<double>(count / lut::warp_width);
}

void validate_gpu_decoder(
    const std::array<dn::segment_coefficients, dn::segment_count> &coefficients,
    const dn::segment_coefficients *device_coefficients,
    const std::string &output_path) {
  std::vector<std::uint32_t> codes;
  for (std::uint32_t segment = 0; segment < dn::segment_count; ++segment) {
    const auto maximum_payload = dn::payload_mask >> segment;
    for (const auto payload : {0u, maximum_payload}) {
      const auto rank =
          dn::rank_from_segment_payload_unchecked(segment, payload);
      codes.push_back(rank);
      codes.push_back(rank | 0x80000000u);
    }
  }
  for (std::size_t index = 0; index < 4096; ++index) {
    codes.push_back(sample_code(index, 0x243f6a8885a308d3ULL, 1.0));
  }
  device_buffer<std::uint32_t> device_codes(codes.size());
  device_buffer<double> device_values(codes.size());
  CUDA_CHECK(cudaMemcpy(device_codes.get(), codes.data(),
                        codes.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  decode_validation_kernel<<<32, threads>>>(device_codes.get(), codes.size(),
                                            device_coefficients,
                                            device_values.get());
  CUDA_CHECK(cudaGetLastError());
  std::vector<double> gpu_values(codes.size());
  CUDA_CHECK(cudaMemcpy(gpu_values.data(), device_values.get(),
                        gpu_values.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));

  std::size_t mismatches{};
  double maximum_absolute_error{};
  for (std::size_t index = 0; index < codes.size(); ++index) {
    const auto expected = dn::decode(codes[index], coefficients);
    std::uint64_t expected_bits{};
    std::uint64_t actual_bits{};
    std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
    std::memcpy(&actual_bits, &gpu_values[index], sizeof(actual_bits));
    if (expected_bits != actual_bits) {
      ++mismatches;
    }
    maximum_absolute_error = std::max(maximum_absolute_error,
                                      std::abs(expected - gpu_values[index]));
  }

  constexpr std::size_t dot_validation_count = 8192;
  constexpr int dot_validation_blocks = dot_blocks;
  std::vector<std::uint32_t> dot_left(dot_validation_count);
  std::vector<std::uint32_t> dot_right(dot_validation_count);
  long double cpu_dot{};
  for (std::size_t index = 0; index < dot_validation_count; ++index) {
    dot_left[index] =
        sample_code(index, 0x13198a2e03707344ULL, 1.0) & dn::magnitude_mask;
    dot_right[index] =
        sample_code(index, 0xa4093822299f31d0ULL, 1.0) & dn::magnitude_mask;
    cpu_dot +=
        static_cast<long double>(dn::decode(dot_left[index], coefficients)) *
        static_cast<long double>(dn::decode(dot_right[index], coefficients));
  }
  device_buffer<std::uint32_t> device_dot_left(dot_validation_count);
  device_buffer<std::uint32_t> device_dot_right(dot_validation_count);
  device_buffer<double> device_dot_partials(dot_validation_blocks);
  device_buffer<double> device_dot_result(1);
  CUDA_CHECK(cudaMemcpy(device_dot_left.get(), dot_left.data(),
                        dot_left.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_dot_right.get(), dot_right.data(),
                        dot_right.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  dot_dyadic_normal32_kernel<<<dot_validation_blocks, threads,
                               timed_shared_bytes>>>(
      device_dot_left.get(), device_dot_right.get(), dot_validation_count,
      device_coefficients, device_dot_partials.get());
  finalize_dot_kernel<<<1, threads>>>(device_dot_partials.get(),
                                      dot_validation_blocks,
                                      device_dot_result.get());
  CUDA_CHECK(cudaGetLastError());
  double gpu_dot{};
  CUDA_CHECK(cudaMemcpy(&gpu_dot, device_dot_result.get(), sizeof(gpu_dot),
                        cudaMemcpyDeviceToHost));
  const auto cpu_dot_double = static_cast<double>(cpu_dot);
  const auto dot_error = std::abs(gpu_dot - cpu_dot_double);
  const auto dot_tolerance = 2e-12 * std::max(1.0, std::abs(cpu_dot_double));
  const auto dot_passed = std::isfinite(gpu_dot) && dot_error <= dot_tolerance;

  ensure_parent(output_path);
  std::ofstream output(output_path);
  if (!output) {
    throw benchmark_error("cannot open correctness output");
  }
  output << "cpu_gpu_cases=" << codes.size() << '\n'
         << "cpu_gpu_bit_mismatches=" << mismatches << '\n'
         << "cpu_gpu_maximum_absolute_error=" << std::setprecision(17)
         << maximum_absolute_error << '\n'
         << "segments_covered=32\n"
         << "delimiter_endpoints_per_sign=64\n"
         << "terminal_rank=0x7fffffff\n"
         << "terminal_value=" << coefficients[31].start << '\n'
         << "coefficient_bytes=" << coefficient_bytes << '\n'
         << "genuine_n01_expected_unique_segments="
         << dn::expected_unique_segments_for_probabilities(
                dn::genuine_standard_normal_segment_probabilities())
         << '\n'
         << "genuine_n01_expected_x="
         << dn::genuine_standard_normal_dispersion() << '\n'
         << "dot_validation_elements=" << dot_validation_count << '\n'
         << "dot_validation_cpu=" << cpu_dot_double << '\n'
         << "dot_validation_gpu=" << gpu_dot << '\n'
         << "dot_validation_absolute_error=" << dot_error << '\n'
         << "dot_validation_tolerance=" << dot_tolerance << '\n'
         << "dot_validation_passed=" << (dot_passed ? 1 : 0) << '\n';
  if (mismatches != 0) {
    throw benchmark_error("CPU/GPU decoder bit mismatch");
  }
  if (!dot_passed) {
    throw benchmark_error("CPU/GPU end-to-end DOT mismatch");
  }
}

struct csv_context {
  std::ofstream &samples;
  std::ofstream &metrics;
  std::string gpu;
  settings configuration;
};

void run_raw_fp64(csv_context &context, const std::string &phase,
                  int sample_begin, int sample_count, int warmups_this_batch,
                  int execution_order) {
  const auto &settings = context.configuration;
  device_buffer<double> left(settings.n);
  device_buffer<double> right(settings.n);
  device_buffer<double> partials(dot_blocks);
  device_buffer<double> result(1);
  const auto generation_blocks = static_cast<int>(
      std::min<std::size_t>(4096, (settings.n + threads - 1) / threads));
  generate_raw_values_kernel<<<generation_blocks, threads>>>(
      left.get(), settings.n, raw_left_seed);
  generate_raw_values_kernel<<<generation_blocks, threads>>>(
      right.get(), settings.n, raw_right_seed);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto launch = [&] {
    dot_raw_fp64_kernel<<<dot_blocks, threads>>>(left.get(), right.get(),
                                                 settings.n, partials.get());
    finalize_dot_kernel<<<1, threads>>>(partials.get(), dot_blocks,
                                        result.get());
  };
  for (int warmup = 0; warmup < warmups_this_batch; ++warmup) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  event_timer timer;
  for (int local_sample = 0; local_sample < sample_count; ++local_sample) {
    const auto sample = sample_begin + local_sample;
    const auto milliseconds = timer.measure(launch);
    double host_result{};
    CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(host_result),
                          cudaMemcpyDeviceToHost));
    if (!std::isfinite(host_result)) {
      throw benchmark_error("nonfinite raw FP64 result");
    }
    context.samples << utc_timestamp() << ',' << context.gpu << ','
                    << settings.mode << ",raw_fp64,Raw FP64,raw," << phase
                    << ",dot," << settings.n << ',' << dot_blocks << ','
                    << threads << ",64,fp64,none,0,0,0,nan,nan,nan,nan,"
                    << settings.warmup << ',' << warmups_this_batch << ','
                    << sample << ',' << execution_order << ','
                    << std::setprecision(17) << milliseconds << ','
                    << host_result << '\n';
  }
  context.samples.flush();
  std::cout << "raw_fp64/" << phase << " complete\n";
}

void run_raw_fp32(csv_context &context, const std::string &phase,
                  int sample_begin, int sample_count, int warmups_this_batch,
                  int execution_order) {
  const auto &settings = context.configuration;
  device_buffer<float> left(settings.n);
  device_buffer<float> right(settings.n);
  device_buffer<float> partials(dot_blocks);
  device_buffer<float> result(1);
  const auto generation_blocks = static_cast<int>(
      std::min<std::size_t>(4096, (settings.n + threads - 1) / threads));
  generate_raw_values_kernel<<<generation_blocks, threads>>>(
      left.get(), settings.n, raw_left_seed);
  generate_raw_values_kernel<<<generation_blocks, threads>>>(
      right.get(), settings.n, raw_right_seed);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto launch = [&] {
    dot_raw_fp32_kernel<<<dot_blocks, threads>>>(left.get(), right.get(),
                                                 settings.n, partials.get());
    finalize_dot_kernel<<<1, threads>>>(partials.get(), dot_blocks,
                                        result.get());
  };
  for (int warmup = 0; warmup < warmups_this_batch; ++warmup) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  event_timer timer;
  for (int local_sample = 0; local_sample < sample_count; ++local_sample) {
    const auto sample = sample_begin + local_sample;
    const auto milliseconds = timer.measure(launch);
    float host_result{};
    CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(host_result),
                          cudaMemcpyDeviceToHost));
    if (!std::isfinite(host_result)) {
      throw benchmark_error("nonfinite raw FP32 result");
    }
    context.samples << utc_timestamp() << ',' << context.gpu << ','
                    << settings.mode << ",raw_fp32,Raw FP32,raw," << phase
                    << ",dot," << settings.n << ',' << dot_blocks << ','
                    << threads << ",32,fp32,none,0,0,0,nan,nan,nan,nan,"
                    << settings.warmup << ',' << warmups_this_batch << ','
                    << sample << ',' << execution_order << ','
                    << std::setprecision(17) << milliseconds << ','
                    << host_result << '\n';
  }
  context.samples.flush();
  std::cout << "raw_fp32/" << phase << " complete\n";
}

void run_fp32_to_fp64(csv_context &context, const std::string &phase,
                      int sample_begin, int sample_count,
                      int warmups_this_batch, int execution_order) {
  const auto &settings = context.configuration;
  device_buffer<float> left(settings.n);
  device_buffer<float> right(settings.n);
  device_buffer<double> partials(dot_blocks);
  device_buffer<double> result(1);
  const auto generation_blocks = static_cast<int>(
      std::min<std::size_t>(4096, (settings.n + threads - 1) / threads));
  generate_raw_values_kernel<<<generation_blocks, threads>>>(
      left.get(), settings.n, raw_left_seed);
  generate_raw_values_kernel<<<generation_blocks, threads>>>(
      right.get(), settings.n, raw_right_seed);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto launch = [&] {
    dot_fp32_to_fp64_kernel<<<dot_blocks, threads>>>(
        left.get(), right.get(), settings.n, partials.get());
    finalize_dot_kernel<<<1, threads>>>(partials.get(), dot_blocks,
                                        result.get());
  };
  for (int warmup = 0; warmup < warmups_this_batch; ++warmup) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  event_timer timer;
  for (int local_sample = 0; local_sample < sample_count; ++local_sample) {
    const auto sample = sample_begin + local_sample;
    const auto milliseconds = timer.measure(launch);
    double host_result{};
    CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(host_result),
                          cudaMemcpyDeviceToHost));
    if (!std::isfinite(host_result)) {
      throw benchmark_error("nonfinite FP32-to-FP64 result");
    }
    context.samples << utc_timestamp() << ',' << context.gpu << ','
                    << settings.mode
                    << ",fp32_to_fp64,FP32 to FP64,raw," << phase << ",dot,"
                    << settings.n << ',' << dot_blocks << ',' << threads
                    << ",32,fp64,none,0,0,0,nan,nan,nan,nan,"
                    << settings.warmup << ',' << warmups_this_batch << ','
                    << sample << ',' << execution_order << ','
                    << std::setprecision(17) << milliseconds << ','
                    << host_result << '\n';
  }
  context.samples.flush();
  std::cout << "fp32_to_fp64/" << phase << " complete\n";
}

int run_dyadic_normal32(csv_context &context,
                        const dn::segment_coefficients *device_coefficients,
                        int first_execution_order) {
  const auto &settings = context.configuration;
  device_buffer<std::uint32_t> left(settings.n);
  device_buffer<std::uint32_t> right(settings.n);
  device_buffer<double> partials(dot_blocks);
  device_buffer<double> result(1);
  const auto generation_blocks = static_cast<int>(
      std::min<std::size_t>(4096, (settings.n + threads - 1) / threads));
  const std::vector<double> targets =
      settings.mode == "smoke"
          ? std::vector<double>{0.0, 0.5, 1.0}
          : std::vector<double>{0.0,   0.125, 0.25,  0.375, 0.5,
                                0.625, 0.75,  0.875, 1.0};
  const auto genuine_probabilities =
      dn::genuine_standard_normal_segment_probabilities();
  std::array<double, dn::segment_count> genuine_cumulative{};
  double cumulative{};
  for (std::size_t segment = 0; segment < dn::segment_count; ++segment) {
    cumulative += genuine_probabilities[segment];
    genuine_cumulative[segment] = cumulative;
  }
  genuine_cumulative.back() = 1.0;
  device_buffer<double> device_genuine_cumulative(dn::segment_count);
  CUDA_CHECK(cudaMemcpy(
      device_genuine_cumulative.get(), genuine_cumulative.data(),
      genuine_cumulative.size() * sizeof(double), cudaMemcpyHostToDevice));
  event_timer timer;
  volatile double result_sink{};
  int execution_order = first_execution_order;
  const auto first_batch_samples = (settings.samples + 1) / 2;
  const auto second_batch_samples = settings.samples / 2;
  const auto warmups_per_batch = (settings.warmup + 1) / 2;

  const auto run_batch = [&](const std::string &distribution,
                             const std::string &phase, double target_x,
                             double q, int sample_begin, int sample_count,
                             int warmups_this_batch, bool record_metrics) {
    const auto genuine = distribution == "genuine_n01";
    if (genuine) {
      generate_genuine_codes_kernel<<<generation_blocks, threads>>>(
          left.get(), settings.n, left_seed, device_genuine_cumulative.get());
      generate_genuine_codes_kernel<<<generation_blocks, threads>>>(
          right.get(), settings.n, right_seed, device_genuine_cumulative.get());
    } else {
      generate_codes_kernel<<<generation_blocks, threads>>>(
          left.get(), settings.n, left_seed, q);
      generate_codes_kernel<<<generation_blocks, threads>>>(
          right.get(), settings.n, right_seed, q);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto left_unique = measure_unique_segments(left.get(), settings.n);
    const auto right_unique = measure_unique_segments(right.get(), settings.n);
    const auto mean_unique = 0.5 * (left_unique + right_unique);
    const auto actual_x =
        lut::normalized_lookup_dispersion(dn::segment_count, mean_unique);
    if (record_metrics) {
      context.metrics << distribution << ',' << std::setprecision(17)
                      << target_x << ',' << q << ',' << settings.n << ','
                      << left_unique << ',' << right_unique << ','
                      << mean_unique << ',' << actual_x << '\n';
    }

    const auto launch = [&] {
      dot_dyadic_normal32_kernel<<<dot_blocks, threads, timed_shared_bytes>>>(
          left.get(), right.get(), settings.n, device_coefficients,
          partials.get());
      finalize_dot_kernel<<<1, threads>>>(partials.get(), dot_blocks,
                                          result.get());
    };
    for (int warmup = 0; warmup < warmups_this_batch; ++warmup) {
      launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int local_sample = 0; local_sample < sample_count; ++local_sample) {
      const auto sample = sample_begin + local_sample;
      const auto milliseconds = timer.measure(launch);
      double host_result{};
      CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(host_result),
                            cudaMemcpyDeviceToHost));
      if (!std::isfinite(host_result)) {
        throw benchmark_error("nonfinite DyadicNormal32 result");
      }
      result_sink += host_result;
      context.samples << utc_timestamp() << ',' << context.gpu << ','
                      << settings.mode << ",dyadic_normal32,DyadicNormal32,"
                      << distribution << ',' << phase << ",dot," << settings.n
                      << ',' << dot_blocks << ',' << threads
                      << ",32,fp64,shared,32,512," << std::setprecision(17)
                      << target_x << ',' << q << ',' << mean_unique << ','
                      << actual_x << ','
                      << dn::genuine_standard_normal_dispersion() << ','
                      << settings.warmup << ',' << warmups_this_batch << ','
                      << sample << ',' << execution_order << ',' << milliseconds
                      << ',' << host_result << '\n';
    }
    ++execution_order;
    context.samples.flush();
    context.metrics.flush();
    std::cout << "dyadic_normal32/" << distribution << '/' << phase
              << " target X=" << target_x << " actual X=" << actual_x
              << " q=" << q << '\n';
  };

  for (const auto target_x : targets) {
    const auto q =
        lut::q_for_normalized_dispersion(dn::segment_count, target_x);
    run_batch("hot_uniform", "forward", target_x, q, 0, first_batch_samples,
              warmups_per_batch, true);
  }
  run_batch("genuine_n01", "forward", dn::genuine_standard_normal_dispersion(),
            std::numeric_limits<double>::quiet_NaN(), 0, first_batch_samples,
            warmups_per_batch, true);
  run_batch("genuine_n01", "reverse", dn::genuine_standard_normal_dispersion(),
            std::numeric_limits<double>::quiet_NaN(), first_batch_samples,
            second_batch_samples, warmups_per_batch, false);
  for (auto iterator = targets.rbegin(); iterator != targets.rend();
       ++iterator) {
    const auto target_x = *iterator;
    const auto q =
        lut::q_for_normalized_dispersion(dn::segment_count, target_x);
    run_batch("hot_uniform", "reverse", target_x, q, first_batch_samples,
              second_batch_samples, warmups_per_batch, false);
  }
  std::cout << "dyadic_normal32 sink=" << result_sink << '\n';
  return execution_order;
}

void run(const settings &settings) {
  CUDA_CHECK(cudaSetDevice(0));
  int device_count{};
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count != 1) {
    throw benchmark_error("benchmark requires exactly one visible GPU");
  }
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  const std::string gpu = properties.name;
  if (gpu.find("H200") == std::string::npos) {
    throw benchmark_error("benchmark requires an H200 GPU, found " + gpu);
  }
  std::cout << "GPU: " << gpu << '\n';
  std::cout << "N=" << settings.n << ", warmups=" << settings.warmup
            << ", samples=" << settings.samples << '\n';

  const auto coefficients = dn::make_coefficients();
  ensure_parent(settings.coefficients_output);
  std::ofstream coefficient_output(settings.coefficients_output);
  if (!coefficient_output) {
    throw benchmark_error("cannot open coefficient output");
  }
  coefficient_output << "segment,lower_boundary,upper_boundary,payload_bits,"
                        "levels,A,B\n";
  for (std::uint32_t segment = 0; segment < dn::segment_count; ++segment) {
    const auto payload_bits = segment < 30 ? 30u - segment : 0u;
    const auto levels = segment < 31 ? std::uint64_t{1} << payload_bits : 1u;
    coefficient_output << segment << ',' << std::setprecision(17)
                       << dn::half_normal_density_boundary(segment) << ','
                       << dn::half_normal_density_boundary(segment + 1) << ','
                       << payload_bits << ',' << levels << ','
                       << coefficients[segment].start << ','
                       << coefficients[segment].step << '\n';
  }
  coefficient_output.close();
  device_buffer<dn::segment_coefficients> device_coefficients(
      dn::segment_count);
  CUDA_CHECK(cudaMemcpy(device_coefficients.get(), coefficients.data(),
                        coefficient_bytes, cudaMemcpyHostToDevice));
  validate_gpu_decoder(coefficients, device_coefficients.get(),
                       settings.correctness_output);

  ensure_parent(settings.output);
  ensure_parent(settings.metrics_output);
  std::ofstream samples(settings.output);
  std::ofstream metrics(settings.metrics_output);
  if (!samples || !metrics) {
    throw benchmark_error("cannot open output CSV files");
  }
  samples
      << "timestamp,gpu,mode,variant,variant_label,distribution,phase,kernel,"
         "N,blocks,threads,"
         "storage_bits,arithmetic_type,table_location,segments,"
         "coefficient_bytes,target_x,q,mean_unique_segments,actual_x,"
         "genuine_n01_expected_x,warmup,warmups_this_batch,sample,"
         "execution_order,kernel_ms,"
         "result\n";
  metrics << "distribution,target_x,q,N,left_unique_segments,"
             "right_unique_segments,"
             "mean_unique_segments,actual_x\n";
  csv_context context{samples, metrics, gpu, settings};

  const auto first_batch_samples = (settings.samples + 1) / 2;
  const auto second_batch_samples = settings.samples / 2;
  const auto warmups_per_batch = (settings.warmup + 1) / 2;
  run_raw_fp64(context, "before", 0, first_batch_samples, warmups_per_batch, 0);
  run_raw_fp32(context, "before", 0, first_batch_samples, warmups_per_batch, 1);
  run_fp32_to_fp64(context, "before", 0, first_batch_samples,
                   warmups_per_batch, 2);
  const auto final_execution_order =
      run_dyadic_normal32(context, device_coefficients.get(), 3);
  run_fp32_to_fp64(context, "after", first_batch_samples,
                   second_batch_samples, warmups_per_batch,
                   final_execution_order);
  run_raw_fp32(context, "after", first_batch_samples, second_batch_samples,
               warmups_per_batch, final_execution_order + 1);
  run_raw_fp64(context, "after", first_batch_samples, second_batch_samples,
               warmups_per_batch, final_execution_order + 2);
  std::cout << "Wrote " << settings.output << '\n';
  std::cout << "Wrote " << settings.metrics_output << '\n';
  std::cout << "Wrote " << settings.correctness_output << '\n';
  std::cout << "Wrote " << settings.coefficients_output << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    run(parse_arguments(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
