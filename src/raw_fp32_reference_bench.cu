#include "lut_distribution_core.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

namespace lut = aut::lut_distribution;

constexpr int threads = 256;
constexpr int dot_blocks = 512;
constexpr std::uint64_t left_seed = 0xa4093822299f31d0ULL;
constexpr std::uint64_t right_seed = 0x082efa98ec4e6c89ULL;

class benchmark_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void cuda_check(cudaError_t status, const char *expression, const char *file,
                int line) {
  if (status != cudaSuccess) {
    throw benchmark_error(std::string(file) + ':' + std::to_string(line) +
                          " " + expression + ": " +
                          cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expression)                                                 \
  cuda_check((expression), #expression, __FILE__, __LINE__)

template <typename T> class device_buffer {
public:
  explicit device_buffer(std::size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_),
                          count * sizeof(T)));
  }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
  ~device_buffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }
  T *get() { return data_; }
  const T *get() const { return data_; }

private:
  T *data_{};
  std::size_t count_{};
};

struct settings {
  std::string mode{"full"};
  std::size_t n{std::size_t{1} << 26};
  int warmup{10};
  int samples{50};
  std::string output{"raw_fp32_timing_samples.csv"};
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
    } else {
      throw benchmark_error("unknown argument: " + argument);
    }
  }
  if (result.mode != "smoke" && result.mode != "full") {
    throw benchmark_error("--mode must be smoke or full");
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

__global__ void generate_values_kernel(float *values, std::size_t count,
                                       std::uint64_t seed) {
  constexpr float scale = 1.0f / 16777216.0f;
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto bits = static_cast<std::uint32_t>(lut::splitmix64(
        seed ^ static_cast<std::uint64_t>(index)) >> 40);
    values[index] = static_cast<float>(bits) * scale - 0.5f;
  }
}

template <typename T>
__device__ __forceinline__ T block_sum(T value, T *shared) {
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

__global__ void dot_raw_fp32_kernel(const float *left, const float *right,
                                    std::size_t count, float *partials) {
  __shared__ float shared[threads];
  float sum{};
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum = fmaf(left[index], right[index], sum);
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

__global__ void finalize_dot_kernel(const float *partials, std::size_t count,
                                    float *result) {
  __shared__ float shared[threads];
  float sum{};
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    sum += partials[index];
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    *result = reduced;
  }
}

void launch_dot(const float *left, const float *right, std::size_t count,
                float *partials, float *result) {
  dot_raw_fp32_kernel<<<dot_blocks, threads>>>(left, right, count, partials);
  finalize_dot_kernel<<<1, threads>>>(partials, dot_blocks, result);
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

  double measure(const float *left, const float *right, std::size_t count,
                 float *partials, float *result) {
    CUDA_CHECK(cudaEventRecord(start_));
    launch_dot(left, right, count, partials, result);
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

void ensure_parent(const std::string &filename) {
  const std::filesystem::path path(filename);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

void run(const settings &settings) {
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  const std::string gpu = properties.name;
  std::cout << "GPU: " << gpu << '\n';
  std::cout << "Raw FP32 DOT: N=" << settings.n
            << ", warmups=" << settings.warmup
            << ", samples=" << settings.samples << '\n';

  device_buffer<float> left(settings.n);
  device_buffer<float> right(settings.n);
  device_buffer<float> partials(dot_blocks);
  device_buffer<float> result(1);
  const auto generation_blocks = static_cast<int>(std::min<std::size_t>(
      4096, (settings.n + threads - 1) / threads));
  generate_values_kernel<<<generation_blocks, threads>>>(left.get(), settings.n,
                                                          left_seed);
  generate_values_kernel<<<generation_blocks, threads>>>(right.get(), settings.n,
                                                           right_seed);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int warmup = 0; warmup < settings.warmup; ++warmup) {
    launch_dot(left.get(), right.get(), settings.n, partials.get(), result.get());
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  ensure_parent(settings.output);
  std::ofstream output(settings.output);
  if (!output) {
    throw benchmark_error("cannot open output CSV: " + settings.output);
  }
  output << "timestamp,gpu,mode,kernel,N,blocks,threads,storage_bits,"
            "arithmetic_type,storage_layout,access_method,packet_values,"
            "lut_entries,lut_bytes,format,x_semantics,left_seed,right_seed,"
            "warmup,sample,kernel_ms,result\n";

  event_timer timer;
  volatile float result_sink{};
  for (int sample = 0; sample < settings.samples; ++sample) {
    const auto milliseconds = timer.measure(left.get(), right.get(), settings.n,
                                             partials.get(), result.get());
    float host_result{};
    CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(host_result),
                          cudaMemcpyDeviceToHost));
    if (!std::isfinite(host_result)) {
      throw benchmark_error("raw FP32 DOT produced a nonfinite result");
    }
    result_sink += host_result;
    output << utc_timestamp() << ',' << gpu << ',' << settings.mode
           << ",dot," << settings.n
           << ",512,256,32,fp32,natural,scalar,1,0,0,raw_fp32,"
              "undefined_no_lut,"
           << left_seed << ',' << right_seed << ',' << settings.warmup << ','
           << sample << ',' << std::setprecision(17) << milliseconds << ','
           << host_result << '\n';
  }
  output.flush();
  std::cout << "Result sink: " << result_sink << '\n';
  std::cout << "Wrote " << settings.output << '\n';
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
