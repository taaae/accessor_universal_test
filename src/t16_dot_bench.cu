#include "bitwidth_benchmark_kernels.cuh"
#include "t16_codebook.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace bw = aut::bitwidth;
namespace storage = aut::storage;

using e6m9_storage = bw::padded_storage_t<storage::e6m9>;
using e8m15_storage = bw::padded_storage_t<storage::e8m15>;

static_assert(sizeof(e6m9_storage) == 2);
static_assert(sizeof(e8m15_storage) == 4);

void check_cuda(cudaError_t status, const char *expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

void check_curand(curandStatus_t status, const char *expression) {
  if (status != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(expression) + " failed with code " +
                             std::to_string(static_cast<int>(status)));
  }
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression)
#define CURAND_CHECK(expression) check_curand((expression), #expression)

struct options {
  std::string mode{"full"};
  std::size_t count{std::size_t{1} << 27};
  int warmup{10};
  int samples{30};
  double target_sample_ms{20.0};
  std::uint64_t seed{0x243f6a8885a308d3ull};
  std::string output{"timing_samples.csv"};
};

options parse_options(int argc, char **argv) {
  options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto require_value = [&]() -> std::string {
      if (++index >= argc) {
        throw std::runtime_error("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--mode") {
      result.mode = require_value();
    } else if (argument == "--n") {
      result.count = std::stoull(require_value());
    } else if (argument == "--warmup") {
      result.warmup = std::stoi(require_value());
    } else if (argument == "--samples") {
      result.samples = std::stoi(require_value());
    } else if (argument == "--target-sample-ms") {
      result.target_sample_ms = std::stod(require_value());
    } else if (argument == "--seed") {
      result.seed = std::stoull(require_value());
    } else if (argument == "--output") {
      result.output = require_value();
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (result.mode == "smoke") {
    result.count = std::size_t{1} << 20;
    result.warmup = 1;
    result.samples = 2;
    result.target_sample_ms = 0.0;
  } else if (result.mode != "full") {
    throw std::runtime_error("mode must be smoke or full");
  }
  if (result.count == 0 || result.warmup < 0 || result.samples <= 0 ||
      result.target_sample_ms < 0.0) {
    throw std::runtime_error("invalid benchmark settings");
  }
  return result;
}

std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  std::ostringstream stream;
  stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return stream.str();
}

template <typename T> class device_buffer {
public:
  explicit device_buffer(std::size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_),
                          std::max<std::size_t>(count, 1) * sizeof(T)));
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

  double measure(int iterations, const std::function<void()> &launch) {
    CUDA_CHECK(cudaEventRecord(start_));
    for (int iteration = 0; iteration < iterations; ++iteration) {
      launch();
    }
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

__global__ void truncated_normal_from_uniform_kernel(float *values,
                                                     std::size_t count,
                                                     float cdf_lower,
                                                     float retained_mass,
                                                     float sigma) {
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto probability = cdf_lower + retained_mass * values[index];
    values[index] =
        sigma * 1.4142135623730950488f * erfinvf(2.0f * probability - 1.0f);
  }
}

__device__ __forceinline__ std::uint16_t encode_t16(float value,
                                                    const float *thresholds) {
  std::uint32_t lower = 0;
  std::uint32_t upper = static_cast<std::uint32_t>(aut::t16::code_count - 1);
  while (lower < upper) {
    const auto middle = lower + ((upper - lower) >> 1);
    if (value > thresholds[middle]) {
      lower = middle + 1;
    } else {
      upper = middle;
    }
  }
  return static_cast<std::uint16_t>(lower);
}

__global__ void encode_formats_kernel(const float *source, __half *fp16,
                                      e6m9_storage *e6m9, e8m15_storage *e8m15,
                                      std::uint16_t *t16,
                                      const float *t16_thresholds,
                                      std::size_t count) {
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto value = source[index];
    fp16[index] = __float2half_rn(value);
    e6m9[index] = storage::encode<storage::e6m9>(static_cast<double>(value));
    e8m15[index] = storage::encode<storage::e8m15>(static_cast<double>(value));
    t16[index] = encode_t16(value, t16_thresholds);
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

struct raw_fp32_view {
  const float *values{};
  __device__ __forceinline__ float load(std::size_t index) const {
    return values[index];
  }
};

struct fp16_view {
  const __half *values{};
  __device__ __forceinline__ float load(std::size_t index) const {
    return __half2float(values[index]);
  }
};

struct e6m9_view {
  const e6m9_storage *values{};
  __device__ __forceinline__ float load(std::size_t index) const {
    return bw::decode_direct_fp32<storage::e6m9>(
        static_cast<std::uint32_t>(values[index]));
  }
};

struct e8m15_view {
  const std::uint32_t *words{};
  __device__ __forceinline__ float load(std::size_t index) const {
    const auto raw =
        bw::load_dense_scalar<storage::e8m15::total_bits>(words, index);
    return bw::decode_direct_fp32<storage::e8m15>(raw);
  }
};

struct t16_view {
  const std::uint16_t *codes{};
  const float *table{};
  __device__ __forceinline__ float load(std::size_t index) const {
    return __ldg(table + codes[index]);
  }
};

template <typename View>
__global__ void dot_kernel(View left, View right, std::size_t count,
                           float *partials) {
  __shared__ float shared[256];
  float sum{};
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum = left.load(index) * right.load(index) + sum;
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

__global__ void finalize_dot_kernel(const float *partials, std::size_t count,
                                    float *result) {
  __shared__ float shared[256];
  float sum{};
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    sum += partials[index];
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    *result = reduced;
  }
}

template <typename View>
void launch_dot(View left, View right, std::size_t count, int blocks,
                float *partials, float *result) {
  dot_kernel<<<blocks, 256>>>(left, right, count, partials);
  finalize_dot_kernel<<<1, 256>>>(partials, static_cast<std::size_t>(blocks),
                                  result);
}

struct timing_sample {
  int sample{};
  int execution_order{};
  double total_ms{};
};

struct benchmark_variant {
  std::string format;
  int bits{};
  std::string storage_layout;
  std::string decoder;
  std::string strategy_id;
  std::size_t physical_input_bytes{};
  std::size_t table_bytes{};
  std::function<void()> launch;
  int iterations{1};
  float result{};
  bool valid{};
  std::vector<timing_sample> timings;
};

double median_mean_ms(const benchmark_variant &variant) {
  std::vector<double> values;
  values.reserve(variant.timings.size());
  for (const auto &sample : variant.timings) {
    values.push_back(sample.total_ms / variant.iterations);
  }
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  if ((values.size() & 1u) != 0) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1] + values[middle]);
}

void write_output(const options &settings, const std::string &gpu, int blocks,
                  const std::vector<benchmark_variant> &variants) {
  const std::filesystem::path output_path(settings.output);
  if (output_path.has_parent_path()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(settings.output);
  if (!output) {
    throw std::runtime_error("cannot open output: " + settings.output);
  }
  output << "timestamp,gpu,mode,distribution,sigma,cutoff_sigma,kernel,format,"
            "bits,arithmetic_type,storage_layout,access_method,packet_values,"
            "decoder,strategy_id,N,logical_input_bytes,physical_input_bytes,"
            "table_bytes,blocks,threads,warmup,sample,execution_order,"
            "iterations,total_ms,mean_ms,result,valid\n";
  const auto timestamp = utc_timestamp();
  for (const auto &variant : variants) {
    for (const auto &sample : variant.timings) {
      output << timestamp << ',' << gpu << ',' << settings.mode
             << ",truncated_normal," << std::setprecision(17)
             << aut::t16::normal_sigma << ',' << aut::t16::normal_cutoff_sigma
             << ",dot," << variant.format << ',' << variant.bits << ",fp32,"
             << variant.storage_layout << ",scalar,1," << variant.decoder << ','
             << variant.strategy_id << ',' << settings.count << ','
             << (2.0 * settings.count * variant.bits / 8.0) << ','
             << variant.physical_input_bytes << ',' << variant.table_bytes
             << ',' << blocks << ",256," << settings.warmup << ','
             << sample.sample << ',' << sample.execution_order << ','
             << variant.iterations << ',' << sample.total_ms << ','
             << sample.total_ms / variant.iterations << ',' << variant.result
             << ',' << (variant.valid ? 1 : 0) << '\n';
    }
  }
}

void run(const options &settings) {
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  const std::string gpu = properties.name;
  const auto blocks = static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(512, (settings.count + 255) / 256)));
  constexpr int threads = 256;
  const auto preparation_blocks = static_cast<int>(
      std::min<std::size_t>(4096, (settings.count + threads - 1) / threads));

  std::cout << "Building the 65,536-entry T16 normal codebook\n";
  const auto host_codebook = aut::t16::build_normal_codebook();
  device_buffer<float> t16_table(host_codebook.values.size());
  device_buffer<float> t16_thresholds(host_codebook.thresholds.size());
  CUDA_CHECK(cudaMemcpy(t16_table.get(), host_codebook.values.data(),
                        host_codebook.values.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(t16_thresholds.get(), host_codebook.thresholds.data(),
                        host_codebook.thresholds.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  device_buffer<float> source_left(settings.count);
  device_buffer<float> source_right(settings.count);
  curandGenerator_t generator{};
  CURAND_CHECK(
      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, settings.seed));
  CURAND_CHECK(
      curandGenerateUniform(generator, source_left.get(), settings.count));
  CURAND_CHECK(
      curandGenerateUniform(generator, source_right.get(), settings.count));
  CURAND_CHECK(curandDestroyGenerator(generator));

  const auto cdf_lower = static_cast<float>(
      aut::t16::standard_normal_cdf(-aut::t16::normal_cutoff_sigma));
  const auto retained_mass = static_cast<float>(
      aut::t16::standard_normal_cdf(aut::t16::normal_cutoff_sigma) -
      aut::t16::standard_normal_cdf(-aut::t16::normal_cutoff_sigma));
  truncated_normal_from_uniform_kernel<<<preparation_blocks, threads>>>(
      source_left.get(), settings.count, cdf_lower, retained_mass,
      static_cast<float>(aut::t16::normal_sigma));
  truncated_normal_from_uniform_kernel<<<preparation_blocks, threads>>>(
      source_right.get(), settings.count, cdf_lower, retained_mass,
      static_cast<float>(aut::t16::normal_sigma));

  device_buffer<__half> fp16_left(settings.count);
  device_buffer<__half> fp16_right(settings.count);
  device_buffer<e6m9_storage> e6m9_left(settings.count);
  device_buffer<e6m9_storage> e6m9_right(settings.count);
  device_buffer<std::uint16_t> t16_left(settings.count);
  device_buffer<std::uint16_t> t16_right(settings.count);
  device_buffer<std::uint32_t> e8m15_left(
      bw::dense_word_count<storage::e8m15::total_bits>(settings.count));
  device_buffer<std::uint32_t> e8m15_right(
      bw::dense_word_count<storage::e8m15::total_bits>(settings.count));
  CUDA_CHECK(cudaMemset(e8m15_left.get(), 0,
                        e8m15_left.size() * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMemset(e8m15_right.get(), 0,
                        e8m15_right.size() * sizeof(std::uint32_t)));

  {
    device_buffer<e8m15_storage> e8m15_left_padded(settings.count);
    device_buffer<e8m15_storage> e8m15_right_padded(settings.count);
    encode_formats_kernel<<<preparation_blocks, threads>>>(
        source_left.get(), fp16_left.get(), e6m9_left.get(),
        e8m15_left_padded.get(), t16_left.get(), t16_thresholds.get(),
        settings.count);
    encode_formats_kernel<<<preparation_blocks, threads>>>(
        source_right.get(), fp16_right.get(), e6m9_right.get(),
        e8m15_right_padded.get(), t16_right.get(), t16_thresholds.get(),
        settings.count);

    using geometry = bw::dense_geometry<storage::e8m15::total_bits>;
    const auto chunks =
        (settings.count + geometry::values_per_aligned_chunk - 1) /
        geometry::values_per_aligned_chunk;
    const auto pack_blocks = static_cast<int>(
        std::min<std::size_t>(4096, (chunks + threads - 1) / threads));
    bw::pack_dense_kernel<storage::e8m15><<<pack_blocks, threads>>>(
        e8m15_left_padded.get(), e8m15_left.get(), settings.count);
    bw::pack_dense_kernel<storage::e8m15><<<pack_blocks, threads>>>(
        e8m15_right_padded.get(), e8m15_right.get(), settings.count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  device_buffer<float> partials(static_cast<std::size_t>(blocks));
  device_buffer<float> result(1);
  const auto make_launch = [&](auto left, auto right) {
    return [=, &partials, &result] {
      launch_dot(left, right, settings.count, blocks, partials.get(),
                 result.get());
    };
  };

  std::vector<benchmark_variant> variants;
  variants.push_back({"t16", 16, "natural", "full_lut_global", "global_lut_x1",
                      2 * settings.count * sizeof(std::uint16_t),
                      aut::t16::code_count * sizeof(float),
                      make_launch(t16_view{t16_left.get(), t16_table.get()},
                                  t16_view{t16_right.get(), t16_table.get()})});
  variants.push_back(
      {"fp16_e5m10", 16, "natural", "native_scalar", "native_scalar_x1",
       2 * settings.count * sizeof(__half), 0,
       make_launch(fp16_view{fp16_left.get()}, fp16_view{fp16_right.get()})});
  variants.push_back(
      {"e6m9", 16, "natural", "direct_branchy", "direct_branchy_x1",
       2 * settings.count * sizeof(e6m9_storage), 0,
       make_launch(e6m9_view{e6m9_left.get()}, e6m9_view{e6m9_right.get()})});
  variants.push_back(
      {"e8m15", 24, "dense", "direct_shift", "direct_shift_x1",
       2 * bw::dense_data_bytes<storage::e8m15::total_bits>(settings.count), 0,
       make_launch(e8m15_view{e8m15_left.get()},
                   e8m15_view{e8m15_right.get()})});
  variants.push_back({"raw_fp32", 32, "natural", "raw", "raw_x1",
                      2 * settings.count * sizeof(float), 0,
                      make_launch(raw_fp32_view{source_left.get()},
                                  raw_fp32_view{source_right.get()})});

  std::mt19937_64 order_generator(settings.seed ^ 0x9e3779b97f4a7c15ull);
  std::vector<std::size_t> order(variants.size());
  std::iota(order.begin(), order.end(), 0);
  for (int warmup = 0; warmup < settings.warmup; ++warmup) {
    std::shuffle(order.begin(), order.end(), order_generator);
    for (const auto index : order) {
      variants[index].launch();
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  event_timer timer;
  for (auto &variant : variants) {
    const auto pilot = timer.measure(1, variant.launch);
    if (settings.target_sample_ms > 0.0 && pilot > 0.0) {
      variant.iterations = std::clamp(
          static_cast<int>(std::ceil(settings.target_sample_ms / pilot)), 1,
          10000);
    }
  }

  for (int sample = 0; sample < settings.samples; ++sample) {
    std::shuffle(order.begin(), order.end(), order_generator);
    for (std::size_t execution_order = 0; execution_order < order.size();
         ++execution_order) {
      auto &variant = variants[order[execution_order]];
      variant.timings.push_back(
          {sample, static_cast<int>(execution_order),
           timer.measure(variant.iterations, variant.launch)});
    }
  }

  for (auto &variant : variants) {
    variant.launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&variant.result, result.get(), sizeof(float),
                          cudaMemcpyDeviceToHost));
    variant.valid = std::isfinite(variant.result);
    if (!variant.valid) {
      throw std::runtime_error(variant.format + " produced a non-finite DOT");
    }
  }

  write_output(settings, gpu, blocks, variants);
  std::cout << "T16 DOT benchmark complete on " << gpu
            << " at N=" << settings.count << '\n';
  for (const auto &variant : variants) {
    std::cout << "  " << std::setw(12) << std::left << variant.format << ' '
              << std::fixed << std::setprecision(6) << median_mean_ms(variant)
              << " ms\n";
  }
  std::cout << "Raw samples: " << settings.output << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    run(parse_options(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << "t16_dot_bench: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
