#include "bitwidth_benchmark_kernels.cuh"
#include "normal32_formats.hpp"

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
namespace normal32 = aut::normal32;
namespace storage = aut::storage;

using e9m22_storage = bw::padded_storage_t<storage::e9m22>;
using e11m20_storage = bw::padded_storage_t<storage::e11m20>;

static_assert(sizeof(e9m22_storage) == 4);
static_assert(sizeof(e11m20_storage) == 4);

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
  std::uint64_t seed{0x13198a2e03707344ull};
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

__global__ void truncated_normal_from_uniform_kernel(double *values,
                                                     std::size_t count,
                                                     double cdf_lower,
                                                     double retained_mass,
                                                     double sigma) {
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto probability = cdf_lower + retained_mass * values[index];
    const auto generated =
        sigma * 1.4142135623730950488 * erfinv(2.0 * probability - 1.0);
    constexpr double fp32_max =
        static_cast<double>(std::numeric_limits<float>::max());
    values[index] = fmin(fmax(generated, -fp32_max), fp32_max);
  }
}

__device__ __forceinline__ std::uint32_t encode_compander_device(double value) {
  constexpr double code_count = 4294967296.0;
  const auto normalized =
      value / (normal32::normal_sigma * 1.7320508075688772935);
  const auto probability = 0.5 * erfc(-normalized * 0.7071067811865475244);
  const auto scaled = fmin(probability * code_count, code_count - 1.0);
  return static_cast<std::uint32_t>(static_cast<std::uint64_t>(scaled));
}

__device__ __forceinline__ std::uint32_t
encode_piecewise_device(double value, int local_bits) {
  const auto rank = encode_compander_device(value);
  const auto local_mask = (std::uint32_t{1} << local_bits) - 1u;
  const auto local_half = std::uint32_t{1} << (local_bits - 1);
  const auto segment = rank >> local_bits;
  const auto local_rank = rank & local_mask;
  const auto local = (local_rank - local_half) & local_mask;
  return (segment << local_bits) | local;
}

__device__ __forceinline__ std::int32_t encode_qn32_device(double value) {
  const auto z = fabs(value) / normal32::normal_sigma;
  const auto t =
      (sqrt(normal32::qn_a * normal32::qn_a + 4.0 * normal32::qn_b * z) -
       normal32::qn_a) /
      (2.0 * normal32::qn_b);
  const auto magnitude = nearbyint(fmin(t, 1.0) * normal32::qn_integer_scale);
  const auto signed_value = copysign(magnitude, value);
  const auto clipped =
      fmin(fmax(signed_value,
                static_cast<double>(std::numeric_limits<std::int32_t>::min())),
           static_cast<double>(std::numeric_limits<std::int32_t>::max()));
  return static_cast<std::int32_t>(clipped);
}

__global__ void
encode_formats_kernel(const double *source, float *fp32, e11m20_storage *e11m20,
                      e9m22_storage *e9m22, std::uint32_t *pwl,
                      std::uint32_t *pwq, std::int32_t *qn32,
                      std::uint64_t *e8m29, std::uint64_t *e8m30,
                      std::uint64_t *e11m36, std::size_t count) {
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto value = source[index];
    fp32[index] = static_cast<float>(value);
    e11m20[index] = storage::encode<storage::e11m20>(value);
    e9m22[index] = storage::encode<storage::e9m22>(value);
    pwl[index] = encode_piecewise_device(value, normal32::pwl_local_bits);
    pwq[index] = encode_piecewise_device(value, normal32::pwq_local_bits);
    qn32[index] = encode_qn32_device(value);
    e8m29[index] = normal32::encode_ieee<8, 29>(value);
    e8m30[index] = normal32::encode_ieee<8, 30>(value);
    e11m36[index] = normal32::encode_ieee<11, 36>(value);
  }
}

template <int Bits>
__global__ void pack_extended_kernel(const std::uint64_t *padded,
                                     std::uint32_t *words, std::size_t count) {
  const auto output_words =
      (count * static_cast<std::size_t>(Bits) + 31u) / 32u;
  for (auto output =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       output < output_words;
       output += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto bit = output * 32u;
    const auto index = bit / Bits;
    const auto offset = static_cast<unsigned>(bit % Bits);
    auto value = padded[index] >> offset;
    const auto available = Bits - static_cast<int>(offset);
    if (available < 32 && index + 1 < count) {
      value |= padded[index + 1] << available;
    }
    words[output] = static_cast<std::uint32_t>(value);
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

__device__ __forceinline__ double
decode_e9m22_prefix(std::uint32_t raw, const std::uint32_t *prefix_table) {
  constexpr auto fraction_bits = storage::e9m22::fraction_bits;
  constexpr auto fraction_mask = (std::uint32_t{1} << fraction_bits) - 1u;
  constexpr auto exponent_mask =
      (std::uint32_t{1} << storage::e9m22::exponent_bits) - 1u;
  const auto exponent = (raw >> fraction_bits) & exponent_mask;
  if (exponent == 0 || exponent == exponent_mask) {
    return bw::decode_direct_fp64_branchy<storage::e9m22>(raw);
  }
  const auto fraction = raw & fraction_mask;
  const auto prefix = raw >> fraction_bits;
  return aut::decoder::words_to_double(
      {__ldg(prefix_table + prefix) | (fraction >> (fraction_bits - 20)),
       fraction << (52 - fraction_bits)});
}

template <int FractionBits>
__device__ __forceinline__ double
decode_e8_prefix(std::uint64_t raw, const std::uint32_t *prefix_table) {
  constexpr auto fraction_mask = (std::uint64_t{1} << FractionBits) - 1u;
  const auto exponent = (raw >> FractionBits) & 0xffu;
  if (exponent == 0 || exponent == 0xffu) {
    return normal32::decode_ieee<8, FractionBits>(raw);
  }
  const auto fraction = raw & fraction_mask;
  const auto prefix = static_cast<std::uint32_t>(raw >> FractionBits);
  return aut::decoder::words_to_double(
      {__ldg(prefix_table + prefix) |
           static_cast<std::uint32_t>(fraction >> (FractionBits - 20)),
       static_cast<std::uint32_t>(fraction << (52 - FractionBits))});
}

struct raw_fp64_view {
  const double *values{};
  __device__ __forceinline__ double load(std::size_t index) const {
    return values[index];
  }
};

struct fp32_view {
  const float *values{};
  __device__ __forceinline__ double load(std::size_t index) const {
    return static_cast<double>(values[index]);
  }
};

struct e11m20_view {
  const e11m20_storage *values{};
  __device__ __forceinline__ double load(std::size_t index) const {
    const auto raw = static_cast<std::uint32_t>(values[index]);
    return aut::decoder::words_to_double({raw, 0u});
  }
};

struct e9m22_branchy_view {
  const e9m22_storage *values{};
  __device__ __forceinline__ double load(std::size_t index) const {
    return bw::decode_direct_fp64_branchy<storage::e9m22>(
        static_cast<std::uint32_t>(values[index]));
  }
};

struct e9m22_prefix_view {
  const e9m22_storage *values{};
  const std::uint32_t *prefix{};
  __device__ __forceinline__ double load(std::size_t index) const {
    return decode_e9m22_prefix(static_cast<std::uint32_t>(values[index]),
                               prefix);
  }
};

struct pwl_view {
  const std::uint32_t *codes{};
  const normal32::pwl_coeff *table{};
  __device__ __forceinline__ double load(std::size_t index) const {
    const auto code = codes[index];
    const auto segment = code >> normal32::pwl_local_bits;
    const auto local = static_cast<std::int16_t>(code & 0xffffu);
    const auto coefficient = table[segment];
    return fma(static_cast<double>(local), coefficient.b, coefficient.a);
  }
};

struct qn32_view {
  const std::int32_t *codes{};
  double c1{};
  double c2{};
  __device__ __forceinline__ double load(std::size_t index) const {
    const auto value = static_cast<double>(codes[index]);
    return value * fma(fabs(value), c2, c1);
  }
};

template <int Bits, int FractionBits> struct e8_prefix_view {
  const std::uint32_t *words{};
  const std::uint32_t *prefix{};
  __device__ __forceinline__ double load(std::size_t index) const {
    const auto raw = normal32::load_dense<Bits>(words, index);
    return decode_e8_prefix<FractionBits>(raw, prefix);
  }
};

struct e11m36_view {
  const std::uint32_t *words{};
  __device__ __forceinline__ double load(std::size_t index) const {
    const auto raw = normal32::load_dense<48>(words, index);
    return __longlong_as_double(static_cast<long long>(raw << 16));
  }
};

template <typename View>
__global__ void dot_kernel(View left, View right, std::size_t count,
                           double *partials) {
  __shared__ double shared[256];
  double sum{};
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum = fma(left.load(index), right.load(index), sum);
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

__device__ __forceinline__ double
decode_pwq_shared(std::uint32_t code, const normal32::pwq_coeff *table) {
  const auto segment = code >> normal32::pwq_local_bits;
  const auto local_bits = code & 0x00ffffffu;
  const auto local = (local_bits & 0x00800000u) != 0
                         ? static_cast<std::int32_t>(local_bits | 0xff000000u)
                         : static_cast<std::int32_t>(local_bits);
  const auto coefficient = table[segment];
  const auto position = static_cast<double>(local);
  return fma(position, fma(position, coefficient.c, coefficient.b),
             coefficient.a);
}

__global__ void pwq_dot_kernel(const std::uint32_t *left,
                               const std::uint32_t *right,
                               const normal32::pwq_coeff *global_table,
                               std::size_t count, double *partials) {
  __shared__ normal32::pwq_coeff table[normal32::pwq_segment_count];
  __shared__ double shared[256];
  table[threadIdx.x] = global_table[threadIdx.x];
  __syncthreads();

  double sum{};
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum = fma(decode_pwq_shared(left[index], table),
              decode_pwq_shared(right[index], table), sum);
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

__global__ void finalize_dot_kernel(const double *partials, std::size_t count,
                                    double *result) {
  __shared__ double shared[256];
  double sum{};
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
                double *partials, double *result) {
  dot_kernel<<<blocks, 256>>>(left, right, count, partials);
  finalize_dot_kernel<<<1, 256>>>(partials, static_cast<std::size_t>(blocks),
                                  result);
}

void launch_pwq_dot(const std::uint32_t *left, const std::uint32_t *right,
                    const normal32::pwq_coeff *table, std::size_t count,
                    int blocks, double *partials, double *result) {
  pwq_dot_kernel<<<blocks, 256>>>(left, right, table, count, partials);
  finalize_dot_kernel<<<1, 256>>>(partials, static_cast<std::size_t>(blocks),
                                  result);
}

std::vector<std::uint32_t> make_prefix_table(int exponent_bits) {
  const auto entries = std::size_t{1} << (exponent_bits + 1);
  const auto exponent_mask = (std::uint32_t{1} << exponent_bits) - 1u;
  const auto exponent_bias = (1 << (exponent_bits - 1)) - 1;
  std::vector<std::uint32_t> result(entries);
  for (std::size_t prefix = 0; prefix < entries; ++prefix) {
    const auto sign = static_cast<std::uint32_t>(prefix >> exponent_bits);
    const auto exponent = static_cast<std::uint32_t>(prefix) & exponent_mask;
    if (exponent == 0) {
      result[prefix] = sign << 31;
    } else if (exponent == exponent_mask) {
      result[prefix] = (sign << 31) | 0x7ff00000u;
    } else {
      const auto exponent64 = static_cast<std::uint32_t>(
          static_cast<int>(exponent) - exponent_bias + 1023);
      result[prefix] = (sign << 31) | (exponent64 << 20);
    }
  }
  return result;
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
  double result{};
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
             << normal32::normal_sigma << ',' << normal32::normal_cutoff_sigma
             << ",dot," << variant.format << ',' << variant.bits << ",fp64,"
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

template <int Bits>
void pack_codes(const device_buffer<std::uint64_t> &padded,
                device_buffer<std::uint32_t> &dense, std::size_t count,
                int preparation_blocks, int threads) {
  CUDA_CHECK(cudaMemset(dense.get(), 0, dense.size() * sizeof(std::uint32_t)));
  pack_extended_kernel<Bits>
      <<<preparation_blocks, threads>>>(padded.get(), dense.get(), count);
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

  std::cout << "Building PWLNormal32 and PWQNormal32 coefficient tables\n";
  const auto host_tables = normal32::build_codebook_tables();
  device_buffer<normal32::pwl_coeff> pwl_table(host_tables.pwl.size());
  device_buffer<normal32::pwq_coeff> pwq_table(host_tables.pwq.size());
  CUDA_CHECK(cudaMemcpy(pwl_table.get(), host_tables.pwl.data(),
                        host_tables.pwl.size() * sizeof(normal32::pwl_coeff),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pwq_table.get(), host_tables.pwq.data(),
                        host_tables.pwq.size() * sizeof(normal32::pwq_coeff),
                        cudaMemcpyHostToDevice));

  const auto host_e8_prefix = make_prefix_table(8);
  const auto host_e9_prefix = make_prefix_table(9);
  device_buffer<std::uint32_t> e8_prefix(host_e8_prefix.size());
  device_buffer<std::uint32_t> e9_prefix(host_e9_prefix.size());
  CUDA_CHECK(cudaMemcpy(e8_prefix.get(), host_e8_prefix.data(),
                        host_e8_prefix.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(e9_prefix.get(), host_e9_prefix.data(),
                        host_e9_prefix.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));

  device_buffer<double> source_left(settings.count);
  device_buffer<double> source_right(settings.count);
  curandGenerator_t generator{};
  CURAND_CHECK(
      curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, settings.seed));
  CURAND_CHECK(curandGenerateUniformDouble(generator, source_left.get(),
                                           settings.count));
  CURAND_CHECK(curandGenerateUniformDouble(generator, source_right.get(),
                                           settings.count));
  CURAND_CHECK(curandDestroyGenerator(generator));

  const auto cdf_lower =
      normal32::standard_normal_cdf(-normal32::normal_cutoff_sigma);
  const auto retained_mass =
      normal32::standard_normal_cdf(normal32::normal_cutoff_sigma) - cdf_lower;
  truncated_normal_from_uniform_kernel<<<preparation_blocks, threads>>>(
      source_left.get(), settings.count, cdf_lower, retained_mass,
      normal32::normal_sigma);
  truncated_normal_from_uniform_kernel<<<preparation_blocks, threads>>>(
      source_right.get(), settings.count, cdf_lower, retained_mass,
      normal32::normal_sigma);

  device_buffer<float> fp32_left(settings.count);
  device_buffer<float> fp32_right(settings.count);
  device_buffer<e11m20_storage> e11m20_left(settings.count);
  device_buffer<e11m20_storage> e11m20_right(settings.count);
  device_buffer<e9m22_storage> e9m22_left(settings.count);
  device_buffer<e9m22_storage> e9m22_right(settings.count);
  device_buffer<std::uint32_t> pwl_left(settings.count);
  device_buffer<std::uint32_t> pwl_right(settings.count);
  device_buffer<std::uint32_t> pwq_left(settings.count);
  device_buffer<std::uint32_t> pwq_right(settings.count);
  device_buffer<std::int32_t> qn32_left(settings.count);
  device_buffer<std::int32_t> qn32_right(settings.count);

  device_buffer<std::uint64_t> e8m29_left_padded(settings.count);
  device_buffer<std::uint64_t> e8m29_right_padded(settings.count);
  device_buffer<std::uint64_t> e8m30_left_padded(settings.count);
  device_buffer<std::uint64_t> e8m30_right_padded(settings.count);
  device_buffer<std::uint64_t> e11m36_left_padded(settings.count);
  device_buffer<std::uint64_t> e11m36_right_padded(settings.count);

  encode_formats_kernel<<<preparation_blocks, threads>>>(
      source_left.get(), fp32_left.get(), e11m20_left.get(), e9m22_left.get(),
      pwl_left.get(), pwq_left.get(), qn32_left.get(), e8m29_left_padded.get(),
      e8m30_left_padded.get(), e11m36_left_padded.get(), settings.count);
  encode_formats_kernel<<<preparation_blocks, threads>>>(
      source_right.get(), fp32_right.get(), e11m20_right.get(),
      e9m22_right.get(), pwl_right.get(), pwq_right.get(), qn32_right.get(),
      e8m29_right_padded.get(), e8m30_right_padded.get(),
      e11m36_right_padded.get(), settings.count);

  device_buffer<std::uint32_t> e8m29_left(
      normal32::dense_word_count<38>(settings.count));
  device_buffer<std::uint32_t> e8m29_right(
      normal32::dense_word_count<38>(settings.count));
  device_buffer<std::uint32_t> e8m30_left(
      normal32::dense_word_count<39>(settings.count));
  device_buffer<std::uint32_t> e8m30_right(
      normal32::dense_word_count<39>(settings.count));
  device_buffer<std::uint32_t> e11m36_left(
      normal32::dense_word_count<48>(settings.count));
  device_buffer<std::uint32_t> e11m36_right(
      normal32::dense_word_count<48>(settings.count));
  pack_codes<38>(e8m29_left_padded, e8m29_left, settings.count,
                 preparation_blocks, threads);
  pack_codes<38>(e8m29_right_padded, e8m29_right, settings.count,
                 preparation_blocks, threads);
  pack_codes<39>(e8m30_left_padded, e8m30_left, settings.count,
                 preparation_blocks, threads);
  pack_codes<39>(e8m30_right_padded, e8m30_right, settings.count,
                 preparation_blocks, threads);
  pack_codes<48>(e11m36_left_padded, e11m36_left, settings.count,
                 preparation_blocks, threads);
  pack_codes<48>(e11m36_right_padded, e11m36_right, settings.count,
                 preparation_blocks, threads);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  device_buffer<double> partials(static_cast<std::size_t>(blocks));
  device_buffer<double> result(1);
  const auto make_launch = [&](auto left, auto right) {
    return [=, &partials, &result] {
      launch_dot(left, right, settings.count, blocks, partials.get(),
                 result.get());
    };
  };

  const auto c1 =
      normal32::normal_sigma * normal32::qn_a / normal32::qn_integer_scale;
  const auto c2 = normal32::normal_sigma * normal32::qn_b /
                  (normal32::qn_integer_scale * normal32::qn_integer_scale);
  std::vector<benchmark_variant> variants;
  variants.push_back({"pwl_normal32_16_16", 32, "natural",
                      "global_coefficients", "pwl_global_x1",
                      2 * settings.count * sizeof(std::uint32_t),
                      host_tables.pwl.size() * sizeof(normal32::pwl_coeff),
                      make_launch(pwl_view{pwl_left.get(), pwl_table.get()},
                                  pwl_view{pwl_right.get(), pwl_table.get()})});
  variants.push_back(
      {"pwq_normal32_8_24", 32, "natural", "shared_coefficients",
       "pwq_shared_x1", 2 * settings.count * sizeof(std::uint32_t),
       host_tables.pwq.size() * sizeof(normal32::pwq_coeff),
       [=, &partials, &result] {
         launch_pwq_dot(pwq_left.get(), pwq_right.get(), pwq_table.get(),
                        settings.count, blocks, partials.get(), result.get());
       }});
  variants.push_back({"qn32", 32, "natural", "direct_quadratic", "qn_direct_x1",
                      2 * settings.count * sizeof(std::int32_t), 0,
                      make_launch(qn32_view{qn32_left.get(), c1, c2},
                                  qn32_view{qn32_right.get(), c1, c2})});
  variants.push_back(
      {"fp32_e8m23", 32, "natural", "native_numeric", "native_f64_x1",
       2 * settings.count * sizeof(float), 0,
       make_launch(fp32_view{fp32_left.get()}, fp32_view{fp32_right.get()})});
  variants.push_back({"e11m20", 32, "natural", "direct_shift",
                      "direct_shift_x1",
                      2 * settings.count * sizeof(e11m20_storage), 0,
                      make_launch(e11m20_view{e11m20_left.get()},
                                  e11m20_view{e11m20_right.get()})});
  variants.push_back(
      {"e9m22", 32, "natural", "prefix_lut_global", "prefix_global_x1",
       2 * settings.count * sizeof(e9m22_storage),
       host_e9_prefix.size() * sizeof(std::uint32_t),
       make_launch(e9m22_prefix_view{e9m22_left.get(), e9_prefix.get()},
                   e9m22_prefix_view{e9m22_right.get(), e9_prefix.get()})});
  variants.push_back({"e9m22", 32, "natural", "direct_branchy",
                      "word_branchy_x1",
                      2 * settings.count * sizeof(e9m22_storage), 0,
                      make_launch(e9m22_branchy_view{e9m22_left.get()},
                                  e9m22_branchy_view{e9m22_right.get()})});
  variants.push_back(
      {"e8m29", 38, "dense", "prefix_lut_global", "prefix_global_x1",
       2 * normal32::dense_data_bytes<38>(settings.count),
       host_e8_prefix.size() * sizeof(std::uint32_t),
       make_launch(
           e8_prefix_view<38, 29>{e8m29_left.get(), e8_prefix.get()},
           e8_prefix_view<38, 29>{e8m29_right.get(), e8_prefix.get()})});
  variants.push_back(
      {"e8m30", 39, "dense", "prefix_lut_global", "prefix_global_x1",
       2 * normal32::dense_data_bytes<39>(settings.count),
       host_e8_prefix.size() * sizeof(std::uint32_t),
       make_launch(
           e8_prefix_view<39, 30>{e8m30_left.get(), e8_prefix.get()},
           e8_prefix_view<39, 30>{e8m30_right.get(), e8_prefix.get()})});
  variants.push_back({"e11m36", 48, "dense", "direct_shift", "direct_shift_x1",
                      2 * normal32::dense_data_bytes<48>(settings.count), 0,
                      make_launch(e11m36_view{e11m36_left.get()},
                                  e11m36_view{e11m36_right.get()})});
  variants.push_back({"raw_fp64", 64, "natural", "raw_pointer",
                      "raw_pointer_x1", 2 * settings.count * sizeof(double), 0,
                      make_launch(raw_fp64_view{source_left.get()},
                                  raw_fp64_view{source_right.get()})});

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
    CUDA_CHECK(cudaMemcpy(&variant.result, result.get(), sizeof(double),
                          cudaMemcpyDeviceToHost));
    variant.valid = std::isfinite(variant.result);
    if (!variant.valid) {
      throw std::runtime_error(variant.format + "/" + variant.strategy_id +
                               " produced a non-finite DOT");
    }
  }

  write_output(settings, gpu, blocks, variants);
  std::cout << "Normal32 DOT benchmark complete on " << gpu
            << " at N=" << settings.count << '\n';
  for (const auto &variant : variants) {
    std::cout << "  " << std::setw(22) << std::left << variant.format << ' '
              << std::setw(20) << variant.strategy_id << ' ' << std::fixed
              << std::setprecision(6) << median_mean_ms(variant) << " ms\n";
  }
  std::cout << "Raw samples: " << settings.output << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    run(parse_options(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << "normal32_dot_bench: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
