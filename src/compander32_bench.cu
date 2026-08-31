#include "bitwidth_benchmark_kernels.cuh"
#include "compander32_core.hpp"
#include "dyadic_normal32_core.hpp"
#include "lns_decoder_strategies.cuh"
#include "posit_takum_core.hpp"

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
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace bw = aut::bitwidth;
namespace c32 = aut::compander32;
namespace dn = aut::dyadic_normal32;
namespace lk = aut::lns_strategy;
namespace lns = aut::lns;
namespace pt = aut::pt;
namespace storage = aut::storage;

using e10_storage = bw::padded_storage_t<storage::e10m21>;
using e11_storage = bw::padded_storage_t<storage::e11m20>;
using lns_format = lns::lns32_r23;

constexpr int threads = 256;
constexpr int blocks = 512;
constexpr std::size_t validation_n = 4096;
constexpr std::uint64_t left_seed = 0x243f6a8885a308d3ULL;
constexpr std::uint64_t right_seed = 0x13198a2e03707344ULL;

static_assert(sizeof(e10_storage) == 4);
static_assert(sizeof(e11_storage) == 4);

class benchmark_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void cuda_check(cudaError_t status, const char *expression, const char *file,
                int line) {
  if (status != cudaSuccess) {
    throw benchmark_error(std::string(file) + ':' + std::to_string(line) + ' ' +
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
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  device_buffer &operator=(device_buffer &&other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  ~device_buffer() { release(); }

  void reset(std::size_t count) {
    release();
    count_ = count;
    if (count != 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_),
                            count * sizeof(T)));
    }
  }

  T *get() { return data_; }
  const T *get() const { return data_; }
  std::size_t size() const { return count_; }

private:
  void release() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = nullptr;
    count_ = 0;
  }

  T *data_{};
  std::size_t count_{};
};

struct options {
  std::string mode{"full"};
  int warmups{10};
  int samples{50};
  std::string output{"timing_samples.csv"};
  std::string correctness_output{"correctness_checks.txt"};
};

options parse_options(int argc, char **argv) {
  options result;
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
    } else if (argument == "--warmups") {
      result.warmups = std::stoi(value());
    } else if (argument == "--samples") {
      result.samples = std::stoi(value());
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--correctness-output") {
      result.correctness_output = value();
    } else {
      throw benchmark_error("unknown option: " + argument);
    }
  }
  if (result.mode == "smoke") {
    result.warmups = 1;
    result.samples = 3;
  } else if (result.mode != "full") {
    throw benchmark_error("--mode must be smoke or full");
  }
  if (result.warmups <= 0 || result.samples <= 0) {
    throw benchmark_error("warmups and samples must be positive");
  }
  return result;
}

std::vector<std::size_t> benchmark_sizes(const options &settings) {
  if (settings.mode == "smoke") {
    return {std::size_t{1} << 20};
  }
  return {std::size_t{1} << 20, std::size_t{1} << 22,
          std::size_t{1} << 24, std::size_t{1} << 26,
          std::size_t{1} << 28};
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

__host__ __device__ __forceinline__ std::uint64_t
splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__device__ __forceinline__ double logical_normal(std::size_t index,
                                                 std::uint64_t seed) {
  constexpr double inverse_53 = 1.0 / 9007199254740992.0;
  constexpr double two_pi = 6.283185307179586476925286766559;
  const auto first = splitmix64(seed ^ static_cast<std::uint64_t>(index));
  const auto second = splitmix64(seed ^ static_cast<std::uint64_t>(index) ^
                                 0x9e3779b97f4a7c15ULL);
  const auto u1 = (static_cast<double>(first >> 11) + 0.5) * inverse_53;
  const auto u2 = (static_cast<double>(second >> 11) + 0.5) * inverse_53;
  return c32::clamp_source(sqrt(-2.0 * log(u1)) * cos(two_pi * u2));
}

__global__ void generate_source_kernel(double *raw64, float *raw32,
                                       std::size_t count, std::uint64_t seed) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto value = logical_normal(index, seed);
    raw64[index] = value;
    raw32[index] = static_cast<float>(value);
  }
}

__global__ void encode_simple_kernel(
    const double *source, std::int32_t *integer, std::int32_t *quadratic,
    std::int32_t *blended_quadratic, std::int32_t *blended_cubic,
    std::uint32_t *pwl2, std::uint32_t *pwl4, e11_storage *e11,
    e10_storage *e10, std::uint32_t *lns_codes, std::size_t count) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto value = source[index];
    integer[index] = c32::encode_integer(value);
    quadratic[index] = c32::encode_quadratic(value);
    blended_quadratic[index] = c32::encode_blended_quadratic(value);
    blended_cubic[index] = c32::encode_blended_cubic(value);
    pwl2[index] = c32::encode_pwl2(value);
    pwl4[index] = c32::encode_pwl4(value);
    e11[index] = storage::encode<storage::e11m20>(value);
    e10[index] = storage::encode<storage::e10m21>(value);
    lns_codes[index] = lns::encode<lns_format>(value);
  }
}

template <pt::family Family>
__device__ __forceinline__ std::uint32_t encode_posit_takum(double value) {
  if (value == 0.0) {
    return 0u;
  }
  const auto negative = value < 0.0;
  const auto target = fabs(value);
  std::uint32_t low = 1u;
  std::uint32_t high = 0x7fffffffu;
#pragma unroll 1
  while (low < high) {
    const auto middle = low + ((high - low) >> 1);
    const auto decoded = pt::decode<Family, 32, 2, double>(middle);
    if (decoded < target) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  auto magnitude = low;
  if (low > 1u) {
    const auto upper_value = pt::decode<Family, 32, 2, double>(low);
    const auto lower_value = pt::decode<Family, 32, 2, double>(low - 1u);
    if (target - lower_value <= upper_value - target) {
      magnitude = low - 1u;
    }
  }
  return negative ? (~magnitude + 1u) : magnitude;
}

__global__ void encode_posit_takum_kernel(const double *source,
                                          std::uint32_t *posit,
                                          std::uint32_t *takum,
                                          std::size_t count) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    posit[index] = encode_posit_takum<pt::family::posit>(source[index]);
    takum[index] =
        encode_posit_takum<pt::family::takum_linear>(source[index]);
  }
}

__global__ void encode_dyadic_kernel(
    const double *source, std::uint32_t *codes,
    const dn::segment_coefficients *coefficients, const double *boundaries,
    std::size_t count) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto value = source[index];
    const auto magnitude = fabs(value);
    std::uint32_t segment{};
#pragma unroll
    for (std::uint32_t candidate = 1; candidate < dn::segment_count;
         ++candidate) {
      segment += magnitude >= boundaries[candidate];
    }
    std::uint32_t payload{};
    if (segment < dn::segment_count - 1) {
      const auto coefficient = coefficients[segment];
      const auto maximum = dn::payload_mask >> segment;
      auto rounded = nearbyint((magnitude - coefficient.start) /
                               coefficient.step);
      rounded = fmin(fmax(rounded, 0.0), static_cast<double>(maximum));
      payload = static_cast<std::uint32_t>(rounded);
    }
    const auto rank = dn::rank_from_segment_payload_unchecked(segment, payload);
    const auto sign = static_cast<std::uint32_t>(value < 0.0) << 31;
    codes[index] = rank | sign;
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

template <typename Decoder, typename Stored>
__device__ __forceinline__ void dot_double_body(const Stored *left,
                                                const Stored *right,
                                                std::size_t count,
                                                double *partials,
                                                Decoder decoder) {
  __shared__ double shared[threads];
  double sum{};
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum = fma(decoder(left[index]), decoder(right[index]), sum);
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

struct raw64_decoder {
  __device__ __forceinline__ double operator()(double value) const {
    return value;
  }
};
struct fp32_to_fp64_decoder {
  __device__ __forceinline__ double operator()(float value) const {
    return static_cast<double>(value);
  }
};
struct int_decoder {
  __device__ __forceinline__ double operator()(std::int32_t value) const {
    return c32::decode_integer(value);
  }
};
struct quadratic_decoder {
  __device__ __forceinline__ double operator()(std::int32_t value) const {
    return c32::decode_quadratic(value);
  }
};
struct blended_quadratic_decoder {
  __device__ __forceinline__ double operator()(std::int32_t value) const {
    return c32::decode_blended_quadratic(value);
  }
};
struct blended_cubic_decoder {
  __device__ __forceinline__ double operator()(std::int32_t value) const {
    return c32::decode_blended_cubic(value);
  }
};
struct pwl2_decoder {
  __device__ __forceinline__ double operator()(std::uint32_t value) const {
    return c32::decode_pwl2(value);
  }
};
struct pwl4_decoder {
  __device__ __forceinline__ double operator()(std::uint32_t value) const {
    return c32::decode_pwl4(value);
  }
};
struct e11_decoder {
  __device__ __forceinline__ double operator()(e11_storage value) const {
    return bw::decode_raw<storage::e11m20, bw::compute_kind::fp64,
                          bw::decoder_kind::direct_masked>(
        static_cast<std::uint32_t>(value));
  }
};
struct e10_decoder {
  __device__ __forceinline__ double operator()(e10_storage value) const {
    return bw::decode_raw<storage::e10m21, bw::compute_kind::fp64,
                          bw::decoder_kind::direct_branchy>(
        static_cast<std::uint32_t>(value));
  }
};
struct posit_decoder {
  __device__ __forceinline__ double operator()(std::uint32_t value) const {
    return pt::decode_posit<32, 2, double>(value);
  }
};
struct takum_decoder {
  __device__ __forceinline__ double operator()(std::uint32_t value) const {
    return pt::decode_linear_takum<32, double>(value);
  }
};
struct lns_decoder {
  __device__ __forceinline__ double operator()(std::uint32_t value) const {
    const lk::decoder_context<lns_format, bw::compute_kind::fp64,
                              lk::decoder_kind::ex2_approx>
        context{};
    return lk::decode_raw<lns_format, bw::compute_kind::fp64,
                          lk::decoder_kind::ex2_approx>(value, context);
  }
};

extern "C" __global__ void dot_raw_fp64_kernel(const double *left,
                                                const double *right,
                                                std::size_t count,
                                                double *partials) {
  dot_double_body(left, right, count, partials, raw64_decoder{});
}
extern "C" __global__ void dot_fp32_to_fp64_kernel(const float *left,
                                                    const float *right,
                                                    std::size_t count,
                                                    double *partials) {
  dot_double_body(left, right, count, partials, fp32_to_fp64_decoder{});
}
extern "C" __global__ void dot_int32_kernel(const std::int32_t *left,
                                             const std::int32_t *right,
                                             std::size_t count,
                                             double *partials) {
  dot_double_body(left, right, count, partials, int_decoder{});
}
extern "C" __global__ void dot_quadratic32_kernel(const std::int32_t *left,
                                                   const std::int32_t *right,
                                                   std::size_t count,
                                                   double *partials) {
  dot_double_body(left, right, count, partials, quadratic_decoder{});
}
extern "C" __global__ void
dot_blended_quadratic32_kernel(const std::int32_t *left,
                               const std::int32_t *right, std::size_t count,
                               double *partials) {
  dot_double_body(left, right, count, partials, blended_quadratic_decoder{});
}
extern "C" __global__ void
dot_blended_cubic32_kernel(const std::int32_t *left,
                           const std::int32_t *right, std::size_t count,
                           double *partials) {
  dot_double_body(left, right, count, partials, blended_cubic_decoder{});
}
extern "C" __global__ void dot_pwl2_compand32_kernel(
    const std::uint32_t *left, const std::uint32_t *right, std::size_t count,
    double *partials) {
  dot_double_body(left, right, count, partials, pwl2_decoder{});
}
extern "C" __global__ void dot_pwl4_compand32_kernel(
    const std::uint32_t *left, const std::uint32_t *right, std::size_t count,
    double *partials) {
  dot_double_body(left, right, count, partials, pwl4_decoder{});
}
extern "C" __global__ void dot_e11m20_kernel(const e11_storage *left,
                                              const e11_storage *right,
                                              std::size_t count,
                                              double *partials) {
  dot_double_body(left, right, count, partials, e11_decoder{});
}
extern "C" __global__ void dot_e10m21_kernel(const e10_storage *left,
                                              const e10_storage *right,
                                              std::size_t count,
                                              double *partials) {
  dot_double_body(left, right, count, partials, e10_decoder{});
}
extern "C" __global__ void dot_posit32_kernel(const std::uint32_t *left,
                                               const std::uint32_t *right,
                                               std::size_t count,
                                               double *partials) {
  dot_double_body(left, right, count, partials, posit_decoder{});
}
extern "C" __global__ void dot_takum32_kernel(const std::uint32_t *left,
                                               const std::uint32_t *right,
                                               std::size_t count,
                                               double *partials) {
  dot_double_body(left, right, count, partials, takum_decoder{});
}
extern "C" __global__ void dot_lns32_r23_kernel(const std::uint32_t *left,
                                                 const std::uint32_t *right,
                                                 std::size_t count,
                                                 double *partials) {
  dot_double_body(left, right, count, partials, lns_decoder{});
}

__device__ __forceinline__ double
dyadic_magnitude(std::uint32_t code,
                 const dn::segment_coefficients *coefficients) {
  const auto rank = code & dn::magnitude_mask;
  const auto segment = dn::segment_from_rank(rank);
  const auto payload = dn::payload_for_segment(rank, segment);
  const auto coefficient = coefficients[segment];
  return fma(static_cast<double>(payload), coefficient.step,
             coefficient.start);
}

extern "C" __global__ void dot_dyadic_normal32_sign_fused_kernel(
    const std::uint32_t *left, const std::uint32_t *right, std::size_t count,
    const dn::segment_coefficients *global_coefficients, double *partials) {
  __shared__ dn::segment_coefficients coefficients[dn::segment_count];
  __shared__ double shared[threads];
  if (threadIdx.x < dn::segment_count) {
    coefficients[threadIdx.x] = global_coefficients[threadIdx.x];
  }
  __syncthreads();
  double sum{};
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto left_code = left[index];
    auto a = dyadic_magnitude(left_code, coefficients);
    const auto b = dyadic_magnitude(right[index], coefficients);
    a = c32::apply_sign(a, left_code ^ right[index]);
    sum = fma(a, b, sum);
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

extern "C" __global__ void dot_raw_fp32_kernel(const float *left,
                                                const float *right,
                                                std::size_t count,
                                                float *partials) {
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

extern "C" __global__ void finalize_dot_fp64_kernel(const double *partials,
                                                     std::size_t count,
                                                     double *result) {
  __shared__ double shared[threads];
  double sum{};
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    sum += partials[index];
  }
  const auto reduced = block_sum(sum, shared);
  if (threadIdx.x == 0) {
    *result = reduced;
  }
}

extern "C" __global__ void finalize_dot_fp32_kernel(const float *partials,
                                                     std::size_t count,
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

  double measure(const std::function<void()> &launch) {
    CUDA_CHECK(cudaEventRecord(start_));
    launch();
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    CUDA_CHECK(cudaGetLastError());
    float milliseconds{};
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

struct timing_sample {
  std::size_t n{};
  int sample{};
  int execution_order{};
  double kernel_ms{};
};

struct result_point {
  std::size_t n{};
  double value{};
  bool valid{};
};

struct variant {
  std::string id;
  std::string label;
  std::string group;
  std::string decoder;
  std::string arithmetic;
  int storage_bits{};
  double dot_scale{1.0};
  std::function<void(std::size_t)> launch;
  std::function<double()> read_result;
  std::vector<timing_sample> timings;
  std::vector<result_point> results;
};

template <auto Kernel, typename Stored>
std::function<void(std::size_t)> make_double_launch(
    const Stored *left, const Stored *right, double *partials, double *result) {
  return [=](std::size_t count) {
    Kernel<<<blocks, threads>>>(left, right, count, partials);
    finalize_dot_fp64_kernel<<<1, threads>>>(partials, blocks, result);
  };
}

template <typename T>
std::vector<T> copy_prefix(const T *device, std::size_t count) {
  std::vector<T> host(count);
  CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return host;
}

template <typename Stored, typename Decoder>
double host_dot(const std::vector<Stored> &left,
                const std::vector<Stored> &right, Decoder decoder,
                double dot_scale) {
  long double sum{};
  for (std::size_t index = 0; index < left.size(); ++index) {
    sum += static_cast<long double>(decoder(left[index])) *
           static_cast<long double>(decoder(right[index]));
  }
  return static_cast<double>(sum * dot_scale);
}

template <typename Stored, typename Decoder>
double maximum_encoding_error(const std::vector<double> &source,
                              const std::vector<Stored> &codes,
                              Decoder decoder, double value_scale = 1.0) {
  double maximum{};
  for (std::size_t index = 0; index < source.size(); ++index) {
    const auto decoded = decoder(codes[index]) * value_scale;
    if (!std::isfinite(decoded)) {
      return std::numeric_limits<double>::infinity();
    }
    maximum = std::max(maximum, std::abs(decoded - source[index]));
  }
  return maximum;
}

double relative_error(double actual, double expected) {
  return std::abs(actual - expected) /
         std::max({1.0, std::abs(actual), std::abs(expected)});
}

struct buffers {
  explicit buffers(std::size_t count)
      : raw64_left(count), raw64_right(count), raw32_left(count),
        raw32_right(count), integer_left(count), integer_right(count),
        quadratic_left(count), quadratic_right(count),
        blended_quadratic_left(count), blended_quadratic_right(count),
        blended_cubic_left(count), blended_cubic_right(count), pwl2_left(count),
        pwl2_right(count), pwl4_left(count), pwl4_right(count), e11_left(count),
        e11_right(count), e10_left(count), e10_right(count), dyadic_left(count),
        dyadic_right(count), posit_left(count), posit_right(count),
        takum_left(count), takum_right(count), lns_left(count), lns_right(count) {}

  device_buffer<double> raw64_left, raw64_right;
  device_buffer<float> raw32_left, raw32_right;
  device_buffer<std::int32_t> integer_left, integer_right;
  device_buffer<std::int32_t> quadratic_left, quadratic_right;
  device_buffer<std::int32_t> blended_quadratic_left,
      blended_quadratic_right;
  device_buffer<std::int32_t> blended_cubic_left, blended_cubic_right;
  device_buffer<std::uint32_t> pwl2_left, pwl2_right;
  device_buffer<std::uint32_t> pwl4_left, pwl4_right;
  device_buffer<e11_storage> e11_left, e11_right;
  device_buffer<e10_storage> e10_left, e10_right;
  device_buffer<std::uint32_t> dyadic_left, dyadic_right;
  device_buffer<std::uint32_t> posit_left, posit_right;
  device_buffer<std::uint32_t> takum_left, takum_right;
  device_buffer<std::uint32_t> lns_left, lns_right;
};

void generate_inputs(buffers &data, std::size_t count,
                     const dn::segment_coefficients *coefficients,
                     const double *boundaries) {
  constexpr int generation_blocks = 4096;
  generate_source_kernel<<<generation_blocks, threads>>>(
      data.raw64_left.get(), data.raw32_left.get(), count, left_seed);
  generate_source_kernel<<<generation_blocks, threads>>>(
      data.raw64_right.get(), data.raw32_right.get(), count, right_seed);
  encode_simple_kernel<<<generation_blocks, threads>>>(
      data.raw64_left.get(), data.integer_left.get(), data.quadratic_left.get(),
      data.blended_quadratic_left.get(), data.blended_cubic_left.get(),
      data.pwl2_left.get(), data.pwl4_left.get(), data.e11_left.get(),
      data.e10_left.get(), data.lns_left.get(), count);
  encode_simple_kernel<<<generation_blocks, threads>>>(
      data.raw64_right.get(), data.integer_right.get(),
      data.quadratic_right.get(), data.blended_quadratic_right.get(),
      data.blended_cubic_right.get(), data.pwl2_right.get(),
      data.pwl4_right.get(), data.e11_right.get(), data.e10_right.get(),
      data.lns_right.get(), count);
  encode_dyadic_kernel<<<generation_blocks, threads>>>(
      data.raw64_left.get(), data.dyadic_left.get(), coefficients, boundaries,
      count);
  encode_dyadic_kernel<<<generation_blocks, threads>>>(
      data.raw64_right.get(), data.dyadic_right.get(), coefficients, boundaries,
      count);
  encode_posit_takum_kernel<<<generation_blocks, threads>>>(
      data.raw64_left.get(), data.posit_left.get(), data.takum_left.get(), count);
  encode_posit_takum_kernel<<<generation_blocks, threads>>>(
      data.raw64_right.get(), data.posit_right.get(), data.takum_right.get(),
      count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<variant> make_variants(
    buffers &data, const dn::segment_coefficients *dyadic_coefficients,
    device_buffer<double> &partials, device_buffer<double> &result,
    device_buffer<float> &partials_fp32, device_buffer<float> &result_fp32) {
  const auto read_double = [&result] {
    double value{};
    CUDA_CHECK(cudaMemcpy(&value, result.get(), sizeof(value),
                          cudaMemcpyDeviceToHost));
    return value;
  };
  const auto read_float = [&result_fp32] {
    float value{};
    CUDA_CHECK(cudaMemcpy(&value, result_fp32.get(), sizeof(value),
                          cudaMemcpyDeviceToHost));
    return static_cast<double>(value);
  };
  const auto *raw32_left = data.raw32_left.get();
  const auto *raw32_right = data.raw32_right.get();
  const auto *dyadic_left = data.dyadic_left.get();
  const auto *dyadic_right = data.dyadic_right.get();

  std::vector<variant> variants;
  variants.push_back({"raw_fp32", "Raw FP32", "baseline", "native_fp32",
                      "fp32", 32, 1.0,
                      [=, &partials_fp32, &result_fp32](std::size_t count) {
                        dot_raw_fp32_kernel<<<blocks, threads>>>(
                            raw32_left, raw32_right, count, partials_fp32.get());
                        finalize_dot_fp32_kernel<<<1, threads>>>(
                            partials_fp32.get(), blocks, result_fp32.get());
                      },
                      read_float});
  variants.push_back({"fp32_to_fp64", "FP32 to FP64", "baseline",
                      "native_convert", "fp64", 32, 1.0,
                      make_double_launch<dot_fp32_to_fp64_kernel>(
                          data.raw32_left.get(), data.raw32_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"raw_fp64", "Raw FP64", "baseline", "native_fp64",
                      "fp64", 64, 1.0,
                      make_double_launch<dot_raw_fp64_kernel>(
                          data.raw64_left.get(), data.raw64_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"int32", "Int32", "new", "int_to_fp64", "fp64", 32,
                      c32::int_value_scale * c32::int_value_scale,
                      make_double_launch<dot_int32_kernel>(
                          data.integer_left.get(), data.integer_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"quadratic32", "Quadratic32", "new",
                      "int_to_fp64_abs_mul", "fp64", 32,
                      c32::quadratic_value_scale * c32::quadratic_value_scale,
                      make_double_launch<dot_quadratic32_kernel>(
                          data.quadratic_left.get(), data.quadratic_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"blended_quadratic32", "BlendedQuadratic32", "new",
                      "int_to_fp64_abs_fma_mul", "fp64", 32,
                      c32::blended_value_scale * c32::blended_value_scale,
                      make_double_launch<dot_blended_quadratic32_kernel>(
                          data.blended_quadratic_left.get(),
                          data.blended_quadratic_right.get(), partials.get(),
                          result.get()),
                      read_double});
  variants.push_back({"blended_cubic32", "BlendedCubic32", "new",
                      "int_to_fp64_square_fma_mul", "fp64", 32,
                      c32::blended_value_scale * c32::blended_value_scale,
                      make_double_launch<dot_blended_cubic32_kernel>(
                          data.blended_cubic_left.get(),
                          data.blended_cubic_right.get(), partials.get(),
                          result.get()),
                      read_double});
  variants.push_back({"pwl2_compand32", "PWL2Compand32", "new",
                      "branchless_2_zone_fma", "fp64", 32, 1.0,
                      make_double_launch<dot_pwl2_compand32_kernel>(
                          data.pwl2_left.get(), data.pwl2_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"pwl4_compand32", "PWL4Compand32", "new",
                      "branchless_4_zone_fma", "fp64", 32, 1.0,
                      make_double_launch<dot_pwl4_compand32_kernel>(
                          data.pwl4_left.get(), data.pwl4_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"e11m20", "E11M20", "reference", "direct_masked",
                      "fp64", 32, 1.0,
                      make_double_launch<dot_e11m20_kernel>(
                          data.e11_left.get(), data.e11_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"e10m21", "E10M21", "reference", "direct_branchy",
                      "fp64", 32, 1.0,
                      make_double_launch<dot_e10m21_kernel>(
                          data.e10_left.get(), data.e10_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"dyadic_normal32", "DyadicNormal32", "reference",
                      "sign_fused", "fp64", 32, 1.0,
                      [=, &partials, &result](std::size_t count) {
                        dot_dyadic_normal32_sign_fused_kernel<<<blocks,
                                                                threads>>>(
                            dyadic_left, dyadic_right, count,
                            dyadic_coefficients, partials.get());
                        finalize_dot_fp64_kernel<<<1, threads>>>(
                            partials.get(), blocks, result.get());
                      },
                      read_double});
  variants.push_back({"posit32_es2", "Posit<32,2>", "reference", "direct",
                      "fp64", 32, 1.0,
                      make_double_launch<dot_posit32_kernel>(
                          data.posit_left.get(), data.posit_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"takum32", "Takum32", "reference", "direct", "fp64",
                      32, 1.0,
                      make_double_launch<dot_takum32_kernel>(
                          data.takum_left.get(), data.takum_right.get(),
                          partials.get(), result.get()),
                      read_double});
  variants.push_back({"lns32_r23", "LNS<32,23>", "reference",
                      "ex2_approx", "fp64", 32, 1.0,
                      make_double_launch<dot_lns32_r23_kernel>(
                          data.lns_left.get(), data.lns_right.get(),
                          partials.get(), result.get()),
                      read_double});
  return variants;
}

void validate_variants(const std::string &path, buffers &data,
                       const std::array<dn::segment_coefficients,
                                        dn::segment_count> &coefficients,
                       std::vector<variant> &variants) {
  const auto raw64_left = copy_prefix(data.raw64_left.get(), validation_n);
  const auto raw64_right = copy_prefix(data.raw64_right.get(), validation_n);
  const auto raw32_left = copy_prefix(data.raw32_left.get(), validation_n);
  const auto raw32_right = copy_prefix(data.raw32_right.get(), validation_n);
  const auto integer_left = copy_prefix(data.integer_left.get(), validation_n);
  const auto integer_right = copy_prefix(data.integer_right.get(), validation_n);
  const auto quadratic_left = copy_prefix(data.quadratic_left.get(), validation_n);
  const auto quadratic_right =
      copy_prefix(data.quadratic_right.get(), validation_n);
  const auto bq_left =
      copy_prefix(data.blended_quadratic_left.get(), validation_n);
  const auto bq_right =
      copy_prefix(data.blended_quadratic_right.get(), validation_n);
  const auto bc_left = copy_prefix(data.blended_cubic_left.get(), validation_n);
  const auto bc_right = copy_prefix(data.blended_cubic_right.get(), validation_n);
  const auto pwl2_left = copy_prefix(data.pwl2_left.get(), validation_n);
  const auto pwl2_right = copy_prefix(data.pwl2_right.get(), validation_n);
  const auto pwl4_left = copy_prefix(data.pwl4_left.get(), validation_n);
  const auto pwl4_right = copy_prefix(data.pwl4_right.get(), validation_n);
  const auto e11_left = copy_prefix(data.e11_left.get(), validation_n);
  const auto e11_right = copy_prefix(data.e11_right.get(), validation_n);
  const auto e10_left = copy_prefix(data.e10_left.get(), validation_n);
  const auto e10_right = copy_prefix(data.e10_right.get(), validation_n);
  const auto dyadic_left = copy_prefix(data.dyadic_left.get(), validation_n);
  const auto dyadic_right = copy_prefix(data.dyadic_right.get(), validation_n);
  const auto posit_left = copy_prefix(data.posit_left.get(), validation_n);
  const auto posit_right = copy_prefix(data.posit_right.get(), validation_n);
  const auto takum_left = copy_prefix(data.takum_left.get(), validation_n);
  const auto takum_right = copy_prefix(data.takum_right.get(), validation_n);
  const auto lns_left = copy_prefix(data.lns_left.get(), validation_n);
  const auto lns_right = copy_prefix(data.lns_right.get(), validation_n);

  const auto find = [&](const std::string &id) -> variant & {
    return *std::find_if(variants.begin(), variants.end(),
                         [&](const auto &entry) { return entry.id == id; });
  };
  struct encoding_check {
    std::string id;
    double maximum_absolute_error{};
    double tolerance{};
  };
  std::vector<encoding_check> encoding_checks;
  encoding_checks.push_back(
      {"raw_fp32",
       maximum_encoding_error(
           raw64_left, raw32_left,
           [](float value) { return static_cast<double>(value); }),
       5.0e-7});
  encoding_checks.push_back(
      {"int32",
       maximum_encoding_error(
           raw64_left, integer_left,
           [](std::int32_t code) { return c32::decode_integer(code); },
           c32::int_value_scale),
       1.0e-8});
  encoding_checks.push_back(
      {"quadratic32",
       maximum_encoding_error(
           raw64_left, quadratic_left,
           [](std::int32_t code) { return c32::decode_quadratic(code); },
           c32::quadratic_value_scale),
       2.0e-8});
  encoding_checks.push_back(
      {"blended_quadratic32",
       maximum_encoding_error(
           raw64_left, bq_left,
           [](std::int32_t code) {
             return c32::decode_blended_quadratic(code);
           },
           c32::blended_value_scale),
       2.0e-8});
  encoding_checks.push_back(
      {"blended_cubic32",
       maximum_encoding_error(
           raw64_left, bc_left,
           [](std::int32_t code) {
             return c32::decode_blended_cubic(code);
           },
           c32::blended_value_scale),
       2.0e-8});
  encoding_checks.push_back(
      {"pwl2_compand32",
       maximum_encoding_error(
           raw64_left, pwl2_left,
           [](std::uint32_t code) { return c32::decode_pwl2(code); }),
       1.0e-8});
  encoding_checks.push_back(
      {"pwl4_compand32",
       maximum_encoding_error(
           raw64_left, pwl4_left,
           [](std::uint32_t code) { return c32::decode_pwl4(code); }),
       1.0e-8});
  encoding_checks.push_back(
      {"e11m20",
       maximum_encoding_error(
           raw64_left, e11_left,
           [](e11_storage code) { return storage::decode<storage::e11m20>(code); }),
       2.1e-6});
  encoding_checks.push_back(
      {"e10m21",
       maximum_encoding_error(
           raw64_left, e10_left,
           [](e10_storage code) { return storage::decode<storage::e10m21>(code); }),
       1.1e-6});
  encoding_checks.push_back(
      {"dyadic_normal32",
       maximum_encoding_error(
           raw64_left, dyadic_left,
           [&](std::uint32_t code) { return dn::decode(code, coefficients); }),
       5.0e-9});
  encoding_checks.push_back(
      {"posit32_es2",
       maximum_encoding_error(
           raw64_left, posit_left,
           [](std::uint32_t code) {
             return pt::decode_posit<32, 2, double>(code);
           }),
       2.0e-8});
  encoding_checks.push_back(
      {"takum32",
       maximum_encoding_error(
           raw64_left, takum_left,
           [](std::uint32_t code) {
             return pt::decode_linear_takum<32, double>(code);
           }),
       2.0e-8});
  encoding_checks.push_back(
      {"lns32_r23",
       maximum_encoding_error(
           raw64_left, lns_left,
           [](std::uint32_t code) {
             return lns::decode<lns_format, double>(code);
           }),
       4.0e-7});
  std::vector<std::pair<std::string, double>> expected;
  expected.emplace_back("raw_fp64",
                        host_dot(raw64_left, raw64_right,
                                 [](double value) { return value; }, 1.0));
  expected.emplace_back(
      "fp32_to_fp64",
      host_dot(raw32_left, raw32_right,
               [](float value) { return static_cast<double>(value); }, 1.0));
  long double fp32_sum{};
  for (std::size_t index = 0; index < validation_n; ++index) {
    fp32_sum += static_cast<long double>(raw32_left[index]) *
                static_cast<long double>(raw32_right[index]);
  }
  expected.emplace_back("raw_fp32", static_cast<double>(fp32_sum));
  expected.emplace_back(
      "int32", host_dot(integer_left, integer_right,
                        [](std::int32_t code) { return c32::decode_integer(code); },
                        c32::int_value_scale * c32::int_value_scale));
  expected.emplace_back(
      "quadratic32",
      host_dot(quadratic_left, quadratic_right,
               [](std::int32_t code) { return c32::decode_quadratic(code); },
               c32::quadratic_value_scale * c32::quadratic_value_scale));
  expected.emplace_back(
      "blended_quadratic32",
      host_dot(bq_left, bq_right,
               [](std::int32_t code) {
                 return c32::decode_blended_quadratic(code);
               },
               c32::blended_value_scale * c32::blended_value_scale));
  expected.emplace_back(
      "blended_cubic32",
      host_dot(bc_left, bc_right,
               [](std::int32_t code) {
                 return c32::decode_blended_cubic(code);
               },
               c32::blended_value_scale * c32::blended_value_scale));
  expected.emplace_back(
      "pwl2_compand32",
      host_dot(pwl2_left, pwl2_right,
               [](std::uint32_t code) { return c32::decode_pwl2(code); }, 1.0));
  expected.emplace_back(
      "pwl4_compand32",
      host_dot(pwl4_left, pwl4_right,
               [](std::uint32_t code) { return c32::decode_pwl4(code); }, 1.0));
  expected.emplace_back(
      "e11m20",
      host_dot(e11_left, e11_right,
               [](e11_storage code) {
                 return bw::decode_raw<storage::e11m20,
                                       bw::compute_kind::fp64,
                                       bw::decoder_kind::direct_masked>(
                     static_cast<std::uint32_t>(code));
               },
               1.0));
  expected.emplace_back(
      "e10m21",
      host_dot(e10_left, e10_right,
               [](e10_storage code) {
                 return bw::decode_raw<storage::e10m21,
                                       bw::compute_kind::fp64,
                                       bw::decoder_kind::direct_branchy>(
                     static_cast<std::uint32_t>(code));
               },
               1.0));
  expected.emplace_back(
      "dyadic_normal32",
      host_dot(dyadic_left, dyadic_right,
               [&](std::uint32_t code) { return dn::decode(code, coefficients); },
               1.0));
  expected.emplace_back(
      "posit32_es2",
      host_dot(posit_left, posit_right,
               [](std::uint32_t code) {
                 return pt::decode_posit<32, 2, double>(code);
               },
               1.0));
  expected.emplace_back(
      "takum32",
      host_dot(takum_left, takum_right,
               [](std::uint32_t code) {
                 return pt::decode_linear_takum<32, double>(code);
               },
               1.0));
  expected.emplace_back(
      "lns32_r23",
      host_dot(lns_left, lns_right,
               [](std::uint32_t code) {
                 if (lns::is_zero<lns_format>(code)) {
                   return 0.0;
                 }
                 const auto magnitude = std::exp2(
                     static_cast<float>(lns::log_code<lns_format>(code)) /
                     static_cast<float>(std::uint32_t{1} << 23));
                 return lns::sign<lns_format>(code) ? -magnitude : magnitude;
               },
               1.0));

  ensure_parent(path);
  std::ofstream output(path);
  if (!output) {
    throw benchmark_error("cannot open correctness output: " + path);
  }
  output << "validation_n=" << validation_n << '\n';
  bool all_passed = true;
  for (const auto &check : encoding_checks) {
    const auto passed = check.maximum_absolute_error <= check.tolerance;
    all_passed = all_passed && passed;
    output << check.id << "_encoding_max_abs_error=" << std::setprecision(17)
           << check.maximum_absolute_error << '\n'
           << check.id << "_encoding_tolerance=" << check.tolerance << '\n'
           << check.id << "_encoding_passed=" << (passed ? 1 : 0) << '\n';
  }
  for (const auto &[id, reference] : expected) {
    auto &entry = find(id);
    entry.launch(validation_n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto measured = entry.read_result() * entry.dot_scale;
    const auto error = relative_error(measured, reference);
    const auto tolerance = id == "raw_fp32"   ? 2.0e-5
                           : id == "lns32_r23" ? 5.0e-5
                                                : 2.0e-10;
    const auto passed = std::isfinite(measured) && error <= tolerance;
    all_passed = all_passed && passed;
    output << id << "_expected=" << std::setprecision(17) << reference << '\n'
           << id << "_actual=" << measured << '\n'
           << id << "_relative_error=" << error << '\n'
           << id << "_passed=" << (passed ? 1 : 0) << '\n';
  }
  output << "all_passed=" << (all_passed ? 1 : 0) << '\n';
  if (!all_passed) {
    throw benchmark_error("one or more correctness checks failed");
  }
}

void write_samples(const options &settings, const std::string &gpu,
                   const std::vector<variant> &variants) {
  ensure_parent(settings.output);
  std::ofstream output(settings.output);
  if (!output) {
    throw benchmark_error("cannot open timing output: " + settings.output);
  }
  output << "timestamp,gpu,mode,distribution,clip_low,clip_high,kernel,"
            "format,label,group,storage_bits,arithmetic_type,storage_layout,"
            "access_method,packet_values,decoder,strategy_id,N,"
            "physical_input_bytes,blocks,threads,warmups,sample,"
            "execution_order,kernel_ms,scaled_result,valid\n";
  const auto timestamp = utc_timestamp();
  for (const auto &entry : variants) {
    for (const auto &sample : entry.timings) {
      const auto result = std::find_if(
          entry.results.begin(), entry.results.end(),
          [&](const auto &point) { return point.n == sample.n; });
      output << timestamp << ',' << gpu << ',' << settings.mode
             << ",normal_clipped,-8,8,dot," << entry.id << ',' << entry.label
             << ',' << entry.group << ',' << entry.storage_bits << ','
             << entry.arithmetic << ",natural,scalar,1," << entry.decoder << ','
             << entry.id << ',' << sample.n << ','
             << (2ull * sample.n * static_cast<std::size_t>(entry.storage_bits) /
                 8ull)
             << ',' << blocks << ',' << threads << ',' << settings.warmups << ','
             << sample.sample << ',' << sample.execution_order << ','
             << std::setprecision(9) << sample.kernel_ms << ','
             << std::setprecision(17) << result->value << ','
             << (result->valid ? 1 : 0) << '\n';
    }
  }
}

void run(const options &settings) {
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  const std::string gpu = properties.name;
  const auto sizes = benchmark_sizes(settings);
  const auto maximum_n = sizes.back();

  const auto host_coefficients = dn::make_coefficients();
  std::array<double, dn::segment_count + 1> host_boundaries{};
  for (std::uint32_t index = 0; index <= dn::segment_count; ++index) {
    host_boundaries[index] = dn::half_normal_density_boundary(index);
  }
  device_buffer<dn::segment_coefficients> dyadic_coefficients(
      dn::segment_count);
  device_buffer<double> dyadic_boundaries(dn::segment_count + 1);
  CUDA_CHECK(cudaMemcpy(dyadic_coefficients.get(), host_coefficients.data(),
                        sizeof(host_coefficients), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dyadic_boundaries.get(), host_boundaries.data(),
                        sizeof(host_boundaries), cudaMemcpyHostToDevice));

  std::cout << "Allocating maximum-size inputs for N=" << maximum_n << '\n';
  buffers data(maximum_n);
  std::cout << "Generating one deterministic clipped N(0,1) input and all "
               "offline encodings\n";
  generate_inputs(data, maximum_n, dyadic_coefficients.get(),
                  dyadic_boundaries.get());

  device_buffer<double> partials(blocks);
  device_buffer<double> result(1);
  device_buffer<float> partials_fp32(blocks);
  device_buffer<float> result_fp32(1);
  auto variants = make_variants(data, dyadic_coefficients.get(), partials,
                                result, partials_fp32, result_fp32);
  if (variants.size() != 15) {
    throw benchmark_error("benchmark must contain exactly 15 variants");
  }
  validate_variants(settings.correctness_output, data, host_coefficients,
                    variants);
  std::cout << "Correctness validation passed for all 15 variants\n";

  event_timer timer;
  for (std::size_t size_index = 0; size_index < sizes.size(); ++size_index) {
    const auto n = sizes[size_index];
    std::cout << "Timing N=" << n << '\n';
    for (int warmup = 0; warmup < settings.warmups; ++warmup) {
      for (std::size_t position = 0; position < variants.size(); ++position) {
        const auto index =
            (position + warmup + 3 * size_index) % variants.size();
        variants[index].launch(n);
      }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int sample = 0; sample < settings.samples; ++sample) {
      for (std::size_t position = 0; position < variants.size(); ++position) {
        const auto index =
            (position + sample + 7 * size_index) % variants.size();
        auto &entry = variants[index];
        entry.timings.push_back(
            {n, sample, static_cast<int>(position),
             timer.measure([&] { entry.launch(n); })});
      }
    }

    for (auto &entry : variants) {
      entry.launch(n);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      const auto scaled_result = entry.read_result() * entry.dot_scale;
      entry.results.push_back(
          {n, scaled_result, std::isfinite(scaled_result)});
      if (!std::isfinite(scaled_result)) {
        throw benchmark_error(entry.id + " produced a non-finite DOT result");
      }
    }
  }

  write_samples(settings, gpu, variants);
  std::cout << "Compander32 benchmark complete on " << gpu << '\n'
            << "Raw timing samples: " << settings.output << '\n';
}

} // namespace

int main(int argc, char **argv) {
  try {
    run(parse_options(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << "compander32_bench: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
