#include "lns_benchmark_core.hpp"
#include "lut_distribution_core.hpp"
#include "posit_takum_core.hpp"
#include "t16_codebook.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
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

namespace lns = aut::lns;
namespace lut = aut::lut_distribution;
namespace pt = aut::pt;

constexpr int threads = 256;
constexpr int dot_blocks = 512;
constexpr int metric_blocks = 4096;
constexpr std::uint64_t left_seed = 0x243f6a8885a308d3ULL;
constexpr std::uint64_t right_seed = 0x13198a2e03707344ULL;

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

struct settings {
  std::string mode{"full"};
  std::size_t n{std::size_t{1} << 26};
  int warmup{10};
  int samples{50};
  std::string output{"timing_samples.csv"};
  std::string metrics_output{"access_metrics.csv"};
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

__global__ void generate_codes_kernel(std::uint16_t *codes, std::size_t count,
                                      std::uint64_t seed, int q_eighths) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    codes[index] = lut::sample_code(index, seed, q_eighths);
  }
}

__global__ void unique_sector_sum_kernel(const std::uint16_t *codes,
                                         std::size_t count,
                                         unsigned long long *total) {
  constexpr int warps_per_block = threads / lut::warp_width;
  const int lane = threadIdx.x & (lut::warp_width - 1);
  const int warp_in_block = threadIdx.x / lut::warp_width;
  const auto first_warp = static_cast<std::size_t>(blockIdx.x) * warps_per_block +
                          static_cast<std::size_t>(warp_in_block);
  const auto warp_stride =
      static_cast<std::size_t>(gridDim.x) * warps_per_block;
  const auto warp_count = count / lut::warp_width;
  unsigned long long local_sum{};
  for (auto warp = first_warp; warp < warp_count; warp += warp_stride) {
    const auto code = codes[warp * lut::warp_width + lane];
    const auto sector = static_cast<unsigned>(code) >> 3;
    const auto peers = __match_any_sync(0xffffffffu, sector);
    const bool leader = lane == (__ffs(static_cast<int>(peers)) - 1);
    const auto leaders = __ballot_sync(0xffffffffu, leader);
    if (lane == 0) {
      local_sum += static_cast<unsigned long long>(__popc(leaders));
    }
  }
  if (lane == 0) {
    atomicAdd(total, local_sum);
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

__global__ void dot_lut_fp32_kernel(const std::uint16_t *left,
                                    const std::uint16_t *right,
                                    std::size_t count, const float *table,
                                    float *partials) {
  __shared__ float shared[threads];
  float sum{};
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto a = __ldg(table + left[index]);
    const auto b = __ldg(table + right[index]);
    sum = fmaf(a, b, sum);
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

void launch_dot(const std::uint16_t *left, const std::uint16_t *right,
                std::size_t count, const float *table, float *partials,
                float *result) {
  dot_lut_fp32_kernel<<<dot_blocks, threads>>>(left, right, count, table,
                                               partials);
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

  double measure(const std::uint16_t *left, const std::uint16_t *right,
                 std::size_t count, const float *table, float *partials,
                 float *result) {
    CUDA_CHECK(cudaEventRecord(start_));
    launch_dot(left, right, count, table, partials, result);
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

struct host_table {
  std::string id;
  std::string label;
  std::vector<float> values;
  std::size_t sanitized{};
};

host_table make_t16_table() {
  auto codebook = aut::t16::build_normal_codebook();
  return {"t16", "T16", std::move(codebook.values), 0};
}

host_table make_posit_table() {
  host_table result{"posit16_es1", "Posit 16 es1",
                    std::vector<float>(lut::code_count), 0};
  for (std::size_t raw = 0; raw < result.values.size(); ++raw) {
    auto value = pt::decode_posit<16, 1, float>(static_cast<std::uint32_t>(raw));
    if (!std::isfinite(value)) {
      value = 0.0f;
      ++result.sanitized;
    }
    result.values[raw] = value;
  }
  return result;
}

host_table make_lns_table() {
  using format = lns::lns16_r11;
  host_table result{"lns16_r11", "LNS 16 r11",
                    std::vector<float>(lut::code_count), 0};
  for (std::size_t raw = 0; raw < result.values.size(); ++raw) {
    auto value = lns::decode<format, float>(static_cast<std::uint32_t>(raw));
    if (!std::isfinite(value)) {
      value = 0.0f;
      ++result.sanitized;
    }
    result.values[raw] = value;
  }
  return result;
}

double mean_unique_sectors(const std::uint16_t *codes, std::size_t count,
                           device_buffer<unsigned long long> &counter) {
  CUDA_CHECK(cudaMemset(counter.get(), 0, sizeof(unsigned long long)));
  unique_sector_sum_kernel<<<metric_blocks, threads>>>(codes, count,
                                                        counter.get());
  CUDA_CHECK(cudaGetLastError());
  unsigned long long host_total{};
  CUDA_CHECK(cudaMemcpy(&host_total, counter.get(), sizeof(host_total),
                        cudaMemcpyDeviceToHost));
  return static_cast<double>(host_total) /
         static_cast<double>(count / lut::warp_width);
}

struct device_table {
  host_table metadata;
  device_buffer<float> values;
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
  std::cout << "N: " << settings.n << ", warmups: " << settings.warmup
            << ", samples: " << settings.samples << '\n';

  std::array<host_table, 3> host_tables{
      make_t16_table(), make_posit_table(), make_lns_table()};
  std::vector<device_table> tables;
  tables.reserve(host_tables.size());
  for (auto &host : host_tables) {
    if (host.values.size() != lut::code_count) {
      throw benchmark_error("LUT has the wrong entry count: " + host.id);
    }
    device_table table{std::move(host), device_buffer<float>(lut::code_count)};
    CUDA_CHECK(cudaMemcpy(table.values.get(), table.metadata.values.data(),
                          lut::code_count * sizeof(float),
                          cudaMemcpyHostToDevice));
    std::cout << table.metadata.label << ": 65536 FP32 entries, sanitized "
              << table.metadata.sanitized << " nonfinite entries\n";
    tables.push_back(std::move(table));
  }

  device_buffer<std::uint16_t> left(settings.n);
  device_buffer<std::uint16_t> right(settings.n);
  device_buffer<float> partials(dot_blocks);
  device_buffer<float> result(1);
  device_buffer<unsigned long long> metric_counter(1);

  ensure_parent(settings.output);
  ensure_parent(settings.metrics_output);
  std::ofstream samples(settings.output);
  std::ofstream metrics(settings.metrics_output);
  if (!samples || !metrics) {
    throw benchmark_error("cannot open output CSV files");
  }
  samples << "timestamp,gpu,mode,kernel,N,storage_bits,arithmetic_type,"
             "access_method,packet_values,lut_entries,lut_bytes,q,"
             "q_eighths,mean_unique_left,mean_unique_right,mean_unique_both,"
             "normalized_sector_dispersion,format,format_label,"
             "sanitized_lut_entries,warmup,sample,execution_order,kernel_ms,"
             "result\n";
  metrics << "q,q_eighths,N,left_seed,right_seed,hot_sector_base,"
             "mean_unique_left,mean_unique_right,mean_unique_both,"
             "uniform_mean_unique,normalized_sector_dispersion\n";

  const std::vector<int> q_values = settings.mode == "smoke"
                                        ? std::vector<int>{0, 4, 8}
                                        : std::vector<int>{0, 1, 2, 3, 4,
                                                           5, 6, 7, 8};
  event_timer timer;
  volatile float result_sink{};
  for (std::size_t q_index = 0; q_index < q_values.size(); ++q_index) {
    const auto q_eighths = q_values[q_index];
    const auto q = static_cast<double>(q_eighths) / 8.0;
    const auto generation_blocks = static_cast<int>(std::min<std::size_t>(
        4096, (settings.n + threads - 1) / threads));
    generate_codes_kernel<<<generation_blocks, threads>>>(
        left.get(), settings.n, left_seed, q_eighths);
    generate_codes_kernel<<<generation_blocks, threads>>>(
        right.get(), settings.n, right_seed, q_eighths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto left_unique =
        mean_unique_sectors(left.get(), settings.n, metric_counter);
    const auto right_unique =
        mean_unique_sectors(right.get(), settings.n, metric_counter);
    const auto mean_unique = 0.5 * (left_unique + right_unique);
    const auto x = lut::normalized_dispersion(mean_unique);
    metrics << std::setprecision(17) << q << ',' << q_eighths << ','
            << settings.n << ',' << left_seed << ',' << right_seed << ','
            << lut::default_hot_sector_base << ',' << left_unique << ','
            << right_unique << ',' << mean_unique << ','
            << lut::uniform_unique_sectors() << ',' << x << '\n';
    std::cout << "q=" << q << ", X=" << x
              << ", mean unique sectors=" << mean_unique << '\n';

    for (int warmup = 0; warmup < settings.warmup; ++warmup) {
      const auto start = (q_index + static_cast<std::size_t>(warmup)) %
                         tables.size();
      for (std::size_t position = 0; position < tables.size(); ++position) {
        const auto &table = tables[(start + position) % tables.size()];
        launch_dot(left.get(), right.get(), settings.n, table.values.get(),
                   partials.get(), result.get());
      }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int sample = 0; sample < settings.samples; ++sample) {
      const auto start =
          (q_index + static_cast<std::size_t>(sample)) % tables.size();
      for (std::size_t position = 0; position < tables.size(); ++position) {
        const auto &table = tables[(start + position) % tables.size()];
        const auto milliseconds = timer.measure(
            left.get(), right.get(), settings.n, table.values.get(),
            partials.get(), result.get());
        float host_result{};
        CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(host_result),
                              cudaMemcpyDeviceToHost));
        if (!std::isfinite(host_result)) {
          throw benchmark_error("nonfinite DOT result for " +
                                table.metadata.id);
        }
        result_sink += host_result;
        samples << utc_timestamp() << ',' << gpu << ',' << settings.mode
                << ",dot," << settings.n
                << ",16,fp32,scalar,1,65536,262144," << std::setprecision(17)
                << q << ',' << q_eighths << ',' << left_unique << ','
                << right_unique << ',' << mean_unique << ',' << x << ','
                << table.metadata.id << ',' << table.metadata.label << ','
                << table.metadata.sanitized << ',' << settings.warmup << ','
                << sample << ',' << position << ',' << milliseconds << ','
                << host_result << '\n';
      }
    }
    samples.flush();
    metrics.flush();
  }
  std::cout << "Result sink: " << result_sink << '\n';
  std::cout << "Wrote " << settings.output << '\n';
  std::cout << "Wrote " << settings.metrics_output << '\n';
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
