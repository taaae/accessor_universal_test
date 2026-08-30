#include "lut_decomposition_core.hpp"

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
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace lut = aut::lut_decomposition;

constexpr int threads = 256;
constexpr int dot_blocks = 512;
constexpr int metric_blocks = 4096;
constexpr std::uint64_t left_seed = 0x6a09e667f3bcc909ULL;
constexpr std::uint64_t right_seed = 0xbb67ae8584caa73bULL;
constexpr std::uint64_t table_seed = 0x3c6ef372fe94f82bULL;
constexpr std::uint64_t raw_left_seed = 0xa54ff53a5f1d36f1ULL;
constexpr std::uint64_t raw_right_seed = 0x510e527fade682d1ULL;

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

void ensure_parent(const std::string &filename) {
  const std::filesystem::path path(filename);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
}

template <typename Storage, int FieldBits, int Components>
__global__ void generate_packed_codes_kernel(Storage *codes,
                                             std::size_t count,
                                             std::uint64_t seed, double q) {
  static_assert(FieldBits * Components == int(sizeof(Storage) * 8));
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::uint64_t packed{};
#pragma unroll
    for (int component = 0; component < Components; ++component) {
      const auto component_seed =
          seed ^ (0x9e3779b97f4a7c15ULL * (component + 1));
      const auto field =
          lut::sample_index(index, component_seed, FieldBits, q);
      packed |= static_cast<std::uint64_t>(field)
                << static_cast<unsigned>(component * FieldBits);
    }
    codes[index] = static_cast<Storage>(packed);
  }
}

template <typename T>
__global__ void generate_raw_values_kernel(T *values, std::size_t count,
                                           std::uint64_t seed) {
  constexpr double scale = 1.0 / 9007199254740992.0;
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto bits = aut::lut_distribution::splitmix64(
                          seed ^ static_cast<std::uint64_t>(index)) >>
                      11;
    values[index] = static_cast<T>(static_cast<double>(bits) * scale - 0.5);
  }
}

template <typename T>
__device__ __forceinline__ T fused_accumulate(T left, T right, T sum) {
  if constexpr (std::is_same_v<T, float>) {
    return fmaf(left, right, sum);
  } else {
    return fma(left, right, sum);
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

template <typename Storage, typename T, int FieldBits, int Components>
__global__ void dot_global_tables_kernel(const Storage *left,
                                         const Storage *right,
                                         std::size_t count, const T *tables,
                                         T *partials) {
  constexpr std::uint32_t mask =
      FieldBits == 32 ? std::numeric_limits<std::uint32_t>::max()
                      : ((std::uint32_t{1} << FieldBits) - 1u);
  constexpr std::size_t entries = std::size_t{1} << FieldBits;
  __shared__ T reduction[threads];
  T sum{};
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto left_code = static_cast<std::uint32_t>(left[index]);
    const auto right_code = static_cast<std::uint32_t>(right[index]);
    T a{};
    T b{};
#pragma unroll
    for (int component = 0; component < Components; ++component) {
      const auto shift = component * FieldBits;
      const auto left_field = (left_code >> shift) & mask;
      const auto right_field = (right_code >> shift) & mask;
      a += __ldg(tables + component * entries + left_field);
      b += __ldg(tables + component * entries + right_field);
    }
    sum = fused_accumulate(a, b, sum);
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename Storage, typename T, int FieldBits, int Components>
__global__ void dot_shared_tables_kernel(const Storage *left,
                                         const Storage *right,
                                         std::size_t count,
                                         const T *global_tables,
                                         T *partials) {
  constexpr std::uint32_t mask =
      FieldBits == 32 ? std::numeric_limits<std::uint32_t>::max()
                      : ((std::uint32_t{1} << FieldBits) - 1u);
  constexpr std::size_t entries = std::size_t{1} << FieldBits;
  constexpr std::size_t total_entries = entries * Components;
  extern __shared__ unsigned char storage[];
  auto *tables = reinterpret_cast<T *>(storage);
  auto *reduction = tables + total_entries;
  for (std::size_t index = threadIdx.x; index < total_entries;
       index += blockDim.x) {
    tables[index] = global_tables[index];
  }
  __syncthreads();

  T sum{};
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto left_code = static_cast<std::uint32_t>(left[index]);
    const auto right_code = static_cast<std::uint32_t>(right[index]);
    T a{};
    T b{};
#pragma unroll
    for (int component = 0; component < Components; ++component) {
      const auto shift = component * FieldBits;
      const auto left_field = (left_code >> shift) & mask;
      const auto right_field = (right_code >> shift) & mask;
      a += tables[component * entries + left_field];
      b += tables[component * entries + right_field];
    }
    sum = fused_accumulate(a, b, sum);
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename T>
__global__ void dot_raw_kernel(const T *left, const T *right,
                               std::size_t count, T *partials) {
  __shared__ T reduction[threads];
  T sum{};
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    sum = fused_accumulate(left[index], right[index], sum);
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = reduced;
  }
}

template <typename T>
__global__ void finalize_dot_kernel(const T *partials, std::size_t count,
                                    T *result) {
  __shared__ T reduction[threads];
  T sum{};
  for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
    sum += partials[index];
  }
  const auto reduced = block_sum(sum, reduction);
  if (threadIdx.x == 0) {
    *result = reduced;
  }
}

template <typename Storage, int FieldBits, int Components, int EntryBytes,
          bool Shared>
__global__ void access_metrics_kernel(const Storage *codes,
                                      std::size_t count,
                                      unsigned long long *unique_totals,
                                      unsigned long long *physical_totals) {
  static_assert(EntryBytes == 4 || EntryBytes == 8);
  constexpr std::uint32_t mask =
      (std::uint32_t{1} << FieldBits) - 1u;
  constexpr int warps_per_block = threads / lut::warp_width;
  const int lane = threadIdx.x & (lut::warp_width - 1);
  const int warp_in_block = threadIdx.x / lut::warp_width;
  const auto first_warp = static_cast<std::size_t>(blockIdx.x) *
                              warps_per_block +
                          static_cast<std::size_t>(warp_in_block);
  const auto stride = static_cast<std::size_t>(gridDim.x) * warps_per_block;
  const auto warp_count = count / lut::warp_width;
  unsigned long long local_unique[Components]{};
  unsigned long long local_physical[Components]{};
  for (auto warp = first_warp; warp < warp_count; warp += stride) {
    const auto code = static_cast<std::uint32_t>(
        codes[warp * lut::warp_width + lane]);
#pragma unroll
    for (int component = 0; component < Components; ++component) {
      const auto field = (code >> (component * FieldBits)) & mask;
      const auto peers = __match_any_sync(0xffffffffu, field);
      const bool leader = lane == (__ffs(static_cast<int>(peers)) - 1);
      const auto leaders = __ballot_sync(0xffffffffu, leader);
      if (lane == 0) {
        local_unique[component] += __popc(leaders);
      }
      if constexpr (Shared) {
        int maximum_conflict{};
#pragma unroll
        for (int word = 0; word < EntryBytes / 4; ++word) {
          const auto bank = (field * (EntryBytes / 4) + word) & 31u;
#pragma unroll
          for (int candidate = 0; candidate < 32; ++candidate) {
            const auto bank_leaders = __ballot_sync(
                0xffffffffu, leader && bank == static_cast<unsigned>(candidate));
            maximum_conflict = max(maximum_conflict, __popc(bank_leaders));
          }
        }
        if (lane == 0) {
          local_physical[component] += maximum_conflict;
        }
      } else {
        const auto sector = (field * EntryBytes) >> 5;
        const auto sector_peers = __match_any_sync(0xffffffffu, sector);
        const bool sector_leader =
            lane == (__ffs(static_cast<int>(sector_peers)) - 1);
        const auto sector_leaders =
            __ballot_sync(0xffffffffu, sector_leader);
        if (lane == 0) {
          local_physical[component] += __popc(sector_leaders);
        }
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int component = 0; component < Components; ++component) {
      atomicAdd(unique_totals + component, local_unique[component]);
      atomicAdd(physical_totals + component, local_physical[component]);
    }
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

template <typename T>
std::vector<T> make_tables(int field_bits, int components,
                           std::uint64_t salt) {
  const auto entries = std::size_t{1} << field_bits;
  std::vector<T> values(entries * static_cast<std::size_t>(components));
  constexpr double inverse_24 = 1.0 / 16777216.0;
  for (int component = 0; component < components; ++component) {
    for (std::size_t index = 0; index < entries; ++index) {
      const auto bits = static_cast<std::uint32_t>(
          aut::lut_distribution::splitmix64(
              table_seed ^ salt ^
              (static_cast<std::uint64_t>(component) << 48) ^ index) >>
          40);
      const auto base = 0.125 + static_cast<double>(bits) * inverse_24 * 0.125;
      values[static_cast<std::size_t>(component) * entries + index] =
          static_cast<T>(base + 0.0078125 * component);
    }
  }
  return values;
}

struct metric_values {
  std::vector<double> unique;
  std::vector<double> physical;
};

template <typename Storage, int FieldBits, int Components, int EntryBytes,
          bool Shared>
metric_values measure_metrics(const Storage *codes, std::size_t count) {
  device_buffer<unsigned long long> unique(Components);
  device_buffer<unsigned long long> physical(Components);
  CUDA_CHECK(cudaMemset(unique.get(), 0,
                        Components * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(physical.get(), 0,
                        Components * sizeof(unsigned long long)));
  access_metrics_kernel<Storage, FieldBits, Components, EntryBytes, Shared>
      <<<metric_blocks, threads>>>(codes, count, unique.get(), physical.get());
  CUDA_CHECK(cudaGetLastError());
  std::vector<unsigned long long> host_unique(Components);
  std::vector<unsigned long long> host_physical(Components);
  CUDA_CHECK(cudaMemcpy(host_unique.data(), unique.get(),
                        Components * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_physical.data(), physical.get(),
                        Components * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost));
  const auto warps = static_cast<double>(count / lut::warp_width);
  metric_values result{std::vector<double>(Components),
                       std::vector<double>(Components)};
  for (int component = 0; component < Components; ++component) {
    result.unique[component] =
        static_cast<double>(host_unique[component]) / warps;
    result.physical[component] =
        static_cast<double>(host_physical[component]) / warps;
  }
  return result;
}

struct csv_context {
  std::ofstream &samples;
  std::ofstream &metrics;
  std::string gpu;
  settings configuration;
};

template <typename Storage, typename T, int FieldBits, int Components,
          bool Shared>
void run_variant(csv_context &context, const std::string &graph_id,
                 const std::string &variant, const std::string &label,
                 std::uint64_t salt) {
  constexpr auto entries = std::size_t{1} << FieldBits;
  constexpr auto entry_bytes = int(sizeof(T));
  constexpr auto table_entries = entries * Components;
  constexpr auto table_bytes = table_entries * sizeof(T);
  constexpr auto shared_bytes =
      Shared ? table_bytes + threads * sizeof(T) : 0;
  const auto &settings = context.configuration;
  device_buffer<Storage> left(settings.n);
  device_buffer<Storage> right(settings.n);
  device_buffer<T> partials(dot_blocks);
  device_buffer<T> result(1);
  const auto host_tables = make_tables<T>(FieldBits, Components, salt);
  device_buffer<T> tables(table_entries);
  CUDA_CHECK(cudaMemcpy(tables.get(), host_tables.data(), table_bytes,
                        cudaMemcpyHostToDevice));
  const auto generation_blocks = static_cast<int>(std::min<std::size_t>(
      4096, (settings.n + threads - 1) / threads));
  const std::vector<double> targets = settings.mode == "smoke"
                                          ? std::vector<double>{0.0, 0.5, 1.0}
                                          : std::vector<double>{0.0, 0.125, 0.25,
                                                                0.375, 0.5, 0.625,
                                                                0.75, 0.875, 1.0};
  event_timer timer;
  volatile double result_sink{};
  for (const auto target_x : targets) {
    const auto q = lut::q_for_normalized_dispersion(entries, target_x);
    generate_packed_codes_kernel<Storage, FieldBits, Components>
        <<<generation_blocks, threads>>>(left.get(), settings.n, left_seed, q);
    generate_packed_codes_kernel<Storage, FieldBits, Components>
        <<<generation_blocks, threads>>>(right.get(), settings.n, right_seed, q);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto left_metrics =
        measure_metrics<Storage, FieldBits, Components, entry_bytes, Shared>(
            left.get(), settings.n);
    const auto right_metrics =
        measure_metrics<Storage, FieldBits, Components, entry_bytes, Shared>(
            right.get(), settings.n);
    double mean_unique{};
    double mean_physical{};
    for (int component = 0; component < Components; ++component) {
      const auto unique =
          0.5 * (left_metrics.unique[component] +
                 right_metrics.unique[component]);
      const auto physical =
          0.5 * (left_metrics.physical[component] +
                 right_metrics.physical[component]);
      const auto actual_x =
          lut::normalized_lookup_dispersion(entries, unique);
      context.metrics << graph_id << ',' << variant << ',' << std::setprecision(17)
                      << target_x << ',' << q << ',' << component << ','
                      << FieldBits << ',' << (Shared ? "shared" : "global")
                      << ',' << entry_bytes << ',' << settings.n << ','
                      << left_metrics.unique[component] << ','
                      << right_metrics.unique[component] << ',' << unique << ','
                      << actual_x << ',';
      if constexpr (Shared) {
        context.metrics << "nan,nan,nan,"
                        << left_metrics.physical[component] << ','
                        << right_metrics.physical[component] << ',' << physical;
      } else {
        context.metrics << left_metrics.physical[component] << ','
                        << right_metrics.physical[component] << ',' << physical
                        << ",nan,nan,nan";
      }
      context.metrics << '\n';
      mean_unique += unique;
      mean_physical += physical;
    }
    mean_unique /= Components;
    mean_physical /= Components;
    const auto actual_x =
        lut::normalized_lookup_dispersion(entries, mean_unique);
    const auto launch = [&] {
      if constexpr (Shared) {
        dot_shared_tables_kernel<Storage, T, FieldBits, Components>
            <<<dot_blocks, threads, shared_bytes>>>(
                left.get(), right.get(), settings.n, tables.get(),
                partials.get());
      } else {
        dot_global_tables_kernel<Storage, T, FieldBits, Components>
            <<<dot_blocks, threads>>>(left.get(), right.get(), settings.n,
                                     tables.get(), partials.get());
      }
      finalize_dot_kernel<T><<<1, threads>>>(partials.get(), dot_blocks,
                                             result.get());
    };
    for (int warmup = 0; warmup < settings.warmup; ++warmup) {
      launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int sample = 0; sample < settings.samples; ++sample) {
      const auto milliseconds = timer.measure(launch);
      T host_result{};
      CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(T),
                            cudaMemcpyDeviceToHost));
      if (!std::isfinite(static_cast<double>(host_result))) {
        throw benchmark_error("nonfinite result for " + variant);
      }
      result_sink += static_cast<double>(host_result);
      context.samples
          << utc_timestamp() << ',' << context.gpu << ',' << settings.mode
          << ',' << graph_id << ',' << variant << ',' << label << ",dot,"
          << settings.n << ',' << dot_blocks << ',' << threads << ','
          << sizeof(Storage) * 8 << ','
          << (std::is_same_v<T, float> ? "fp32" : "fp64") << ','
          << (Shared ? "shared" : "global") << ',' << FieldBits << ','
          << Components << ',' << entries << ',' << table_entries << ','
          << table_bytes << ',' << shared_bytes << ',' << std::setprecision(17)
          << target_x << ',' << q << ',' << mean_unique << ',' << actual_x
          << ',';
      if constexpr (Shared) {
        context.samples << "nan," << mean_physical;
      } else {
        context.samples << mean_physical << ",nan";
      }
      context.samples << ',' << settings.warmup << ',' << sample << ",0,"
                      << milliseconds << ',' << host_result << '\n';
    }
    context.samples.flush();
    context.metrics.flush();
    std::cout << graph_id << '/' << variant << " target X=" << target_x
              << " actual X=" << actual_x << " q=" << q << '\n';
  }
  std::cout << graph_id << '/' << variant << " sink=" << result_sink << '\n';
}

void run_raw_fp64(csv_context &context) {
  const auto &settings = context.configuration;
  device_buffer<double> left(settings.n);
  device_buffer<double> right(settings.n);
  device_buffer<double> partials(dot_blocks);
  device_buffer<double> result(1);
  const auto generation_blocks = static_cast<int>(std::min<std::size_t>(
      4096, (settings.n + threads - 1) / threads));
  generate_raw_values_kernel<double><<<generation_blocks, threads>>>(
      left.get(), settings.n, raw_left_seed);
  generate_raw_values_kernel<double><<<generation_blocks, threads>>>(
      right.get(), settings.n, raw_right_seed);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto launch = [&] {
    dot_raw_kernel<double><<<dot_blocks, threads>>>(
        left.get(), right.get(), settings.n, partials.get());
    finalize_dot_kernel<double><<<1, threads>>>(partials.get(), dot_blocks,
                                                result.get());
  };
  for (int warmup = 0; warmup < settings.warmup; ++warmup) {
    launch();
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  event_timer timer;
  for (int sample = 0; sample < settings.samples; ++sample) {
    const auto milliseconds = timer.measure(launch);
    double host_result{};
    CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(host_result),
                          cudaMemcpyDeviceToHost));
    if (!std::isfinite(host_result)) {
      throw benchmark_error("nonfinite raw FP64 result");
    }
    context.samples
        << utc_timestamp() << ',' << context.gpu << ',' << settings.mode
        << ",raw_fp64,raw_fp64,Raw FP64,dot," << settings.n << ','
        << dot_blocks << ',' << threads
        << ",64,fp64,none,0,0,0,0,0,0,nan,nan,nan,nan,nan,nan,"
        << settings.warmup << ',' << sample << ",0," << std::setprecision(17)
        << milliseconds << ',' << host_result << '\n';
  }
  context.samples.flush();
  std::cout << "raw_fp64 complete\n";
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

  ensure_parent(settings.output);
  ensure_parent(settings.metrics_output);
  std::ofstream samples(settings.output);
  std::ofstream metrics(settings.metrics_output);
  if (!samples || !metrics) {
    throw benchmark_error("cannot open output CSV files");
  }
  samples << "timestamp,gpu,mode,graph_id,variant,variant_label,kernel,N,"
             "blocks,threads,storage_bits,arithmetic_type,table_location,"
             "component_bits,components,lut_entries_per_component,"
             "total_lut_entries,total_lut_bytes,dynamic_shared_bytes,"
             "target_x,q,mean_unique_indices,actual_x,mean_unique_sectors,"
             "mean_shared_wavefronts,warmup,sample,execution_order,kernel_ms,"
             "result\n";
  metrics << "graph_id,variant,target_x,q,component,component_bits,"
             "table_location,entry_bytes,N,left_unique_indices,"
             "right_unique_indices,mean_unique_indices,actual_x,"
             "left_unique_sectors,right_unique_sectors,mean_unique_sectors,"
             "left_shared_wavefronts,right_shared_wavefronts,"
             "mean_shared_wavefronts\n";
  csv_context context{samples, metrics, gpu, settings};

  run_raw_fp64(context);
  run_variant<std::uint16_t, double, 16, 1, false>(
      context, "g1_u16_global_fp64", "u16_global_fp64",
      "16-bit full global LUT to FP64", 0x101);
  run_variant<std::uint8_t, float, 8, 1, true>(
      context, "g2_u8_shared_fp32", "u8_shared_fp32",
      "8-bit shared LUT to FP32", 0x202);
  run_variant<std::uint8_t, double, 8, 1, true>(
      context, "g3_u8_shared_fp64", "u8_shared_fp64",
      "8-bit shared LUT to FP64", 0x303);
  run_variant<std::uint16_t, float, 8, 2, true>(
      context, "g4_u16_split_fp32", "u16_2x8_shared_fp32",
      "16-bit as 2 x 8-bit shared LUTs", 0x408);
  run_variant<std::uint16_t, float, 4, 4, true>(
      context, "g4_u16_split_fp32", "u16_4x4_shared_fp32",
      "16-bit as 4 x 4-bit shared LUTs", 0x404);
  run_variant<std::uint32_t, double, 16, 2, false>(
      context, "g5_u32_split_fp64", "u32_2x16_global_fp64",
      "32-bit as 2 x 16-bit global LUTs", 0x516);
  run_variant<std::uint32_t, double, 8, 4, true>(
      context, "g5_u32_split_fp64", "u32_4x8_shared_fp64",
      "32-bit as 4 x 8-bit shared LUTs", 0x508);
  run_variant<std::uint32_t, double, 4, 8, true>(
      context, "g5_u32_split_fp64", "u32_8x4_shared_fp64",
      "32-bit as 8 x 4-bit shared LUTs", 0x504);
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
