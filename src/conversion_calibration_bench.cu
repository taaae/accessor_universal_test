#include "conversion_calibration_cases.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace calibration = aut::calibration;
namespace normal32 = aut::normal32;

__constant__ double aut::calibration::constant_lut[4096];

namespace {

constexpr std::uint64_t left_key = 0x6bd87c012a53f9e1ULL;
constexpr std::uint64_t right_key = left_key ^ 0x9e3779b97f4a7c15ULL;
constexpr std::size_t full_n = std::size_t{1} << 27;
constexpr int fixed_blocks = 512;
constexpr int fixed_threads = 256;
constexpr std::uint32_t shuffle_seed = 0x5c41b3a7u;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const auto status_ = (call);                                                \
    if (status_ != cudaSuccess)                                                 \
      throw std::runtime_error(std::string(#call) + ": " +                    \
                               cudaGetErrorString(status_));                    \
  } while (false)

struct options {
  std::string mode{"full"};
  std::string output{"timing_samples.csv"};
  std::string resources{"kernel_resources.csv"};
  std::size_t n{full_n};
  int warmups{10};
  int rounds{3};
  int samples{10};
  float interval_ms{20.0f};
  std::string profile_cases{};
};

options parse_options(int argc, char **argv) {
  options result;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (++i == argc) throw std::runtime_error("missing value for " + arg);
      return argv[i];
    };
    if (arg == "--mode") result.mode = value();
    else if (arg == "--output") result.output = value();
    else if (arg == "--resources") result.resources = value();
    else if (arg == "--n") result.n = std::stoull(value());
    else if (arg == "--warmups") result.warmups = std::stoi(value());
    else if (arg == "--rounds") result.rounds = std::stoi(value());
    else if (arg == "--samples") result.samples = std::stoi(value());
    else if (arg == "--interval-ms") result.interval_ms = std::stof(value());
    else if (arg == "--profile-cases") result.profile_cases = value();
    else throw std::runtime_error("unknown argument: " + arg);
  }
  if (result.mode == "smoke") {
    if (result.n == full_n) result.n = std::size_t{1} << 20;
    result.warmups = std::min(result.warmups, 1);
    result.rounds = std::min(result.rounds, 1);
    result.samples = std::min(result.samples, 1);
    result.interval_ms = 0.0f;
  } else if (result.mode != "full") {
    throw std::runtime_error("mode must be smoke or full");
  }
  if (result.mode == "full" && result.n != full_n)
    throw std::runtime_error("full mode requires N=2^27");
  if (result.n == 0 || result.warmups < 0 || result.rounds < 1 ||
      result.samples < 1 || result.interval_ms < 0.0f)
    throw std::runtime_error("invalid benchmark options");
  return result;
}

template <typename T> class device_buffer {
public:
  explicit device_buffer(std::size_t count) : count_(count) {
    CUDA_CHECK(cudaMalloc(&pointer_, count * sizeof(T)));
  }
  ~device_buffer() { cudaFree(pointer_); }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
  T *get() { return pointer_; }
  const T *get() const { return pointer_; }
private:
  T *pointer_{};
  std::size_t count_{};
};

__device__ __forceinline__ void philox_round(std::uint32_t &c0,
                                             std::uint32_t &c1,
                                             std::uint32_t &c2,
                                             std::uint32_t &c3,
                                             std::uint32_t k0,
                                             std::uint32_t k1) {
  const auto hi0 = __umulhi(0xd2511f53u, c0);
  const auto lo0 = 0xd2511f53u * c0;
  const auto hi1 = __umulhi(0xcd9e8d57u, c2);
  const auto lo1 = 0xcd9e8d57u * c2;
  const auto n0 = hi1 ^ c1 ^ k0;
  const auto n1 = lo1;
  const auto n2 = hi0 ^ c3 ^ k1;
  const auto n3 = lo0;
  c0 = n0; c1 = n1; c2 = n2; c3 = n3;
}

__global__ void generate_philox(std::uint32_t *output, std::size_t count,
                                std::uint64_t key) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::uint32_t c0 = static_cast<std::uint32_t>(index);
    std::uint32_t c1 = static_cast<std::uint32_t>(index >> 32);
    std::uint32_t c2 = 0u, c3 = 0u;
    std::uint32_t k0 = static_cast<std::uint32_t>(key);
    std::uint32_t k1 = static_cast<std::uint32_t>(key >> 32);
#pragma unroll
    for (int round = 0; round < 10; ++round) {
      philox_round(c0, c1, c2, c3, k0, k1);
      k0 += 0x9e3779b9u;
      k1 += 0xbb67ae85u;
    }
    output[index] = c0;
  }
}

__global__ void initialize_lut(double *table, std::size_t count,
                               std::uint32_t salt) {
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto mixed = static_cast<std::uint32_t>(index) * 1664525u + salt;
    table[index] = 1.0 + static_cast<double>(mixed & 0xffffu) * 0x1p-24;
  }
}

__global__ void finalize_kernel(const double *partials, double *result) {
  __shared__ double shared[fixed_threads];
  double sum = 0.0;
  for (int index = threadIdx.x; index < fixed_blocks; index += blockDim.x)
    sum += partials[index];
  shared[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = fixed_threads / 2; offset; offset >>= 1) {
    if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) result[0] = shared[0];
}

struct event_timer {
  cudaEvent_t start{}, stop{};
  event_timer() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
  ~event_timer() { cudaEventDestroy(stop); cudaEventDestroy(start); }
  template <typename F> float measure(F operation) {
    CUDA_CHECK(cudaEventRecord(start)); operation(); CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); float value{};
    CUDA_CHECK(cudaEventElapsedTime(&value, start, stop)); return value;
  }
};

void write_resources(const std::string &path) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open resources output");
  output << "case_id,split,group,registers,static_shared_bytes,local_bytes,max_threads\n";
  for (std::size_t i = 0; i < std::size(calibration::case_info); ++i) {
    cudaFuncAttributes attr{};
    CUDA_CHECK(calibration::case_attributes(static_cast<int>(i), &attr));
    const auto &info = calibration::case_info[i];
    output << info.id << ',' << info.split << ',' << info.group << ','
           << attr.numRegs << ',' << attr.sharedSizeBytes << ','
           << attr.localSizeBytes << ',' << attr.maxThreadsPerBlock << '\n';
  }
}

void run(const options &settings) {
  int device{}; CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{}; CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  if (settings.mode == "full" && (properties.major != 9 || properties.minor != 0))
    throw std::runtime_error("full experiment requires SM90");
  if (settings.mode == "full" && std::string(properties.name).find("H200") == std::string::npos)
    throw std::runtime_error("full experiment requires an NVIDIA H200");

  device_buffer<std::uint32_t> left(settings.n), right(settings.n);
  device_buffer<double> partials(fixed_blocks), result(1);
  generate_philox<<<4096, fixed_threads>>>(left.get(), settings.n, left_key);
  generate_philox<<<4096, fixed_threads>>>(right.get(), settings.n, right_key);
  CUDA_CHECK(cudaGetLastError());

  constexpr std::size_t max_lut_entries = std::size_t{1} << 24;
  device_buffer<double> lut0(max_lut_entries), lut1(max_lut_entries), lut2(max_lut_entries);
  initialize_lut<<<4096, fixed_threads>>>(lut0.get(), max_lut_entries, 0x12345678u);
  initialize_lut<<<4096, fixed_threads>>>(lut1.get(), max_lut_entries, 0x87654321u);
  initialize_lut<<<4096, fixed_threads>>>(lut2.get(), max_lut_entries, 0x31415926u);

  std::vector<double> constant(4096);
  for (std::size_t i = 0; i < constant.size(); ++i)
    constant[i] = 1.0 + static_cast<double>(i) * 0x1p-20;
  CUDA_CHECK(cudaMemcpyToSymbol(calibration::constant_lut, constant.data(),
                                constant.size() * sizeof(double)));

  using e9layout = aut::bitwidth::format_layout_t<aut::storage::e9m22>;
  std::vector<std::uint32_t> prefix(1u << 10);
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    const auto raw = static_cast<std::uint32_t>(i << e9layout::fraction_bits);
    prefix[i] = aut::decoder::decode_words_branchy<e9layout>(raw).high;
  }
  device_buffer<std::uint32_t> prefix_device(prefix.size());
  CUDA_CHECK(cudaMemcpy(prefix_device.get(), prefix.data(),
                        prefix.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice));

  const auto normal_tables = normal32::build_codebook_tables();
  device_buffer<normal32::pwl_coeff> pwl(normal_tables.pwl.size());
  device_buffer<normal32::pwq_coeff> pwq(normal_tables.pwq.size());
  CUDA_CHECK(cudaMemcpy(pwl.get(), normal_tables.pwl.data(),
                        normal_tables.pwl.size() * sizeof(normal32::pwl_coeff),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(pwq.get(), normal_tables.pwq.data(),
                        normal_tables.pwq.size() * sizeof(normal32::pwq_coeff),
                        cudaMemcpyHostToDevice));
  calibration::device_tables tables{lut0.get(), lut1.get(), lut2.get(),
                                    prefix_device.get(), pwl.get(), pwq.get()};
  CUDA_CHECK(cudaDeviceSynchronize());

  write_resources(settings.resources);
  if (!settings.profile_cases.empty()) {
    std::size_t begin{};
    while (begin < settings.profile_cases.size()) {
      const auto end = settings.profile_cases.find(',', begin);
      const auto token = settings.profile_cases.substr(
          begin, end == std::string::npos ? end : end - begin);
      const auto case_index = std::stoi(token);
      if (case_index < 0 || case_index >= static_cast<int>(std::size(calibration::case_info)))
        throw std::runtime_error("profile case index out of range");
      calibration::launch_case(case_index, fixed_blocks, fixed_threads, 0,
                               left.get(), right.get(), settings.n, tables,
                               partials.get());
      finalize_kernel<<<1, fixed_threads>>>(partials.get(), result.get());
      begin = end == std::string::npos ? settings.profile_cases.size() : end + 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double consumed{}; CUDA_CHECK(cudaMemcpy(&consumed, result.get(), sizeof(consumed), cudaMemcpyDeviceToHost));
    std::cout << "profile cases complete; consumed=" << consumed << '\n';
    return;
  }
  std::ofstream output(settings.output);
  if (!output) throw std::runtime_error("cannot open timing output");
  output << "case_id,split,group,round,position,sample,stage,repeats,n,elapsed_ms,per_dot_ms,result_bits,status\n";
  event_timer timer;
  auto launch = [&](int case_index) {
    calibration::launch_case(case_index, fixed_blocks, fixed_threads, 0,
                             left.get(), right.get(), settings.n, tables,
                             partials.get());
    finalize_kernel<<<1, fixed_threads>>>(partials.get(), result.get());
  };
  auto timed = [&](int case_index, int repeats) {
    return timer.measure([&] { for (int i = 0; i < repeats; ++i) launch(case_index); });
  };

  const int case_count = static_cast<int>(std::size(calibration::case_info));
  std::vector<int> repeats(case_count, 1);
  for (int case_index = 0; case_index < case_count; ++case_index) {
    for (int warm = 0; warm < settings.warmups; ++warm) launch(case_index);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    const auto one = timed(case_index, 1);
    if (settings.interval_ms > 0.0f && one > 0.0f)
      repeats[case_index] = std::clamp(static_cast<int>(std::ceil(settings.interval_ms / one)), 1, 100000);
  }

  auto record = [&](int case_index, int round, int position, int sample,
                    const char *stage) {
    const auto elapsed = timed(case_index, repeats[case_index]);
    double host_result{};
    CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(host_result), cudaMemcpyDeviceToHost));
    std::uint64_t result_bits{}; std::memcpy(&result_bits, &host_result, sizeof(result_bits));
    const auto &info = calibration::case_info[case_index];
    output << info.id << ',' << info.split << ',' << info.group << ','
           << round << ',' << position << ',' << sample << ',' << stage << ','
           << repeats[case_index] << ',' << settings.n << ','
           << std::setprecision(9) << elapsed << ',' << elapsed / repeats[case_index]
           << ",0x" << std::hex << result_bits << std::dec << ",ok\n";
    output.flush();
  };

  constexpr int anchor_index = 1;
  for (int round = 0; round < settings.rounds; ++round) {
    record(anchor_index, round, -1, 0, "round_start_anchor");
    std::vector<int> order(case_count); std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(shuffle_seed + static_cast<std::uint32_t>(round));
    std::shuffle(order.begin(), order.end(), rng);
    for (int position = 0; position < case_count; ++position)
      for (int sample = 0; sample < settings.samples; ++sample)
        record(order[position], round, position, sample, "measurement");
    record(anchor_index, round, case_count, 0, "round_end_anchor");
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto settings = parse_options(argc, argv);
    run(settings);
    std::cout << "conversion calibration " << settings.mode << " complete\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "conversion calibration failed: " << error.what() << '\n';
    return 1;
  }
}
