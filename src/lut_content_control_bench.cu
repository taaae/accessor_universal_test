#include "bitwidth_benchmark_kernels.cuh"
#include "posit_takum_core.hpp"
#include "posit_takum_kernels.cuh"
#include "storage_formats.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef AUT_LUT_CONTROL_BITS
#error AUT_LUT_CONTROL_BITS is required
#endif
#ifndef AUT_LUT_CONTROL_COMPUTE
#error AUT_LUT_CONTROL_COMPUTE is required
#endif

namespace bw = aut::bitwidth;
namespace pt = aut::pt;
namespace storage = aut::storage;

namespace {

constexpr int bits = AUT_LUT_CONTROL_BITS;
constexpr auto compute = static_cast<bw::compute_kind>(AUT_LUT_CONTROL_COMPUTE);
using Float = bw::compute_t<compute>;
using ieee_format =
    std::conditional_t<bits == 8, storage::fp8_e4m3, storage::e5m8>;
constexpr int threads = 256;
constexpr int dot_blocks = 512;
constexpr std::size_t table_entries = std::size_t{1} << bits;

static_assert(bits == 8 || bits == 14);

class benchmark_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void check_cuda(cudaError_t status, const char *expression, const char *file,
                int line) {
  if (status != cudaSuccess) {
    throw benchmark_error(std::string(file) + ':' + std::to_string(line) +
                          " " + expression + ": " +
                          cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expression)                                                 \
  check_cuda((expression), #expression, __FILE__, __LINE__)

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
      reset(0);
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  ~device_buffer() { reset(0); }
  void reset(std::size_t count) {
    if (data_ != nullptr) cudaFree(data_);
    data_ = nullptr;
    count_ = count;
    if (count != 0) CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
  }
  T *get() const { return data_; }
  std::size_t size() const { return count_; }

private:
  T *data_{};
  std::size_t count_{};
};

struct settings {
  std::string mode{"full"};
  std::string output;
  std::string validation_output;
  std::string histogram_output;
  std::uint64_t seed{0x6bd87c012a53f9e1ULL};
  std::size_t dot_n{std::size_t{1} << 27};
  std::size_t gemv_m{1024};
  std::size_t gemv_n{65536};
  int warmup{10};
  int samples{30};
};

std::size_t parse_size(const std::string &text, const std::string &option) {
  std::size_t used{};
  const auto value = std::stoull(text, &used);
  if (used != text.size() || value == 0)
    throw benchmark_error(option + " requires a positive integer");
  return value;
}

settings parse_arguments(int argc, char **argv) {
  settings result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&] {
      if (++index >= argc) throw benchmark_error("missing value after " + argument);
      return std::string(argv[index]);
    };
    if (argument == "--mode") result.mode = value();
    else if (argument == "--output") result.output = value();
    else if (argument == "--validation-output") result.validation_output = value();
    else if (argument == "--histogram-output") result.histogram_output = value();
    else if (argument == "--seed") result.seed = parse_size(value(), argument);
    else if (argument == "--dot-n") result.dot_n = parse_size(value(), argument);
    else if (argument == "--gemv-m") result.gemv_m = parse_size(value(), argument);
    else if (argument == "--gemv-n") result.gemv_n = parse_size(value(), argument);
    else if (argument == "--warmup") result.warmup = parse_size(value(), argument);
    else if (argument == "--samples") result.samples = parse_size(value(), argument);
    else throw benchmark_error("unknown argument: " + argument);
  }
  if (result.mode == "smoke") {
    result.dot_n = std::min<std::size_t>(result.dot_n, 65536);
    result.gemv_m = std::min<std::size_t>(result.gemv_m, 16);
    result.gemv_n = std::min<std::size_t>(result.gemv_n, 256);
    result.warmup = 1;
    result.samples = std::min(result.samples, 2);
  } else if (result.mode != "full") {
    throw benchmark_error("--mode must be smoke or full");
  }
  if (result.output.empty() || result.validation_output.empty() ||
      result.histogram_output.empty())
    throw benchmark_error("all three output paths are required");
  return result;
}

const char *arithmetic_name() {
  return compute == bw::compute_kind::fp32 ? "fp32" : "fp64";
}

Float ieee_value(std::uint32_t raw) {
  if constexpr (bits == 8) {
    storage::storage_type_t<ieee_format> value;
    value.__x = static_cast<__nv_fp8_storage_t>(raw);
    return static_cast<Float>(storage::decode<ieee_format>(value));
  } else {
    return static_cast<Float>(
        bw::decode_reference<ieee_format, bw::compute_kind::fp64>(raw));
  }
}

enum class content_kind { ieee, posit, takum, takum_log };

struct content_description {
  const char *format;
  const char *family;
  content_kind kind;
};

constexpr std::array<content_description, 4> contents{{
    {bits == 8 ? "fp8_e4m3" : "e5m8", "ieee", content_kind::ieee},
    {bits == 8 ? "posit8_es0" : "posit14_es1", "posit", content_kind::posit},
    {bits == 8 ? "takum8" : "takum14", "takum", content_kind::takum},
    {bits == 8 ? "takum_log8" : "takum_log14", "takum_log",
     content_kind::takum_log},
}};

Float table_value(content_kind kind, std::uint32_t raw) {
  switch (kind) {
  case content_kind::ieee:
    return ieee_value(raw);
  case content_kind::posit:
    if constexpr (bits == 8)
      return pt::decode<pt::family::posit, 8, 0, Float>(raw);
    else
      return pt::decode<pt::family::posit, 14, 1, Float>(raw);
  case content_kind::takum:
    if constexpr (bits == 8)
      return pt::decode<pt::family::takum_linear, 8, 0, Float>(raw);
    else
      return pt::decode<pt::family::takum_linear, 14, 0, Float>(raw);
  case content_kind::takum_log:
    if constexpr (bits == 8)
      return pt::decode<pt::family::takum_log, 8, 0, Float>(raw);
    else
      return pt::decode<pt::family::takum_log, 14, 0, Float>(raw);
  }
  return Float{};
}

std::size_t storage_bytes(std::size_t count) {
  return (count * std::size_t(bits) + 7u) / 8u;
}

std::vector<std::uint8_t> pack(std::size_t count,
                               const std::vector<std::uint32_t> &codes) {
  const auto packed_bytes = storage_bytes(count);
  const auto allocation =
      bits == 14 ? ((packed_bytes + 3u) & ~std::size_t{3}) + 4u
                 : packed_bytes;
  std::vector<std::uint8_t> output(allocation, 0u);
  for (std::size_t index = 0; index < count; ++index) {
    const auto raw = codes[index % codes.size()] & pt::mask<bits>();
    if constexpr (bits == 8) {
      output[index] = static_cast<std::uint8_t>(raw);
    } else {
      const auto bit = index * std::size_t{14};
      const auto byte = bit >> 3;
      const auto shift = static_cast<unsigned>(bit & 7u);
      const auto placed = raw << shift;
      for (int part = 0; part < 3; ++part)
        output[byte + part] |=
            static_cast<std::uint8_t>(placed >> (part * 8));
    }
  }
  return output;
}

struct raw_trace {
  std::vector<std::uint32_t> left;
  std::vector<std::uint32_t> right;
};

raw_trace make_trace(std::size_t count, bool concentrated) {
  raw_trace result;
  result.left.resize(count);
  result.right.resize(count);
  const auto trace_mask = concentrated ? std::uint32_t{255} : pt::mask<bits>();
  for (std::size_t index = 0; index < count; ++index) {
    result.left[index] = static_cast<std::uint32_t>(index) & trace_mask;
    result.right[index] =
        static_cast<std::uint32_t>(index * 40503u + 17u) & trace_mask;
  }
  return result;
}

class event_timer {
public:
  event_timer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~event_timer() {
    cudaEventDestroy(stop_);
    cudaEventDestroy(start_);
  }
  template <typename Launch> double measure(Launch launch) {
    CUDA_CHECK(cudaEventRecord(start_));
    launch();
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float milliseconds{};
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

struct variant {
  const content_description *content{};
  const char *strategy{};
  std::size_t shared_bytes{};
  bool feasible{};
  std::function<void()> launch;
};

class runner {
public:
  runner(const settings &configuration, std::ofstream &samples,
         std::ofstream &histograms, int max_shared)
      : settings_(configuration), samples_(samples), histograms_(histograms),
        max_shared_(max_shared), partials_(dot_blocks),
        result_(configuration.gemv_m) {
    for (std::size_t content = 0; content < contents.size(); ++content) {
      std::vector<Float> host(table_entries);
      for (std::size_t raw = 0; raw < table_entries; ++raw)
        host[raw] = table_value(contents[content].kind, raw);
      tables_[content].reset(table_entries);
      CUDA_CHECK(cudaMemcpy(tables_[content].get(), host.data(),
                            host.size() * sizeof(Float),
                            cudaMemcpyHostToDevice));
    }
  }

  void run_trace(const std::string &distribution, const raw_trace &trace) {
    const auto [left_min, left_max] =
        std::minmax_element(trace.left.begin(), trace.left.end());
    const auto [right_min, right_max] =
        std::minmax_element(trace.right.begin(), trace.right.end());
    const auto realized_min = std::min(*left_min, *right_min);
    const auto realized_max = std::max(*left_max, *right_max);
    std::vector<bool> seen(table_entries, false);
    for (const auto raw : trace.left) seen[raw] = true;
    for (const auto raw : trace.right) seen[raw] = true;
    const auto distinct =
        static_cast<std::size_t>(std::count(seen.begin(), seen.end(), true));
    for (const auto &content : contents) {
      histograms_ << content.format << ',' << arithmetic_name() << ','
                  << distribution << ",raw_index_trace,all,"
                  << trace.left.size() << ',' << realized_min << ','
                  << realized_max << '\n';
      histograms_ << content.format << ',' << arithmetic_name() << ','
                  << distribution << ",raw_index_distinct,all," << distinct
                  << ',' << realized_min << ',' << realized_max << '\n';
    }
    run_dot(distribution, trace);
    run_gemv(distribution, trace);
  }

private:
  template <pt::strategy Strategy>
  variant make_variant(const content_description &content,
                       const Float *table, pt::storage_view<bits> left,
                       pt::storage_view<bits> right, std::size_t count,
                       bool gemv) {
    using decoder = pt::alternative_decoder<pt::family::posit, bits, 0, Float,
                                             Strategy>;
    constexpr bool shared = Strategy == pt::strategy::full_lut_shared;
    const auto shared_bytes =
        (shared ? table_entries * sizeof(Float) : 0u) + threads * sizeof(Float);
    variant result{&content,
                   shared ? "full_lut_shared" : "full_lut_global",
                   shared_bytes,
                   shared_bytes <= std::size_t(max_shared_),
                   {}};
    if (!result.feasible) return result;
    if constexpr (shared) {
      if (gemv) {
        CUDA_CHECK(cudaFuncSetAttribute(
            pt::gemv_kernel<decoder, bits, Float, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
      } else {
        CUDA_CHECK(cudaFuncSetAttribute(
            pt::dot_kernel<decoder, bits, Float, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
      }
    }
    if (gemv) {
      result.launch = [this, left, right, table, shared_bytes] {
        pt::gemv_kernel<decoder, bits, Float, shared>
            <<<settings_.gemv_m, threads, shared_bytes>>>(
                left, right, settings_.gemv_m, settings_.gemv_n, {table},
                result_.get());
      };
    } else {
      result.launch = [this, left, right, table, shared_bytes, count] {
        pt::dot_kernel<decoder, bits, Float, shared>
            <<<dot_blocks, threads, shared_bytes>>>(left, right, count, {table},
                                                    partials_.get());
        pt::finalize_dot_kernel<<<1, threads>>>(partials_.get(), dot_blocks,
                                                result_.get());
      };
    }
    return result;
  }

  std::vector<variant> make_variants(pt::storage_view<bits> left,
                                     pt::storage_view<bits> right,
                                     std::size_t count, bool gemv) {
    std::vector<variant> result;
    result.reserve(contents.size() * 2);
    for (std::size_t index = 0; index < contents.size(); ++index) {
      result.push_back(make_variant<pt::strategy::full_lut_global>(
          contents[index], tables_[index].get(), left, right, count, gemv));
      result.push_back(make_variant<pt::strategy::full_lut_shared>(
          contents[index], tables_[index].get(), left, right, count, gemv));
    }
    return result;
  }

  void time(const std::string &distribution, const char *kernel,
            std::size_t n, std::size_t m, std::size_t input_bytes,
            std::vector<variant> &variants) {
    for (auto &entry : variants) {
      if (!entry.feasible) {
        samples_ << entry.content->format << ',' << entry.content->family << ','
                 << bits << ',' << arithmetic_name() << ',' << distribution
                 << ',' << kernel << ',' << entry.strategy
                 << ",dense,scalar,1," << n << ',' << m << ',' << input_bytes << ','
                 << table_entries * sizeof(Float) << ',' << entry.shared_bytes
                 << ",-1,-1,-1,infeasible_shared_memory\n";
        continue;
      }
      for (int warmup = 0; warmup < settings_.warmup; ++warmup) entry.launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int round = 0; round < settings_.samples; ++round) {
      for (std::size_t position = 0; position < variants.size(); ++position) {
        auto &entry = variants[(round + position) % variants.size()];
        if (!entry.feasible) continue;
        const auto milliseconds = timer_.measure(entry.launch);
        samples_ << entry.content->format << ',' << entry.content->family << ','
                 << bits << ',' << arithmetic_name() << ',' << distribution
                 << ',' << kernel << ',' << entry.strategy
                 << ",dense,scalar,1," << n << ',' << m << ',' << input_bytes << ','
                 << table_entries * sizeof(Float) << ',' << entry.shared_bytes
                 << ',' << round << ',' << position << ','
                 << std::setprecision(9) << milliseconds << ",ok\n";
      }
    }
  }

  void run_dot(const std::string &distribution, const raw_trace &trace) {
    const auto left_host = pack(settings_.dot_n, trace.left);
    const auto right_host = pack(settings_.dot_n, trace.right);
    device_buffer<std::uint8_t> left(left_host.size()), right(right_host.size());
    CUDA_CHECK(cudaMemcpy(left.get(), left_host.data(), left_host.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(right.get(), right_host.data(), right_host.size(),
                          cudaMemcpyHostToDevice));
    auto variants = make_variants({left.get()}, {right.get()}, settings_.dot_n,
                                  false);
    time(distribution, "dot", settings_.dot_n, 1,
         left.size() + right.size(), variants);
  }

  void run_gemv(const std::string &distribution, const raw_trace &trace) {
    const auto matrix_host =
        pack(settings_.gemv_m * settings_.gemv_n, trace.left);
    const auto vector_host = pack(settings_.gemv_n, trace.right);
    device_buffer<std::uint8_t> matrix(matrix_host.size()),
        vector(vector_host.size());
    CUDA_CHECK(cudaMemcpy(matrix.get(), matrix_host.data(), matrix_host.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vector.get(), vector_host.data(), vector_host.size(),
                          cudaMemcpyHostToDevice));
    auto variants = make_variants({matrix.get()}, {vector.get()},
                                  settings_.gemv_n, true);
    time(distribution, "gemv", settings_.gemv_n, settings_.gemv_m,
         matrix.size() + vector.size(), variants);
  }

  const settings &settings_;
  std::ofstream &samples_;
  std::ofstream &histograms_;
  int max_shared_{};
  std::array<device_buffer<Float>, contents.size()> tables_;
  device_buffer<Float> partials_;
  device_buffer<Float> result_;
  event_timer timer_;
};

} // namespace

int main(int argc, char **argv) try {
  const auto configuration = parse_arguments(argc, argv);
  int max_shared{};
  CUDA_CHECK(cudaDeviceGetAttribute(&max_shared,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  std::ofstream samples(configuration.output);
  std::ofstream validation(configuration.validation_output);
  std::ofstream histograms(configuration.histogram_output);
  if (!samples || !validation || !histograms)
    throw benchmark_error("failed to open an output file");
  samples << "format,family,bits,arithmetic,distribution,kernel,strategy,"
             "storage_layout,access_method,packet_values,N,M,input_bytes,lut_bytes,"
             "dynamic_shared_bytes,round,order_position,kernel_ms,status\n";
  validation <<
      "format,arithmetic,strategy,codes_checked,failures,max_ulp,reference,status\n";
  histograms << "format,arithmetic,distribution,histogram,bucket,count,"
                "realized_q_min,realized_q_max\n";

  runner benchmark(configuration, samples, histograms, max_shared);
  const auto trace_size =
      configuration.mode == "smoke" ? std::size_t{4096} : std::size_t{65536};
  benchmark.run_trace("lut_scattered_control", make_trace(trace_size, false));
  benchmark.run_trace("lut_concentrated_control", make_trace(trace_size, true));
  std::cout << "completed " << bits << "-bit " << arithmetic_name()
            << " LUT-content control\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
