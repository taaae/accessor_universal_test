#include "bitwidth_benchmark_kernels.cuh"

#include <curand.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef AUT_BITWIDTH_TOTAL_BITS
#error "AUT_BITWIDTH_TOTAL_BITS must select a registered width cohort"
#endif

namespace {

namespace bw = aut::bitwidth;
namespace storage = aut::storage;

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

std::vector<int> parse_powers(const std::string &text) {
  std::vector<int> powers;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      powers.push_back(std::stoi(token));
    }
  }
  if (powers.empty()) {
    throw std::runtime_error("power list cannot be empty");
  }
  return powers;
}

std::vector<std::string> parse_strings(const std::string &text) {
  std::vector<std::string> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      values.push_back(token);
    }
  }
  return values;
}

struct options {
  std::string mode{"screen"};
  std::vector<int> dot_powers{24};
  std::vector<int> gemv_powers{14};
  std::vector<std::string> distributions{"uniform_0_1", "normal_0_1"};
  std::size_t gemv_rows{1024};
  int warmup{5};
  int samples{3};
  double target_sample_ms{10.0};
  std::uint64_t seed{0x243f6a8885a308d3ull};
  std::string output{"timing_samples.csv"};
  std::string variant_file;
  std::set<std::string> enabled_variants;
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
    } else if (argument == "--dot-powers") {
      result.dot_powers = parse_powers(require_value());
    } else if (argument == "--gemv-powers") {
      result.gemv_powers = parse_powers(require_value());
    } else if (argument == "--distributions") {
      result.distributions = parse_strings(require_value());
    } else if (argument == "--gemv-rows") {
      result.gemv_rows = std::stoull(require_value());
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
    } else if (argument == "--variant-file") {
      result.variant_file = require_value();
    } else if (argument == "--variants") {
      for (const auto &value : parse_strings(require_value())) {
        result.enabled_variants.insert(value);
      }
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (!result.variant_file.empty()) {
    std::ifstream input(result.variant_file);
    if (!input) {
      throw std::runtime_error("cannot open variant file: " +
                               result.variant_file);
    }
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty() && line.front() != '#') {
        result.enabled_variants.insert(line);
      }
    }
  }
  if (result.mode == "smoke") {
    result.dot_powers = {12};
    result.gemv_powers = {8};
    result.gemv_rows = 32;
    result.warmup = 1;
    result.samples = 1;
    result.target_sample_ms = 0.0;
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

template <typename T> struct device_buffer {
  T *data{};
  std::size_t count{};

  device_buffer() = default;
  explicit device_buffer(std::size_t size) : count(size) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data),
                          std::max<std::size_t>(size, 1) * sizeof(T)));
  }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
  device_buffer(device_buffer &&other) noexcept
      : data(std::exchange(other.data, nullptr)), count(other.count) {}
  ~device_buffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
};

struct event_timer {
  cudaEvent_t start{};
  cudaEvent_t stop{};
  event_timer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~event_timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  template <typename Function> double measure(int iterations, Function launch) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
      launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds{};
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return static_cast<double>(milliseconds);
  }
};

struct source_buffers {
  device_buffer<double> left;
  device_buffer<double> right;
  curandGenerator_t generator{};

  source_buffers(std::size_t left_count, std::size_t right_count)
      : left(left_count + (left_count & 1u)),
        right(right_count + (right_count & 1u)) {
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  }
  ~source_buffers() { curandDestroyGenerator(generator); }

  void generate(const std::string &distribution, std::uint64_t seed) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));
    if (distribution == "uniform_0_1") {
      CURAND_CHECK(curandGenerateUniformDouble(generator, left.data, left.count));
      CURAND_CHECK(curandGenerateUniformDouble(generator, right.data, right.count));
    } else if (distribution == "normal_0_1") {
      CURAND_CHECK(curandGenerateNormalDouble(generator, left.data, left.count,
                                              0.0, 1.0));
      CURAND_CHECK(curandGenerateNormalDouble(generator, right.data, right.count,
                                              0.0, 1.0));
    } else {
      throw std::runtime_error("unsupported distribution: " + distribution);
    }
  }
};

template <typename Format> struct encoded_buffers {
  device_buffer<bw::padded_storage_t<Format>> left_padded;
  device_buffer<bw::padded_storage_t<Format>> right_padded;
  device_buffer<std::uint32_t> left_dense;
  device_buffer<std::uint32_t> right_dense;

  encoded_buffers(std::size_t left_count, std::size_t right_count)
      : left_padded(left_count + 8), right_padded(right_count + 8),
        left_dense(bw::dense_word_count<Format::total_bits>(left_count) + 8),
        right_dense(bw::dense_word_count<Format::total_bits>(right_count) + 8) {}

  void encode(const source_buffers &source, std::size_t left_count,
              std::size_t right_count) {
    constexpr int threads = 256;
    const auto left_blocks = static_cast<int>(
        std::min<std::size_t>((left_count + threads - 1) / threads, 4096));
    const auto right_blocks = static_cast<int>(
        std::min<std::size_t>((right_count + threads - 1) / threads, 4096));
    bw::encode_padded_kernel<Format><<<left_blocks, threads>>>(
        source.left.data, left_padded.data, left_count);
    bw::encode_padded_kernel<Format><<<right_blocks, threads>>>(
        source.right.data, right_padded.data, right_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(left_dense.data, 0,
                          left_dense.count * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(right_dense.data, 0,
                          right_dense.count * sizeof(std::uint32_t)));
    using geometry = bw::dense_geometry<Format::total_bits>;
    const auto left_chunks =
        (left_count + geometry::values_per_aligned_chunk - 1) /
        geometry::values_per_aligned_chunk;
    const auto right_chunks =
        (right_count + geometry::values_per_aligned_chunk - 1) /
        geometry::values_per_aligned_chunk;
    bw::pack_dense_kernel<Format><<<
        static_cast<int>(std::min<std::size_t>((left_chunks + threads - 1) /
                                                   threads,
                                               4096)),
        threads>>>(left_padded.data, left_dense.data, left_count);
    bw::pack_dense_kernel<Format><<<
        static_cast<int>(std::min<std::size_t>((right_chunks + threads - 1) /
                                                   threads,
                                               4096)),
        threads>>>(right_padded.data, right_dense.data, right_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  template <bw::storage_layout Layout>
  bw::storage_view<Format, Layout> left_view() const {
    return {left_dense.data, left_padded.data};
  }
  template <bw::storage_layout Layout>
  bw::storage_view<Format, Layout> right_view() const {
    return {right_dense.data, right_padded.data};
  }
};

template <typename Format, bw::compute_kind Compute> struct table_storage {
  device_buffer<std::uint32_t> full;
  device_buffer<std::uint32_t> prefix;
  device_buffer<std::uint32_t> subnormal;

  static std::uint32_t target_high(std::uint32_t raw) {
    if constexpr (Compute == bw::compute_kind::fp32) {
      return bw::float_bits(
          bw::decode_reference<Format, bw::compute_kind::fp32>(raw));
    } else {
      const auto value =
          bw::decode_reference<Format, bw::compute_kind::fp64>(raw);
      std::uint64_t bits{};
      std::memcpy(&bits, &value, sizeof(bits));
      return static_cast<std::uint32_t>(bits >> 32);
    }
  }

  table_storage()
      : full(std::size_t{1} << Format::total_bits),
        prefix(std::size_t{1} << (Format::exponent_bits + 1)),
        subnormal(std::size_t{1} << Format::fraction_bits) {
    std::vector<std::uint32_t> host_full(full.count);
    for (std::size_t raw = 0; raw < host_full.size(); ++raw) {
      host_full[raw] = target_high(static_cast<std::uint32_t>(raw));
    }
    CUDA_CHECK(cudaMemcpy(full.data, host_full.data(),
                          host_full.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));

    std::vector<std::uint32_t> host_prefix(prefix.count);
    for (std::size_t value = 0; value < host_prefix.size(); ++value) {
      host_prefix[value] = target_high(static_cast<std::uint32_t>(
          value << Format::fraction_bits));
    }
    CUDA_CHECK(cudaMemcpy(prefix.data, host_prefix.data(),
                          host_prefix.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));

    std::vector<std::uint32_t> host_subnormal(subnormal.count);
    for (std::size_t value = 0; value < host_subnormal.size(); ++value) {
      host_subnormal[value] =
          target_high(static_cast<std::uint32_t>(value));
    }
    CUDA_CHECK(cudaMemcpy(subnormal.data, host_subnormal.data(),
                          host_subnormal.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice));
  }

  bw::decoder_tables view() const {
    return {full.data, prefix.data, subnormal.data};
  }
};

const char *decoder_name(bw::decoder_kind decoder) {
  switch (decoder) {
  case bw::decoder_kind::generic_reference:
    return "generic";
  case bw::decoder_kind::direct_branchy:
    return "direct_branchy";
  case bw::decoder_kind::direct_masked:
    return "direct_masked";
  case bw::decoder_kind::fixed_integer:
    return "fixed_integer";
  case bw::decoder_kind::e1_integer:
    return "e1_integer";
  case bw::decoder_kind::exponent_only:
    return "exponent_only";
  case bw::decoder_kind::full_lut_global:
    return "full_lut_global";
  case bw::decoder_kind::full_lut_shared:
    return "full_lut_shared";
  case bw::decoder_kind::prefix_lut_global:
    return "prefix_lut_global";
  case bw::decoder_kind::prefix_lut_shared:
    return "prefix_lut_shared";
  case bw::decoder_kind::subnormal_lut_global:
    return "subnormal_lut_global";
  case bw::decoder_kind::subnormal_lut_shared:
    return "subnormal_lut_shared";
  case bw::decoder_kind::native_scalar:
    return "native_scalar";
  case bw::decoder_kind::native_packed:
    return "native_packed";
  }
  return "unknown";
}

template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
          bw::decoder_kind Decoder>
std::string variant_id() {
  const auto layout = Layout == bw::storage_layout::dense ? "dense" : "padded";
  const auto access = Access == bw::access_method::scalar
                          ? "scalar"
                          : Access == bw::access_method::thread_packet
                                ? "thread_packet"
                                : "cooperative_shuffle";
  const auto packet = Access == bw::access_method::cooperative_shuffle
                          ? bw::cooperative_geometry<AUT_BITWIDTH_TOTAL_BITS>::values
                          : Lanes;
  return std::string(layout) + "/" + access + "/x" +
         std::to_string(packet) + "/" + decoder_name(Decoder);
}

struct csv_output {
  std::ofstream stream;
  std::string timestamp{utc_timestamp()};
  std::string gpu;

  csv_output(const std::string &path, std::string gpu_name) : gpu(std::move(gpu_name)) {
    std::ifstream existing(path, std::ios::binary | std::ios::ate);
    const auto needs_header = !existing || existing.tellg() == 0;
    stream.open(path, std::ios::app);
    if (!stream) {
      throw std::runtime_error("cannot open output: " + path);
    }
    if (needs_header) {
      stream << "timestamp,gpu,mode,distribution,kernel,format,bits,exponent_bits,"
                "mantissa_bits,arithmetic_type,storage_layout,access_method,"
                "packet_values,load_word_bits,loader_threads,consumer_threads,"
                "values_per_consumer,redistribution,decoder,strategy_id,N,M,rows,"
                "logical_input_bytes,physical_input_bytes,table_bytes,blocks,"
                "threads,warmup,sample,iterations,total_ms,mean_ms,result,valid\n";
    }
  }
};

template <typename Format, bw::compute_kind Compute> class format_runner {
public:
  using value_type = bw::compute_t<Compute>;

  format_runner(const options &settings, const std::string &distribution,
                encoded_buffers<Format> &encoded, table_storage<Format, Compute> &tables,
                csv_output &output)
      : settings_(settings), distribution_(distribution), encoded_(encoded),
        tables_(tables), output_(output), partials_(512), result_(settings.gemv_rows),
        timer_() {}

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            bw::decoder_kind Decoder>
  void run_variant() {
    const auto id = variant_id<Layout, Access, Lanes, Decoder>();
    if (!settings_.enabled_variants.empty() &&
        settings_.enabled_variants.count(id) == 0) {
      return;
    }
    for (const auto power : settings_.dot_powers) {
      run_dot<Layout, Access, Lanes, Decoder>(std::size_t{1} << power, id);
    }
    for (const auto power : settings_.gemv_powers) {
      run_gemv<Layout, Access, Lanes, Decoder>(std::size_t{1} << power, id);
    }
  }

private:
  template <bw::decoder_kind Decoder> static constexpr std::size_t table_bytes() {
    return bw::table_entries<Format, Decoder>() * sizeof(std::uint32_t);
  }

  template <bw::decoder_kind Decoder>
  static constexpr std::size_t shared_bytes() {
    return (bw::uses_shared_table<Decoder>() ? table_bytes<Decoder>() : 0) +
           256 * sizeof(value_type);
  }

  template <bw::storage_layout Layout>
  static std::size_t physical_bytes(std::size_t values) {
    if constexpr (Layout == bw::storage_layout::dense) {
      return bw::dense_data_bytes<Format::total_bits>(values);
    } else {
      return values * sizeof(bw::padded_storage_t<Format>);
    }
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            bw::decoder_kind Decoder>
  void launch_dot(std::size_t count, int blocks) {
    if constexpr (Access == bw::access_method::cooperative_shuffle) {
      static_assert(Layout == bw::storage_layout::dense);
      bw::dot_cooperative_kernel<Format, Compute, Decoder>
          <<<blocks, 256, shared_bytes<Decoder>()>>>(
              encoded_.left_dense.data, encoded_.right_dense.data, count,
              tables_.view(), partials_.data);
    } else {
      bw::dot_thread_kernel<Format, Compute, Layout, Lanes, Decoder>
          <<<blocks, 256, shared_bytes<Decoder>()>>>(
              encoded_.template left_view<Layout>(),
              encoded_.template right_view<Layout>(), count, tables_.view(),
              partials_.data);
    }
    bw::finalize_dot_kernel<<<1, 256>>>(partials_.data,
                                        static_cast<std::size_t>(blocks),
                                        result_.data);
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            bw::decoder_kind Decoder>
  void launch_gemv(std::size_t columns) {
    if constexpr (Access == bw::access_method::cooperative_shuffle) {
      static_assert(Layout == bw::storage_layout::dense);
      bw::gemv_cooperative_kernel<Format, Compute, Decoder>
          <<<static_cast<int>(settings_.gemv_rows), 256,
             shared_bytes<Decoder>()>>>(
              encoded_.left_dense.data, encoded_.right_dense.data,
              settings_.gemv_rows, columns, tables_.view(), result_.data);
    } else {
      bw::gemv_thread_kernel<Format, Compute, Layout, Lanes, Decoder>
          <<<static_cast<int>(settings_.gemv_rows), 256,
             shared_bytes<Decoder>()>>>(
              encoded_.template left_view<Layout>(),
              encoded_.template right_view<Layout>(), settings_.gemv_rows,
              columns, tables_.view(), result_.data);
    }
  }

  template <typename Launch>
  std::pair<int, std::vector<double>> measure(Launch launch) {
    for (int warmup = 0; warmup < settings_.warmup; ++warmup) {
      launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto pilot = timer_.measure(1, launch);
    auto iterations = 1;
    if (settings_.target_sample_ms > 0 && pilot > 0) {
      iterations = std::clamp(
          static_cast<int>(std::ceil(settings_.target_sample_ms / pilot)), 1,
          10000);
    }
    std::vector<double> samples;
    for (int sample = 0; sample < settings_.samples; ++sample) {
      samples.push_back(timer_.measure(iterations, launch));
    }
    return {iterations, samples};
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            bw::decoder_kind Decoder>
  void metadata(std::ostream &stream) const {
    if constexpr (Access == bw::access_method::cooperative_shuffle) {
      using geometry = bw::cooperative_geometry<Format::total_bits>;
      stream << "dense,cooperative_shuffle," << geometry::values
             << ",32," << geometry::words << ',' << geometry::consumers << ','
             << geometry::values_per_consumer << ",warp_shuffle,";
    } else {
      stream << (Layout == bw::storage_layout::dense ? "dense" : "padded")
             << ','
             << (Access == bw::access_method::scalar ? "scalar"
                                                      : "thread_packet")
             << ',' << Lanes << ','
             << (Layout == bw::storage_layout::dense
                     ? 32
                     : static_cast<int>(sizeof(bw::padded_storage_t<Format>) *
                                        8))
             << ",1,1," << Lanes << ",none,";
    }
    stream << decoder_name(Decoder) << ',';
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            bw::decoder_kind Decoder>
  void run_dot(std::size_t count, const std::string &id) {
    const auto packets = (count + Lanes - 1) / Lanes;
    const auto blocks = static_cast<int>(
        std::max<std::size_t>(1, std::min<std::size_t>(512, (packets + 255) / 256)));
    const auto launch = [&] {
      launch_dot<Layout, Access, Lanes, Decoder>(count, blocks);
    };
    const auto [iterations, times] = measure(launch);
    value_type result{};
    CUDA_CHECK(cudaMemcpy(&result, result_.data, sizeof(result),
                          cudaMemcpyDeviceToHost));
    const auto valid = std::isfinite(static_cast<double>(result));
    for (std::size_t sample = 0; sample < times.size(); ++sample) {
      auto &stream = output_.stream;
      stream << output_.timestamp << ',' << output_.gpu << ',' << settings_.mode
             << ',' << distribution_ << ",dot," << Format::name << ','
             << Format::total_bits << ',' << Format::exponent_bits << ','
             << Format::fraction_bits << ',' << bw::compute_traits<Compute>::name
             << ',';
      metadata<Layout, Access, Lanes, Decoder>(stream);
      stream << id << ',' << count << ",0,1,"
             << (2.0 * count * Format::total_bits / 8.0) << ','
             << 2 * physical_bytes<Layout>(count) << ',' << table_bytes<Decoder>()
             << ',' << blocks << ",256," << settings_.warmup << ',' << sample
             << ',' << iterations << ',' << times[sample] << ','
             << times[sample] / iterations << ',' << std::setprecision(17)
             << static_cast<double>(result) << ',' << (valid ? 1 : 0) << '\n';
    }
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            bw::decoder_kind Decoder>
  void run_gemv(std::size_t columns, const std::string &id) {
    if constexpr (Access == bw::access_method::cooperative_shuffle) {
      using geometry = bw::cooperative_geometry<Format::total_bits>;
      if (columns % geometry::values != 0) {
        throw std::runtime_error("cooperative GEMV requires aligned row width");
      }
    }
    const auto launch = [&] { launch_gemv<Layout, Access, Lanes, Decoder>(columns); };
    const auto [iterations, times] = measure(launch);
    value_type first{};
    CUDA_CHECK(cudaMemcpy(&first, result_.data, sizeof(first),
                          cudaMemcpyDeviceToHost));
    const auto valid = std::isfinite(static_cast<double>(first));
    const auto values = settings_.gemv_rows * columns + columns;
    for (std::size_t sample = 0; sample < times.size(); ++sample) {
      auto &stream = output_.stream;
      stream << output_.timestamp << ',' << output_.gpu << ',' << settings_.mode
             << ',' << distribution_ << ",gemv," << Format::name << ','
             << Format::total_bits << ',' << Format::exponent_bits << ','
             << Format::fraction_bits << ',' << bw::compute_traits<Compute>::name
             << ',';
      metadata<Layout, Access, Lanes, Decoder>(stream);
      stream << id << ',' << columns << ',' << settings_.gemv_rows << ','
             << settings_.gemv_rows << ','
             << (static_cast<double>(values) * Format::total_bits / 8.0) << ','
             << physical_bytes<Layout>(values) << ',' << table_bytes<Decoder>()
             << ',' << settings_.gemv_rows << ",256," << settings_.warmup << ','
             << sample << ',' << iterations << ',' << times[sample] << ','
             << times[sample] / iterations << ',' << std::setprecision(17)
             << static_cast<double>(first) << ',' << (valid ? 1 : 0) << '\n';
    }
  }

  const options &settings_;
  const std::string &distribution_;
  encoded_buffers<Format> &encoded_;
  table_storage<Format, Compute> &tables_;
  csv_output &output_;
  device_buffer<value_type> partials_;
  device_buffer<value_type> result_;
  event_timer timer_;
};

template <typename Format, bw::compute_kind Compute,
          bw::decoder_kind Decoder>
void run_scalar_family(format_runner<Format, Compute> &runner) {
  runner.template run_variant<bw::storage_layout::dense,
                              bw::access_method::scalar, 1, Decoder>();
  runner.template run_variant<bw::storage_layout::padded,
                              bw::access_method::scalar, 1, Decoder>();
}

template <typename Format, bw::compute_kind Compute,
          bw::decoder_kind Decoder>
void run_access_family(format_runner<Format, Compute> &runner) {
  run_scalar_family<Format, Compute, Decoder>(runner);
  runner.template run_variant<bw::storage_layout::dense,
                              bw::access_method::thread_packet, 2, Decoder>();
  runner.template run_variant<bw::storage_layout::dense,
                              bw::access_method::thread_packet, 4, Decoder>();
  runner.template run_variant<bw::storage_layout::dense,
                              bw::access_method::thread_packet, 8, Decoder>();
  runner.template run_variant<bw::storage_layout::padded,
                              bw::access_method::thread_packet, 2, Decoder>();
  runner.template run_variant<bw::storage_layout::padded,
                              bw::access_method::thread_packet, 4, Decoder>();
  runner.template run_variant<bw::storage_layout::padded,
                              bw::access_method::thread_packet, 8, Decoder>();
  runner.template run_variant<bw::storage_layout::dense,
                              bw::access_method::cooperative_shuffle, 1,
                              Decoder>();
}

template <typename Format, bw::compute_kind Compute>
void run_strategies(format_runner<Format, Compute> &runner) {
  run_scalar_family<Format, Compute, bw::decoder_kind::generic_reference>(runner);
  run_access_family<Format, Compute, bw::decoder_kind::direct_branchy>(runner);
  if constexpr (Compute == bw::compute_kind::fp64) {
    run_access_family<Format, Compute, bw::decoder_kind::direct_masked>(runner);
  }
  if constexpr (Format::exponent_bits == 0) {
    run_access_family<Format, Compute, bw::decoder_kind::fixed_integer>(runner);
  } else if constexpr (Format::exponent_bits == 1 && Format::finite) {
    run_access_family<Format, Compute, bw::decoder_kind::e1_integer>(runner);
  } else if constexpr (Format::fraction_bits == 0) {
    run_access_family<Format, Compute, bw::decoder_kind::exponent_only>(runner);
  }
  if constexpr (Format::total_bits <= 14) {
    run_access_family<Format, Compute, bw::decoder_kind::full_lut_global>(runner);
    run_access_family<Format, Compute, bw::decoder_kind::full_lut_shared>(runner);
  }
}

template <typename Format, bw::compute_kind Compute>
void run_one_format(const options &settings, const std::string &distribution,
                    source_buffers &sources, csv_output &output,
                    std::size_t left_count, std::size_t right_count) {
  std::cout << "  " << Format::name << " -> "
            << bw::compute_traits<Compute>::name << std::endl;
  encoded_buffers<Format> encoded(left_count, right_count);
  encoded.encode(sources, left_count, right_count);
  table_storage<Format, Compute> tables;
  format_runner<Format, Compute> runner(settings, distribution, encoded, tables,
                                         output);
  run_strategies(runner);
}

template <bw::compute_kind Compute>
void run_registered_formats(const options &settings,
                            const std::string &distribution,
                            source_buffers &sources, csv_output &output,
                            std::size_t left_count, std::size_t right_count) {
#if AUT_BITWIDTH_TOTAL_BITS == 3
  run_one_format<storage::e0m2, Compute>(settings, distribution, sources, output,
                                         left_count, right_count);
  run_one_format<storage::e1m1, Compute>(settings, distribution, sources, output,
                                         left_count, right_count);
  run_one_format<storage::e2m0, Compute>(settings, distribution, sources, output,
                                         left_count, right_count);
#elif AUT_BITWIDTH_TOTAL_BITS == 5
  run_one_format<storage::e0m4, Compute>(settings, distribution, sources, output,
                                         left_count, right_count);
  if constexpr (Compute == bw::compute_kind::fp64) {
    run_one_format<storage::e1m3, Compute>(settings, distribution, sources,
                                           output, left_count, right_count);
  }
  run_one_format<storage::e2m2, Compute>(settings, distribution, sources, output,
                                         left_count, right_count);
  if constexpr (Compute == bw::compute_kind::fp64) {
    run_one_format<storage::e3m1, Compute>(settings, distribution, sources,
                                           output, left_count, right_count);
  }
  run_one_format<storage::e4m0, Compute>(settings, distribution, sources, output,
                                         left_count, right_count);
#else
#error "the selected width has not been registered"
#endif
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto settings = parse_options(argc, argv);
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::cout << "Arbitrary-width strategy benchmark: "
              << AUT_BITWIDTH_TOTAL_BITS << " bits\nGPU: " << properties.name
              << " (sm_" << properties.major << properties.minor << ")\n";

    const auto maximum_dot = std::size_t{1}
                             << *std::max_element(settings.dot_powers.begin(),
                                                  settings.dot_powers.end());
    const auto maximum_columns =
        std::size_t{1}
        << *std::max_element(settings.gemv_powers.begin(),
                             settings.gemv_powers.end());
    const auto left_count =
        std::max(maximum_dot, settings.gemv_rows * maximum_columns);
    const auto right_count = std::max(maximum_dot, maximum_columns);
    source_buffers sources(left_count, right_count);
    csv_output output(settings.output, properties.name);

    for (std::size_t dataset = 0; dataset < settings.distributions.size();
         ++dataset) {
      const auto &distribution = settings.distributions[dataset];
      std::cout << distribution << std::endl;
      sources.generate(distribution, settings.seed + dataset * 0x9e3779b97f4a7c15ull);
      run_registered_formats<bw::compute_kind::fp32>(
          settings, distribution, sources, output, left_count, right_count);
      run_registered_formats<bw::compute_kind::fp64>(
          settings, distribution, sources, output, left_count, right_count);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Wrote " << settings.output << std::endl;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
