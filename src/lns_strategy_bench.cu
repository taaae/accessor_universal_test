#include "bitwidth_benchmark_kernels.cuh"
#include "lns_benchmark_kernels.cuh"

#include <curand.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef AUT_LNS_FORMAT_ID
#error "AUT_LNS_FORMAT_ID must select one LNS format"
#endif

#ifndef AUT_LNS_ENABLE_FP32
#define AUT_LNS_ENABLE_FP32 1
#endif

namespace {

namespace bw = aut::bitwidth;
namespace lk = aut::lns_strategy;
namespace lns = aut::lns;

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
  std::vector<int> result;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      result.push_back(std::stoi(token));
    }
  }
  if (result.empty()) {
    throw std::runtime_error("power list cannot be empty");
  }
  return result;
}

std::vector<std::string> parse_strings(const std::string &text) {
  std::vector<std::string> result;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      result.push_back(token);
    }
  }
  return result;
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
  std::string output{"lns_timing_samples.csv"};
  std::string validation_output;
  std::set<std::string> enabled_variants;
};

options parse_options(int argc, char **argv) {
  options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw std::runtime_error("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--mode") {
      result.mode = value();
    } else if (argument == "--dot-powers") {
      result.dot_powers = parse_powers(value());
    } else if (argument == "--gemv-powers") {
      result.gemv_powers = parse_powers(value());
    } else if (argument == "--distributions") {
      result.distributions = parse_strings(value());
    } else if (argument == "--gemv-rows") {
      result.gemv_rows = std::stoull(value());
    } else if (argument == "--warmup") {
      result.warmup = std::stoi(value());
    } else if (argument == "--samples") {
      result.samples = std::stoi(value());
    } else if (argument == "--target-sample-ms") {
      result.target_sample_ms = std::stod(value());
    } else if (argument == "--seed") {
      result.seed = std::stoull(value());
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--validation-output") {
      result.validation_output = value();
    } else if (argument == "--variants") {
      for (const auto &variant : parse_strings(value())) {
        result.enabled_variants.insert(variant);
      }
    } else {
      throw std::runtime_error("unknown option: " + argument);
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

std::string timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&time, &utc);
  std::ostringstream stream;
  stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return stream.str();
}

template <typename T> struct device_buffer {
  T *data{};
  std::size_t count{};

  device_buffer() = default;
  explicit device_buffer(std::size_t size) : count(std::max<std::size_t>(size, 1)) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));
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
  template <typename Launch> double measure(int iterations, Launch launch) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
      launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds{};
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
  }
};

struct source_buffers {
  device_buffer<double> left;
  device_buffer<double> right;
  curandGenerator_t generator{};

  source_buffers(std::size_t left_count, std::size_t right_count)
      : left(left_count + 8), right(right_count + 8) {
    CURAND_CHECK(
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  }
  ~source_buffers() { curandDestroyGenerator(generator); }

  void generate(const std::string &distribution, std::uint64_t seed) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));
    if (distribution == "uniform_0_1") {
      CURAND_CHECK(curandGenerateUniformDouble(generator, left.data, left.count));
      CURAND_CHECK(
          curandGenerateUniformDouble(generator, right.data, right.count));
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
  device_buffer<lns::padded_storage_t<Format>> left_padded;
  device_buffer<lns::padded_storage_t<Format>> right_padded;
  device_buffer<std::uint32_t> left_dense;
  device_buffer<std::uint32_t> right_dense;

  encoded_buffers(std::size_t left_count, std::size_t right_count)
      : left_padded(left_count + 8), right_padded(right_count + 8),
        left_dense(bw::dense_word_count<Format::total_bits>(left_count) + 8),
        right_dense(bw::dense_word_count<Format::total_bits>(right_count) + 8) {}

  void encode(const source_buffers &source, std::size_t left_count,
              std::size_t right_count) {
    constexpr int threads = 256;
    const auto blocks = [](std::size_t count) {
      return static_cast<int>(
          std::min<std::size_t>((count + 255) / 256, 4096));
    };
    lk::encode_padded_kernel<Format><<<blocks(left_count), threads>>>(
        source.left.data, left_padded.data, left_count);
    lk::encode_padded_kernel<Format><<<blocks(right_count), threads>>>(
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
    lk::pack_dense_kernel<Format><<<blocks(left_chunks), threads>>>(
        left_padded.data, left_dense.data, left_count);
    lk::pack_dense_kernel<Format><<<blocks(right_chunks), threads>>>(
        right_padded.data, right_dense.data, right_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  template <bw::storage_layout Layout>
  lk::storage_view<Format, Layout> left_view() const {
    return {left_dense.data, left_padded.data};
  }
  template <bw::storage_layout Layout>
  lk::storage_view<Format, Layout> right_view() const {
    return {right_dense.data, right_padded.data};
  }
};

template <typename Format, bw::compute_kind Compute> struct table_storage {
  using value_type = bw::compute_t<Compute>;
  static constexpr std::size_t full_count =
      Format::total_bits <= 12 ? std::size_t{1} << Format::total_bits : 1;
  static constexpr std::size_t fraction_count =
      Format::log_fraction_bits <= 13
          ? std::size_t{1} << Format::log_fraction_bits
          : 1;
  static constexpr int coarse_bits =
      Format::log_fraction_bits < 8 ? Format::log_fraction_bits : 8;
  static constexpr std::size_t coarse_count = std::size_t{1} << coarse_bits;
  static constexpr std::size_t pair_count =
      Format::total_bits <= 6
          ? std::size_t{1} << (2 * Format::total_bits)
          : 1;

  device_buffer<value_type> full{full_count};
  device_buffer<value_type> fraction{fraction_count};
  device_buffer<value_type> coarse{coarse_count};
  device_buffer<value_type> pair{pair_count};

  table_storage() {
    std::vector<value_type> host_full(full_count);
    for (std::size_t raw = 0; raw < full_count; ++raw) {
      host_full[raw] = lns::decode<Format, value_type>(raw);
    }
    copy(full, host_full);

    std::vector<value_type> host_fraction(fraction_count);
    constexpr auto fraction_scale =
        static_cast<double>(std::uint64_t{1} << Format::log_fraction_bits);
    for (std::size_t index = 0; index < fraction_count; ++index) {
      host_fraction[index] =
          static_cast<value_type>(std::exp2(index / fraction_scale));
    }
    copy(fraction, host_fraction);

    std::vector<value_type> host_coarse(coarse_count);
    constexpr auto coarse_scale =
        static_cast<double>(std::uint64_t{1} << coarse_bits);
    for (std::size_t index = 0; index < coarse_count; ++index) {
      host_coarse[index] =
          static_cast<value_type>(std::exp2(index / coarse_scale));
    }
    copy(coarse, host_coarse);

    std::vector<value_type> host_pair(pair_count);
    if constexpr (Format::total_bits <= 6) {
      constexpr auto mask = lns::raw_mask<Format>();
      for (std::size_t index = 0; index < pair_count; ++index) {
        const auto left = static_cast<std::uint32_t>(index) & mask;
        const auto right = static_cast<std::uint32_t>(index) >>
                           Format::total_bits;
        host_pair[index] =
            lns::multiply_fused<Format, value_type>(left, right);
      }
    }
    copy(pair, host_pair);
  }

  static void copy(device_buffer<value_type> &device,
                   const std::vector<value_type> &host) {
    CUDA_CHECK(cudaMemcpy(device.data, host.data(),
                          host.size() * sizeof(value_type),
                          cudaMemcpyHostToDevice));
  }

  lk::table_bundle<Compute> view() const {
    return {full.data, fraction.data, coarse.data, pair.data};
  }
};

const char *decoder_name(lk::decoder_kind decoder);

template <lk::decoder_kind Decoder> constexpr double relative_tolerance() {
  if constexpr (Decoder == lk::decoder_kind::ex2_approx) {
    return 4e-6;
  } else if constexpr (Decoder == lk::decoder_kind::split_linear) {
    return 1e-5;
  } else if constexpr (Decoder == lk::decoder_kind::split_quadratic) {
    return 1e-7;
  } else if constexpr (Decoder == lk::decoder_kind::split_cubic) {
    return 1e-9;
  } else {
    return 1e-12;
  }
}

template <typename Format, bw::compute_kind Compute, lk::decoder_kind Decoder>
void validate_decoder(const options &settings,
                      table_storage<Format, Compute> &tables,
                      std::ofstream *summary) {
  using value_type = bw::compute_t<Compute>;
  constexpr std::uint64_t domain =
      std::uint64_t{1} << Format::total_bits;
  constexpr std::size_t maximum_cases = 65536;
  const auto cases = static_cast<std::size_t>(
      std::min<std::uint64_t>(domain, maximum_cases));
  std::vector<std::uint32_t> host_raw(cases);
  for (std::size_t index = 0; index < cases; ++index) {
    host_raw[index] = static_cast<std::uint32_t>(
        (static_cast<unsigned long long>(index) * domain) / cases);
  }
  if (cases >= 2) {
    host_raw[cases - 2] = lns::zero_raw<Format>();
    host_raw[cases - 1] = lns::nan_raw<Format>();
  }
  device_buffer<std::uint32_t> raw(cases);
  device_buffer<value_type> observed(cases);
  CUDA_CHECK(cudaMemcpy(raw.data, host_raw.data(),
                        cases * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice));
  constexpr auto shared =
      lk::shared_table_bytes_v<Format, Compute, Decoder>;
  if constexpr (shared > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(
        lk::validate_decode_kernel<Format, Compute, Decoder>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
  }
  lk::validate_decode_kernel<Format, Compute, Decoder>
      <<<std::min<int>(256, static_cast<int>((cases + 255) / 256)), 256,
         shared>>>(raw.data, cases, tables.view(), observed.data);
  CUDA_CHECK(cudaGetLastError());
  std::vector<value_type> host_observed(cases);
  CUDA_CHECK(cudaMemcpy(host_observed.data(), observed.data,
                        cases * sizeof(value_type), cudaMemcpyDeviceToHost));

  std::size_t failures{};
  double maximum_relative{};
  double maximum_absolute{};
  std::uint32_t worst_raw{};
  for (std::size_t index = 0; index < cases; ++index) {
    auto expected = lns::decode<Format, value_type>(host_raw[index]);
    if constexpr (Decoder == lk::decoder_kind::ex2_approx) {
      const auto raw_value = host_raw[index];
      if (!lns::is_special<Format>(raw_value)) {
        constexpr float scale = static_cast<float>(
            std::uint64_t{1} << Format::log_fraction_bits);
        auto projected = std::exp2(
            static_cast<float>(lns::log_code<Format>(raw_value)) / scale);
        if (lns::sign<Format>(raw_value)) {
          projected = -projected;
        }
        expected = static_cast<value_type>(projected);
      }
    }
    const auto actual = host_observed[index];
    bool valid{};
    double absolute{};
    double relative{};
    if (std::isnan(static_cast<double>(expected))) {
      valid = std::isnan(static_cast<double>(actual));
    } else if (std::isinf(static_cast<double>(expected))) {
      valid = std::isinf(static_cast<double>(actual)) &&
              std::signbit(static_cast<double>(expected)) ==
                  std::signbit(static_cast<double>(actual));
    } else {
      absolute = std::abs(static_cast<double>(actual) -
                          static_cast<double>(expected));
      const auto denominator =
          std::max(std::abs(static_cast<double>(expected)),
                   std::is_same_v<value_type, float> ? 1e-37 : 1e-307);
      relative = absolute / denominator;
      const auto tolerance =
          std::is_same_v<value_type, float>
              ? std::max(relative_tolerance<Decoder>(), 2e-6)
              : relative_tolerance<Decoder>();
      valid = relative <= tolerance ||
              (expected == value_type{} && actual == value_type{});
    }
    if (!valid) {
      ++failures;
    }
    if (relative > maximum_relative) {
      maximum_relative = relative;
      maximum_absolute = absolute;
      worst_raw = host_raw[index];
    }
  }
  if (summary != nullptr) {
    *summary << Format::name << ',' << bw::compute_traits<Compute>::name << ','
             << decoder_name(Decoder) << ',' << cases << ',' << failures << ','
             << std::setprecision(17) << maximum_absolute << ','
             << maximum_relative << ',' << worst_raw << '\n';
  }
  if (failures != 0) {
    throw std::runtime_error(std::string("decoder validation failed: ") +
                             Format::name + "/" +
                             bw::compute_traits<Compute>::name + "/" +
                             decoder_name(Decoder) + " failures=" +
                             std::to_string(failures));
  }
}

template <typename Format, bw::compute_kind Compute>
void validate_decoders(const options &settings,
                       table_storage<Format, Compute> &tables) {
  std::ofstream output;
  const auto needs_header = !settings.validation_output.empty() &&
                            !std::ifstream(settings.validation_output).good();
  if (!settings.validation_output.empty()) {
    output.open(settings.validation_output, std::ios::app);
    if (!output) {
      throw std::runtime_error("cannot open validation output: " +
                               settings.validation_output);
    }
    if (needs_header) {
      output << "format,arithmetic_type,decoder,cases,failures,"
                "max_absolute_error,max_relative_error,worst_raw\n";
    }
  }
  auto *stream = output ? &output : nullptr;
  validate_decoder<Format, Compute, lk::decoder_kind::reference_exp2>(
      settings, tables, stream);
  validate_decoder<Format, Compute, lk::decoder_kind::ex2_approx>(
      settings, tables, stream);
  if constexpr (Format::total_bits <= 12) {
    validate_decoder<Format, Compute, lk::decoder_kind::full_lut_global>(
        settings, tables, stream);
    validate_decoder<Format, Compute, lk::decoder_kind::full_lut_shared>(
        settings, tables, stream);
  }
  if constexpr (Format::log_fraction_bits <= 13) {
    validate_decoder<Format, Compute, lk::decoder_kind::fraction_lut_global>(
        settings, tables, stream);
    validate_decoder<Format, Compute, lk::decoder_kind::fraction_lut_shared>(
        settings, tables, stream);
  }
  if constexpr (Format::log_fraction_bits <= 5) {
    validate_decoder<Format, Compute, lk::decoder_kind::fraction_lut_warp>(
        settings, tables, stream);
  }
  if constexpr (Format::log_fraction_bits >= 6) {
    validate_decoder<Format, Compute, lk::decoder_kind::split_linear>(
        settings, tables, stream);
    validate_decoder<Format, Compute, lk::decoder_kind::split_quadratic>(
        settings, tables, stream);
    validate_decoder<Format, Compute, lk::decoder_kind::split_cubic>(
        settings, tables, stream);
  }
  std::cout << "    decoder validation passed" << std::endl;
}

const char *decoder_name(lk::decoder_kind decoder) {
  switch (decoder) {
  case lk::decoder_kind::reference_exp2:
    return "reference_exp2";
  case lk::decoder_kind::ex2_approx:
    return "ex2_approx";
  case lk::decoder_kind::full_lut_global:
    return "full_lut_global";
  case lk::decoder_kind::full_lut_shared:
    return "full_lut_shared";
  case lk::decoder_kind::fraction_lut_global:
    return "fraction_lut_global";
  case lk::decoder_kind::fraction_lut_shared:
    return "fraction_lut_shared";
  case lk::decoder_kind::fraction_lut_warp:
    return "fraction_lut_warp";
  case lk::decoder_kind::split_linear:
    return "split_linear";
  case lk::decoder_kind::split_quadratic:
    return "split_quadratic";
  case lk::decoder_kind::split_cubic:
    return "split_cubic";
  case lk::decoder_kind::pair_lut_global:
    return "pair_lut_global";
  case lk::decoder_kind::pair_lut_shared:
    return "pair_lut_shared";
  }
  return "unknown";
}

const char *multiply_name(lk::multiply_kind multiply) {
  return multiply == lk::multiply_kind::ordinary ? "ordinary" : "fused";
}

struct csv_output {
  std::ofstream stream;
  std::string started{timestamp()};
  std::string gpu;

  csv_output(const std::string &path, std::string gpu_name)
      : gpu(std::move(gpu_name)) {
    std::ifstream existing(path, std::ios::binary | std::ios::ate);
    const auto header = !existing || existing.tellg() == 0;
    stream.open(path, std::ios::app);
    if (!stream) {
      throw std::runtime_error("cannot open output: " + path);
    }
    if (header) {
      stream << "timestamp,gpu,mode,distribution,kernel,number_family,format,"
                "bits,log_integer_bits,log_fraction_bits,arithmetic_type,"
                "multiply_method,storage_layout,access_method,packet_values,"
                "load_word_bits,loader_threads,consumer_threads,"
                "values_per_consumer,redistribution,decoder,strategy_id,N,M,"
                "rows,logical_input_bytes,physical_input_bytes,table_bytes,"
                "blocks,threads,warmup,sample,iterations,total_ms,mean_ms,"
                "result,valid\n";
    }
  }
};

template <typename Format, bw::compute_kind Compute> class runner {
public:
  using value_type = bw::compute_t<Compute>;

  runner(const options &settings, const std::string &distribution,
         encoded_buffers<Format> &encoded,
         table_storage<Format, Compute> &tables, csv_output &output)
      : settings_(settings), distribution_(distribution), encoded_(encoded),
        tables_(tables), output_(output), partials_(512),
        result_(settings.gemv_rows), timer_() {}

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            lk::decoder_kind Decoder, lk::multiply_kind Multiply>
  void run_variant() {
    const auto id = variant_id<Layout, Access, Lanes, Decoder, Multiply>();
    const auto qualified = std::string(Format::name) + "/" +
                           bw::compute_traits<Compute>::name + "/" + id;
    if (!settings_.enabled_variants.empty() &&
        settings_.enabled_variants.count(id) == 0 &&
        settings_.enabled_variants.count(qualified) == 0) {
      return;
    }
    if (settings_.mode == "smoke") {
      std::cout << "    " << qualified << std::endl;
    }
    for (const auto power : settings_.dot_powers) {
      run_dot<Layout, Access, Lanes, Decoder, Multiply>(
          std::size_t{1} << power, id);
    }
    for (const auto power : settings_.gemv_powers) {
      run_gemv<Layout, Access, Lanes, Decoder, Multiply>(
          std::size_t{1} << power, id);
    }
  }

private:
  template <lk::decoder_kind Decoder>
  static constexpr std::size_t table_bytes() {
    return lk::table_entries_v<Format, Decoder> * sizeof(value_type);
  }

  template <lk::decoder_kind Decoder>
  static constexpr std::size_t shared_bytes() {
    return lk::shared_table_bytes_v<Format, Compute, Decoder> +
           256 * sizeof(value_type);
  }

  template <bw::storage_layout Layout>
  static std::size_t physical_bytes(std::size_t count) {
    if constexpr (Layout == bw::storage_layout::dense) {
      return bw::dense_data_bytes<Format::total_bits>(count);
    } else {
      return count * sizeof(lns::padded_storage_t<Format>);
    }
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            lk::decoder_kind Decoder, lk::multiply_kind Multiply>
  static std::string variant_id() {
    const auto layout = Layout == bw::storage_layout::dense ? "dense" : "padded";
    const auto access = Access == bw::access_method::scalar
                            ? "scalar"
                            : Access == bw::access_method::thread_packet
                                  ? "thread_packet"
                                  : "cooperative_shuffle";
    const auto width = [] {
      if constexpr (Access == bw::access_method::cooperative_shuffle) {
        return lk::cooperative_geometry<Format::total_bits>::values;
      } else {
        return Lanes;
      }
    }();
    return std::string(multiply_name(Multiply)) + "/" + layout + "/" +
           access + "/x" + std::to_string(width) + "/" +
           decoder_name(Decoder);
  }

  template <lk::decoder_kind Decoder, typename Kernel>
  static void allow_large_shared(Kernel kernel) {
    if constexpr (shared_bytes<Decoder>() > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(shared_bytes<Decoder>())));
    }
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            lk::decoder_kind Decoder, lk::multiply_kind Multiply>
  void launch_dot(std::size_t count, int blocks) {
    if constexpr (Access == bw::access_method::cooperative_shuffle) {
      static_assert(Layout == bw::storage_layout::dense);
      allow_large_shared<Decoder>(
          lk::dot_cooperative_kernel<Format, Compute, Decoder, Multiply>);
      lk::dot_cooperative_kernel<Format, Compute, Decoder, Multiply>
          <<<blocks, 256, shared_bytes<Decoder>()>>>(
              encoded_.left_dense.data, encoded_.right_dense.data, count,
              tables_.view(), partials_.data);
    } else {
      allow_large_shared<Decoder>(
          lk::dot_thread_kernel<Format, Compute, Layout, Lanes, Decoder,
                                Multiply>);
      lk::dot_thread_kernel<Format, Compute, Layout, Lanes, Decoder, Multiply>
          <<<blocks, 256, shared_bytes<Decoder>()>>>(
              encoded_.template left_view<Layout>(),
              encoded_.template right_view<Layout>(), count, tables_.view(),
              partials_.data);
    }
    lk::finalize_dot_kernel<<<1, 256>>>(
        partials_.data, static_cast<std::size_t>(blocks), result_.data);
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            lk::decoder_kind Decoder, lk::multiply_kind Multiply>
  void launch_gemv(std::size_t columns) {
    if constexpr (Access == bw::access_method::cooperative_shuffle) {
      static_assert(Layout == bw::storage_layout::dense);
      allow_large_shared<Decoder>(
          lk::gemv_cooperative_kernel<Format, Compute, Decoder, Multiply>);
      lk::gemv_cooperative_kernel<Format, Compute, Decoder, Multiply>
          <<<static_cast<int>(settings_.gemv_rows), 256,
             shared_bytes<Decoder>()>>>(
              encoded_.left_dense.data, encoded_.right_dense.data,
              settings_.gemv_rows, columns, tables_.view(), result_.data);
    } else {
      allow_large_shared<Decoder>(
          lk::gemv_thread_kernel<Format, Compute, Layout, Lanes, Decoder,
                                 Multiply>);
      lk::gemv_thread_kernel<Format, Compute, Layout, Lanes, Decoder, Multiply>
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
    const auto pilot = timer_.measure(1, launch);
    auto iterations = 1;
    if (settings_.target_sample_ms > 0 && pilot > 0) {
      iterations = std::clamp(
          static_cast<int>(std::ceil(settings_.target_sample_ms / pilot)), 1,
          10000);
    }
    std::vector<double> times;
    for (int sample = 0; sample < settings_.samples; ++sample) {
      times.push_back(timer_.measure(iterations, launch));
    }
    return {iterations, times};
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            lk::decoder_kind Decoder, lk::multiply_kind Multiply>
  void metadata(std::ostream &stream) const {
    stream << multiply_name(Multiply) << ',';
    if constexpr (Access == bw::access_method::cooperative_shuffle) {
      using geometry = lk::cooperative_geometry<Format::total_bits>;
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
                     : static_cast<int>(sizeof(lns::padded_storage_t<Format>) *
                                        8))
             << ",1,1," << Lanes << ",none,";
    }
    stream << decoder_name(Decoder) << ',';
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            lk::decoder_kind Decoder, lk::multiply_kind Multiply>
  void write_rows(const char *kernel, std::size_t n, std::size_t m,
                  std::size_t rows, int blocks, std::size_t values,
                  const std::string &id, int iterations,
                  const std::vector<double> &times, value_type value) {
    for (std::size_t sample = 0; sample < times.size(); ++sample) {
      auto &stream = output_.stream;
      stream << output_.started << ',' << output_.gpu << ',' << settings_.mode
             << ',' << distribution_ << ',' << kernel << ",lns,"
             << Format::name << ',' << Format::total_bits << ','
             << Format::log_integer_bits << ',' << Format::log_fraction_bits
             << ',' << bw::compute_traits<Compute>::name << ',';
      metadata<Layout, Access, Lanes, Decoder, Multiply>(stream);
      stream << id << ',' << n << ',' << m << ',' << rows << ','
             << (static_cast<double>(values) * Format::total_bits / 8.0) << ','
             << physical_bytes<Layout>(values) << ',' << table_bytes<Decoder>()
             << ',' << blocks << ",256," << settings_.warmup << ',' << sample
             << ',' << iterations << ',' << times[sample] << ','
             << times[sample] / iterations << ',' << std::setprecision(17)
             << static_cast<double>(value) << ','
             << (std::isfinite(static_cast<double>(value)) ? 1 : 0) << '\n';
    }
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            lk::decoder_kind Decoder, lk::multiply_kind Multiply>
  void run_dot(std::size_t count, const std::string &id) {
    const auto work = [&] {
      if constexpr (Access == bw::access_method::cooperative_shuffle) {
        return (count +
                lk::cooperative_geometry<Format::total_bits>::values - 1) /
               lk::cooperative_geometry<Format::total_bits>::values;
      } else {
        return (count + Lanes - 1) / Lanes;
      }
    }();
    const auto blocks = static_cast<int>(std::max<std::size_t>(
        1, std::min<std::size_t>(512, (work + 255) / 256)));
    const auto launch = [&] {
      launch_dot<Layout, Access, Lanes, Decoder, Multiply>(count, blocks);
    };
    const auto [iterations, times] = measure(launch);
    value_type result{};
    CUDA_CHECK(cudaMemcpy(&result, result_.data, sizeof(result),
                          cudaMemcpyDeviceToHost));
    write_rows<Layout, Access, Lanes, Decoder, Multiply>(
        "dot", count, 0, 1, blocks, 2 * count, id, iterations, times, result);
  }

  template <bw::storage_layout Layout, bw::access_method Access, int Lanes,
            lk::decoder_kind Decoder, lk::multiply_kind Multiply>
  void run_gemv(std::size_t columns, const std::string &id) {
    if constexpr (Access == bw::access_method::cooperative_shuffle) {
      using geometry = lk::cooperative_geometry<Format::total_bits>;
      if (columns % geometry::values != 0) {
        throw std::runtime_error("cooperative GEMV requires aligned columns");
      }
    }
    const auto launch = [&] {
      launch_gemv<Layout, Access, Lanes, Decoder, Multiply>(columns);
    };
    const auto [iterations, times] = measure(launch);
    value_type result{};
    CUDA_CHECK(cudaMemcpy(&result, result_.data, sizeof(result),
                          cudaMemcpyDeviceToHost));
    const auto values = settings_.gemv_rows * columns + columns;
    write_rows<Layout, Access, Lanes, Decoder, Multiply>(
        "gemv", columns, settings_.gemv_rows, settings_.gemv_rows,
        static_cast<int>(settings_.gemv_rows), values, id, iterations, times,
        result);
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

template <typename Format, bw::compute_kind Compute, lk::decoder_kind Decoder,
          lk::multiply_kind Multiply>
void run_scalar_family(runner<Format, Compute> &run) {
  if constexpr (Format::total_bits == 8 || Format::total_bits == 16 ||
                Format::total_bits == 32) {
    run.template run_variant<bw::storage_layout::padded,
                             bw::access_method::scalar, 1, Decoder, Multiply>();
  } else {
    run.template run_variant<bw::storage_layout::dense,
                             bw::access_method::scalar, 1, Decoder, Multiply>();
    run.template run_variant<bw::storage_layout::padded,
                             bw::access_method::scalar, 1, Decoder, Multiply>();
  }
}

template <typename Format, bw::compute_kind Compute, lk::decoder_kind Decoder,
          lk::multiply_kind Multiply>
void run_access_family(runner<Format, Compute> &run) {
  run_scalar_family<Format, Compute, Decoder, Multiply>(run);
  if constexpr (Format::total_bits == 8 || Format::total_bits == 16 ||
                Format::total_bits == 32) {
    run.template run_variant<bw::storage_layout::padded,
                             bw::access_method::thread_packet, 2, Decoder,
                             Multiply>();
    run.template run_variant<bw::storage_layout::padded,
                             bw::access_method::thread_packet, 4, Decoder,
                             Multiply>();
    run.template run_variant<bw::storage_layout::padded,
                             bw::access_method::thread_packet, 8, Decoder,
                             Multiply>();
  } else {
    run.template run_variant<bw::storage_layout::dense,
                             bw::access_method::thread_packet, 2, Decoder,
                             Multiply>();
    run.template run_variant<bw::storage_layout::dense,
                             bw::access_method::thread_packet, 4, Decoder,
                             Multiply>();
    run.template run_variant<bw::storage_layout::dense,
                             bw::access_method::thread_packet, 8, Decoder,
                             Multiply>();
    run.template run_variant<bw::storage_layout::padded,
                             bw::access_method::thread_packet, 2, Decoder,
                             Multiply>();
    run.template run_variant<bw::storage_layout::padded,
                             bw::access_method::thread_packet, 4, Decoder,
                             Multiply>();
    run.template run_variant<bw::storage_layout::padded,
                             bw::access_method::thread_packet, 8, Decoder,
                             Multiply>();
    if constexpr (lk::cooperative_supported_v<Format::total_bits>) {
      run.template run_variant<bw::storage_layout::dense,
                               bw::access_method::cooperative_shuffle, 1,
                               Decoder, Multiply>();
    }
  }
}

template <typename Format, bw::compute_kind Compute>
void run_strategies(runner<Format, Compute> &run) {
  run_scalar_family<Format, Compute, lk::decoder_kind::reference_exp2,
                    lk::multiply_kind::ordinary>(run);
  run_scalar_family<Format, Compute, lk::decoder_kind::reference_exp2,
                    lk::multiply_kind::fused>(run);
  run_access_family<Format, Compute, lk::decoder_kind::ex2_approx,
                    lk::multiply_kind::ordinary>(run);
  run_access_family<Format, Compute, lk::decoder_kind::ex2_approx,
                    lk::multiply_kind::fused>(run);

  if constexpr (Format::total_bits <= 12) {
    run_access_family<Format, Compute, lk::decoder_kind::full_lut_global,
                      lk::multiply_kind::ordinary>(run);
    run_access_family<Format, Compute, lk::decoder_kind::full_lut_shared,
                      lk::multiply_kind::ordinary>(run);
  }
  if constexpr (Format::log_fraction_bits <= 13) {
    run_access_family<Format, Compute, lk::decoder_kind::fraction_lut_global,
                      lk::multiply_kind::ordinary>(run);
    run_access_family<Format, Compute, lk::decoder_kind::fraction_lut_global,
                      lk::multiply_kind::fused>(run);
    run_access_family<Format, Compute, lk::decoder_kind::fraction_lut_shared,
                      lk::multiply_kind::ordinary>(run);
    run_access_family<Format, Compute, lk::decoder_kind::fraction_lut_shared,
                      lk::multiply_kind::fused>(run);
  }
  if constexpr (Format::log_fraction_bits <= 5) {
    run_access_family<Format, Compute, lk::decoder_kind::fraction_lut_warp,
                      lk::multiply_kind::ordinary>(run);
    run_access_family<Format, Compute, lk::decoder_kind::fraction_lut_warp,
                      lk::multiply_kind::fused>(run);
  }
  if constexpr (Format::log_fraction_bits >= 6) {
    run_access_family<Format, Compute, lk::decoder_kind::split_linear,
                      lk::multiply_kind::ordinary>(run);
    run_access_family<Format, Compute, lk::decoder_kind::split_linear,
                      lk::multiply_kind::fused>(run);
    run_access_family<Format, Compute, lk::decoder_kind::split_quadratic,
                      lk::multiply_kind::ordinary>(run);
    run_access_family<Format, Compute, lk::decoder_kind::split_quadratic,
                      lk::multiply_kind::fused>(run);
    run_access_family<Format, Compute, lk::decoder_kind::split_cubic,
                      lk::multiply_kind::ordinary>(run);
    run_access_family<Format, Compute, lk::decoder_kind::split_cubic,
                      lk::multiply_kind::fused>(run);
  }
  if constexpr (Format::total_bits <= 6) {
    run_access_family<Format, Compute, lk::decoder_kind::pair_lut_global,
                      lk::multiply_kind::fused>(run);
    run_access_family<Format, Compute, lk::decoder_kind::pair_lut_shared,
                      lk::multiply_kind::fused>(run);
  }
}

#if AUT_LNS_FORMAT_ID == 1
using selected_format = lns::lns4_r1;
#elif AUT_LNS_FORMAT_ID == 2
using selected_format = lns::lns6_r2;
#elif AUT_LNS_FORMAT_ID == 3
using selected_format = lns::lns8_r2;
#elif AUT_LNS_FORMAT_ID == 4
using selected_format = lns::lns8_r3;
#elif AUT_LNS_FORMAT_ID == 5
using selected_format = lns::lns8_r4;
#elif AUT_LNS_FORMAT_ID == 6
using selected_format = lns::lns8_r5;
#elif AUT_LNS_FORMAT_ID == 7
using selected_format = lns::lns10_r4;
#elif AUT_LNS_FORMAT_ID == 8
using selected_format = lns::lns12_r6;
#elif AUT_LNS_FORMAT_ID == 9
using selected_format = lns::lns16_r4;
#elif AUT_LNS_FORMAT_ID == 10
using selected_format = lns::lns16_r7;
#elif AUT_LNS_FORMAT_ID == 11
using selected_format = lns::lns16_r10;
#elif AUT_LNS_FORMAT_ID == 12
using selected_format = lns::lns16_r11;
#elif AUT_LNS_FORMAT_ID == 13
using selected_format = lns::lns16_r12;
#elif AUT_LNS_FORMAT_ID == 14
using selected_format = lns::lns16_r13;
#elif AUT_LNS_FORMAT_ID == 15
using selected_format = lns::lns32_r20;
#elif AUT_LNS_FORMAT_ID == 16
using selected_format = lns::lns32_r23;
#elif AUT_LNS_FORMAT_ID == 17
using selected_format = lns::lns32_r28;
#else
#error "unknown AUT_LNS_FORMAT_ID"
#endif

template <bw::compute_kind Compute>
void run_compute(const options &settings, const std::string &distribution,
                 source_buffers &sources, csv_output &output,
                 std::size_t left_count, std::size_t right_count) {
  std::cout << "  " << selected_format::name << " -> "
            << bw::compute_traits<Compute>::name << std::endl;
  encoded_buffers<selected_format> encoded(left_count, right_count);
  encoded.encode(sources, left_count, right_count);
  table_storage<selected_format, Compute> tables;
  if (settings.mode == "smoke" &&
      distribution == settings.distributions.front()) {
    validate_decoders(settings, tables);
  }
  runner<selected_format, Compute> benchmark(settings, distribution, encoded,
                                               tables, output);
  run_strategies(benchmark);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto settings = parse_options(argc, argv);
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::cout << "LNS strategy benchmark: " << selected_format::name
              << "\nGPU: " << properties.name << " (sm_" << properties.major
              << properties.minor << ")\n";

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
      sources.generate(distribution,
                       settings.seed + dataset * 0x9e3779b97f4a7c15ull);
#if AUT_LNS_ENABLE_FP32
      run_compute<bw::compute_kind::fp32>(settings, distribution, sources,
                                           output, left_count, right_count);
#endif
      run_compute<bw::compute_kind::fp64>(settings, distribution, sources,
                                           output, left_count, right_count);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Wrote " << settings.output << std::endl;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << std::endl;
    return 1;
  }
}
