#include "storage_performance_kernels.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace aut::storage;

class benchmark_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

void check_cuda(cudaError_t status, const char *expression, const char *file,
                int line) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << ": "
            << cudaGetErrorName(status) << " (" << cudaGetErrorString(status)
            << ')';
    throw benchmark_error(message.str());
  }
}

void check_curand(curandStatus_t status, const char *expression,
                  const char *file, int line) {
  if (status != CURAND_STATUS_SUCCESS) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line
            << " with cuRAND status " << static_cast<int>(status);
    throw benchmark_error(message.str());
  }
}

#define CUDA_CHECK(expression)                                                 \
  check_cuda((expression), #expression, __FILE__, __LINE__)
#define CURAND_CHECK(expression)                                               \
  check_curand((expression), #expression, __FILE__, __LINE__)

template <typename T> class device_buffer {
public:
  explicit device_buffer(std::size_t count) : count_{count} {
    if (count != 0) {
      CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }
  }
  ~device_buffer() {
    if (data_) {
      cudaFree(data_);
    }
  }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
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

  float measure(const std::function<void()> &launch, std::size_t iterations) {
    CUDA_CHECK(cudaEventRecord(start_));
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      launch();
    }
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float elapsed{};
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_, stop_));
    return elapsed;
  }

private:
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
};

enum class mode { sweep, profile };
enum class distribution { uniform_0_1, normal_0_1 };

const char *name(distribution value) {
  return value == distribution::uniform_0_1 ? "uniform_0_1" : "normal_0_1";
}

distribution parse_distribution(const std::string &text) {
  if (text == "uniform_0_1") {
    return distribution::uniform_0_1;
  }
  if (text == "normal_0_1") {
    return distribution::normal_0_1;
  }
  throw benchmark_error("unknown distribution: " + text);
}

struct options {
  mode run_mode{mode::sweep};
  std::vector<int> dot_powers{12, 16, 20, 24, 27};
  std::vector<int> gemv_powers{8, 10, 12, 14, 16};
  std::size_t gemv_rows{1024};
  int warmup{10};
  int rounds{3};
  int samples{5};
  double target_sample_ms{15.0};
  int decode_repeats{256};
  std::uint64_t base_seed{0x243f6a8885a308d3ULL};
  std::string output{"performance_samples.csv"};
  std::string profile_format{"all"};
  std::string profile_component{"dot"};
  int profile_lanes{1};
  std::size_t profile_n{std::size_t{1} << 27};
  std::size_t profile_m{1024};
  distribution profile_distribution{distribution::normal_0_1};
};

std::size_t parse_positive_size(const std::string &text,
                                const std::string &option) {
  std::size_t consumed{};
  const auto value = std::stoull(text, &consumed, 0);
  if (consumed != text.size() || value == 0) {
    throw benchmark_error(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(value);
}

int parse_positive_int(const std::string &text, const std::string &option) {
  const auto value = parse_positive_size(text, option);
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw benchmark_error(option + " is too large");
  }
  return static_cast<int>(value);
}

std::vector<int> parse_powers(const std::string &text,
                              const std::string &option) {
  std::vector<int> result;
  std::stringstream input{text};
  std::string token;
  while (std::getline(input, token, ',')) {
    const auto power = parse_positive_int(token, option);
    if (power > 30) {
      throw benchmark_error(option + " powers must be in [1,30]");
    }
    result.push_back(power);
  }
  if (result.empty()) {
    throw benchmark_error(option + " must not be empty");
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

options parse_options(int argc, char **argv) {
  options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument{argv[index]};
    auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw benchmark_error("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--mode") {
      const auto selected = value();
      if (selected == "sweep") {
        result.run_mode = mode::sweep;
      } else if (selected == "profile") {
        result.run_mode = mode::profile;
      } else {
        throw benchmark_error("--mode must be sweep or profile");
      }
    } else if (argument == "--dot-powers") {
      result.dot_powers = parse_powers(value(), argument);
    } else if (argument == "--gemv-powers") {
      result.gemv_powers = parse_powers(value(), argument);
    } else if (argument == "--gemv-rows") {
      result.gemv_rows = parse_positive_size(value(), argument);
    } else if (argument == "--warmup") {
      result.warmup = parse_positive_int(value(), argument);
    } else if (argument == "--rounds") {
      result.rounds = parse_positive_int(value(), argument);
    } else if (argument == "--samples") {
      result.samples = parse_positive_int(value(), argument);
    } else if (argument == "--target-sample-ms") {
      result.target_sample_ms = std::stod(value());
      if (!(result.target_sample_ms > 0.0)) {
        throw benchmark_error(argument + " must be positive");
      }
    } else if (argument == "--decode-repeats") {
      result.decode_repeats = parse_positive_int(value(), argument);
    } else if (argument == "--base-seed") {
      result.base_seed = parse_positive_size(value(), argument);
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--format") {
      result.profile_format = value();
    } else if (argument == "--component") {
      result.profile_component = value();
    } else if (argument == "--lanes") {
      result.profile_lanes = parse_positive_int(value(), argument);
    } else if (argument == "--n") {
      result.profile_n = parse_positive_size(value(), argument);
    } else if (argument == "--m") {
      result.profile_m = parse_positive_size(value(), argument);
    } else if (argument == "--distribution") {
      result.profile_distribution = parse_distribution(value());
    } else if (argument == "--help") {
      std::cout
          << "Usage: storage_performance_bench [options]\n"
          << "Sweep: --mode sweep --dot-powers P,... --gemv-powers P,...\n"
          << "       --gemv-rows M --warmup N --rounds N --samples N\n"
          << "       --target-sample-ms X --decode-repeats N --output FILE\n"
          << "Profile: --mode profile --component register_decode|stream_load|"
             "stream_decode|dot|gemv\n"
          << "         --format NAME|all --lanes 1|2|4 --n N --m M\n"
          << "         --distribution uniform_0_1|normal_0_1 --output FILE\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw benchmark_error("unknown argument: " + argument);
    }
  }
  if (result.profile_lanes != 1 && result.profile_lanes != 2 &&
      result.profile_lanes != 4) {
    throw benchmark_error("--lanes must be 1, 2, or 4");
  }
  return result;
}

std::size_t size_from_power(int power) { return std::size_t{1} << power; }

std::uint64_t mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

class random_generator {
public:
  random_generator() {
    CURAND_CHECK(
        curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  }
  ~random_generator() {
    if (generator_) {
      curandDestroyGenerator(generator_);
    }
  }
  void fill(double *values, std::size_t count, distribution kind,
            std::uint64_t seed) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, seed));
    CURAND_CHECK(curandSetGeneratorOffset(generator_, 0));
    if (kind == distribution::uniform_0_1) {
      CURAND_CHECK(curandGenerateUniformDouble(generator_, values, count));
    } else {
      if (count % 2 != 0) {
        throw benchmark_error("normal generation requires an even count");
      }
      CURAND_CHECK(
          curandGenerateNormalDouble(generator_, values, count, 0.0, 1.0));
    }
  }

private:
  curandGenerator_t generator_{};
};

template <typename Format>
__global__ void encode_values_kernel(const double *source,
                                     storage_type_t<Format> *encoded,
                                     std::size_t count) {
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto index = first; index < count; index += stride) {
    encoded[index] = encode<Format>(source[index]);
  }
}

template <typename Format>
void encode_values(const double *source, storage_type_t<Format> *encoded,
                   std::size_t count, int multiprocessors) {
  const auto wanted = (count + 255) / 256;
  const auto blocks = static_cast<unsigned>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
  encode_values_kernel<Format><<<blocks, 256>>>(source, encoded, count);
  CUDA_CHECK(cudaGetLastError());
}

int work_blocks(std::size_t count, int lanes, int multiprocessors) {
  const auto packs = (count + static_cast<std::size_t>(lanes) - 1) / lanes;
  const auto wanted = (packs + aut::performance::block_threads - 1) /
                      aut::performance::block_threads;
  return static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
}

struct device_info {
  std::string name;
  std::string capability;
  int multiprocessors{};
  double theoretical_hbm_gb_s{};
  double modeled_fp64_gflop_s{};
};

device_info query_device() {
  int device{};
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::ostringstream capability;
  capability << "sm_" << properties.major << properties.minor;
  const auto hbm = 2.0 * properties.memoryClockRate * 1000.0 *
                   properties.memoryBusWidth / 8.0 / 1.0e9;
  // Hopper has 64 scalar FP64 lanes per SM. This is a modeled clock-rate
  // ceiling; Nsight Compute supplies the empirical sustained ceiling.
  const auto fp64 = properties.multiProcessorCount * 64.0 * 2.0 *
                    properties.clockRate * 1000.0 / 1.0e9;
  return {properties.name, capability.str(), properties.multiProcessorCount,
          hbm, fp64};
}

struct work_model {
  std::string component;
  int lanes{};
  std::size_t n{};
  std::size_t m{};
  std::size_t leading_dimension{};
  int blocks{};
  int threads{aut::performance::block_threads};
  int decode_repeats{};
  double decoded_values{};
  double unique_storage_bytes{};
  double requested_storage_bytes{};
  double useful_flops{};
  double modeled_flops{};
};

struct timed_variant {
  work_model model;
  std::function<void()> launch;
  std::size_t iterations{1};
};

class sample_output {
public:
  sample_output(const std::string &path, const device_info &device)
      : stream_{path}, device_{device} {
    if (!stream_) {
      throw benchmark_error("could not create " + path);
    }
    stream_ << "gpu,compute_capability,distribution,format,storage_bits,"
               "component,lanes,n,m,leading_dimension,blocks,threads,"
               "decode_repeats,warmup,round,sample,order_position,iterations,"
               "total_time_ms,time_ms,decoded_values,unique_storage_bytes,"
               "requested_storage_bytes,useful_flops,modeled_flops,"
               "arithmetic_intensity_unique,arithmetic_intensity_requested,"
               "decoded_gvalues_per_s,unique_storage_gb_per_s,"
               "requested_storage_gb_per_s,useful_gflop_per_s,"
               "theoretical_hbm_gb_per_s,modeled_fp64_gflop_per_s\n";
    stream_ << std::setprecision(17);
  }

  void write(distribution kind, const char *format, int storage_bits,
             const work_model &model, int warmup, int round, int sample,
             int order_position, std::size_t iterations, double total_ms) {
    const auto time_ms = total_ms / iterations;
    const auto seconds = time_ms * 1.0e-3;
    const auto rate = [&](double amount, double scale) {
      return seconds > 0.0 ? amount / seconds / scale : 0.0;
    };
    const auto intensity = [](double flops, double bytes) {
      return bytes > 0.0 ? flops / bytes : 0.0;
    };
    stream_ << device_.name << ',' << device_.capability << ',' << name(kind)
            << ',' << format << ',' << storage_bits << ',' << model.component
            << ',' << model.lanes << ',' << model.n << ',' << model.m << ','
            << model.leading_dimension << ',' << model.blocks << ','
            << model.threads << ',' << model.decode_repeats << ',' << warmup
            << ',' << round << ',' << sample << ',' << order_position << ','
            << iterations << ',' << total_ms << ',' << time_ms << ','
            << model.decoded_values << ',' << model.unique_storage_bytes << ','
            << model.requested_storage_bytes << ',' << model.useful_flops << ','
            << model.modeled_flops << ','
            << intensity(model.useful_flops, model.unique_storage_bytes) << ','
            << intensity(model.useful_flops, model.requested_storage_bytes)
            << ',' << rate(model.decoded_values, 1.0e9) << ','
            << rate(model.unique_storage_bytes, 1.0e9) << ','
            << rate(model.requested_storage_bytes, 1.0e9) << ','
            << rate(model.useful_flops, 1.0e9) << ','
            << device_.theoretical_hbm_gb_s << ','
            << device_.modeled_fp64_gflop_s << '\n';
    stream_.flush();
  }

private:
  std::ofstream stream_;
  device_info device_;
};

void measure_variants(std::vector<timed_variant> &variants,
                      const options &settings, distribution kind,
                      const char *format, int storage_bits,
                      sample_output &output, event_timer &timer) {
  for (auto &variant : variants) {
    for (int warmup = 0; warmup < settings.warmup; ++warmup) {
      variant.launch();
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto &variant : variants) {
    const auto probe = std::max(0.0001f, timer.measure(variant.launch, 1));
    const auto wanted =
        static_cast<std::size_t>(std::ceil(settings.target_sample_ms / probe));
    variant.iterations =
        std::max<std::size_t>(1, std::min<std::size_t>(wanted, 1000000));
  }

  const auto count = static_cast<int>(variants.size());
  for (int round = 0; round < settings.rounds; ++round) {
    for (int sample = 0; sample < settings.samples; ++sample) {
      const auto offset = (round + sample) % count;
      for (int position = 0; position < count; ++position) {
        auto &variant = variants[(offset + position) % count];
        const auto elapsed = timer.measure(variant.launch, variant.iterations);
        output.write(kind, format, storage_bits, variant.model, settings.warmup,
                     round, sample, position, variant.iterations, elapsed);
      }
    }
  }
}

template <typename Format, int Lanes>
timed_variant make_register_variant(const storage_type_t<Format> *input,
                                    std::size_t count, double *sink, int blocks,
                                    int repeats) {
  const auto threads =
      static_cast<double>(blocks) * aut::performance::block_threads;
  work_model model{"register_decode",
                   Lanes,
                   count,
                   1,
                   count,
                   blocks,
                   aut::performance::block_threads,
                   repeats,
                   threads * Lanes * repeats,
                   threads * Lanes * sizeof(storage_type_t<Format>),
                   threads * Lanes * sizeof(storage_type_t<Format>),
                   threads * Lanes * repeats,
                   threads * Lanes * repeats};
  return {model, [=] {
            aut::performance::register_decode<Format, Lanes>
                <<<blocks, aut::performance::block_threads>>>(input, count,
                                                              repeats, sink);
          }};
}

template <typename Format, int Lanes>
timed_variant
make_stream_load_variant(const storage_type_t<Format> *input, std::size_t count,
                         unsigned long long *sink, int multiprocessors) {
  const auto blocks = work_blocks(count, Lanes, multiprocessors);
  work_model model{"stream_load",
                   Lanes,
                   count,
                   1,
                   count,
                   blocks,
                   aut::performance::block_threads,
                   0,
                   0.0,
                   static_cast<double>(count * sizeof(storage_type_t<Format>)),
                   static_cast<double>(count * sizeof(storage_type_t<Format>)),
                   0.0,
                   0.0};
  return {model, [=] {
            aut::performance::stream_load<Format, Lanes>
                <<<blocks, aut::performance::block_threads>>>(input, count,
                                                              sink);
          }};
}

template <typename Format, int Lanes>
timed_variant make_stream_decode_variant(const storage_type_t<Format> *input,
                                         std::size_t count, double *sink,
                                         int multiprocessors) {
  const auto blocks = work_blocks(count, Lanes, multiprocessors);
  const auto bytes =
      static_cast<double>(count * sizeof(storage_type_t<Format>));
  work_model model{"stream_decode",
                   Lanes,
                   count,
                   1,
                   count,
                   blocks,
                   aut::performance::block_threads,
                   0,
                   static_cast<double>(count),
                   bytes,
                   bytes,
                   static_cast<double>(count),
                   static_cast<double>(count)};
  return {model, [=] {
            aut::performance::stream_load_decode<Format, Lanes>
                <<<blocks, aut::performance::block_threads>>>(input, count,
                                                              sink);
          }};
}

template <typename Format, int Lanes>
timed_variant make_dot_variant(const storage_type_t<Format> *left,
                               const storage_type_t<Format> *right,
                               std::size_t count, double *partials,
                               double *result, int multiprocessors) {
  const auto blocks = work_blocks(count, Lanes, multiprocessors);
  const auto storage_bytes =
      static_cast<double>(2ULL * count * sizeof(storage_type_t<Format>));
  const auto overhead_bytes =
      static_cast<double>(2ULL * blocks * sizeof(double) + sizeof(double));
  const auto reduction_flops = static_cast<double>(blocks - 1);
  work_model model{"dot",
                   Lanes,
                   count,
                   1,
                   count,
                   blocks,
                   aut::performance::block_threads,
                   0,
                   static_cast<double>(2ULL * count),
                   storage_bytes + overhead_bytes,
                   storage_bytes + overhead_bytes,
                   static_cast<double>(2ULL * count),
                   static_cast<double>(2ULL * count) + reduction_flops};
  return {
      model, [=] {
        aut::kernels::storage_dot_map_reduce<Format, Lanes>
            <<<blocks, aut::kernels::reduction_block_threads>>>(
                left, right, count, partials);
        aut::kernels::
            storage_dot_finalize<<<1, aut::kernels::reduction_block_threads>>>(
                partials, static_cast<std::size_t>(blocks), result);
      }};
}

template <typename Format, int Lanes>
timed_variant make_gemv_variant(const storage_type_t<Format> *matrix,
                                const storage_type_t<Format> *vector,
                                std::size_t rows, std::size_t columns,
                                std::size_t leading_dimension, double *result) {
  const auto scalar_bytes = sizeof(storage_type_t<Format>);
  const auto unique = static_cast<double>(
      (rows * columns + columns) * scalar_bytes + rows * sizeof(double));
  const auto requested = static_cast<double>(
      2ULL * rows * columns * scalar_bytes + rows * sizeof(double));
  const auto useful = static_cast<double>(2ULL * rows * columns);
  work_model model{"gemv",
                   Lanes,
                   columns,
                   rows,
                   leading_dimension,
                   static_cast<int>(rows),
                   aut::performance::block_threads,
                   0,
                   static_cast<double>(2ULL * rows * columns),
                   unique,
                   requested,
                   useful,
                   useful + rows * 255.0};
  return {model, [=] {
            aut::kernels::storage_gemv<Format, Lanes>
                <<<static_cast<unsigned>(rows),
                   aut::kernels::reduction_block_threads>>>(
                    matrix, vector, rows, columns, leading_dimension, result);
          }};
}

template <typename Format>
void run_format_sweep(const options &settings, const device_info &device,
                      distribution kind, const double *source_dot_left,
                      const double *source_dot_right,
                      const double *source_matrix, const double *source_vector,
                      sample_output &output, event_timer &timer) {
  const auto max_dot = size_from_power(settings.dot_powers.back());
  const auto max_columns = size_from_power(settings.gemv_powers.back());
  const auto max_matrix = settings.gemv_rows * max_columns;
  device_buffer<storage_type_t<Format>> dot_left{max_dot};
  device_buffer<storage_type_t<Format>> dot_right{max_dot};
  device_buffer<storage_type_t<Format>> matrix{max_matrix};
  device_buffer<storage_type_t<Format>> vector{max_columns};
  const auto max_blocks = device.multiprocessors * 16;
  const auto register_blocks = device.multiprocessors * 4;
  device_buffer<double> double_sink{std::max<std::size_t>(
      settings.gemv_rows, static_cast<std::size_t>(register_blocks) *
                              aut::performance::block_threads)};
  device_buffer<unsigned long long> integer_sink{max_blocks};
  device_buffer<double> partials{max_blocks};
  device_buffer<double> dot_result{1};

  encode_values<Format>(source_dot_left, dot_left.get(), max_dot,
                        device.multiprocessors);
  encode_values<Format>(source_dot_right, dot_right.get(), max_dot,
                        device.multiprocessors);
  CUDA_CHECK(cudaDeviceSynchronize());

  {
    std::vector<timed_variant> variants;
    variants.push_back(make_register_variant<Format, 1>(
        dot_left.get(), max_dot, double_sink.get(), register_blocks,
        settings.decode_repeats));
    variants.push_back(make_register_variant<Format, 2>(
        dot_left.get(), max_dot, double_sink.get(), register_blocks,
        settings.decode_repeats));
    variants.push_back(make_register_variant<Format, 4>(
        dot_left.get(), max_dot, double_sink.get(), register_blocks,
        settings.decode_repeats));
    measure_variants(variants, settings, kind, Format::name, Format::total_bits,
                     output, timer);
  }

  for (const auto power : settings.dot_powers) {
    const auto count = size_from_power(power);
    std::vector<timed_variant> loads;
    loads.push_back(make_stream_load_variant<Format, 1>(
        dot_left.get(), count, integer_sink.get(), device.multiprocessors));
    loads.push_back(make_stream_load_variant<Format, 2>(
        dot_left.get(), count, integer_sink.get(), device.multiprocessors));
    loads.push_back(make_stream_load_variant<Format, 4>(
        dot_left.get(), count, integer_sink.get(), device.multiprocessors));
    measure_variants(loads, settings, kind, Format::name, Format::total_bits,
                     output, timer);

    std::vector<timed_variant> decodes;
    decodes.push_back(make_stream_decode_variant<Format, 1>(
        dot_left.get(), count, double_sink.get(), device.multiprocessors));
    decodes.push_back(make_stream_decode_variant<Format, 2>(
        dot_left.get(), count, double_sink.get(), device.multiprocessors));
    decodes.push_back(make_stream_decode_variant<Format, 4>(
        dot_left.get(), count, double_sink.get(), device.multiprocessors));
    measure_variants(decodes, settings, kind, Format::name, Format::total_bits,
                     output, timer);

    std::vector<timed_variant> dots;
    dots.push_back(make_dot_variant<Format, 1>(
        dot_left.get(), dot_right.get(), count, partials.get(),
        dot_result.get(), device.multiprocessors));
    dots.push_back(make_dot_variant<Format, 2>(
        dot_left.get(), dot_right.get(), count, partials.get(),
        dot_result.get(), device.multiprocessors));
    dots.push_back(make_dot_variant<Format, 4>(
        dot_left.get(), dot_right.get(), count, partials.get(),
        dot_result.get(), device.multiprocessors));
    measure_variants(dots, settings, kind, Format::name, Format::total_bits,
                     output, timer);
  }

  for (const auto power : settings.gemv_powers) {
    const auto columns = size_from_power(power);
    const auto matrix_count = settings.gemv_rows * columns;
    encode_values<Format>(source_matrix, matrix.get(), matrix_count,
                          device.multiprocessors);
    encode_values<Format>(source_vector, vector.get(), columns,
                          device.multiprocessors);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<timed_variant> gemvs;
    gemvs.push_back(make_gemv_variant<Format, 1>(matrix.get(), vector.get(),
                                                 settings.gemv_rows, columns,
                                                 columns, double_sink.get()));
    gemvs.push_back(make_gemv_variant<Format, 2>(matrix.get(), vector.get(),
                                                 settings.gemv_rows, columns,
                                                 columns, double_sink.get()));
    gemvs.push_back(make_gemv_variant<Format, 4>(matrix.get(), vector.get(),
                                                 settings.gemv_rows, columns,
                                                 columns, double_sink.get()));
    measure_variants(gemvs, settings, kind, Format::name, Format::total_bits,
                     output, timer);
  }

  std::cout << "  " << std::left << std::setw(14) << Format::name
            << " complete\n";
}

template <typename... Formats>
void run_sweep_formats(const options &settings, const device_info &device,
                       distribution kind, const double *source_dot_left,
                       const double *source_dot_right,
                       const double *source_matrix, const double *source_vector,
                       sample_output &output, event_timer &timer) {
  (run_format_sweep<Formats>(settings, device, kind, source_dot_left,
                             source_dot_right, source_matrix, source_vector,
                             output, timer),
   ...);
}

void run_sweep(const options &settings, const device_info &device) {
  const auto max_dot = size_from_power(settings.dot_powers.back());
  const auto max_columns = size_from_power(settings.gemv_powers.back());
  const auto max_matrix = settings.gemv_rows * max_columns;
  device_buffer<double> source_dot_left{max_dot};
  device_buffer<double> source_dot_right{max_dot};
  device_buffer<double> source_matrix{max_matrix};
  device_buffer<double> source_vector{max_columns};
  random_generator generator;
  sample_output output{settings.output, device};
  event_timer timer;

  for (const auto kind :
       {distribution::uniform_0_1, distribution::normal_0_1}) {
    const auto tag = static_cast<std::uint64_t>(kind);
    generator.fill(source_dot_left.get(), max_dot, kind,
                   mix(settings.base_seed ^ tag ^ 0x444f544cULL));
    generator.fill(source_dot_right.get(), max_dot, kind,
                   mix(settings.base_seed ^ tag ^ 0x444f5452ULL));
    generator.fill(source_matrix.get(), max_matrix, kind,
                   mix(settings.base_seed ^ tag ^ 0x47454d41ULL));
    generator.fill(source_vector.get(), max_columns, kind,
                   mix(settings.base_seed ^ tag ^ 0x47454d58ULL));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << name(kind) << '\n';
    run_sweep_formats<e1m6, e2m5, e3m4, fp8_e4m3, fp8_e5m2, e1m14, e2m13, e3m12,
                      fp16_e5m10, bf16_e8m7, e11m4, e1m30, e2m29, e3m28,
                      fp32_e8m23, e11m20, fp64_e11m52>(
        settings, device, kind, source_dot_left.get(), source_dot_right.get(),
        source_matrix.get(), source_vector.get(), output, timer);
  }
}

class profile_output {
public:
  explicit profile_output(const std::string &path) : stream_{path} {
    if (!stream_) {
      throw benchmark_error("could not create " + path);
    }
    stream_ << "profile_sequence,distribution,format,storage_bits,component,"
               "lanes,n,m,leading_dimension,kernel_count,decoded_values,"
               "unique_storage_bytes,requested_storage_bytes,useful_flops,"
               "modeled_flops,timing_status\n";
    stream_ << std::setprecision(17);
  }
  void write(std::size_t sequence, distribution kind, const char *format,
             int storage_bits, const work_model &model, int kernel_count) {
    stream_ << sequence << ',' << name(kind) << ',' << format << ','
            << storage_bits << ',' << model.component << ',' << model.lanes
            << ',' << model.n << ',' << model.m << ','
            << model.leading_dimension << ',' << kernel_count << ','
            << model.decoded_values << ',' << model.unique_storage_bytes << ','
            << model.requested_storage_bytes << ',' << model.useful_flops << ','
            << model.modeled_flops << ",profiler_contaminated\n";
    stream_.flush();
  }

private:
  std::ofstream stream_;
};

template <typename Format, int Lanes>
void profile_format_lane(const options &settings, const device_info &device,
                         const double *source_a, const double *source_b,
                         profile_output &output, std::size_t sequence) {
  const auto component = settings.profile_component;
  const auto count = component == "gemv"
                         ? settings.profile_m * settings.profile_n
                         : settings.profile_n;
  device_buffer<storage_type_t<Format>> encoded_a{count};
  device_buffer<storage_type_t<Format>> encoded_b{component == "dot" ? count
                                                  : component == "gemv"
                                                      ? settings.profile_n
                                                      : 0};
  encode_values<Format>(source_a, encoded_a.get(), count,
                        device.multiprocessors);
  if (component == "dot") {
    encode_values<Format>(source_b, encoded_b.get(), count,
                          device.multiprocessors);
  } else if (component == "gemv") {
    encode_values<Format>(source_b, encoded_b.get(), settings.profile_n,
                          device.multiprocessors);
  }
  const auto register_blocks = device.multiprocessors * 4;
  device_buffer<double> double_sink{std::max<std::size_t>(
      settings.profile_m, static_cast<std::size_t>(register_blocks) *
                              aut::performance::block_threads)};
  device_buffer<unsigned long long> integer_sink{device.multiprocessors * 16};
  device_buffer<double> partials{device.multiprocessors * 16};
  device_buffer<double> result{1};
  timed_variant variant;
  int kernel_count{1};
  if (component == "register_decode") {
    variant = make_register_variant<Format, Lanes>(
        encoded_a.get(), settings.profile_n, double_sink.get(), register_blocks,
        settings.decode_repeats);
  } else if (component == "stream_load") {
    variant = make_stream_load_variant<Format, Lanes>(
        encoded_a.get(), settings.profile_n, integer_sink.get(),
        device.multiprocessors);
  } else if (component == "stream_decode") {
    variant = make_stream_decode_variant<Format, Lanes>(
        encoded_a.get(), settings.profile_n, double_sink.get(),
        device.multiprocessors);
  } else if (component == "dot") {
    variant = make_dot_variant<Format, Lanes>(
        encoded_a.get(), encoded_b.get(), settings.profile_n, partials.get(),
        result.get(), device.multiprocessors);
    kernel_count = 2;
  } else if (component == "gemv") {
    variant = make_gemv_variant<Format, Lanes>(
        encoded_a.get(), encoded_b.get(), settings.profile_m,
        settings.profile_n, settings.profile_n, double_sink.get());
  } else {
    throw benchmark_error("unknown profile component: " + component);
  }
  for (int warmup = 0; warmup < settings.warmup; ++warmup) {
    variant.launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  output.write(sequence, settings.profile_distribution, Format::name,
               Format::total_bits, variant.model, kernel_count);
  CUDA_CHECK(cudaProfilerStart());
  variant.launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaProfilerStop());
  std::cout << "  profiled " << Format::name << '\n';
}

template <typename Format>
bool try_profile_format(const options &settings, const device_info &device,
                        const double *source_a, const double *source_b,
                        profile_output &output, std::size_t &sequence) {
  if (settings.profile_format != "all" &&
      settings.profile_format != Format::name) {
    return false;
  }
  if (settings.profile_lanes == 1) {
    profile_format_lane<Format, 1>(settings, device, source_a, source_b, output,
                                   sequence++);
  } else if (settings.profile_lanes == 2) {
    profile_format_lane<Format, 2>(settings, device, source_a, source_b, output,
                                   sequence++);
  } else {
    profile_format_lane<Format, 4>(settings, device, source_a, source_b, output,
                                   sequence++);
  }
  return true;
}

template <typename... Formats>
void profile_formats(const options &settings, const device_info &device,
                     const double *source_a, const double *source_b,
                     profile_output &output) {
  std::size_t sequence{};
  const std::array<bool, sizeof...(Formats)> matches{
      try_profile_format<Formats>(settings, device, source_a, source_b, output,
                                  sequence)...};
  const auto matched = std::any_of(matches.begin(), matches.end(),
                                   [](bool value) { return value; });
  if (!matched) {
    throw benchmark_error("unknown profile format: " + settings.profile_format);
  }
}

void run_profile(const options &settings, const device_info &device) {
  const auto count = settings.profile_component == "gemv"
                         ? settings.profile_m * settings.profile_n
                         : settings.profile_n;
  const auto second_count = settings.profile_component == "dot" ? count
                            : settings.profile_component == "gemv"
                                ? settings.profile_n
                                : 0;
  device_buffer<double> source_a{count};
  device_buffer<double> source_b{second_count};
  random_generator generator;
  generator.fill(source_a.get(), count, settings.profile_distribution,
                 mix(settings.base_seed ^ 0x50524f4641ULL));
  if (second_count != 0) {
    generator.fill(source_b.get(), second_count, settings.profile_distribution,
                   mix(settings.base_seed ^ 0x50524f4642ULL));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  profile_output output{settings.output};
  profile_formats<e1m6, e2m5, e3m4, fp8_e4m3, fp8_e5m2, e1m14, e2m13, e3m12,
                  fp16_e5m10, bf16_e8m7, e11m4, e1m30, e2m29, e3m28, fp32_e8m23,
                  e11m20, fp64_e11m52>(settings, device, source_a.get(),
                                       source_b.get(), output);
}

} // namespace

int main(int argc, char **argv) try {
  const auto settings = parse_options(argc, argv);
  const auto device = query_device();
  if (device.capability != "sm_90") {
    throw benchmark_error("this experiment is calibrated for an sm_90 H200");
  }
  const auto parent = std::filesystem::path{settings.output}.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::cout << "Storage performance benchmark\n"
            << "GPU: " << device.name << " (" << device.capability << ")\n"
            << "Mode: "
            << (settings.run_mode == mode::sweep ? "sweep" : "profile") << "\n";
  if (settings.run_mode == mode::sweep) {
    run_sweep(settings, device);
  } else {
    run_profile(settings, device);
  }
  std::cout << "Wrote " << settings.output << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return EXIT_FAILURE;
}
