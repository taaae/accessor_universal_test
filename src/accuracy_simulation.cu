#include "accuracy_simulation.cuh"
#include "accuracy_statistics.hpp"

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
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace aut::accuracy;
using namespace aut::accuracy::statistics;
using namespace aut::storage;

class simulation_error : public std::runtime_error {
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
    throw simulation_error(message.str());
  }
}

void check_curand(curandStatus_t status, const char *expression,
                  const char *file, int line) {
  if (status != CURAND_STATUS_SUCCESS) {
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line
            << " with cuRAND status " << static_cast<int>(status);
    throw simulation_error(message.str());
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

enum class distribution { uniform_0_1, normal_0_1 };

const char *name(distribution value) {
  return value == distribution::uniform_0_1 ? "uniform_0_1" : "normal_0_1";
}

struct options {
  std::vector<int> dot_powers{10, 14, 18, 22};
  std::vector<int> gemv_powers{8, 10, 12, 14, 16};
  std::size_t dot_samples{8192};
  std::size_t dot_statistical_batches{32};
  std::size_t gemv_rows{1024};
  std::size_t gemv_replicates{16};
  std::uint64_t base_seed{0x243f6a8885a308d3ULL};
  double workspace_gib{12.0};
  std::string output_dir{"accuracy_simulation"};
};

std::size_t parse_positive_size(const std::string &text,
                                const std::string &option) {
  std::size_t consumed{};
  const auto value = std::stoull(text, &consumed, 0);
  if (consumed != text.size() || value == 0) {
    throw simulation_error(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(value);
}

std::vector<int> parse_powers(const std::string &text,
                              const std::string &option) {
  std::vector<int> result;
  std::stringstream input{text};
  std::string token;
  while (std::getline(input, token, ',')) {
    std::size_t consumed{};
    const auto power = std::stoi(token, &consumed);
    if (consumed != token.size() || power < 1 || power > 40) {
      throw simulation_error(option +
                             " requires comma-separated powers in [1,40]");
    }
    result.push_back(power);
  }
  if (result.empty()) {
    throw simulation_error(option + " must not be empty");
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
        throw simulation_error("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--dot-powers") {
      result.dot_powers = parse_powers(value(), argument);
    } else if (argument == "--gemv-powers") {
      result.gemv_powers = parse_powers(value(), argument);
    } else if (argument == "--dot-samples") {
      result.dot_samples = parse_positive_size(value(), argument);
    } else if (argument == "--dot-statistical-batches") {
      result.dot_statistical_batches = parse_positive_size(value(), argument);
    } else if (argument == "--gemv-rows") {
      result.gemv_rows = parse_positive_size(value(), argument);
    } else if (argument == "--gemv-replicates") {
      result.gemv_replicates = parse_positive_size(value(), argument);
    } else if (argument == "--base-seed") {
      result.base_seed = parse_positive_size(value(), argument);
    } else if (argument == "--workspace-gib") {
      result.workspace_gib = std::stod(value());
      if (!(result.workspace_gib > 0.0)) {
        throw simulation_error(argument + " must be positive");
      }
    } else if (argument == "--output-dir") {
      result.output_dir = value();
    } else if (argument == "--help") {
      std::cout
          << "Usage: accuracy_simulation [options]\n"
          << "  --dot-powers P,...             default 10,14,18,22\n"
          << "  --gemv-powers P,...            default 8,10,12,14,16\n"
          << "  --dot-samples N                default 8192 per case\n"
          << "  --dot-statistical-batches N    default 32\n"
          << "  --gemv-rows M                  default 1024\n"
          << "  --gemv-replicates N            default 16\n"
          << "  --base-seed N                  deterministic seed\n"
          << "  --workspace-gib X              DOT chunk budget, default 12\n"
          << "  --output-dir PATH              CSV output directory\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw simulation_error("unknown argument: " + argument);
    }
  }
  result.dot_statistical_batches =
      std::min(result.dot_statistical_batches, result.dot_samples);
  return result;
}

std::size_t size_from_power(int power) { return std::size_t{1} << power; }

std::string join_powers(const std::vector<int> &powers) {
  std::ostringstream result;
  for (std::size_t index = 0; index < powers.size(); ++index) {
    if (index != 0) {
      result << ',';
    }
    result << powers[index];
  }
  return result.str();
}

std::uint64_t mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

std::uint64_t derive_seed(std::uint64_t base, std::uint64_t kernel_tag,
                          distribution kind, std::size_t n, std::size_t group,
                          std::uint64_t operand_tag) {
  auto result = mix(base ^ kernel_tag);
  result = mix(result ^ static_cast<std::uint64_t>(kind));
  result = mix(result ^ static_cast<std::uint64_t>(n));
  result = mix(result ^ static_cast<std::uint64_t>(group));
  return mix(result ^ operand_tag);
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
  random_generator(const random_generator &) = delete;
  random_generator &operator=(const random_generator &) = delete;

  void fill(double *values, std::size_t count, distribution kind,
            std::uint64_t seed) {
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, seed));
    CURAND_CHECK(curandSetGeneratorOffset(generator_, 0));
    if (kind == distribution::uniform_0_1) {
      CURAND_CHECK(curandGenerateUniformDouble(generator_, values, count));
    } else {
      if (count % 2 != 0) {
        throw simulation_error("normal generation count must be even");
      }
      CURAND_CHECK(
          curandGenerateNormalDouble(generator_, values, count, 0.0, 1.0));
    }
  }

private:
  curandGenerator_t generator_{};
};

long double extended(double_double value) {
  if (!std::isfinite(value.hi)) {
    return static_cast<long double>(value.hi);
  }
  return static_cast<long double>(value.hi) +
         static_cast<long double>(value.lo);
}

template <typename Format> struct format_saturation {
  static constexpr bool enabled = false;
  static double threshold() { return std::numeric_limits<double>::infinity(); }
};

template <> struct format_saturation<e1m6> {
  static constexpr bool enabled = true;
  static double threshold() { return std::ldexp(2.0 - std::ldexp(1.0, -7), 1); }
};
template <> struct format_saturation<e1m14> {
  static constexpr bool enabled = true;
  static double threshold() {
    return std::ldexp(2.0 - std::ldexp(1.0, -15), 1);
  }
};
template <> struct format_saturation<e1m30> {
  static constexpr bool enabled = true;
  static double threshold() {
    return std::ldexp(2.0 - std::ldexp(1.0, -31), 1);
  }
};
template <> struct format_saturation<fp8_e4m3> {
  static constexpr bool enabled = true;
  static double threshold() { return 464.0; }
};
template <> struct format_saturation<fp8_e5m2> {
  static constexpr bool enabled = true;
  static double threshold() { return 61440.0; }
};

encoding_counts add_counts(encoding_counts left, encoding_counts right) {
  return {left.zeros + right.zeros, left.infinities + right.infinities,
          left.nans + right.nans, left.saturations + right.saturations};
}

template <typename Format>
encoding_counts encode_values(const double *source,
                              storage_type_t<Format> *encoded,
                              std::size_t count, int multiprocessors) {
  device_buffer<encoding_counts> device_counts{1};
  CUDA_CHECK(cudaMemset(device_counts.get(), 0, sizeof(encoding_counts)));
  const auto wanted = (count + 255) / 256;
  const auto blocks = static_cast<unsigned>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
  encode_and_count_kernel<Format><<<blocks, 256>>>(
      source, encoded, count, format_saturation<Format>::enabled,
      format_saturation<Format>::threshold(), device_counts.get());
  CUDA_CHECK(cudaGetLastError());
  encoding_counts result{};
  CUDA_CHECK(cudaMemcpy(&result, device_counts.get(), sizeof(result),
                        cudaMemcpyDeviceToHost));
  return result;
}

std::vector<reference_pair> source_batched_references(const double *left,
                                                      const double *right,
                                                      std::size_t n,
                                                      std::size_t samples) {
  device_buffer<reference_pair> device_result{samples};
  source_batched_reference_kernel<<<static_cast<unsigned>(samples), 256>>>(
      left, right, n, samples, device_result.get());
  CUDA_CHECK(cudaGetLastError());
  std::vector<reference_pair> result(samples);
  CUDA_CHECK(cudaMemcpy(result.data(), device_result.get(),
                        samples * sizeof(reference_pair),
                        cudaMemcpyDeviceToHost));
  return result;
}

template <typename Format>
std::vector<reference_pair>
storage_batched_references(const storage_type_t<Format> *left,
                           const storage_type_t<Format> *right, std::size_t n,
                           std::size_t samples) {
  device_buffer<reference_pair> device_result{samples};
  storage_batched_reference_kernel<Format>
      <<<static_cast<unsigned>(samples), 256>>>(left, right, n, samples,
                                                device_result.get());
  CUDA_CHECK(cudaGetLastError());
  std::vector<reference_pair> result(samples);
  CUDA_CHECK(cudaMemcpy(result.data(), device_result.get(),
                        samples * sizeof(reference_pair),
                        cudaMemcpyDeviceToHost));
  return result;
}

int first_stage_blocks(std::size_t count, int lanes, int multiprocessors) {
  const auto packs = (count + static_cast<std::size_t>(lanes) - 1) / lanes;
  const auto wanted = (packs + aut::kernels::reduction_block_threads - 1) /
                      aut::kernels::reduction_block_threads;
  return static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(wanted, multiprocessors * 16ULL)));
}

template <typename Format, int Lanes>
std::vector<double> run_batched_dot(const storage_type_t<Format> *left,
                                    const storage_type_t<Format> *right,
                                    std::size_t n, std::size_t samples,
                                    int multiprocessors) {
  const auto blocks = first_stage_blocks(n, Lanes, multiprocessors);
  device_buffer<double> partials{samples * static_cast<std::size_t>(blocks)};
  device_buffer<double> result{samples};
  const dim3 grid(static_cast<unsigned>(blocks),
                  static_cast<unsigned>(samples));
  storage_dot_map_reduce_batched<Format, Lanes>
      <<<grid, aut::kernels::reduction_block_threads>>>(left, right, n, samples,
                                                        partials.get());
  CUDA_CHECK(cudaGetLastError());
  storage_dot_finalize_batched<<<static_cast<unsigned>(samples),
                                 aut::kernels::reduction_block_threads>>>(
      partials.get(), static_cast<std::size_t>(blocks), samples, result.get());
  CUDA_CHECK(cudaGetLastError());
  std::vector<double> host(samples);
  CUDA_CHECK(cudaMemcpy(host.data(), result.get(), samples * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return host;
}

std::vector<reference_pair> source_gemv_references(const double *matrix,
                                                   const double *vector,
                                                   std::size_t rows,
                                                   std::size_t columns) {
  device_buffer<reference_pair> device_result{rows};
  source_gemv_reference_kernel<<<static_cast<unsigned>(rows), 256>>>(
      matrix, vector, rows, columns, device_result.get());
  CUDA_CHECK(cudaGetLastError());
  std::vector<reference_pair> result(rows);
  CUDA_CHECK(cudaMemcpy(result.data(), device_result.get(),
                        rows * sizeof(reference_pair), cudaMemcpyDeviceToHost));
  return result;
}

template <typename Format>
std::vector<reference_pair>
storage_gemv_references(const storage_type_t<Format> *matrix,
                        const storage_type_t<Format> *vector, std::size_t rows,
                        std::size_t columns) {
  device_buffer<reference_pair> device_result{rows};
  storage_gemv_reference_kernel<Format><<<static_cast<unsigned>(rows), 256>>>(
      matrix, vector, rows, columns, device_result.get());
  CUDA_CHECK(cudaGetLastError());
  std::vector<reference_pair> result(rows);
  CUDA_CHECK(cudaMemcpy(result.data(), device_result.get(),
                        rows * sizeof(reference_pair), cudaMemcpyDeviceToHost));
  return result;
}

template <typename Format, int Lanes>
std::vector<double> run_gemv(const storage_type_t<Format> *matrix,
                             const storage_type_t<Format> *vector,
                             std::size_t rows, std::size_t columns) {
  device_buffer<double> result{rows};
  aut::kernels::storage_gemv<Format, Lanes>
      <<<static_cast<unsigned>(rows), aut::kernels::reduction_block_threads>>>(
          matrix, vector, rows, columns, columns, result.get());
  CUDA_CHECK(cudaGetLastError());
  std::vector<double> host(rows);
  CUDA_CHECK(cudaMemcpy(host.data(), result.get(), rows * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return host;
}

struct output_files {
  explicit output_files(const std::filesystem::path &directory)
      : summary{directory / "simulation_summary.csv"},
        quantiles{directory / "empirical_quantiles.csv"},
        batches{directory / "batch_estimates.csv"},
        encodings{directory / "encoding_stats.csv"},
        seeds{directory / "generation_seeds.csv"},
        self_test{directory / "reference_self_test.csv"} {
    if (!summary || !quantiles || !batches || !encodings || !seeds ||
        !self_test) {
      throw simulation_error("could not create simulation CSV files");
    }
    write_headers(summary, quantiles, batches);
    encodings
        << "kernel,distribution,format,storage_bits,n,m,operand,source_values,"
           "decoded_zeros,decoded_infinities,decoded_nans,saturated_values\n";
    seeds << "kernel,distribution,n,m,group_start,group_count,operand_a_seed,"
             "operand_b_seed\n";
    self_test << "reference,n,source_reference,host_long_double_reference,"
                 "sum_abs,normalized_difference,status\n";
    summary << std::setprecision(21);
    quantiles << std::setprecision(21);
    batches << std::setprecision(21);
    encodings << std::setprecision(21);
    seeds << std::setprecision(21);
    self_test << std::setprecision(21);
  }

  void flush_data() {
    summary.flush();
    quantiles.flush();
    batches.flush();
    encodings.flush();
    seeds.flush();
    self_test.flush();
  }

  std::ofstream summary;
  std::ofstream quantiles;
  std::ofstream batches;
  std::ofstream encodings;
  std::ofstream seeds;
  std::ofstream self_test;
};

void write_encoding_row(std::ostream &output, const case_identity &identity,
                        const char *operand, std::size_t source_values,
                        const encoding_counts &counts) {
  output << identity.kernel << ',' << identity.distribution << ','
         << identity.format << ',' << identity.storage_bits << ',' << identity.n
         << ',' << identity.m << ',' << operand << ',' << source_values << ','
         << counts.zeros << ',' << counts.infinities << ',' << counts.nans
         << ',' << counts.saturations << '\n';
}

std::size_t dot_chunk_capacity(const options &settings, std::size_t n,
                               std::size_t workspace_bytes,
                               int multiprocessors) {
  const auto partial_bytes =
      static_cast<std::size_t>(multiprocessors * 16) * sizeof(double);
  const auto bytes_per_sample = 32ULL * n + partial_bytes + 256;
  return std::max<std::size_t>(
      1, std::min<std::size_t>({settings.dot_samples,
                                workspace_bytes / bytes_per_sample,
                                std::size_t{65535}}));
}

template <typename Format>
void run_dot_cases(const options &settings, std::size_t workspace_bytes,
                   int multiprocessors, random_generator &generator,
                   output_files &output) {
  for (const auto kind :
       {distribution::uniform_0_1, distribution::normal_0_1}) {
    for (const auto power : settings.dot_powers) {
      const auto n = size_from_power(power);
      const auto chunk_capacity =
          dot_chunk_capacity(settings, n, workspace_bytes, multiprocessors);
      case_accumulator accumulated{settings.dot_statistical_batches};
      encoding_counts left_counts{};
      encoding_counts right_counts{};
      std::size_t encoded_values{};

      for (std::size_t start = 0; start < settings.dot_samples;
           start += chunk_capacity) {
        const auto samples =
            std::min(chunk_capacity, settings.dot_samples - start);
        const auto elements = samples * n;
        device_buffer<double> source_left{elements};
        device_buffer<double> source_right{elements};
        device_buffer<storage_type_t<Format>> encoded_left{elements};
        device_buffer<storage_type_t<Format>> encoded_right{elements};
        const auto left_seed = derive_seed(settings.base_seed, 0x444f54ULL,
                                           kind, n, start, 0x4cULL);
        const auto right_seed = derive_seed(settings.base_seed, 0x444f54ULL,
                                            kind, n, start, 0x52ULL);
        generator.fill(source_left.get(), elements, kind, left_seed);
        generator.fill(source_right.get(), elements, kind, right_seed);
        const auto source_reference = source_batched_references(
            source_left.get(), source_right.get(), n, samples);
        left_counts = add_counts(
            left_counts,
            encode_values<Format>(source_left.get(), encoded_left.get(),
                                  elements, multiprocessors));
        right_counts = add_counts(
            right_counts,
            encode_values<Format>(source_right.get(), encoded_right.get(),
                                  elements, multiprocessors));
        encoded_values += elements;
        const auto storage_reference = storage_batched_references<Format>(
            encoded_left.get(), encoded_right.get(), n, samples);
        const auto gpu_x1 =
            run_batched_dot<Format, 1>(encoded_left.get(), encoded_right.get(),
                                       n, samples, multiprocessors);
        const auto gpu_x2 =
            run_batched_dot<Format, 2>(encoded_left.get(), encoded_right.get(),
                                       n, samples, multiprocessors);
        const auto gpu_x4 =
            run_batched_dot<Format, 4>(encoded_left.get(), encoded_right.get(),
                                       n, samples, multiprocessors);

        for (std::size_t sample = 0; sample < samples; ++sample) {
          const auto global_sample = start + sample;
          const auto batch =
              std::min(settings.dot_statistical_batches - 1,
                       global_sample * settings.dot_statistical_batches /
                           settings.dot_samples);
          accumulated.add(extended(source_reference[sample].value),
                          extended(source_reference[sample].sum_abs),
                          extended(storage_reference[sample].value),
                          extended(storage_reference[sample].sum_abs),
                          {gpu_x1[sample], gpu_x2[sample], gpu_x4[sample]},
                          batch);
        }

        if constexpr (std::is_same_v<Format, e1m6>) {
          output.seeds << "dot," << name(kind) << ',' << n << ",1," << start
                       << ',' << samples << ',' << left_seed << ','
                       << right_seed << '\n';
        }
      }

      const case_identity identity{
          "dot", name(kind), Format::name, Format::total_bits, n, 1};
      accumulated.write(output.summary, output.quantiles, output.batches,
                        identity);
      write_encoding_row(output.encodings, identity, "left", encoded_values,
                         left_counts);
      write_encoding_row(output.encodings, identity, "right", encoded_values,
                         right_counts);
      output.flush_data();
      std::cout << "DOT " << name(kind) << ' ' << Format::name << " N=2^"
                << power << " samples=" << settings.dot_samples
                << " chunk=" << chunk_capacity << '\n';
    }
  }
}

template <typename Format>
void run_gemv_cases(const options &settings, int multiprocessors,
                    random_generator &generator, output_files &output) {
  for (const auto kind :
       {distribution::uniform_0_1, distribution::normal_0_1}) {
    for (const auto power : settings.gemv_powers) {
      const auto n = size_from_power(power);
      const auto matrix_elements = settings.gemv_rows * n;
      case_accumulator accumulated{settings.gemv_replicates};
      encoding_counts matrix_counts{};
      encoding_counts vector_counts{};

      device_buffer<double> source_matrix{matrix_elements};
      device_buffer<double> source_vector{n};
      device_buffer<storage_type_t<Format>> encoded_matrix{matrix_elements};
      device_buffer<storage_type_t<Format>> encoded_vector{n};
      for (std::size_t replicate = 0; replicate < settings.gemv_replicates;
           ++replicate) {
        const auto matrix_seed = derive_seed(settings.base_seed, 0x47454d56ULL,
                                             kind, n, replicate, 0x41ULL);
        const auto vector_seed = derive_seed(settings.base_seed, 0x47454d56ULL,
                                             kind, n, replicate, 0x58ULL);
        generator.fill(source_matrix.get(), matrix_elements, kind, matrix_seed);
        generator.fill(source_vector.get(), n, kind, vector_seed);
        const auto source_reference = source_gemv_references(
            source_matrix.get(), source_vector.get(), settings.gemv_rows, n);
        matrix_counts = add_counts(
            matrix_counts,
            encode_values<Format>(source_matrix.get(), encoded_matrix.get(),
                                  matrix_elements, multiprocessors));
        vector_counts = add_counts(vector_counts,
                                   encode_values<Format>(source_vector.get(),
                                                         encoded_vector.get(),
                                                         n, multiprocessors));
        const auto storage_reference = storage_gemv_references<Format>(
            encoded_matrix.get(), encoded_vector.get(), settings.gemv_rows, n);
        const auto gpu_x1 = run_gemv<Format, 1>(
            encoded_matrix.get(), encoded_vector.get(), settings.gemv_rows, n);
        const auto gpu_x2 = run_gemv<Format, 2>(
            encoded_matrix.get(), encoded_vector.get(), settings.gemv_rows, n);
        const auto gpu_x4 = run_gemv<Format, 4>(
            encoded_matrix.get(), encoded_vector.get(), settings.gemv_rows, n);

        for (std::size_t row = 0; row < settings.gemv_rows; ++row) {
          accumulated.add(extended(source_reference[row].value),
                          extended(source_reference[row].sum_abs),
                          extended(storage_reference[row].value),
                          extended(storage_reference[row].sum_abs),
                          {gpu_x1[row], gpu_x2[row], gpu_x4[row]}, replicate);
        }
        if constexpr (std::is_same_v<Format, e1m6>) {
          output.seeds << "gemv," << name(kind) << ',' << n << ','
                       << settings.gemv_rows << ',' << replicate << ",1,"
                       << matrix_seed << ',' << vector_seed << '\n';
        }
      }

      const case_identity identity{
          "gemv", name(kind),        Format::name, Format::total_bits,
          n,      settings.gemv_rows};
      accumulated.write(output.summary, output.quantiles, output.batches,
                        identity);
      write_encoding_row(output.encodings, identity, "matrix",
                         matrix_elements * settings.gemv_replicates,
                         matrix_counts);
      write_encoding_row(output.encodings, identity, "vector",
                         n * settings.gemv_replicates, vector_counts);
      output.flush_data();
      std::cout << "GEMV " << name(kind) << ' ' << Format::name
                << " M=" << settings.gemv_rows << " N=2^" << power
                << " replicates=" << settings.gemv_replicates << '\n';
    }
  }
}

class compensated_sum {
public:
  void add(long double value) {
    const auto updated = sum_ + value;
    if (std::fabs(sum_) >= std::fabs(value)) {
      correction_ += (sum_ - updated) + value;
    } else {
      correction_ += (value - updated) + sum_;
    }
    sum_ = updated;
  }
  long double value() const { return sum_ + correction_; }

private:
  long double sum_{};
  long double correction_{};
};

template <typename Format>
void reference_self_test(random_generator &generator, int multiprocessors,
                         output_files &output) {
  constexpr std::size_t n = 16384;
  device_buffer<double> device_left{n};
  device_buffer<double> device_right{n};
  generator.fill(device_left.get(), n, distribution::normal_0_1,
                 0x13198a2e03707344ULL);
  generator.fill(device_right.get(), n, distribution::normal_0_1,
                 0xa4093822299f31d0ULL);
  const auto source_reference =
      source_batched_references(device_left.get(), device_right.get(), n, 1);
  std::vector<double> left(n);
  std::vector<double> right(n);
  CUDA_CHECK(cudaMemcpy(left.data(), device_left.get(), n * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(right.data(), device_right.get(), n * sizeof(double),
                        cudaMemcpyDeviceToHost));
  compensated_sum host_source;
  compensated_sum host_source_abs;
  for (std::size_t index = 0; index < n; ++index) {
    const auto term = static_cast<long double>(left[index]) * right[index];
    host_source.add(term);
    host_source_abs.add(std::fabs(term));
  }
  const auto source_value = extended(source_reference[0].value);
  const auto source_difference =
      std::fabs(source_value - host_source.value()) / host_source_abs.value();
  const auto source_pass = source_difference <= 2.0e-18L;
  output.self_test << "source_double_double," << n << ',' << source_value << ','
                   << host_source.value() << ',' << host_source_abs.value()
                   << ',' << source_difference << ','
                   << (source_pass ? "pass" : "fail") << '\n';

  device_buffer<storage_type_t<Format>> encoded_left{n};
  device_buffer<storage_type_t<Format>> encoded_right{n};
  encode_values<Format>(device_left.get(), encoded_left.get(), n,
                        multiprocessors);
  encode_values<Format>(device_right.get(), encoded_right.get(), n,
                        multiprocessors);
  const auto storage_reference = storage_batched_references<Format>(
      encoded_left.get(), encoded_right.get(), n, 1);
  std::vector<storage_type_t<Format>> host_left(n);
  std::vector<storage_type_t<Format>> host_right(n);
  CUDA_CHECK(cudaMemcpy(host_left.data(), encoded_left.get(),
                        n * sizeof(storage_type_t<Format>),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_right.data(), encoded_right.get(),
                        n * sizeof(storage_type_t<Format>),
                        cudaMemcpyDeviceToHost));
  compensated_sum host_storage;
  compensated_sum host_storage_abs;
  for (std::size_t index = 0; index < n; ++index) {
    const auto term =
        static_cast<long double>(decode<Format>(host_left[index])) *
        decode<Format>(host_right[index]);
    host_storage.add(term);
    host_storage_abs.add(std::fabs(term));
  }
  const auto storage_value = extended(storage_reference[0].value);
  const auto storage_difference =
      std::fabs(storage_value - host_storage.value()) /
      host_storage_abs.value();
  const auto storage_pass = storage_difference <= 2.0e-18L;
  output.self_test << "storage_double_double_" << Format::name << ',' << n
                   << ',' << storage_value << ',' << host_storage.value() << ','
                   << host_storage_abs.value() << ',' << storage_difference
                   << ',' << (storage_pass ? "pass" : "fail") << '\n';
  if (!source_pass || !storage_pass) {
    throw simulation_error("GPU double-double reference self-test failed");
  }
}

template <typename... Formats>
void run_all_formats(const options &settings, std::size_t workspace_bytes,
                     int multiprocessors, random_generator &generator,
                     output_files &output) {
  (run_dot_cases<Formats>(settings, workspace_bytes, multiprocessors, generator,
                          output),
   ...);
  (run_gemv_cases<Formats>(settings, multiprocessors, generator, output), ...);
}

} // namespace

int main(int argc, char **argv) try {
  const auto settings = parse_options(argc, argv);
  int device{};
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::size_t free_bytes{};
  std::size_t total_bytes{};
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  const auto requested_workspace = static_cast<std::size_t>(
      settings.workspace_gib * 1024.0 * 1024.0 * 1024.0);
  const auto workspace_bytes =
      std::min(requested_workspace, free_bytes * 3 / 4);

  std::filesystem::create_directories(settings.output_dir);
  output_files output{settings.output_dir};
  random_generator generator;

  std::cout << "GPU accuracy simulation\n"
            << "GPU: " << properties.name << " (sm_" << properties.major
            << properties.minor << ")\n"
            << "DOT powers: " << join_powers(settings.dot_powers)
            << ", samples per case: " << settings.dot_samples << '\n'
            << "GEMV powers: " << join_powers(settings.gemv_powers)
            << ", M=" << settings.gemv_rows
            << ", replicates: " << settings.gemv_replicates << '\n'
            << "Workspace: "
            << static_cast<double>(workspace_bytes) / (1024.0 * 1024.0 * 1024.0)
            << " GiB ("
            << static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0)
            << " GiB free of "
            << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0)
            << " GiB)\n";

  reference_self_test<e3m4>(generator, properties.multiProcessorCount, output);
  run_all_formats<e1m6, e2m5, e3m4, fp8_e4m3, fp8_e5m2, e1m14, e2m13, e3m12,
                  fp16_e5m10, bf16_e8m7, e11m4, e1m30, e2m29, e3m28, fp32_e8m23,
                  e11m20, fp64_e11m52>(settings, workspace_bytes,
                                       properties.multiProcessorCount,
                                       generator, output);

  std::cout << "Wrote simulation data to " << settings.output_dir << '\n';
  return EXIT_SUCCESS;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return EXIT_FAILURE;
}
