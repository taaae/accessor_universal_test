#include "storage_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace aut::storage;

class test_error : public std::runtime_error {
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
    throw test_error(message.str());
  }
}

#define CUDA_CHECK(expression)                                                 \
  check_cuda((expression), #expression, __FILE__, __LINE__)

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

struct options {
  std::size_t dot_count{(std::size_t{1} << 18) + 3};
  std::size_t gemv_rows{257};
  std::size_t gemv_columns{1027};
  std::string output_dir{"storage_kernel_validation"};
};

std::size_t parse_positive_size(const std::string &text,
                                const std::string &option) {
  std::size_t consumed{};
  const auto value = std::stoull(text, &consumed, 0);
  if (consumed != text.size() || value == 0) {
    throw test_error(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(value);
}

options parse_options(int argc, char **argv) {
  options result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument{argv[index]};
    auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw test_error("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--dot-count") {
      result.dot_count = parse_positive_size(value(), argument);
    } else if (argument == "--gemv-rows") {
      result.gemv_rows = parse_positive_size(value(), argument);
    } else if (argument == "--gemv-columns") {
      result.gemv_columns = parse_positive_size(value(), argument);
    } else if (argument == "--output-dir") {
      result.output_dir = value();
    } else if (argument == "--help") {
      std::cout
          << "Usage: storage_kernels_cuda_test [options]\n"
          << "  --dot-count N       logical values in each DOT input\n"
          << "  --gemv-rows M       GEMV output rows\n"
          << "  --gemv-columns N    logical columns (padding is automatic)\n"
          << "  --output-dir PATH   directory for dot.csv and gemv.csv\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw test_error("unknown argument: " + argument);
    }
  }
  return result;
}

std::size_t round_up(std::size_t value, std::size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

enum class distribution { uniform, normal };

const char *name(distribution value) {
  return value == distribution::uniform ? "uniform_-1_1" : "normal_0_1";
}

struct source_data {
  std::vector<double> dot_left;
  std::vector<double> dot_right;
  std::vector<double> matrix;
  std::vector<double> vector;
  std::size_t leading_dimension{};
};

source_data make_source(distribution kind, const options &settings) {
  const auto leading_dimension = round_up(settings.gemv_columns, 4);
  source_data result;
  result.leading_dimension = leading_dimension;
  result.dot_left.resize(settings.dot_count);
  result.dot_right.resize(settings.dot_count);
  result.matrix.assign(settings.gemv_rows * leading_dimension, 0.0);
  result.vector.resize(settings.gemv_columns);

  std::mt19937_64 engine{kind == distribution::uniform
                             ? std::uint64_t{0x6a09e667f3bcc909ULL}
                             : std::uint64_t{0xbb67ae8584caa73bULL}};
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};
  std::normal_distribution<double> normal{0.0, 1.0};
  auto next = [&]() {
    if (kind == distribution::uniform) {
      return uniform(engine);
    }
    return normal(engine);
  };

  for (auto &value : result.dot_left) {
    value = next();
  }
  for (auto &value : result.dot_right) {
    value = next();
  }
  for (std::size_t row = 0; row < settings.gemv_rows; ++row) {
    for (std::size_t column = 0; column < settings.gemv_columns; ++column) {
      result.matrix[row * leading_dimension + column] = next();
    }
  }
  for (auto &value : result.vector) {
    value = next();
  }
  return result;
}

template <typename Format>
std::vector<storage_type_t<Format>>
encode_values(const std::vector<double> &source) {
  std::vector<storage_type_t<Format>> result(source.size());
  for (std::size_t index = 0; index < source.size(); ++index) {
    result[index] = encode<Format>(source[index]);
  }
  return result;
}

class compensated_sum {
public:
  void add(long double value) {
    if (!std::isfinite(value) || !std::isfinite(sum_)) {
      sum_ += value;
      correction_ = 0.0L;
      return;
    }
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

struct dot_reference {
  long double value{};
  long double sum_abs{};
};

dot_reference reference_dot(const std::vector<double> &left,
                            const std::vector<double> &right) {
  compensated_sum value;
  compensated_sum sum_abs;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const auto product = static_cast<long double>(left[index]) * right[index];
    value.add(product);
    sum_abs.add(std::fabs(product));
  }
  return {value.value(), sum_abs.value()};
}

template <typename Format>
dot_reference reference_dot(const std::vector<storage_type_t<Format>> &left,
                            const std::vector<storage_type_t<Format>> &right) {
  compensated_sum value;
  compensated_sum sum_abs;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const auto product = static_cast<long double>(decode<Format>(left[index])) *
                         decode<Format>(right[index]);
    value.add(product);
    sum_abs.add(std::fabs(product));
  }
  return {value.value(), sum_abs.value()};
}

struct gemv_reference {
  std::vector<long double> values;
  std::vector<long double> sum_abs;
};

gemv_reference reference_gemv(const std::vector<double> &matrix,
                              const std::vector<double> &vector,
                              std::size_t rows, std::size_t columns,
                              std::size_t leading_dimension) {
  gemv_reference result;
  result.values.resize(rows);
  result.sum_abs.resize(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    compensated_sum value;
    compensated_sum sum_abs;
    for (std::size_t column = 0; column < columns; ++column) {
      const auto product =
          static_cast<long double>(matrix[row * leading_dimension + column]) *
          vector[column];
      value.add(product);
      sum_abs.add(std::fabs(product));
    }
    result.values[row] = value.value();
    result.sum_abs[row] = sum_abs.value();
  }
  return result;
}

template <typename Format>
gemv_reference reference_gemv(const std::vector<storage_type_t<Format>> &matrix,
                              const std::vector<storage_type_t<Format>> &vector,
                              std::size_t rows, std::size_t columns,
                              std::size_t leading_dimension) {
  gemv_reference result;
  result.values.resize(rows);
  result.sum_abs.resize(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    compensated_sum value;
    compensated_sum sum_abs;
    for (std::size_t column = 0; column < columns; ++column) {
      const auto product = static_cast<long double>(decode<Format>(
                               matrix[row * leading_dimension + column])) *
                           decode<Format>(vector[column]);
      value.add(product);
      sum_abs.add(std::fabs(product));
    }
    result.values[row] = value.value();
    result.sum_abs[row] = sum_abs.value();
  }
  return result;
}

enum class value_class { finite, positive_infinity, negative_infinity, nan };

value_class classify(long double value) {
  if (std::isnan(value)) {
    return value_class::nan;
  }
  if (std::isinf(value)) {
    return std::signbit(value) ? value_class::negative_infinity
                               : value_class::positive_infinity;
  }
  return value_class::finite;
}

const char *name(value_class value) {
  switch (value) {
  case value_class::finite:
    return "finite";
  case value_class::positive_infinity:
    return "positive_infinity";
  case value_class::negative_infinity:
    return "negative_infinity";
  case value_class::nan:
    return "nan";
  }
  return "unknown";
}

long double finite_or_infinity(long double value) {
  return std::isfinite(value) ? value
                              : std::numeric_limits<long double>::infinity();
}

struct vector_comparison {
  std::size_t finite_pairs{};
  std::size_t matching_nonfinite{};
  std::size_t class_mismatches{};
  long double max_absolute_error{};
  long double max_normalized_error{};
  long double error_squared{};
  long double reference_squared{};

  long double relative_l2() const {
    if (class_mismatches != 0) {
      return std::numeric_limits<long double>::infinity();
    }
    if (reference_squared == 0.0L) {
      return error_squared == 0.0L
                 ? 0.0L
                 : std::numeric_limits<long double>::infinity();
    }
    return std::sqrt(error_squared / reference_squared);
  }
};

template <typename Actual>
vector_comparison compare_vectors(const std::vector<Actual> &actual,
                                  const std::vector<long double> &reference,
                                  const std::vector<long double> &normalizer) {
  vector_comparison result;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const auto actual_value = static_cast<long double>(actual[index]);
    const auto actual_class = classify(actual_value);
    const auto reference_class = classify(reference[index]);
    if (actual_class == value_class::finite &&
        reference_class == value_class::finite) {
      const auto error = std::fabs(actual_value - reference[index]);
      result.max_absolute_error = std::max(result.max_absolute_error, error);
      const auto normalized =
          normalizer[index] == 0.0L
              ? (error == 0.0L ? 0.0L
                               : std::numeric_limits<long double>::infinity())
              : error / normalizer[index];
      result.max_normalized_error =
          std::max(result.max_normalized_error, normalized);
      result.error_squared += error * error;
      result.reference_squared += reference[index] * reference[index];
      ++result.finite_pairs;
    } else if (actual_class == reference_class) {
      ++result.matching_nonfinite;
    } else {
      ++result.class_mismatches;
    }
  }
  return result;
}

int first_stage_blocks(std::size_t count, int lanes, int multiprocessors) {
  const auto packs = (count + static_cast<std::size_t>(lanes) - 1) / lanes;
  const auto wanted = (packs + aut::kernels::reduction_block_threads - 1) /
                      aut::kernels::reduction_block_threads;
  return static_cast<int>(std::max<std::size_t>(
      1, std::min<std::size_t>(
             wanted, static_cast<std::size_t>(multiprocessors * 16))));
}

template <typename Format, int Lanes>
double run_dot(const device_buffer<storage_type_t<Format>> &left,
               const device_buffer<storage_type_t<Format>> &right,
               std::size_t count, int multiprocessors,
               device_buffer<double> &partials, device_buffer<double> &result) {
  const auto blocks = first_stage_blocks(count, Lanes, multiprocessors);
  aut::kernels::storage_dot_map_reduce<Format, Lanes>
      <<<blocks, aut::kernels::reduction_block_threads>>>(
          left.get(), right.get(), count, partials.get());
  CUDA_CHECK(cudaGetLastError());
  aut::kernels::
      storage_dot_finalize<<<1, aut::kernels::reduction_block_threads>>>(
          partials.get(), static_cast<std::size_t>(blocks), result.get());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  double host_result{};
  CUDA_CHECK(cudaMemcpy(&host_result, result.get(), sizeof(host_result),
                        cudaMemcpyDeviceToHost));
  return host_result;
}

template <typename Format, int Lanes>
std::vector<double>
run_gemv(const device_buffer<storage_type_t<Format>> &matrix,
         const device_buffer<storage_type_t<Format>> &vector, std::size_t rows,
         std::size_t columns, std::size_t leading_dimension,
         device_buffer<double> &result) {
  aut::kernels::storage_gemv<Format, Lanes>
      <<<static_cast<unsigned>(rows), aut::kernels::reduction_block_threads>>>(
          matrix.get(), vector.get(), rows, columns, leading_dimension,
          result.get());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> host_result(rows);
  CUDA_CHECK(cudaMemcpy(host_result.data(), result.get(), rows * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return host_result;
}

void write_dot_row(std::ofstream &output, distribution kind, const char *format,
                   int storage_bits, int lanes, std::size_t count,
                   double actual, const dot_reference &storage_reference,
                   const dot_reference &original_reference, bool pass) {
  const auto actual_class = classify(actual);
  const auto reference_class = classify(storage_reference.value);
  long double kernel_absolute =
      actual_class == value_class::finite &&
              reference_class == value_class::finite
          ? std::fabs(static_cast<long double>(actual) -
                      storage_reference.value)
          : (actual_class == reference_class
                 ? 0.0L
                 : std::numeric_limits<long double>::infinity());
  const auto kernel_normalized =
      storage_reference.sum_abs == 0.0L
          ? kernel_absolute
          : kernel_absolute / storage_reference.sum_abs;
  const auto storage_absolute =
      classify(storage_reference.value) == value_class::finite
          ? std::fabs(storage_reference.value - original_reference.value)
          : std::numeric_limits<long double>::infinity();
  const auto storage_normalized =
      original_reference.sum_abs == 0.0L
          ? storage_absolute
          : storage_absolute / original_reference.sum_abs;

  output << name(kind) << ',' << format << ',' << storage_bits << ',' << lanes
         << ',' << count << ',' << std::setprecision(21) << actual << ','
         << storage_reference.value << ',' << original_reference.value << ','
         << finite_or_infinity(kernel_absolute) << ','
         << finite_or_infinity(kernel_normalized) << ','
         << finite_or_infinity(storage_absolute) << ','
         << finite_or_infinity(storage_normalized) << ',' << name(actual_class)
         << ',' << name(reference_class) << ',' << (pass ? "pass" : "fail")
         << '\n';
}

void write_gemv_row(std::ofstream &output, distribution kind,
                    const char *format, int storage_bits, int lanes,
                    const options &settings, std::size_t leading_dimension,
                    const vector_comparison &kernel,
                    const vector_comparison &storage, bool pass) {
  output << name(kind) << ',' << format << ',' << storage_bits << ',' << lanes
         << ',' << settings.gemv_rows << ',' << settings.gemv_columns << ','
         << leading_dimension << ',' << kernel.finite_pairs << ','
         << kernel.matching_nonfinite << ',' << kernel.class_mismatches << ','
         << std::setprecision(21)
         << finite_or_infinity(kernel.max_absolute_error) << ','
         << finite_or_infinity(kernel.max_normalized_error) << ','
         << finite_or_infinity(kernel.relative_l2()) << ','
         << storage.class_mismatches << ','
         << finite_or_infinity(storage.max_absolute_error) << ','
         << finite_or_infinity(storage.relative_l2()) << ','
         << (pass ? "pass" : "fail") << '\n';
}

template <typename Format, int Lanes>
int validate_lane(distribution kind, const options &settings,
                  std::size_t leading_dimension, int multiprocessors,
                  const device_buffer<storage_type_t<Format>> &dot_left,
                  const device_buffer<storage_type_t<Format>> &dot_right,
                  const device_buffer<storage_type_t<Format>> &matrix,
                  const device_buffer<storage_type_t<Format>> &vector,
                  const dot_reference &dot_storage_reference,
                  const dot_reference &dot_original_reference,
                  const gemv_reference &gemv_storage_reference,
                  const gemv_reference &gemv_original_reference,
                  std::ofstream &dot_output, std::ofstream &gemv_output,
                  device_buffer<double> &partials,
                  device_buffer<double> &dot_result,
                  device_buffer<double> &gemv_result) {
  const auto dot_actual =
      run_dot<Format, Lanes>(dot_left, dot_right, settings.dot_count,
                             multiprocessors, partials, dot_result);
  const auto dot_actual_class = classify(dot_actual);
  const auto dot_reference_class = classify(dot_storage_reference.value);
  const auto dot_kernel_error =
      dot_actual_class == value_class::finite &&
              dot_reference_class == value_class::finite
          ? std::fabs(static_cast<long double>(dot_actual) -
                      dot_storage_reference.value) /
                std::max(dot_storage_reference.sum_abs,
                         std::numeric_limits<long double>::min())
          : 0.0L;
  const bool dot_pass =
      dot_actual_class == dot_reference_class && dot_kernel_error <= 5.0e-12L;
  write_dot_row(dot_output, kind, Format::name, Format::total_bits, Lanes,
                settings.dot_count, dot_actual, dot_storage_reference,
                dot_original_reference, dot_pass);

  const auto gemv_actual = run_gemv<Format, Lanes>(
      matrix, vector, settings.gemv_rows, settings.gemv_columns,
      leading_dimension, gemv_result);
  const auto gemv_kernel =
      compare_vectors(gemv_actual, gemv_storage_reference.values,
                      gemv_storage_reference.sum_abs);
  const auto gemv_storage = compare_vectors(gemv_storage_reference.values,
                                            gemv_original_reference.values,
                                            gemv_original_reference.sum_abs);
  const bool gemv_pass = gemv_kernel.class_mismatches == 0 &&
                         gemv_kernel.max_normalized_error <= 5.0e-12L;
  write_gemv_row(gemv_output, kind, Format::name, Format::total_bits, Lanes,
                 settings, leading_dimension, gemv_kernel, gemv_storage,
                 gemv_pass);
  return static_cast<int>(!dot_pass) + static_cast<int>(!gemv_pass);
}

template <typename Format>
int validate_format(distribution kind, const source_data &source,
                    const options &settings, int multiprocessors,
                    const dot_reference &dot_original_reference,
                    const gemv_reference &gemv_original_reference,
                    std::ofstream &dot_output, std::ofstream &gemv_output) {
  const auto host_dot_left = encode_values<Format>(source.dot_left);
  const auto host_dot_right = encode_values<Format>(source.dot_right);
  const auto host_matrix = encode_values<Format>(source.matrix);
  const auto host_vector = encode_values<Format>(source.vector);

  const auto dot_storage_reference =
      reference_dot<Format>(host_dot_left, host_dot_right);
  const auto gemv_storage_reference =
      reference_gemv<Format>(host_matrix, host_vector, settings.gemv_rows,
                             settings.gemv_columns, source.leading_dimension);

  device_buffer<storage_type_t<Format>> device_dot_left{host_dot_left.size()};
  device_buffer<storage_type_t<Format>> device_dot_right{host_dot_right.size()};
  device_buffer<storage_type_t<Format>> device_matrix{host_matrix.size()};
  device_buffer<storage_type_t<Format>> device_vector{host_vector.size()};
  CUDA_CHECK(cudaMemcpy(device_dot_left.get(), host_dot_left.data(),
                        host_dot_left.size() * sizeof(storage_type_t<Format>),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_dot_right.get(), host_dot_right.data(),
                        host_dot_right.size() * sizeof(storage_type_t<Format>),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_matrix.get(), host_matrix.data(),
                        host_matrix.size() * sizeof(storage_type_t<Format>),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_vector.get(), host_vector.data(),
                        host_vector.size() * sizeof(storage_type_t<Format>),
                        cudaMemcpyHostToDevice));

  device_buffer<double> partials{
      static_cast<std::size_t>(multiprocessors * 16)};
  device_buffer<double> dot_result{1};
  device_buffer<double> gemv_result{settings.gemv_rows};

  int failures{};
  failures += validate_lane<Format, 1>(
      kind, settings, source.leading_dimension, multiprocessors,
      device_dot_left, device_dot_right, device_matrix, device_vector,
      dot_storage_reference, dot_original_reference, gemv_storage_reference,
      gemv_original_reference, dot_output, gemv_output, partials, dot_result,
      gemv_result);
  failures += validate_lane<Format, 2>(
      kind, settings, source.leading_dimension, multiprocessors,
      device_dot_left, device_dot_right, device_matrix, device_vector,
      dot_storage_reference, dot_original_reference, gemv_storage_reference,
      gemv_original_reference, dot_output, gemv_output, partials, dot_result,
      gemv_result);
  failures += validate_lane<Format, 4>(
      kind, settings, source.leading_dimension, multiprocessors,
      device_dot_left, device_dot_right, device_matrix, device_vector,
      dot_storage_reference, dot_original_reference, gemv_storage_reference,
      gemv_original_reference, dot_output, gemv_output, partials, dot_result,
      gemv_result);

  std::cout << std::left << std::setw(14) << Format::name << ' '
            << std::setw(13) << name(kind) << " failed_tests=" << failures
            << " dot_reference_class="
            << name(classify(dot_storage_reference.value)) << '\n';
  dot_output.flush();
  gemv_output.flush();
  return failures;
}

template <typename... Formats>
int validate_all(distribution kind, const source_data &source,
                 const options &settings, int multiprocessors,
                 const dot_reference &dot_original_reference,
                 const gemv_reference &gemv_original_reference,
                 std::ofstream &dot_output, std::ofstream &gemv_output) {
  return (validate_format<Formats>(
              kind, source, settings, multiprocessors, dot_original_reference,
              gemv_original_reference, dot_output, gemv_output) +
          ... + 0);
}

} // namespace

int main(int argc, char **argv) try {
  const auto settings = parse_options(argc, argv);
  int device{};
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  std::filesystem::create_directories(settings.output_dir);
  std::ofstream dot_output{std::filesystem::path{settings.output_dir} /
                           "dot_validation.csv"};
  std::ofstream gemv_output{std::filesystem::path{settings.output_dir} /
                            "gemv_validation.csv"};
  if (!dot_output || !gemv_output) {
    throw test_error("could not create validation CSV files in " +
                     settings.output_dir);
  }
  dot_output
      << "distribution,format,storage_bits,lanes,n,gpu_result,"
         "decoded_storage_reference,fp64_source_reference,kernel_abs_error,"
         "kernel_normalized_error,storage_abs_error,storage_normalized_error,"
         "gpu_class,reference_class,status\n";
  gemv_output
      << "distribution,format,storage_bits,lanes,rows,columns,"
         "leading_dimension,finite_output_pairs,matching_nonfinite_outputs,"
         "classification_mismatches,kernel_max_abs_error,"
         "kernel_max_normalized_error,kernel_relative_l2_error,"
         "storage_classification_mismatches,storage_max_abs_error,"
         "storage_relative_l2_error,status\n";

  std::cout << "storage DOT/GEMV CUDA validation\n"
            << "GPU: " << properties.name << " (sm_" << properties.major
            << properties.minor << ")\n"
            << "DOT logical values: " << settings.dot_count << '\n'
            << "GEMV: " << settings.gemv_rows << " x " << settings.gemv_columns
            << " (row padding to x4 alignment)\n";

  int failures{};
  for (const auto kind : {distribution::uniform, distribution::normal}) {
    const auto source = make_source(kind, settings);
    const auto dot_original_reference =
        reference_dot(source.dot_left, source.dot_right);
    const auto gemv_original_reference =
        reference_gemv(source.matrix, source.vector, settings.gemv_rows,
                       settings.gemv_columns, source.leading_dimension);
    failures += validate_all<e1m6, e2m5, e3m4, fp8_e4m3, fp8_e5m2, e1m14, e2m13,
                             e3m12, fp16_e5m10, bf16_e8m7, e11m4, e1m30, e2m29,
                             e3m28, fp32_e8m23, e11m20, fp64_e11m52>(
        kind, source, settings, properties.multiProcessorCount,
        dot_original_reference, gemv_original_reference, dot_output,
        gemv_output);
  }

  std::cout << "Wrote " << settings.output_dir << "/dot_validation.csv\n"
            << "Wrote " << settings.output_dir << "/gemv_validation.csv\n";
  if (failures != 0) {
    std::cerr << failures << " DOT/GEMV validations failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "All scalar, x2, and x4 DOT/GEMV validations passed.\n";
  return EXIT_SUCCESS;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return EXIT_FAILURE;
}
