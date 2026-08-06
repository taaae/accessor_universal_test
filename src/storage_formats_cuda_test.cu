#include "cuda_storage_formats.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace aut::storage;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const auto aut_cuda_error = (call);                                        \
    if (aut_cuda_error != cudaSuccess) {                                       \
      throw std::runtime_error(std::string{#call} + ": " +                     \
                               cudaGetErrorString(aut_cuda_error));            \
    }                                                                          \
  } while (false)

template <typename T> class device_buffer {
public:
  explicit device_buffer(std::size_t count) : count_{count} {
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)));
  }

  ~device_buffer() { cudaFree(data_); }

  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;

  T *get() { return data_; }
  const T *get() const { return data_; }
  std::size_t size() const { return count_; }

private:
  T *data_{};
  std::size_t count_{};
};

template <typename Format>
__global__ void decode_scalar_kernel(const storage_type_t<Format> *input,
                                     double *output, std::size_t count) {
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto i = first; i < count; i += stride) {
    output[i] = decode<Format>(input[i]);
  }
}

template <typename Format>
__global__ void decode_pack2_kernel(const storage_type_t<Format> *input,
                                    double *output, std::size_t pack_count) {
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto pack = first; pack < pack_count; pack += stride) {
    const auto values = packed_decoder<Format>::load2(input + 2 * pack);
    output[2 * pack] = values.x;
    output[2 * pack + 1] = values.y;
  }
}

template <typename Format>
__global__ void decode_pack4_kernel(const storage_type_t<Format> *input,
                                    double *output, std::size_t pack_count) {
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto pack = first; pack < pack_count; pack += stride) {
    const auto values = packed_decoder<Format>::load4(input + 4 * pack);
    output[4 * pack] = values.x;
    output[4 * pack + 1] = values.y;
    output[4 * pack + 2] = values.z;
    output[4 * pack + 3] = values.w;
  }
}

struct options {
  std::size_t count = 1u << 16;
  std::string output = "storage_formats_validation.csv";
};

options parse_options(int argc, char **argv) {
  options result;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) {
        throw std::runtime_error("missing value after " + argument);
      }
      return argv[i];
    };
    if (argument == "--count") {
      result.count = std::stoull(value());
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--help") {
      std::cout << "Usage: storage_formats_cuda_test [options]\n"
                << "  --count N     values per distribution (multiple of 4)\n"
                << "  --output PATH validation CSV\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (result.count == 0 || result.count % 4 != 0) {
    throw std::runtime_error("--count must be a positive multiple of 4");
  }
  return result;
}

std::vector<double> generate_values(const std::string &distribution,
                                    std::size_t count) {
  std::mt19937_64 engine{distribution == "uniform" ? 0x159a55e5ULL
                                                   : 0x71c3d29bULL};
  std::vector<double> values(count);
  if (distribution == "uniform") {
    std::uniform_real_distribution<double> random{-1.0, 1.0};
    for (auto &value : values) {
      value = random(engine);
    }
  } else {
    std::normal_distribution<double> random{0.0, 1.0};
    for (auto &value : values) {
      value = random(engine);
    }
  }

  const double probes[]{0.0,
                        -0.0,
                        1.0,
                        -1.0,
                        0.5,
                        -0.5,
                        0.25,
                        -0.25,
                        std::ldexp(1.0, -20),
                        -std::ldexp(1.0, -20),
                        3.75,
                        -3.75};
  for (std::size_t i = 0; i < std::size(probes) && i < values.size(); ++i) {
    values[i] = probes[i];
  }
  return values;
}

std::uint64_t bits_of(double value) {
  std::uint64_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct encoding_metrics {
  double nmse{};
  double max_abs_error{};
  std::size_t zeros{};
  std::size_t infinities{};
};

encoding_metrics measure_encoding(const std::vector<double> &source,
                                  const std::vector<double> &decoded) {
  long double error_squared{};
  long double signal_squared{};
  double max_abs_error{};
  std::size_t zeros{};
  std::size_t infinities{};
  for (std::size_t i = 0; i < source.size(); ++i) {
    if (decoded[i] == 0.0) {
      ++zeros;
    }
    if (std::isinf(decoded[i])) {
      ++infinities;
    }
    const auto error = decoded[i] - source[i];
    max_abs_error = std::max(max_abs_error, std::abs(error));
    error_squared += static_cast<long double>(error) * error;
    signal_squared += static_cast<long double>(source[i]) * source[i];
  }
  return {static_cast<double>(error_squared / signal_squared), max_abs_error,
          zeros, infinities};
}

struct comparison {
  std::size_t mismatches{};
  double max_abs_difference{};
};

comparison compare_decodes(const std::vector<double> &expected,
                           const std::vector<double> &actual,
                           const char *format, int lanes) {
  comparison result;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const bool equal = bits_of(expected[i]) == bits_of(actual[i]) ||
                       (std::isnan(expected[i]) && std::isnan(actual[i]));
    if (!equal) {
      if (result.mismatches < 8) {
        std::cerr << format << " pack" << lanes << " mismatch at " << i
                  << ": expected=" << std::setprecision(17) << expected[i]
                  << " actual=" << actual[i] << '\n';
      }
      ++result.mismatches;
      result.max_abs_difference = std::max(result.max_abs_difference,
                                           std::abs(expected[i] - actual[i]));
    }
  }
  return result;
}

int launch_blocks(std::size_t work_items) {
  constexpr int threads = 256;
  return static_cast<int>(
      std::min<std::size_t>((work_items + threads - 1) / threads, 4096));
}

template <typename Format>
int validate_format(const std::string &distribution,
                    const std::vector<double> &source, std::ofstream &output) {
  using storage_type = storage_type_t<Format>;
  std::vector<storage_type> encoded(source.size());
  std::vector<double> expected(source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    encoded[i] = encode<Format>(source[i]);
    expected[i] = decode<Format>(encoded[i]);
  }
  const auto metrics = measure_encoding(source, expected);

  device_buffer<storage_type> device_input(encoded.size());
  device_buffer<double> device_output(encoded.size());
  CUDA_CHECK(cudaMemcpy(device_input.get(), encoded.data(),
                        encoded.size() * sizeof(storage_type),
                        cudaMemcpyHostToDevice));

  std::vector<double> actual(source.size());
  int failures{};
  constexpr int threads = 256;
  for (const int lanes : {1, 2, 4}) {
    const auto work_items = source.size() / static_cast<std::size_t>(lanes);
    const auto blocks = launch_blocks(work_items);
    if (lanes == 1) {
      decode_scalar_kernel<Format><<<blocks, threads>>>(
          device_input.get(), device_output.get(), source.size());
    } else if (lanes == 2) {
      decode_pack2_kernel<Format><<<blocks, threads>>>(
          device_input.get(), device_output.get(), work_items);
    } else {
      decode_pack4_kernel<Format><<<blocks, threads>>>(
          device_input.get(), device_output.get(), work_items);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), device_output.get(),
                          actual.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    const auto compared =
        compare_decodes(expected, actual, Format::name, lanes);
    failures += compared.mismatches != 0;
    output << distribution << ',' << Format::name << ',' << Format::total_bits
           << ',' << lanes << ',' << source.size() << ',' << compared.mismatches
           << ',' << std::setprecision(17) << compared.max_abs_difference << ','
           << metrics.nmse << ',' << metrics.max_abs_error << ','
           << metrics.zeros << ',' << metrics.infinities << '\n';
  }

  std::cout << std::left << std::setw(14) << Format::name << " " << std::setw(7)
            << distribution << " failed_lane_widths=" << failures
            << " nmse=" << std::scientific << metrics.nmse
            << " zeros=" << metrics.zeros << " infs=" << metrics.infinities
            << '\n';
  return failures;
}

template <typename... Formats>
int validate_all(const std::string &distribution,
                 const std::vector<double> &source, std::ofstream &output) {
  return (validate_format<Formats>(distribution, source, output) + ... + 0);
}

} // namespace

int main(int argc, char **argv) try {
  const auto settings = parse_options(argc, argv);
  int device{};
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::cout << "storage format CUDA validation\n"
            << "GPU: " << properties.name << " (sm_" << properties.major
            << properties.minor << ")\n"
            << "values per distribution: " << settings.count << "\n";

  const auto parent = std::filesystem::path{settings.output}.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream output{settings.output};
  if (!output) {
    throw std::runtime_error("cannot open output: " + settings.output);
  }
  output << "distribution,format,storage_bits,lanes,count,mismatches,"
            "max_decode_difference,encoding_nmse,max_encoding_abs_error,"
            "decoded_zeros,decoded_infinities\n";

  int failures{};
  for (const std::string distribution : {"uniform", "normal"}) {
    const auto source = generate_values(distribution, settings.count);
    failures += validate_all<e1m6, e2m5, e3m4, fp8_e4m3, fp8_e5m2, e1m14, e2m13,
                             e3m12, fp16_e5m10, bf16_e8m7, e11m4, e1m30, e2m29,
                             e3m28, fp32_e8m23, e11m20, fp64_e11m52>(
        distribution, source, output);
  }

  std::cout << "Wrote " << settings.output << '\n';
  if (failures != 0) {
    std::cerr << failures << " format/distribution/lane validations failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "All scalar, pack2, and pack4 decoders agree exactly.\n";
  return EXIT_SUCCESS;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return EXIT_FAILURE;
}
