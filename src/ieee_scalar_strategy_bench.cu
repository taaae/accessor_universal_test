#include "ieee_scalar_decoder.cuh"
#include "posit_takum_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef AUT_IEEE_FORMAT_ID
#error AUT_IEEE_FORMAT_ID is required
#endif
#ifndef AUT_IEEE_COMPUTE
#error AUT_IEEE_COMPUTE is required
#endif
#ifndef AUT_IEEE_STRATEGY_MASK
#error AUT_IEEE_STRATEGY_MASK is required
#endif
#ifndef AUT_IEEE_Q_LOWER
#error AUT_IEEE_Q_LOWER is required
#endif
#ifndef AUT_IEEE_Q_UPPER
#error AUT_IEEE_Q_UPPER is required
#endif

namespace bw = aut::bitwidth;
namespace storage = aut::storage;
namespace pt = aut::pt;

namespace {

template <int Id> struct selected_format;
template <> struct selected_format<1> { using type = storage::fp8_e4m3; };
template <> struct selected_format<2> { using type = storage::fp8_e5m2; };
template <> struct selected_format<3> { using type = storage::e3m4; };
template <> struct selected_format<4> { using type = storage::e6m1; };
template <> struct selected_format<5> { using type = storage::e8m5; };
template <> struct selected_format<6> { using type = storage::e11m2; };
template <> struct selected_format<7> { using type = storage::e2m11; };
template <> struct selected_format<8> { using type = storage::e5m8; };
template <> struct selected_format<9> { using type = storage::fp16_e5m10; };
template <> struct selected_format<10> { using type = storage::bf16_e8m7; };
template <> struct selected_format<11> { using type = storage::e11m4; };
template <> struct selected_format<12> { using type = storage::e3m12; };
template <> struct selected_format<13> { using type = storage::e6m9; };
template <> struct selected_format<14> { using type = storage::fp32_e8m23; };
template <> struct selected_format<15> { using type = storage::e11m20; };
template <> struct selected_format<16> { using type = storage::e4m27; };
template <> struct selected_format<17> { using type = storage::e10m21; };

using Format = typename selected_format<AUT_IEEE_FORMAT_ID>::type;
constexpr auto Compute = static_cast<bw::compute_kind>(AUT_IEEE_COMPUTE);
using Float = bw::compute_t<Compute>;
constexpr int Bits = Format::total_bits;
constexpr int threads = 256;
constexpr int dot_blocks = 512;
constexpr std::uint32_t strategy_mask = AUT_IEEE_STRATEGY_MASK;
constexpr long double q_lower = AUT_IEEE_Q_LOWER;
constexpr long double q_upper = AUT_IEEE_Q_UPPER;

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
      : data_(std::exchange(other.data_, nullptr)), count_(other.count_) {}
  device_buffer &operator=(device_buffer &&other) noexcept {
    if (this != &other) {
      reset(0);
      data_ = std::exchange(other.data_, nullptr);
      count_ = other.count_;
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

template <typename Selected = Format>
double reference_value(std::uint32_t raw) {
  if constexpr (std::is_same_v<Selected, storage::fp8_e4m3>) {
    storage::storage_type_t<Selected> value;
    value.__x = static_cast<__nv_fp8_storage_t>(raw);
    return storage::decode<Selected>(value);
  } else if constexpr (std::is_same_v<Selected, storage::fp8_e5m2>) {
    storage::storage_type_t<Selected> value;
    value.__x = static_cast<__nv_fp8_storage_t>(raw);
    return storage::decode<Selected>(value);
  } else if constexpr (std::is_same_v<Selected, storage::fp16_e5m10>) {
    return storage::decode<Selected>(
        __half{__half_raw{static_cast<unsigned short>(raw)}});
  } else if constexpr (std::is_same_v<Selected, storage::bf16_e8m7>) {
    return storage::decode<Selected>(__nv_bfloat16{
        __nv_bfloat16_raw{static_cast<unsigned short>(raw)}});
  } else if constexpr (std::is_same_v<Selected, storage::fp32_e8m23>) {
    float value{};
    std::memcpy(&value, &raw, sizeof(value));
    return value;
  } else {
    return bw::decode_reference<Selected, bw::compute_kind::fp64>(raw);
  }
}

template <typename Selected = Format>
std::uint32_t encode_raw(double value) {
  const auto encoded = storage::encode<Selected>(value);
  if constexpr (std::is_same_v<Selected, storage::fp8_e4m3> ||
                std::is_same_v<Selected, storage::fp8_e5m2>) {
    return encoded.__x;
  } else if constexpr (std::is_same_v<Selected, storage::fp16_e5m10>) {
    return __half_as_ushort(encoded);
  } else if constexpr (std::is_same_v<Selected, storage::bf16_e8m7>) {
    return __bfloat16_as_ushort(encoded);
  } else if constexpr (std::is_same_v<Selected, storage::fp32_e8m23>) {
    std::uint32_t raw{};
    std::memcpy(&raw, &encoded, sizeof(raw));
    return raw;
  } else {
    return static_cast<std::uint32_t>(encoded);
  }
}

std::uint32_t source_sign(std::uint32_t raw) { return raw >> (Bits - 1); }
std::uint32_t fraction(std::uint32_t raw) {
  return raw & ((std::uint32_t{1} << Format::fraction_bits) - 1u);
}
std::uint32_t exponent(std::uint32_t raw) {
  return (raw >> Format::fraction_bits) &
         ((std::uint32_t{1} << Format::exponent_bits) - 1u);
}

struct pool {
  std::vector<std::uint32_t> left;
  std::vector<std::uint32_t> right;
  std::vector<std::size_t> buckets;
  long double minimum{};
  long double maximum{};
};

long double raw_q(std::uint32_t raw) {
  return std::log2(std::abs(static_cast<long double>(reference_value(raw))));
}

pool make_field_pool(std::size_t count, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const std::uint32_t exponent_codes = std::uint32_t{1} << Format::exponent_bits;
  std::vector<std::vector<std::uint32_t>> normal(exponent_codes);
  std::vector<std::uint32_t> subnormal;
  auto accept = [&](std::uint32_t raw) {
    if (source_sign(raw)) return;
    const auto value = reference_value(raw);
    if (!std::isfinite(value) || value == 0.0) return;
    const auto q = raw_q(raw);
    if (q < q_lower || q > q_upper) return;
    if (exponent(raw) == 0) subnormal.push_back(raw);
    else normal[exponent(raw)].push_back(raw);
  };
  if constexpr (Bits <= 16) {
    for (std::uint32_t raw = 1; raw < (std::uint32_t{1} << (Bits - 1)); ++raw)
      accept(raw);
  } else {
    const std::uint32_t fraction_mask =
        (std::uint32_t{1} << Format::fraction_bits) - 1u;
    for (std::uint32_t exp = 1; exp < exponent_codes; ++exp) {
      for (int sample = 0; sample < 2048; ++sample)
        accept((exp << Format::fraction_bits) |
               (static_cast<std::uint32_t>(rng()) & fraction_mask));
    }
    for (int sample = 0; sample < 262144; ++sample)
      accept(static_cast<std::uint32_t>(rng()) & fraction_mask);
  }
  normal.erase(std::remove_if(normal.begin(), normal.end(),
                              [](const auto &values) { return values.empty(); }),
               normal.end());
  if (normal.empty()) throw benchmark_error("no admissible IEEE normal codes");

  pool result;
  result.left.reserve(count);
  result.buckets.assign(normal.size() + (subnormal.empty() ? 0 : 1), 0);
  const std::size_t subnormal_count = subnormal.empty() ? 0 : count / 2;
  const std::size_t normal_count = count - subnormal_count;
  for (std::size_t index = 0; index < normal_count; ++index) {
    const auto bucket = index % normal.size();
    result.left.push_back(normal[bucket][rng() % normal[bucket].size()]);
    ++result.buckets[bucket];
  }
  for (std::size_t index = 0; index < subnormal_count; ++index) {
    result.left.push_back(subnormal[rng() % subnormal.size()]);
    ++result.buckets.back();
  }
  std::sort(result.left.begin(), result.left.end(), [](auto a, auto b) {
    return raw_q(a) < raw_q(b);
  });
  for (std::size_t index = 0; index < result.left.size(); ++index) {
    if (index & 1u) result.left[index] |= std::uint32_t{1} << (Bits - 1);
  }
  result.right = result.left;
  std::reverse(result.right.begin(), result.right.end());
  result.minimum = raw_q(result.left.front());
  result.maximum = raw_q(result.left.back());
  return result;
}

pool make_log_pool(std::size_t count, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<long double> unit(0.0L, 1.0L);
  pool result;
  result.left.reserve(count);
  result.right.reserve(count);
  result.minimum = std::numeric_limits<long double>::infinity();
  result.maximum = -std::numeric_limits<long double>::infinity();
  for (std::size_t index = 0; index < count; ++index) {
    const auto ql = q_lower + (q_upper - q_lower) * unit(rng);
    const auto qr = q_lower + q_upper - ql;
    auto left = encode_raw(static_cast<double>(std::exp2(ql)));
    auto right = encode_raw(static_cast<double>(std::exp2(qr)));
    if (rng() & 1u) left |= std::uint32_t{1} << (Bits - 1);
    if (rng() & 1u) right |= std::uint32_t{1} << (Bits - 1);
    if (!std::isfinite(reference_value(left)) || reference_value(left) == 0.0 ||
        !std::isfinite(reference_value(right)) || reference_value(right) == 0.0)
      throw benchmark_error("paired log generator produced a special value");
    result.left.push_back(left);
    result.right.push_back(right);
    result.minimum = std::min({result.minimum, raw_q(left), raw_q(right)});
    result.maximum = std::max({result.maximum, raw_q(left), raw_q(right)});
  }
  return result;
}

std::size_t storage_bytes(std::size_t count) {
  return (count * std::size_t(Bits) + 7u) / 8u;
}

std::vector<std::uint8_t> pack(std::size_t count,
                               const std::vector<std::uint32_t> &codes,
                               std::size_t row_length = 0,
                               std::uint64_t seed = 0) {
  std::vector<std::uint8_t> bytes(storage_bytes(count) + (Bits == 14 ? 4 : 0), 0);
  for (std::size_t index = 0; index < count; ++index) {
    auto raw = codes[index % codes.size()];
    if (row_length != 0) {
      const auto sign = ((index / row_length) * 0x9e3779b97f4a7c15ULL + index +
                         seed) & 1u;
      raw = (raw & ~(std::uint32_t{1} << (Bits - 1))) |
            (static_cast<std::uint32_t>(sign) << (Bits - 1));
    }
    if constexpr (Bits == 8) bytes[index] = raw;
    else if constexpr (Bits == 16) {
      const auto value = static_cast<std::uint16_t>(raw);
      std::memcpy(bytes.data() + index * 2, &value, 2);
    } else if constexpr (Bits == 32) {
      std::memcpy(bytes.data() + index * 4, &raw, 4);
    } else {
      const auto bit = index * std::size_t{14};
      const auto byte = bit >> 3;
      const auto shift = bit & 7u;
      const auto placed = (raw & 0x3fffu) << shift;
      for (int part = 0; part < 3; ++part)
        bytes[byte + part] |= static_cast<std::uint8_t>(placed >> (8 * part));
    }
  }
  return bytes;
}

struct event_timer {
  cudaEvent_t start{}, stop{};
  event_timer() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
  ~event_timer() { cudaEventDestroy(stop); cudaEventDestroy(start); }
  template <typename Launch> double measure(Launch launch) {
    CUDA_CHECK(cudaEventRecord(start)); launch(); CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); float ms{};
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); return ms;
  }
};

const char *decoder_name(bw::decoder_kind decoder) {
  switch (decoder) {
  case bw::decoder_kind::native_scalar: return "native_scalar";
  case bw::decoder_kind::direct_branchy: return "direct_branchy";
  case bw::decoder_kind::direct_masked: return "direct_masked";
  case bw::decoder_kind::full_lut_shared: return "full_lut_shared";
  case bw::decoder_kind::full_lut_global: return "full_lut_global";
  case bw::decoder_kind::subnormal_lut_global: return "subnormal_lut_global";
  case bw::decoder_kind::prefix_lut_global: return "prefix_lut_global";
  default: return "unsupported";
  }
}

template <bw::decoder_kind Decoder> constexpr std::size_t table_entries() {
  if constexpr (Decoder == bw::decoder_kind::full_lut_global ||
                Decoder == bw::decoder_kind::full_lut_shared)
    return std::size_t{1} << Bits;
  if constexpr (Decoder == bw::decoder_kind::prefix_lut_global)
    return std::size_t{1} << (Format::exponent_bits + 1);
  if constexpr (Decoder == bw::decoder_kind::subnormal_lut_global)
    return std::size_t{1} << Format::fraction_bits;
  return 0;
}

struct benchmark_variant {
  bw::decoder_kind kind{};
  std::string name;
  std::size_t lut_bytes{};
  std::size_t shared_bytes{};
  bool feasible{true};
  std::function<void()> launch;
};

class runner {
public:
  runner(const settings &settings, std::ofstream &samples,
         std::ofstream &validation, std::ofstream &histograms, int max_shared)
      : settings_(settings), samples_(samples), validation_(validation),
        histograms_(histograms), max_shared_(max_shared), partials_(dot_blocks),
        result_(settings.gemv_m) {}

  void run_distribution(const std::string &distribution, const pool &codes) {
    write_histogram(distribution, codes);
    run_dot(distribution, codes);
    run_gemv(distribution, codes);
  }

  void validate() {
    const std::size_t count = Bits <= 16 ? std::size_t{1} << Bits : 1'000'000;
    std::vector<std::uint32_t> raw(count);
    if constexpr (Bits <= 16) std::iota(raw.begin(), raw.end(), 0u);
    else {
      std::mt19937_64 rng(settings_.seed ^ 0xc6904f51ULL);
      for (auto &value : raw) value = static_cast<std::uint32_t>(rng());
    }
    auto host = pack(count, raw);
    device_buffer<std::uint8_t> input(host.size());
    device_buffer<Float> output(count);
    CUDA_CHECK(cudaMemcpy(input.get(), host.data(), host.size(), cudaMemcpyHostToDevice));
    auto variants = validation_variants(input, output, count);
    for (auto &entry : variants) {
      if (!entry.feasible) {
        validation_ << Format::name << ',' << arithmetic_name() << ','
                    << entry.name << ",0,0,0,infeasible_shared_memory\n";
        continue;
      }
      entry.launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<Float> decoded(count);
      CUDA_CHECK(cudaMemcpy(decoded.data(), output.get(), count * sizeof(Float), cudaMemcpyDeviceToHost));
      std::size_t failures{};
      for (std::size_t index = 0; index < count; ++index) {
        const auto expected = static_cast<Float>(reference_value(raw[index]));
        if (std::isnan(expected) && std::isnan(decoded[index])) continue;
        if (pt::to_bits(expected) != pt::to_bits(decoded[index])) ++failures;
      }
      validation_ << Format::name << ',' << arithmetic_name() << ',' << entry.name
                  << ',' << count << ',' << failures << ",0,"
                  << (failures == 0 ? "pass" : "fail") << '\n';
      if (failures) throw benchmark_error("IEEE decoder validation failed");
    }
  }

private:
  const char *arithmetic_name() const {
    return Compute == bw::compute_kind::fp32 ? "fp32" : "fp64";
  }

  template <bw::decoder_kind Decoder> device_buffer<Float> make_table() {
    constexpr auto entries = table_entries<Decoder>();
    device_buffer<Float> device(entries);
    if constexpr (entries != 0) {
      std::vector<Float> host(entries);
      for (std::size_t index = 0; index < entries; ++index) {
        std::uint32_t raw{};
        if constexpr (Decoder == bw::decoder_kind::prefix_lut_global)
          raw = static_cast<std::uint32_t>(index << Format::fraction_bits);
        else raw = static_cast<std::uint32_t>(index);
        host[index] = static_cast<Float>(reference_value(raw));
      }
      CUDA_CHECK(cudaMemcpy(device.get(), host.data(), entries * sizeof(Float), cudaMemcpyHostToDevice));
    }
    return device;
  }

  template <bw::decoder_kind Decoder>
  benchmark_variant make_variant(device_buffer<std::uint8_t> &left,
                                 device_buffer<std::uint8_t> &right,
                                 std::size_t count, bool gemv) {
    using decoder = pt::ieee_scalar_decoder<Format, Compute, Decoder>;
    constexpr bool shared = pt::ieee_shared_v<Decoder>;
    constexpr auto entries = table_entries<Decoder>();
    auto table = make_table<Decoder>();
    // Keep table ownership alive with a shared pointer-like heap buffer.
    auto *table_ptr = table.get();
    owned_tables_.push_back(std::move(table));
    const auto lut_bytes = entries * sizeof(Float);
    const auto shared_bytes = (shared ? lut_bytes : 0) + threads * sizeof(Float);
    benchmark_variant result{Decoder, decoder_name(Decoder), lut_bytes,
                             shared_bytes, shared_bytes <= std::size_t(max_shared_), {}};
    if (!result.feasible) return result;
    if constexpr (shared) {
      if (gemv)
        CUDA_CHECK(cudaFuncSetAttribute(pt::gemv_kernel<decoder, Bits, Float, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared_bytes)));
      else
        CUDA_CHECK(cudaFuncSetAttribute(pt::dot_kernel<decoder, Bits, Float, true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared_bytes)));
    }
    auto *left_ptr = left.get(); auto *right_ptr = right.get();
    if (gemv) {
      result.launch = [this, left_ptr, right_ptr, table_ptr, shared_bytes] {
        pt::gemv_kernel<decoder, Bits, Float, shared>
            <<<settings_.gemv_m, threads, shared_bytes>>>(
                {left_ptr}, {right_ptr}, settings_.gemv_m, settings_.gemv_n,
                {table_ptr}, result_.get());
      };
    } else {
      result.launch = [this, left_ptr, right_ptr, table_ptr, shared_bytes, count] {
        pt::dot_kernel<decoder, Bits, Float, shared>
            <<<dot_blocks, threads, shared_bytes>>>(
                {left_ptr}, {right_ptr}, count, {table_ptr}, partials_.get());
        pt::finalize_dot_kernel<<<1, threads>>>(partials_.get(), dot_blocks,
                                                result_.get());
      };
    }
    return result;
  }

  template <std::uint32_t Mask = strategy_mask>
  std::vector<benchmark_variant> make_variants(device_buffer<std::uint8_t> &left,
                                               device_buffer<std::uint8_t> &right,
                                               std::size_t count, bool gemv) {
    owned_tables_.clear();
    std::vector<benchmark_variant> result;
    if constexpr (Mask & 1u) result.push_back(make_variant<bw::decoder_kind::native_scalar>(left,right,count,gemv));
    if constexpr (Mask & 2u) result.push_back(make_variant<bw::decoder_kind::direct_branchy>(left,right,count,gemv));
    if constexpr (Mask & 4u) result.push_back(make_variant<bw::decoder_kind::direct_masked>(left,right,count,gemv));
    if constexpr (Mask & 8u) result.push_back(make_variant<bw::decoder_kind::full_lut_shared>(left,right,count,gemv));
    if constexpr (Mask & 16u) result.push_back(make_variant<bw::decoder_kind::full_lut_global>(left,right,count,gemv));
    if constexpr (Mask & 32u) result.push_back(make_variant<bw::decoder_kind::subnormal_lut_global>(left,right,count,gemv));
    if constexpr (Mask & 64u) result.push_back(make_variant<bw::decoder_kind::prefix_lut_global>(left,right,count,gemv));
    return result;
  }

  template <std::uint32_t Mask = strategy_mask>
  std::vector<benchmark_variant> validation_variants(device_buffer<std::uint8_t> &input,
                                                     device_buffer<Float> &output,
                                                     std::size_t count) {
    owned_tables_.clear();
    std::vector<benchmark_variant> variants;
    auto add = [&](auto decoder_tag) {
      constexpr auto Decoder = decltype(decoder_tag)::value;
      using decoder = pt::ieee_scalar_decoder<Format, Compute, Decoder>;
      constexpr bool shared = pt::ieee_shared_v<Decoder>;
      constexpr auto entries = table_entries<Decoder>();
      auto table = make_table<Decoder>(); auto *table_ptr = table.get();
      owned_tables_.push_back(std::move(table));
      const auto shared_bytes = (shared ? entries * sizeof(Float) : 0);
      benchmark_variant entry{Decoder, decoder_name(Decoder), entries*sizeof(Float),
                              shared_bytes, shared_bytes <= std::size_t(max_shared_), {}};
      if (entry.feasible) {
        if constexpr (shared)
          CUDA_CHECK(cudaFuncSetAttribute(
              pt::validate_generic_decoder_kernel<decoder, Bits, Float, true>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(shared_bytes)));
        auto *input_ptr = input.get();
        auto *output_ptr = output.get();
        entry.launch = [input_ptr, output_ptr, table_ptr, shared_bytes, count] {
          pt::validate_generic_decoder_kernel<decoder, Bits, Float, shared>
              <<<std::min<std::size_t>(4096,(count+255)/256),256,shared_bytes>>>(
                  {input_ptr}, {table_ptr}, output_ptr, count);
        };
      }
      variants.push_back(std::move(entry));
    };
    if constexpr (Mask & 1u) add(std::integral_constant<bw::decoder_kind,bw::decoder_kind::native_scalar>{});
    if constexpr (Mask & 2u) add(std::integral_constant<bw::decoder_kind,bw::decoder_kind::direct_branchy>{});
    if constexpr (Mask & 4u) add(std::integral_constant<bw::decoder_kind,bw::decoder_kind::direct_masked>{});
    if constexpr (Mask & 8u) add(std::integral_constant<bw::decoder_kind,bw::decoder_kind::full_lut_shared>{});
    if constexpr (Mask & 16u) add(std::integral_constant<bw::decoder_kind,bw::decoder_kind::full_lut_global>{});
    if constexpr (Mask & 32u) add(std::integral_constant<bw::decoder_kind,bw::decoder_kind::subnormal_lut_global>{});
    if constexpr (Mask & 64u) add(std::integral_constant<bw::decoder_kind,bw::decoder_kind::prefix_lut_global>{});
    return variants;
  }

  void time(const std::string &distribution, const char *kernel,
            std::size_t n, std::size_t m,
            std::vector<benchmark_variant> &variants) {
    for (auto &entry : variants) {
      if (!entry.feasible) {
        samples_ << Format::name << ",ieee," << Bits << ',' << arithmetic_name()
                 << ',' << distribution << ',' << kernel << ',' << entry.name
                 << ",dense,scalar,1," << n << ',' << m << ',' << entry.lut_bytes
                 << ',' << entry.shared_bytes << ",-1,-1,-1,infeasible_shared_memory\n";
      } else {
        for (int warm = 0; warm < settings_.warmup; ++warm) entry.launch();
      }
    }
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    for (int round = 0; round < settings_.samples; ++round) {
      for (std::size_t position = 0; position < variants.size(); ++position) {
        auto &entry = variants[(round + position) % variants.size()];
        if (!entry.feasible) continue;
        const auto ms = timer_.measure(entry.launch);
        samples_ << Format::name << ",ieee," << Bits << ',' << arithmetic_name()
                 << ',' << distribution << ',' << kernel << ',' << entry.name
                 << ",dense,scalar,1," << n << ',' << m << ',' << entry.lut_bytes
                 << ',' << entry.shared_bytes << ',' << round << ',' << position
                 << ',' << std::setprecision(9) << ms << ",ok\n";
      }
    }
  }

  void run_dot(const std::string &distribution, const pool &codes) {
    auto lh = pack(settings_.dot_n, codes.left); auto rh = pack(settings_.dot_n, codes.right);
    device_buffer<std::uint8_t> left(lh.size()), right(rh.size());
    CUDA_CHECK(cudaMemcpy(left.get(),lh.data(),lh.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(right.get(),rh.data(),rh.size(),cudaMemcpyHostToDevice));
    auto variants = make_variants(left,right,settings_.dot_n,false);
    time(distribution,"dot",settings_.dot_n,1,variants);
  }
  void run_gemv(const std::string &distribution, const pool &codes) {
    auto mh = pack(settings_.gemv_m*settings_.gemv_n,codes.left,settings_.gemv_n,settings_.seed);
    auto vh = pack(settings_.gemv_n,codes.right);
    device_buffer<std::uint8_t> matrix(mh.size()), vector(vh.size());
    CUDA_CHECK(cudaMemcpy(matrix.get(),mh.data(),mh.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(vector.get(),vh.data(),vh.size(),cudaMemcpyHostToDevice));
    auto variants = make_variants(matrix,vector,settings_.gemv_n,true);
    time(distribution,"gemv",settings_.gemv_n,settings_.gemv_m,variants);
  }
  void write_histogram(const std::string &distribution, const pool &codes) {
    if (codes.buckets.empty()) {
      histograms_ << Format::name << ',' << arithmetic_name() << ',' << distribution
                  << ",log_interval,all," << codes.left.size() << ','
                  << double(codes.minimum) << ',' << double(codes.maximum) << '\n';
    } else for (std::size_t bucket=0;bucket<codes.buckets.size();++bucket)
      histograms_ << Format::name << ',' << arithmetic_name() << ',' << distribution
                  << ",field_bucket," << bucket << ',' << codes.buckets[bucket]
                  << ',' << double(codes.minimum) << ',' << double(codes.maximum) << '\n';
  }

  const settings &settings_;
  std::ofstream &samples_, &validation_, &histograms_;
  int max_shared_{};
  device_buffer<Float> partials_, result_;
  std::vector<device_buffer<Float>> owned_tables_;
  event_timer timer_;
};

} // namespace

int main(int argc, char **argv) try {
  const auto settings = parse_arguments(argc, argv);
  int max_shared{}; CUDA_CHECK(cudaDeviceGetAttribute(&max_shared,
      cudaDevAttrMaxSharedMemoryPerBlockOptin,0));
  std::ofstream samples(settings.output), validation(settings.validation_output), histograms(settings.histogram_output);
  if (!samples || !validation || !histograms) throw benchmark_error("failed to open output");
  samples << "format,family,bits,arithmetic,distribution,kernel,strategy,storage_layout,access_method,packet_values,N,M,lut_bytes,dynamic_shared_bytes,round,order_position,kernel_ms,status\n";
  validation << "format,arithmetic,strategy,codes_checked,failures,max_ulp,status\n";
  histograms << "format,arithmetic,distribution,histogram,bucket,count,realized_q_min,realized_q_max\n";
  runner benchmark(settings,samples,validation,histograms,max_shared);
  benchmark.validate();
  const auto pool_size = settings.mode == "smoke" ? 4096u : 65536u;
  benchmark.run_distribution("field_balanced_finite",make_field_pool(pool_size,settings.seed));
  benchmark.run_distribution("paired_log_uniform_finite",make_log_pool(pool_size,settings.seed^0xa0187ULL));
  std::cout << "completed " << Format::name << ' ' <<
      (Compute == bw::compute_kind::fp32 ? "fp32" : "fp64") << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n'; return 1;
}
