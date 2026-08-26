#include "posit_takum_core.hpp"
#include "posit_takum_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef AUT_PT_FAMILY
#error AUT_PT_FAMILY must be defined
#endif
#ifndef AUT_PT_BITS
#error AUT_PT_BITS must be defined
#endif
#ifndef AUT_PT_ES
#define AUT_PT_ES 0
#endif

namespace pt = aut::pt;

namespace {

constexpr auto benchmark_family = static_cast<pt::family>(AUT_PT_FAMILY);
constexpr int storage_bits = AUT_PT_BITS;
constexpr int posit_es = AUT_PT_ES;
constexpr int threads = 256;
constexpr int dot_blocks = 512;

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
      : data_(std::exchange(other.data_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  device_buffer &operator=(device_buffer &&other) noexcept {
    if (this != &other) {
      release();
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  ~device_buffer() { release(); }

  void reset(std::size_t count) {
    release();
    count_ = count;
    if (count != 0) {
      CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }
  }
  T *get() { return data_; }
  const T *get() const { return data_; }
  std::size_t size() const { return count_; }

private:
  void release() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = nullptr;
    count_ = 0;
  }
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
  if (used != text.size() || value == 0) {
    throw benchmark_error(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(value);
}

settings parse_arguments(int argc, char **argv) {
  settings result;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw benchmark_error("missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--mode") {
      result.mode = value();
    } else if (argument == "--output") {
      result.output = value();
    } else if (argument == "--validation-output") {
      result.validation_output = value();
    } else if (argument == "--histogram-output") {
      result.histogram_output = value();
    } else if (argument == "--seed") {
      result.seed = parse_size(value(), argument);
    } else if (argument == "--dot-n") {
      result.dot_n = parse_size(value(), argument);
    } else if (argument == "--gemv-m") {
      result.gemv_m = parse_size(value(), argument);
    } else if (argument == "--gemv-n") {
      result.gemv_n = parse_size(value(), argument);
    } else if (argument == "--warmup") {
      result.warmup = static_cast<int>(parse_size(value(), argument));
    } else if (argument == "--samples") {
      result.samples = static_cast<int>(parse_size(value(), argument));
    } else {
      throw benchmark_error("unknown argument: " + argument);
    }
  }
  if (result.mode == "smoke") {
    result.dot_n = std::min<std::size_t>(result.dot_n, 65536);
    result.gemv_m = std::min<std::size_t>(result.gemv_m, 16);
    result.gemv_n = std::min<std::size_t>(result.gemv_n, 256);
    result.warmup = std::min(result.warmup, 1);
    result.samples = std::min(result.samples, 2);
  } else if (result.mode != "full") {
    throw benchmark_error("--mode must be smoke or full");
  }
  if (result.output.empty() || result.validation_output.empty() ||
      result.histogram_output.empty()) {
    throw benchmark_error(
        "--output, --validation-output, and --histogram-output are required");
  }
  return result;
}

const char *family_name() {
  if constexpr (benchmark_family == pt::family::posit) {
    return "posit";
  } else if constexpr (benchmark_family == pt::family::takum_linear) {
    return "takum";
  } else {
    return "takum_log";
  }
}

std::string format_name() {
  if constexpr (benchmark_family == pt::family::posit) {
    return "posit" + std::to_string(storage_bits) + "_es" +
           std::to_string(posit_es);
  } else {
    return std::string(family_name()) + std::to_string(storage_bits);
  }
}

template <typename Float> const char *arithmetic_name() {
  return std::is_same_v<Float, float> ? "fp32" : "fp64";
}

const char *reference_name() {
  return benchmark_family == pt::family::takum_log
             ? "takum_log_paper_formula"
             : "universal_cross_validated_core";
}

std::pair<long double, long double> exponent_interval(bool fp32) {
  if constexpr (benchmark_family == pt::family::posit) {
    if constexpr (storage_bits == 8) return {-5, 5};
    if constexpr (storage_bits == 14) return {-22, 22};
    if constexpr (storage_bits == 16) return {-26, 26};
    return {-112, 112};
  } else if constexpr (benchmark_family == pt::family::takum_linear) {
    return fp32 ? std::pair<long double, long double>{-140, 127}
                : std::pair<long double, long double>{-240, 240};
  } else {
    return fp32 ? std::pair<long double, long double>{-120, 120}
                : std::pair<long double, long double>{-170, 170};
  }
}

long double raw_q(std::uint32_t raw) {
  const auto value =
      pt::decode_long_double<benchmark_family, storage_bits, posit_es>(raw);
  return std::log2(std::abs(value));
}

int field_bucket(std::uint32_t positive_raw) {
  if constexpr (benchmark_family == pt::family::posit) {
    const auto aligned = positive_raw << (33 - storage_bits);
    const bool one = (aligned & 0x80000000u) != 0u;
    int run = one ? pt::leading_zeros(~aligned) : pt::leading_zeros(aligned);
    run = std::min(run, storage_bits - 1);
    return static_cast<int>(one) * storage_bits + run - 1;
  } else {
    const auto fields = pt::split_takum<storage_bits>(positive_raw);
    return static_cast<int>(fields.direction * 8 + fields.regime_code);
  }
}

std::vector<std::vector<std::uint32_t>>
field_candidates(long double lower, long double upper, std::mt19937_64 &rng) {
  const int bucket_count = benchmark_family == pt::family::posit
                               ? 2 * storage_bits
                               : 16;
  std::vector<std::vector<std::uint32_t>> buckets(bucket_count);
  auto accept = [&](std::uint32_t raw) {
    if (raw == 0u || raw >= pt::sign_mask<storage_bits>()) return;
    const auto q = raw_q(raw);
    if (std::isfinite(q) && q >= lower && q <= upper) {
      buckets[field_bucket(raw)].push_back(raw);
    }
  };

  if constexpr (storage_bits <= 16) {
    for (std::uint32_t raw = 1; raw < pt::sign_mask<storage_bits>(); ++raw) {
      accept(raw);
    }
  } else if constexpr (benchmark_family == pt::family::posit) {
    for (int one = 0; one <= 1; ++one) {
      for (int run = 1; run < storage_bits; ++run) {
        const int consumed = run + 1;
        if (consumed > storage_bits - 1) continue;
        for (int sample = 0; sample < 4096; ++sample) {
          std::uint32_t raw{};
          const int first = storage_bits - 2;
          if (one) {
            const auto run_mask = run == 32 ? 0xffffffffu
                                            : ((std::uint32_t{1} << run) - 1u);
            raw |= run_mask << (first - run + 1);
          } else {
            raw |= std::uint32_t{1} << (first - run);
          }
          const int tail_bits = storage_bits - 1 - consumed;
          if (tail_bits > 0) {
            raw |= static_cast<std::uint32_t>(rng()) &
                   ((std::uint32_t{1} << tail_bits) - 1u);
          }
          accept(raw);
        }
      }
    }
  } else {
    constexpr int tail_bits = storage_bits - 5;
    for (std::uint32_t dr = 0; dr < 16; ++dr) {
      for (int sample = 0; sample < 8192; ++sample) {
        const auto tail = static_cast<std::uint32_t>(rng()) &
                          ((std::uint32_t{1} << tail_bits) - 1u);
        accept((dr << tail_bits) | tail);
      }
    }
  }
  buckets.erase(std::remove_if(buckets.begin(), buckets.end(),
                               [](const auto &bucket) { return bucket.empty(); }),
                buckets.end());
  if (buckets.empty()) {
    throw benchmark_error("no admissible field buckets");
  }
  return buckets;
}

struct code_pair_pool {
  std::vector<std::uint32_t> left;
  std::vector<std::uint32_t> right;
  std::vector<std::size_t> bucket_counts;
  long double realized_min{};
  long double realized_max{};
};

code_pair_pool make_field_pool(std::size_t count, long double lower,
                               long double upper, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  auto buckets = field_candidates(lower, upper, rng);
  code_pair_pool result;
  result.left.reserve(count);
  result.bucket_counts.assign(buckets.size(), 0);
  for (std::size_t index = 0; index < count; ++index) {
    const auto bucket = index % buckets.size();
    const auto magnitude = buckets[bucket][rng() % buckets[bucket].size()];
    result.left.push_back(
        pt::apply_sign<storage_bits>(magnitude, (index & 1u) != 0u));
    ++result.bucket_counts[bucket];
  }
  std::sort(result.left.begin(), result.left.end(), [](auto a, auto b) {
    return raw_q(a) < raw_q(b);
  });
  result.right = result.left;
  std::reverse(result.right.begin(), result.right.end());
  result.realized_min = raw_q(result.left.front());
  result.realized_max = raw_q(result.left.back());
  return result;
}

code_pair_pool make_log_uniform_pool(std::size_t count, long double lower,
                                     long double upper, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<long double> unit(0.0L, 1.0L);
  code_pair_pool result;
  result.left.reserve(count);
  result.right.reserve(count);
  result.realized_min = std::numeric_limits<long double>::infinity();
  result.realized_max = -std::numeric_limits<long double>::infinity();
  for (std::size_t index = 0; index < count; ++index) {
    std::uint32_t left_magnitude{};
    std::uint32_t right_magnitude{};
    bool accepted = false;
    for (int attempt = 0; attempt < 4096 && !accepted; ++attempt) {
      const auto q_left = lower + (upper - lower) * unit(rng);
      const auto q_right = lower + upper - q_left;
      left_magnitude =
          pt::encode_positive_log2<benchmark_family, storage_bits, posit_es>(q_left);
      right_magnitude =
          pt::encode_positive_log2<benchmark_family, storage_bits, posit_es>(q_right);
      const auto realized_left = raw_q(left_magnitude);
      const auto realized_right = raw_q(right_magnitude);
      accepted = std::isfinite(realized_left) &&
                 std::isfinite(realized_right) && realized_left >= lower &&
                 realized_left <= upper && realized_right >= lower &&
                 realized_right <= upper;
    }
    if (!accepted) {
      throw benchmark_error(
          "could not quantize a paired-log sample inside its interval");
    }
    result.left.push_back(
        pt::apply_sign<storage_bits>(left_magnitude, (index & 1u) != 0u));
    result.right.push_back(pt::apply_sign<storage_bits>(
        right_magnitude, ((index >> 1) & 1u) != 0u));
    result.realized_min =
        std::min({result.realized_min, raw_q(result.left.back()),
                  raw_q(result.right.back())});
    result.realized_max =
        std::max({result.realized_max, raw_q(result.left.back()),
                  raw_q(result.right.back())});
  }
  return result;
}

void verify_pool(const code_pair_pool &pool, long double lower,
                 long double upper, bool require_interval_coverage) {
  auto verify_side = [&](const std::vector<std::uint32_t> &codes) {
    std::size_t negative{};
    for (const auto raw : codes) {
      const auto value =
          pt::decode_long_double<benchmark_family, storage_bits, posit_es>(raw);
      if (!std::isfinite(value) || value == 0.0L)
        throw benchmark_error("input pool contains zero or a special value");
      const auto q = std::log2(std::abs(value));
      if (q < lower || q > upper)
        throw benchmark_error("input pool escaped its exponent interval");
      negative += value < 0.0L;
    }
    if (negative * 2 + 1 < codes.size() ||
        negative * 2 > codes.size() + 1)
      throw benchmark_error("input signs are not balanced");
  };
  verify_side(pool.left);
  verify_side(pool.right);
  if (require_interval_coverage) {
    const auto allowance = (upper - lower) * 0.02L;
    if (pool.realized_min > lower + allowance ||
        pool.realized_max < upper - allowance)
      throw benchmark_error("paired-log pool does not cover its interval");
  }
}

std::size_t storage_bytes(std::size_t count) {
  return (count * static_cast<std::size_t>(storage_bits) + 7u) / 8u;
}

std::uint64_t mix_index(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

std::uint32_t randomize_significand(std::uint32_t raw, std::uint64_t hash,
                                    long double lower, long double upper,
                                    bool sign) {
  const auto original_magnitude = pt::magnitude_bits<storage_bits>(raw);
  auto magnitude = original_magnitude;
  int fraction_bits{};
  bool may_randomize = true;
  if constexpr (benchmark_family == pt::family::posit) {
    const auto aligned = magnitude << (33 - storage_bits);
    const bool regime_one = (aligned & 0x80000000u) != 0u;
    int run = regime_one ? pt::leading_zeros(~aligned)
                         : pt::leading_zeros(aligned);
    run = std::min(run, storage_bits - 1);
    const int consumed = run + (run < storage_bits - 1 ? 1 : 0);
    const int remaining = storage_bits - 1 - consumed;
    const int exponent_taken = std::min(remaining, posit_es);
    fraction_bits = remaining - exponent_taken;
    const auto tail = magnitude &
                      (consumed == storage_bits - 1
                           ? 0u
                           : ((std::uint32_t{1}
                               << (storage_bits - 1 - consumed)) -
                              1u));
    const auto exponent_raw = exponent_taken == 0
                                  ? 0u
                                  : tail >> fraction_bits;
    const auto exponent_bits = exponent_raw << (posit_es - exponent_taken);
    const int exponent = (regime_one ? run - 1 : -run) * (1 << posit_es) +
                         static_cast<int>(exponent_bits);
    may_randomize = exponent < upper;
  } else {
    const auto fields = pt::split_takum<storage_bits>(magnitude);
    fraction_bits = fields.tail_bits;
    if constexpr (benchmark_family == pt::family::takum_linear) {
      may_randomize = fields.characteristic < upper;
    } else {
      constexpr long double inv_two_ln2 =
          0.721347520444481703679962340500946L;
      may_randomize =
          (static_cast<long double>(fields.characteristic) + 1.0L) *
              inv_two_ln2 <=
          upper;
    }
  }
  if (may_randomize && fraction_bits > 0) {
    const auto fraction_mask =
        (std::uint32_t{1} << fraction_bits) - std::uint32_t{1};
    magnitude = (magnitude & ~fraction_mask) |
                (static_cast<std::uint32_t>(hash) & fraction_mask);
  }
  auto candidate = pt::apply_sign<storage_bits>(magnitude, sign);
  const auto value =
      pt::decode_long_double<benchmark_family, storage_bits, posit_es>(candidate);
  const auto q = std::log2(std::abs(value));
  if (!std::isfinite(value) || value == 0.0L || q < lower || q > upper)
    candidate = pt::apply_sign<storage_bits>(original_magnitude, sign);
  return candidate;
}

std::vector<std::uint8_t>
pack_codes(std::size_t count, const std::vector<std::uint32_t> &pool,
           std::size_t row_length = 0, std::uint64_t sign_seed = 0,
           long double lower = -std::numeric_limits<long double>::infinity(),
           long double upper = std::numeric_limits<long double>::infinity()) {
  const auto packed = storage_bytes(count);
  const auto allocation =
      storage_bits == 14 ? ((packed + 3u) & ~std::size_t{3}) + 4u : packed;
  std::vector<std::uint8_t> bytes(allocation, 0u);
  for (std::size_t index = 0; index < count; ++index) {
    const auto pool_index =
        row_length == 0 ? index % pool.size() : (index % row_length) % pool.size();
    auto raw = pool[pool_index];
    if (row_length != 0) {
      const auto row = index / row_length;
      const auto hash = mix_index(row * 0x9e3779b97f4a7c15ULL + index +
                                  sign_seed);
      const auto sign = ((row + index % row_length) & 1u) != 0u;
      raw = randomize_significand(raw, hash, lower, upper, sign);
    }
    if constexpr (storage_bits == 8) {
      bytes[index] = static_cast<std::uint8_t>(raw);
    } else if constexpr (storage_bits == 16) {
      std::uint16_t value = static_cast<std::uint16_t>(raw);
      std::memcpy(bytes.data() + index * 2, &value, sizeof(value));
    } else if constexpr (storage_bits == 32) {
      std::memcpy(bytes.data() + index * 4, &raw, sizeof(raw));
    } else {
      const auto bit = index * std::size_t{14};
      const auto byte = bit >> 3;
      const auto shift = static_cast<unsigned>(bit & 7u);
      const auto placed = static_cast<std::uint32_t>(raw & 0x3fffu) << shift;
      for (int part = 0; part < 3; ++part) {
        if (byte + part < bytes.size()) {
          bytes[byte + part] |= static_cast<std::uint8_t>(placed >> (part * 8));
        }
      }
    }
  }
  return bytes;
}

template <typename Float> class event_timer {
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

template <typename Float> struct variant {
  std::string strategy_name;
  std::size_t table_bytes{};
  std::size_t shared_bytes{};
  bool feasible{true};
  std::function<void()> launch;
};

template <typename Float> class arithmetic_runner {
public:
  arithmetic_runner(const settings &settings, std::ofstream &samples,
                    std::ofstream &validation, std::ofstream &histograms,
                    int max_shared)
      : settings_(settings), samples_(samples), validation_(validation),
        histograms_(histograms), max_shared_(max_shared), partials_(dot_blocks),
        dot_result_(1), gemv_result_(settings.gemv_m) {
    build_table();
  }

  void validate_decoder() {
    const std::size_t count = storage_bits <= 16
                                  ? (std::size_t{1} << storage_bits)
                                  : std::size_t{1'000'000};
    std::vector<std::uint32_t> raw(count);
    if constexpr (storage_bits <= 16) {
      std::iota(raw.begin(), raw.end(), 0u);
    } else {
      std::mt19937_64 rng(settings_.seed ^ 0xa4d1cb6e43ULL);
      for (auto &value : raw) value = static_cast<std::uint32_t>(rng());
      raw[0] = 0u;
      raw[1] = pt::sign_mask<storage_bits>();
      raw[2] = pt::mask<storage_bits>();
      raw[3] = 1u;
      raw[4] = pt::sign_mask<storage_bits>() - 1u;
      raw[5] = pt::sign_mask<storage_bits>() + 1u;
    }
    auto packed = pack_codes(count, raw);
    device_buffer<std::uint8_t> device_input(packed.size());
    device_buffer<Float> device_output(count);
    CUDA_CHECK(cudaMemcpy(device_input.get(), packed.data(), packed.size(),
                          cudaMemcpyHostToDevice));
    pt::validate_decode_kernel<benchmark_family, storage_bits, posit_es, Float>
        <<<std::min<std::size_t>(4096, (count + 255) / 256), 256>>>(
            {device_input.get()}, device_output.get(), count);
    CUDA_CHECK(cudaGetLastError());
    std::vector<Float> output(count);
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.get(),
                          count * sizeof(Float), cudaMemcpyDeviceToHost));
    std::size_t failures{};
    std::uint64_t max_ulp{};
    for (std::size_t index = 0; index < count; ++index) {
      const auto expected = pt::decode<benchmark_family, storage_bits, posit_es,
                                       Float>(raw[index]);
      if (std::isnan(expected) && std::isnan(output[index])) continue;
      const auto left = pt::to_bits(output[index]);
      const auto right = pt::to_bits(expected);
      const auto distance = left > right ? left - right : right - left;
      max_ulp = std::max<std::uint64_t>(max_ulp, distance);
      const auto allowed = benchmark_family == pt::family::takum_log
                               ? (std::is_same_v<Float, float> ? 2u : 1u)
                               : 0u;
      if (distance > allowed) ++failures;
    }
    validation_ << format_name() << ',' << arithmetic_name<Float>()
                << ",direct," << count << ',' << failures << ',' << max_ulp
                << ',' << reference_name() << ','
                << (failures == 0 ? "pass" : "fail") << '\n';
    if (failures != 0) {
      throw benchmark_error("GPU decoder validation failed");
    }
    if constexpr (storage_bits <= 16) {
      validate_lut<pt::strategy::full_lut_global>(device_input, device_output,
                                                   raw, count);
    }
    if constexpr (storage_bits <= 14) {
      validate_lut<pt::strategy::full_lut_shared>(device_input, device_output,
                                                   raw, count);
    }
  }

  void run_distribution(const std::string &distribution,
                        const code_pair_pool &pool) {
    write_histogram(distribution, pool);
    run_dot(distribution, pool);
    run_gemv(distribution, pool);
  }

private:
  template <pt::strategy Strategy>
  void validate_lut(const device_buffer<std::uint8_t> &input,
                    device_buffer<Float> &output,
                    const std::vector<std::uint32_t> &raw,
                    std::size_t count) {
    using decoder = pt::alternative_decoder<benchmark_family, storage_bits,
                                            posit_es, Float, Strategy>;
    constexpr bool shared = Strategy == pt::strategy::full_lut_shared;
    const auto shared_bytes = shared ? table_.size() * sizeof(Float) : 0u;
    const auto *name = shared ? "full_lut_shared" : "full_lut_global";
    if (shared_bytes > std::size_t(max_shared_)) {
      validation_ << format_name() << ',' << arithmetic_name<Float>() << ','
                  << name << ",0,0,0," << reference_name()
                  << ",infeasible_shared_memory\n";
      return;
    }
    if constexpr (shared) {
      CUDA_CHECK(cudaFuncSetAttribute(
          pt::validate_generic_decoder_kernel<decoder, storage_bits, Float,
                                              true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(shared_bytes)));
    }
    pt::validate_generic_decoder_kernel<decoder, storage_bits, Float, shared>
        <<<std::min<std::size_t>(4096, (count + 255) / 256), 256,
           shared_bytes>>>({input.get()}, {table_.get()}, output.get(), count);
    CUDA_CHECK(cudaGetLastError());
    std::vector<Float> decoded(count);
    CUDA_CHECK(cudaMemcpy(decoded.data(), output.get(), count * sizeof(Float),
                          cudaMemcpyDeviceToHost));
    std::size_t failures{};
    for (std::size_t index = 0; index < count; ++index) {
      const auto expected = pt::decode<benchmark_family, storage_bits, posit_es,
                                       Float>(raw[index]);
      if (std::isnan(expected) && std::isnan(decoded[index])) continue;
      if (pt::to_bits(expected) != pt::to_bits(decoded[index])) ++failures;
    }
    validation_ << format_name() << ',' << arithmetic_name<Float>() << ','
                << name << ',' << count << ',' << failures << ",0,"
                << reference_name() << ','
                << (failures == 0 ? "pass" : "fail") << '\n';
    if (failures != 0) throw benchmark_error("GPU LUT validation failed");
  }

  void build_table() {
    if constexpr (storage_bits <= 16) {
      const std::size_t entries = std::size_t{1} << storage_bits;
      std::vector<Float> host(entries);
      for (std::size_t raw = 0; raw < entries; ++raw) {
        host[raw] = pt::decode<benchmark_family, storage_bits, posit_es, Float>(
            static_cast<std::uint32_t>(raw));
      }
      table_.reset(entries);
      CUDA_CHECK(cudaMemcpy(table_.get(), host.data(), entries * sizeof(Float),
                            cudaMemcpyHostToDevice));
    }
  }

  void write_histogram(const std::string &distribution,
                       const code_pair_pool &pool) {
    const auto write_counts = [&](const char *name,
                                  const std::map<int, std::size_t> &counts) {
      for (const auto &[bucket, count] : counts)
        histograms_ << format_name() << ',' << arithmetic_name<Float>() << ','
                    << distribution << ',' << name << ',' << bucket << ','
                    << count << ',' << static_cast<double>(pool.realized_min)
                    << ',' << static_cast<double>(pool.realized_max) << '\n';
    };
    std::map<int, std::size_t> signs;
    for (const auto raw : pool.left)
      ++signs[(raw & pt::sign_mask<storage_bits>()) != 0u];
    write_counts("sign", signs);
    if (pool.bucket_counts.empty()) {
      histograms_ << format_name() << ',' << arithmetic_name<Float>() << ','
                  << distribution << ",log_interval,all," << pool.left.size()
                  << ',' << static_cast<double>(pool.realized_min) << ','
                  << static_cast<double>(pool.realized_max) << '\n';
    } else {
      for (std::size_t bucket = 0; bucket < pool.bucket_counts.size(); ++bucket) {
        histograms_ << format_name() << ',' << arithmetic_name<Float>() << ','
                    << distribution << ",field_bucket," << bucket << ','
                    << pool.bucket_counts[bucket] << ','
                    << static_cast<double>(pool.realized_min) << ','
                    << static_cast<double>(pool.realized_max) << '\n';
      }
      if constexpr (benchmark_family == pt::family::posit) {
        std::map<int, std::size_t> regimes;
        for (const auto raw : pool.left)
          ++regimes[field_bucket(pt::magnitude_bits<storage_bits>(raw))];
        write_counts("regime", regimes);
      } else {
        std::map<int, std::size_t> directions;
        std::map<int, std::size_t> regimes;
        std::map<int, std::size_t> characteristics;
        for (const auto raw : pool.left) {
          const auto fields = pt::split_takum<storage_bits>(raw);
          ++directions[static_cast<int>(fields.direction)];
          ++regimes[static_cast<int>(fields.regime_code)];
          ++characteristics[fields.characteristic];
        }
        write_counts("direction", directions);
        write_counts("regime", regimes);
        write_counts("characteristic", characteristics);
      }
    }
  }

  template <pt::strategy Strategy>
  variant<Float> make_dot_variant(const device_buffer<std::uint8_t> &left,
                                  const device_buffer<std::uint8_t> &right,
                                  std::size_t count) {
    constexpr bool shared = Strategy == pt::strategy::full_lut_shared;
    using decoder = pt::alternative_decoder<benchmark_family, storage_bits,
                                            posit_es, Float, Strategy>;
    const std::size_t table_bytes =
        Strategy == pt::strategy::direct
            ? 0u
            : (std::size_t{1} << storage_bits) * sizeof(Float);
    const std::size_t shared_bytes =
        (shared ? table_bytes : 0u) + threads * sizeof(Float);
    variant<Float> result{
        Strategy == pt::strategy::direct
            ? "direct"
            : (shared ? "full_lut_shared" : "full_lut_global"),
        table_bytes, shared_bytes, shared_bytes <= std::size_t(max_shared_), {}};
    if (!result.feasible) return result;
    if constexpr (shared) {
      CUDA_CHECK(cudaFuncSetAttribute(
          pt::dot_kernel<decoder, storage_bits, Float, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(shared_bytes)));
    }
    auto *left_ptr = left.get();
    auto *right_ptr = right.get();
    result.launch = [this, left_ptr, right_ptr, count, shared_bytes] {
      const decoder decode{table_.get()};
      pt::dot_kernel<decoder, storage_bits, Float, shared>
          <<<dot_blocks, threads, shared_bytes>>>(
              {left_ptr}, {right_ptr}, count, decode, partials_.get());
      pt::finalize_dot_kernel<<<1, threads>>>(partials_.get(), dot_blocks,
                                              dot_result_.get());
    };
    return result;
  }

  template <pt::strategy Strategy>
  variant<Float> make_gemv_variant(const device_buffer<std::uint8_t> &matrix,
                                   const device_buffer<std::uint8_t> &vector) {
    constexpr bool shared = Strategy == pt::strategy::full_lut_shared;
    using decoder = pt::alternative_decoder<benchmark_family, storage_bits,
                                            posit_es, Float, Strategy>;
    const std::size_t table_bytes =
        Strategy == pt::strategy::direct
            ? 0u
            : (std::size_t{1} << storage_bits) * sizeof(Float);
    const std::size_t shared_bytes =
        (shared ? table_bytes : 0u) + threads * sizeof(Float);
    variant<Float> result{
        Strategy == pt::strategy::direct
            ? "direct"
            : (shared ? "full_lut_shared" : "full_lut_global"),
        table_bytes, shared_bytes, shared_bytes <= std::size_t(max_shared_), {}};
    if (!result.feasible) return result;
    if constexpr (shared) {
      CUDA_CHECK(cudaFuncSetAttribute(
          pt::gemv_kernel<decoder, storage_bits, Float, true>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(shared_bytes)));
    }
    auto *matrix_ptr = matrix.get();
    auto *vector_ptr = vector.get();
    result.launch = [this, matrix_ptr, vector_ptr, shared_bytes] {
      const decoder decode{table_.get()};
      pt::gemv_kernel<decoder, storage_bits, Float, shared>
          <<<settings_.gemv_m, threads, shared_bytes>>>(
              {matrix_ptr}, {vector_ptr}, settings_.gemv_m,
              settings_.gemv_n, decode, gemv_result_.get());
    };
    return result;
  }

  void time_variants(const std::string &distribution, const std::string &kernel,
                     std::size_t n, std::size_t m, std::size_t input_bytes,
                     std::vector<variant<Float>> &variants) {
    for (auto &entry : variants) {
      if (!entry.feasible) {
        samples_ << format_name() << ',' << family_name() << ',' << storage_bits
                 << ',' << arithmetic_name<Float>() << ',' << distribution << ','
                 << kernel << ',' << entry.strategy_name
                 << ",dense,scalar,1," << n << ',' << m << ',' << input_bytes << ','
                 << entry.table_bytes << ',' << entry.shared_bytes
                 << ",-1,-1,-1,infeasible_shared_memory\n";
        continue;
      }
      for (int warm = 0; warm < settings_.warmup; ++warm) entry.launch();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<Float> checked(kernel == "dot" ? 1u : m);
      auto *source = kernel == "dot" ? dot_result_.get() : gemv_result_.get();
      CUDA_CHECK(cudaMemcpy(checked.data(), source,
                            checked.size() * sizeof(Float),
                            cudaMemcpyDeviceToHost));
      if (!std::all_of(checked.begin(), checked.end(),
                       [](Float value) { return std::isfinite(value); }))
        throw benchmark_error("nonfinite kernel output for " +
                              entry.strategy_name + " in " + distribution +
                              " " + kernel);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int round = 0; round < settings_.samples; ++round) {
      for (std::size_t position = 0; position < variants.size(); ++position) {
        auto &entry = variants[(position + round) % variants.size()];
        if (!entry.feasible) continue;
        const auto milliseconds = timer_.measure(entry.launch);
        CUDA_CHECK(cudaGetLastError());
        samples_ << format_name() << ',' << family_name() << ',' << storage_bits
                 << ',' << arithmetic_name<Float>() << ',' << distribution << ','
                 << kernel << ',' << entry.strategy_name
                 << ",dense,scalar,1," << n << ',' << m << ',' << input_bytes << ','
                 << entry.table_bytes << ',' << entry.shared_bytes << ','
                 << round << ',' << position << ',' << std::setprecision(9)
                 << milliseconds << ",ok\n";
      }
    }
  }

  void run_dot(const std::string &distribution, const code_pair_pool &pool) {
    auto left_host = pack_codes(settings_.dot_n, pool.left);
    device_buffer<std::uint8_t> left(left_host.size());
    CUDA_CHECK(cudaMemcpy(left.get(), left_host.data(), left_host.size(),
                          cudaMemcpyHostToDevice));
    auto right_host = pack_codes(settings_.dot_n, pool.right);
    device_buffer<std::uint8_t> right(right_host.size());
    CUDA_CHECK(cudaMemcpy(right.get(), right_host.data(), right_host.size(),
                          cudaMemcpyHostToDevice));
    std::vector<variant<Float>> cases;
    cases.push_back(make_dot_variant<pt::strategy::direct>(
        left, right, settings_.dot_n));
    if constexpr (storage_bits <= 16) {
      cases.push_back(make_dot_variant<pt::strategy::full_lut_global>(
          left, right, settings_.dot_n));
    }
    if constexpr (storage_bits <= 14) {
      cases.push_back(make_dot_variant<pt::strategy::full_lut_shared>(
          left, right, settings_.dot_n));
    }
    time_variants(distribution, "dot", settings_.dot_n, 1,
                  left.size() + right.size(), cases);
  }

  void run_gemv(const std::string &distribution, const code_pair_pool &pool) {
    const auto matrix_count = settings_.gemv_m * settings_.gemv_n;
    const auto [lower, upper] =
        exponent_interval(std::is_same_v<Float, float>);
    auto matrix_host = pack_codes(matrix_count, pool.left, settings_.gemv_n,
                                  settings_.seed, lower, upper);
    device_buffer<std::uint8_t> matrix(matrix_host.size());
    CUDA_CHECK(cudaMemcpy(matrix.get(), matrix_host.data(), matrix_host.size(),
                          cudaMemcpyHostToDevice));
    auto vector_host = pack_codes(settings_.gemv_n, pool.right);
    device_buffer<std::uint8_t> vector(vector_host.size());
    CUDA_CHECK(cudaMemcpy(vector.get(), vector_host.data(), vector_host.size(),
                          cudaMemcpyHostToDevice));
    std::vector<variant<Float>> cases;
    cases.push_back(
        make_gemv_variant<pt::strategy::direct>(matrix, vector));
    if constexpr (storage_bits <= 16) {
      cases.push_back(
          make_gemv_variant<pt::strategy::full_lut_global>(matrix, vector));
    }
    if constexpr (storage_bits <= 14) {
      cases.push_back(
          make_gemv_variant<pt::strategy::full_lut_shared>(matrix, vector));
    }
    time_variants(distribution, "gemv", settings_.gemv_n, settings_.gemv_m,
                  matrix.size() + vector.size(), cases);
  }

  const settings &settings_;
  std::ofstream &samples_;
  std::ofstream &validation_;
  std::ofstream &histograms_;
  int max_shared_{};
  device_buffer<Float> table_;
  device_buffer<Float> partials_;
  device_buffer<Float> dot_result_;
  device_buffer<Float> gemv_result_;
  event_timer<Float> timer_;
};

void write_headers(std::ofstream &samples, std::ofstream &validation,
                   std::ofstream &histograms) {
  samples << "format,family,bits,arithmetic,distribution,kernel,strategy,"
             "storage_layout,access_method,packet_values,N,M,input_bytes,lut_bytes,"
             "dynamic_shared_bytes,round,order_position,kernel_ms,status\n";
  validation << "format,arithmetic,strategy,codes_checked,failures,max_ulp,reference,status\n";
  histograms << "format,arithmetic,distribution,histogram,bucket,count,"
                "realized_q_min,realized_q_max\n";
}

template <typename Float>
void run_arithmetic(const settings &settings, std::ofstream &samples,
                    std::ofstream &validation, std::ofstream &histograms,
                    int max_shared) {
  arithmetic_runner<Float> runner(settings, samples, validation, histograms,
                                  max_shared);
  runner.validate_decoder();
  const auto [lower, upper] =
      exponent_interval(std::is_same_v<Float, float>);
  const std::size_t pool_size = settings.mode == "smoke" ? 4096 : 65536;
  const auto field = make_field_pool(pool_size, lower, upper,
                                     settings.seed ^ sizeof(Float));
  verify_pool(field, lower, upper, false);
  runner.run_distribution("field_balanced_finite", field);
  const auto log = make_log_uniform_pool(pool_size, lower, upper,
                                         settings.seed ^ 0x52b61e19ULL ^
                                             sizeof(Float));
  verify_pool(log, lower, upper, true);
  runner.run_distribution("paired_log_uniform_finite", log);
}

} // namespace

int main(int argc, char **argv) try {
  const auto settings = parse_arguments(argc, argv);
  int device{};
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int max_shared{};
  CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  std::ofstream samples(settings.output);
  std::ofstream validation(settings.validation_output);
  std::ofstream histograms(settings.histogram_output);
  if (!samples || !validation || !histograms) {
    throw benchmark_error("failed to open an output file");
  }
  write_headers(samples, validation, histograms);
  std::cout << "format=" << format_name() << " gpu=" << properties.name
            << " mode=" << settings.mode << '\n';
  run_arithmetic<float>(settings, samples, validation, histograms, max_shared);
  run_arithmetic<double>(settings, samples, validation, histograms, max_shared);
  std::cout << "completed " << format_name() << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << "error: " << error.what() << '\n';
  return 1;
}
