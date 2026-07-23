#include "dot_dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

struct summary {
  long double dot{};
  long double sum_abs{};
  float minimum_nonzero{std::numeric_limits<float>::infinity()};
  float maximum{};
  std::uint64_t zeros{};
  std::uint64_t subnormals{};
  std::uint64_t nonfinite{};
  std::uint64_t x_fingerprint{1469598103934665603ULL};
  std::uint64_t y_fingerprint{1469598103934665603ULL};
};

std::uint32_t float_bits(float value) {
  std::uint32_t result{};
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

void update_fingerprint(std::uint64_t &fingerprint, float value) {
  fingerprint ^= float_bits(value);
  fingerprint *= 1099511628211ULL;
}

summary inspect(aut::dataset::parameters spec, std::uint64_t count) {
  compensated_sum dot;
  compensated_sum sum_abs;
  summary result;
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto x = aut::dataset::value(spec, 0, i, count);
    const auto y = aut::dataset::value(spec, 1, i, count);
    update_fingerprint(result.x_fingerprint, x);
    update_fingerprint(result.y_fingerprint, y);
    for (const auto value : {x, y}) {
      const auto absolute = std::fabs(value);
      if (value == 0.0f) {
        ++result.zeros;
      } else {
        result.minimum_nonzero = std::min(result.minimum_nonzero, absolute);
      }
      result.maximum = std::max(result.maximum, absolute);
      if (std::fpclassify(value) == FP_SUBNORMAL) {
        ++result.subnormals;
      }
      if (!std::isfinite(value)) {
        ++result.nonfinite;
      }
    }
    const auto product =
        static_cast<long double>(x) * static_cast<long double>(y);
    dot.add(product);
    sum_abs.add(std::fabs(product));
  }
  result.dot = dot.value();
  result.sum_abs = sum_abs.value();
  return result;
}

std::uint64_t parse_count(const std::string &text) {
  std::size_t consumed{};
  const auto value = std::stoull(text, &consumed, 0);
  if (consumed != text.size() || value == 0) {
    throw std::invalid_argument("counts must be positive integers");
  }
  return value;
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::vector<std::uint64_t> counts{1ULL << 10, 1ULL << 16, 1ULL << 20};
    if (argc > 1) {
      counts.clear();
      for (int i = 1; i < argc; ++i) {
        counts.push_back(parse_count(argv[i]));
      }
    }

    std::cout
        << "dataset,n,dot_reference,sum_abs,dot_condition_number,"
           "minimum_nonzero_input,maximum_input,zeros,subnormals,nonfinite,"
           "x_fingerprint,y_fingerprint\n";
    std::cout << std::setprecision(21);
    for (const auto &dataset : aut::dataset::accuracy_cases) {
      for (const auto count : counts) {
        const auto result = inspect(dataset.values, count);
        const auto condition =
            result.dot == 0.0L ? std::numeric_limits<long double>::infinity()
                               : result.sum_abs / std::fabs(result.dot);
        std::cout << dataset.id << ',' << count << ',' << result.dot << ','
                  << result.sum_abs << ',' << condition << ','
                  << result.minimum_nonzero << ',' << result.maximum << ','
                  << result.zeros << ',' << result.subnormals << ','
                  << result.nonfinite << ",0x" << std::hex
                  << result.x_fingerprint << ",0x" << result.y_fingerprint
                  << std::dec << '\n';
      }
    }
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
