#ifndef ACCESSOR_UNIVERSAL_TEST_ACCURACY_STATISTICS_HPP_
#define ACCESSOR_UNIVERSAL_TEST_ACCURACY_STATISTICS_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace aut::accuracy::statistics {

enum class value_class { finite, positive_infinity, negative_infinity, nan };

inline value_class classify(long double value) {
  if (std::isnan(value)) {
    return value_class::nan;
  }
  if (std::isinf(value)) {
    return std::signbit(value) ? value_class::negative_infinity
                               : value_class::positive_infinity;
  }
  return value_class::finite;
}

inline const char *name(value_class value) {
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

struct moment_sums {
  std::size_t count{};
  long double error{};
  long double absolute_error{};
  long double error_squared{};
  long double reference_squared{};
  long double normalized_absolute{};
  long double normalized_squared{};

  void add(long double signed_error, long double reference,
           long double normalized_error) {
    ++count;
    error += signed_error;
    absolute_error += std::fabs(signed_error);
    error_squared += signed_error * signed_error;
    reference_squared += reference * reference;
    normalized_absolute += normalized_error;
    normalized_squared += normalized_error * normalized_error;
  }
};

inline long double quiet_nan() {
  return std::numeric_limits<long double>::quiet_NaN();
}

inline long double mean(long double sum, std::size_t count) {
  return count == 0 ? quiet_nan() : sum / static_cast<long double>(count);
}

inline long double quantile(std::vector<long double> values,
                            long double probability) {
  if (values.empty()) {
    return quiet_nan();
  }
  std::sort(values.begin(), values.end());
  if (probability <= 0.0L) {
    return values.front();
  }
  if (probability >= 1.0L) {
    return values.back();
  }
  const auto position =
      probability * static_cast<long double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const auto fraction = position - static_cast<long double>(lower);
  if (lower == upper || values[lower] == values[upper]) {
    return values[lower];
  }
  if (fraction == 0.0L) {
    return values[lower];
  }
  // Linear interpolation is undefined at infinity (for example inf - inf).
  // Treat the quantile on the extended real line instead.
  if (!std::isfinite(values[lower])) {
    return values[lower];
  }
  if (!std::isfinite(values[upper])) {
    return values[upper];
  }
  return values[lower] + fraction * (values[upper] - values[lower]);
}

inline long double standard_error(const std::vector<long double> &values) {
  if (values.size() < 2) {
    return quiet_nan();
  }
  long double average{};
  for (const auto value : values) {
    average += value;
  }
  average /= static_cast<long double>(values.size());
  long double squared{};
  for (const auto value : values) {
    const auto difference = value - average;
    squared += difference * difference;
  }
  const auto sample_variance =
      squared / static_cast<long double>(values.size() - 1);
  return std::sqrt(sample_variance / static_cast<long double>(values.size()));
}

class comparison_accumulator {
public:
  explicit comparison_accumulator(std::size_t statistical_batches = 1)
      : batches_(std::max<std::size_t>(1, statistical_batches)) {}

  void add(long double actual, long double reference, long double normalizer,
           std::size_t batch) {
    ++total_outputs_;
    const auto actual_class = classify(actual);
    const auto reference_class = classify(reference);
    if (actual_class != value_class::finite ||
        reference_class != value_class::finite) {
      if (actual_class == reference_class) {
        ++matching_nonfinite_;
      } else {
        ++class_mismatches_;
      }
      return;
    }

    const auto signed_error = actual - reference;
    const auto absolute_error = std::fabs(signed_error);
    const auto normalized =
        normalizer == 0.0L
            ? (absolute_error == 0.0L
                   ? 0.0L
                   : std::numeric_limits<long double>::infinity())
            : absolute_error / normalizer;
    const auto relative =
        reference == 0.0L ? (absolute_error == 0.0L
                                 ? 0.0L
                                 : std::numeric_limits<long double>::infinity())
                          : absolute_error / std::fabs(reference);
    const auto condition = reference == 0.0L
                               ? std::numeric_limits<long double>::infinity()
                               : normalizer / std::fabs(reference);
    if (reference == 0.0L) {
      ++zero_references_;
    }
    ++finite_pairs_;
    sums_.add(signed_error, reference, normalized);
    batches_.at(batch).add(signed_error, reference, normalized);
    signed_errors_.push_back(signed_error);
    absolute_errors_.push_back(absolute_error);
    normalized_errors_.push_back(normalized);
    relative_errors_.push_back(relative);
    conditions_.push_back(condition);
    maximum_absolute_error_ = std::max(maximum_absolute_error_, absolute_error);
    maximum_normalized_error_ = std::max(maximum_normalized_error_, normalized);
  }

  std::size_t total_outputs() const { return total_outputs_; }
  std::size_t finite_pairs() const { return finite_pairs_; }
  std::size_t matching_nonfinite() const { return matching_nonfinite_; }
  std::size_t class_mismatches() const { return class_mismatches_; }
  std::size_t zero_references() const { return zero_references_; }
  const moment_sums &sums() const { return sums_; }
  const std::vector<moment_sums> &batches() const { return batches_; }

  long double bias() const { return mean(sums_.error, sums_.count); }
  long double mae() const { return mean(sums_.absolute_error, sums_.count); }
  long double mse() const { return mean(sums_.error_squared, sums_.count); }
  long double rmse() const { return std::sqrt(mse()); }
  long double relative_l2() const {
    return sums_.count == 0 || sums_.reference_squared == 0.0L
               ? quiet_nan()
               : std::sqrt(sums_.error_squared / sums_.reference_squared);
  }
  long double normalized_mean() const {
    return mean(sums_.normalized_absolute, sums_.count);
  }
  long double normalized_rms() const {
    return std::sqrt(mean(sums_.normalized_squared, sums_.count));
  }
  long double maximum_absolute_error() const {
    return finite_pairs_ == 0 ? quiet_nan() : maximum_absolute_error_;
  }
  long double maximum_normalized_error() const {
    return finite_pairs_ == 0 ? quiet_nan() : maximum_normalized_error_;
  }

  std::pair<long double, long double> cluster_standard_errors() const {
    std::vector<long double> biases;
    std::vector<long double> mean_squares;
    for (const auto &batch : batches_) {
      if (batch.count == 0) {
        continue;
      }
      biases.push_back(mean(batch.error, batch.count));
      mean_squares.push_back(mean(batch.error_squared, batch.count));
    }
    return {standard_error(biases), standard_error(mean_squares)};
  }

  void write_quantiles(std::ostream &output, const std::string &prefix) const {
    static constexpr std::array<long double, 17> probabilities{
        0.0L,  0.001L, 0.005L, 0.01L,  0.025L, 0.05L,  0.1L,   0.25L, 0.5L,
        0.75L, 0.9L,   0.95L,  0.975L, 0.99L,  0.995L, 0.999L, 1.0L};
    const std::array<std::pair<const char *, const std::vector<long double> *>,
                     5>
        metrics{{{"signed_error", &signed_errors_},
                 {"absolute_error", &absolute_errors_},
                 {"normalized_absolute_error", &normalized_errors_},
                 {"relative_absolute_error", &relative_errors_},
                 {"condition_number", &conditions_}}};
    for (const auto &[metric, values] : metrics) {
      for (const auto probability : probabilities) {
        output << prefix << ',' << metric << ',' << probability << ','
               << quantile(*values, probability) << ',' << values->size()
               << '\n';
      }
    }
  }

private:
  std::size_t total_outputs_{};
  std::size_t finite_pairs_{};
  std::size_t matching_nonfinite_{};
  std::size_t class_mismatches_{};
  std::size_t zero_references_{};
  moment_sums sums_{};
  std::vector<moment_sums> batches_;
  std::vector<long double> signed_errors_;
  std::vector<long double> absolute_errors_;
  std::vector<long double> normalized_errors_;
  std::vector<long double> relative_errors_;
  std::vector<long double> conditions_;
  long double maximum_absolute_error_{};
  long double maximum_normalized_error_{};
};

struct case_identity {
  std::string kernel;
  std::string distribution;
  std::string format;
  int storage_bits{};
  std::size_t n{};
  std::size_t m{};
};

inline std::string prefix(const case_identity &identity,
                          const std::string &comparison) {
  return identity.kernel + ',' + identity.distribution + ',' + identity.format +
         ',' + std::to_string(identity.storage_bits) + ',' +
         std::to_string(identity.n) + ',' + std::to_string(identity.m) + ',' +
         comparison;
}

inline void write_summary(std::ostream &output, const case_identity &identity,
                          const std::string &comparison,
                          const comparison_accumulator &values) {
  const auto [bias_se, mse_se] = values.cluster_standard_errors();
  const auto mse = values.mse();
  output << prefix(identity, comparison) << ',' << values.total_outputs() << ','
         << values.finite_pairs() << ',' << values.matching_nonfinite() << ','
         << values.class_mismatches() << ',' << values.zero_references() << ','
         << values.bias() << ',' << values.mae() << ',' << mse << ','
         << values.rmse() << ',' << values.maximum_absolute_error() << ','
         << values.relative_l2() << ',' << values.normalized_mean() << ','
         << values.normalized_rms() << ',' << values.maximum_normalized_error()
         << ',' << bias_se << ',' << mse_se << ','
         << (mse == 0.0L ? quiet_nan() : mse_se / mse) << ','
         << values.batches().size() << '\n';
}

inline void write_batches(std::ostream &output, const case_identity &identity,
                          const std::string &comparison,
                          const comparison_accumulator &values) {
  for (std::size_t index = 0; index < values.batches().size(); ++index) {
    const auto &batch = values.batches()[index];
    output << prefix(identity, comparison) << ',' << index << ',' << batch.count
           << ',' << mean(batch.error, batch.count) << ','
           << mean(batch.absolute_error, batch.count) << ','
           << mean(batch.error_squared, batch.count) << ','
           << (batch.count == 0 || batch.reference_squared == 0.0L
                   ? quiet_nan()
                   : std::sqrt(batch.error_squared / batch.reference_squared))
           << ',' << mean(batch.normalized_absolute, batch.count) << ','
           << std::sqrt(mean(batch.normalized_squared, batch.count)) << '\n';
  }
}

class case_accumulator {
public:
  explicit case_accumulator(std::size_t statistical_batches)
      : comparisons_{comparison_accumulator{statistical_batches},
                     comparison_accumulator{statistical_batches},
                     comparison_accumulator{statistical_batches},
                     comparison_accumulator{statistical_batches},
                     comparison_accumulator{statistical_batches},
                     comparison_accumulator{statistical_batches},
                     comparison_accumulator{statistical_batches}} {}

  void add(long double source, long double source_sum_abs, long double storage,
           long double storage_sum_abs, const std::array<double, 3> &gpu,
           std::size_t batch) {
    comparisons_[0].add(storage, source, source_sum_abs, batch);
    for (std::size_t lane = 0; lane < gpu.size(); ++lane) {
      comparisons_[1 + lane].add(gpu[lane], storage, storage_sum_abs, batch);
      comparisons_[4 + lane].add(gpu[lane], source, source_sum_abs, batch);
    }
  }

  void write(std::ostream &summary, std::ostream &quantiles,
             std::ostream &batches, const case_identity &identity) const {
    static constexpr std::array<const char *, 7> names{
        "storage",  "kernel_x1", "kernel_x2", "kernel_x4",
        "total_x1", "total_x2",  "total_x4"};
    for (std::size_t index = 0; index < comparisons_.size(); ++index) {
      write_summary(summary, identity, names[index], comparisons_[index]);
      comparisons_[index].write_quantiles(quantiles,
                                          prefix(identity, names[index]));
      write_batches(batches, identity, names[index], comparisons_[index]);
    }
  }

private:
  std::array<comparison_accumulator, 7> comparisons_;
};

inline void write_headers(std::ostream &summary, std::ostream &quantiles,
                          std::ostream &batches) {
  summary
      << "kernel,distribution,format,storage_bits,n,m,comparison,total_outputs,"
         "finite_pairs,matching_nonfinite,class_mismatches,zero_references,"
         "bias,mean_absolute_error,mse,rmse,max_absolute_error,relative_l2,"
         "mean_normalized_absolute_error,rms_normalized_error,"
         "max_normalized_absolute_error,bias_cluster_standard_error,"
         "mse_cluster_standard_error,mse_relative_cluster_standard_error,"
         "statistical_batches\n";
  quantiles << "kernel,distribution,format,storage_bits,n,m,comparison,metric,"
               "quantile,value,finite_value_count\n";
  batches << "kernel,distribution,format,storage_bits,n,m,comparison,batch,"
             "finite_count,bias,mean_absolute_error,mse,relative_l2,"
             "mean_normalized_absolute_error,rms_normalized_error\n";
  summary << std::setprecision(21);
  quantiles << std::setprecision(21);
  batches << std::setprecision(21);
}

} // namespace aut::accuracy::statistics

#endif // ACCESSOR_UNIVERSAL_TEST_ACCURACY_STATISTICS_HPP_
