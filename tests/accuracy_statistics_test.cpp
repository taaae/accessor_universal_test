#include "accuracy_statistics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>

namespace {

using aut::accuracy::statistics::case_accumulator;
using aut::accuracy::statistics::case_identity;
using aut::accuracy::statistics::comparison_accumulator;
using aut::accuracy::statistics::quantile;

bool close(long double left, long double right,
           long double tolerance = 1e-18L) {
  return std::fabs(left - right) <= tolerance;
}

} // namespace

int main() {
  const auto infinity = std::numeric_limits<long double>::infinity();
  if (quantile({1.0L, infinity, infinity}, 0.75L) != infinity ||
      quantile({1.0L, 2.0L, infinity}, 0.5L) != 2.0L ||
      quantile({1.0L, 2.0L, infinity}, 0.75L) != infinity) {
    std::cerr << "non-finite quantiles are incorrect\n";
    return EXIT_FAILURE;
  }

  comparison_accumulator values{2};
  values.add(11.0L, 10.0L, 20.0L, 0);
  values.add(18.0L, 20.0L, 40.0L, 1);
  if (!close(values.bias(), -0.5L) || !close(values.mae(), 1.5L) ||
      !close(values.mse(), 2.5L) || !close(values.rmse(), std::sqrt(2.5L)) ||
      !close(values.normalized_mean(), 0.05L) ||
      !close(values.normalized_rms(), 0.05L)) {
    std::cerr << "finite error moments are incorrect\n";
    return EXIT_FAILURE;
  }

  values.add(std::numeric_limits<long double>::infinity(),
             std::numeric_limits<long double>::infinity(),
             std::numeric_limits<long double>::infinity(), 0);
  values.add(std::numeric_limits<long double>::quiet_NaN(), 1.0L, 1.0L, 0);
  if (values.finite_pairs() != 2 || values.matching_nonfinite() != 1 ||
      values.class_mismatches() != 1) {
    std::cerr << "non-finite classification counts are incorrect\n";
    return EXIT_FAILURE;
  }

  case_accumulator complete{2};
  complete.add(10.0L, 20.0L, 9.0L, 18.0L, {9.0, 9.0, 9.0}, 0);
  complete.add(20.0L, 40.0L, 18.0L, 36.0L, {18.0, 18.0, 18.0}, 1);
  std::ostringstream summary;
  std::ostringstream quantiles;
  std::ostringstream batches;
  complete.write(summary, quantiles, batches,
                 case_identity{"dot", "uniform_0_1", "test", 8, 2, 1});
  const auto summary_text = summary.str();
  const auto quantile_text = quantiles.str();
  const auto batch_text = batches.str();
  if (std::count(summary_text.begin(), summary_text.end(), '\n') != 7 ||
      std::count(quantile_text.begin(), quantile_text.end(), '\n') !=
          7 * 5 * 17 ||
      std::count(batch_text.begin(), batch_text.end(), '\n') != 14) {
    std::cerr << "case output row counts are incorrect\n";
    return EXIT_FAILURE;
  }

  std::cout << "accuracy statistics tests passed\n";
  return EXIT_SUCCESS;
}
