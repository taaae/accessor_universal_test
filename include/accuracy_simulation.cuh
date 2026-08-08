#ifndef ACCESSOR_UNIVERSAL_TEST_ACCURACY_SIMULATION_CUH_
#define ACCESSOR_UNIVERSAL_TEST_ACCURACY_SIMULATION_CUH_

#include "storage_kernels.cuh"

#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace aut::accuracy {

inline constexpr int reference_block_threads = 256;

struct double_double {
  double hi{};
  double lo{};
};

struct reference_pair {
  double_double value{};
  double_double sum_abs{};
};

struct encoding_counts {
  unsigned long long zeros{};
  unsigned long long infinities{};
  unsigned long long nans{};
  unsigned long long saturations{};
};

__device__ __forceinline__ double_double add(double_double left,
                                             double_double right) {
  if (!isfinite(left.hi) || !isfinite(right.hi)) {
    return {left.hi + right.hi, 0.0};
  }
  const auto sum = left.hi + right.hi;
  const auto virtual_right = sum - left.hi;
  auto error = (left.hi - (sum - virtual_right)) + (right.hi - virtual_right);
  error += left.lo + right.lo;
  const auto hi = sum + error;
  const auto lo = error - (hi - sum);
  return {hi, lo};
}

__device__ __forceinline__ double_double product(double left, double right) {
  const auto hi = left * right;
  if (!isfinite(hi)) {
    return {hi, 0.0};
  }
  return {hi, fma(left, right, -hi)};
}

__device__ __forceinline__ double_double absolute(double_double value) {
  if (signbit(value.hi) || (value.hi == 0.0 && signbit(value.lo))) {
    return {-value.hi, -value.lo};
  }
  return value;
}

struct add_reference_pairs {
  __device__ __forceinline__ reference_pair
  operator()(const reference_pair &left, const reference_pair &right) const {
    return {add(left.value, right.value), add(left.sum_abs, right.sum_abs)};
  }
};

template <typename Format>
__global__ void
encode_and_count_kernel(const double *source,
                        storage::storage_type_t<Format> *encoded,
                        std::size_t count, bool saturating,
                        double saturation_threshold, encoding_counts *counts) {
  unsigned long long zeros{};
  unsigned long long infinities{};
  unsigned long long nans{};
  unsigned long long saturations{};
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto index = first; index < count; index += stride) {
    const auto stored = storage::encode<Format>(source[index]);
    encoded[index] = stored;
    const auto decoded = storage::decode<Format>(stored);
    zeros += decoded == 0.0;
    infinities += isinf(decoded);
    nans += isnan(decoded);
    saturations += saturating && fabs(source[index]) >= saturation_threshold;
  }
  if (zeros != 0) {
    atomicAdd(&counts->zeros, zeros);
  }
  if (infinities != 0) {
    atomicAdd(&counts->infinities, infinities);
  }
  if (nans != 0) {
    atomicAdd(&counts->nans, nans);
  }
  if (saturations != 0) {
    atomicAdd(&counts->saturations, saturations);
  }
}

__global__ void source_batched_reference_kernel(const double *left,
                                                const double *right,
                                                std::size_t count,
                                                std::size_t batch_count,
                                                reference_pair *result) {
  using block_reduce =
      cub::BlockReduce<reference_pair, reference_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto sample = static_cast<std::size_t>(blockIdx.x);
  if (sample >= batch_count) {
    return;
  }
  const auto offset = sample * count;
  reference_pair local{};
  for (auto index = static_cast<std::size_t>(threadIdx.x); index < count;
       index += blockDim.x) {
    const auto term = product(left[offset + index], right[offset + index]);
    local.value = add(local.value, term);
    local.sum_abs = add(local.sum_abs, absolute(term));
  }
  const auto reduced =
      block_reduce(temporary).Reduce(local, add_reference_pairs{});
  if (threadIdx.x == 0) {
    result[sample] = reduced;
  }
}

template <typename Format>
__global__ void
storage_batched_reference_kernel(const storage::storage_type_t<Format> *left,
                                 const storage::storage_type_t<Format> *right,
                                 std::size_t count, std::size_t batch_count,
                                 reference_pair *result) {
  using block_reduce =
      cub::BlockReduce<reference_pair, reference_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto sample = static_cast<std::size_t>(blockIdx.x);
  if (sample >= batch_count) {
    return;
  }
  const auto offset = sample * count;
  reference_pair local{};
  for (auto index = static_cast<std::size_t>(threadIdx.x); index < count;
       index += blockDim.x) {
    const auto term = product(storage::decode<Format>(left[offset + index]),
                              storage::decode<Format>(right[offset + index]));
    local.value = add(local.value, term);
    local.sum_abs = add(local.sum_abs, absolute(term));
  }
  const auto reduced =
      block_reduce(temporary).Reduce(local, add_reference_pairs{});
  if (threadIdx.x == 0) {
    result[sample] = reduced;
  }
}

__global__ void source_gemv_reference_kernel(const double *matrix,
                                             const double *vector,
                                             std::size_t rows,
                                             std::size_t columns,
                                             reference_pair *result) {
  using block_reduce =
      cub::BlockReduce<reference_pair, reference_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  reference_pair local{};
  for (auto column = static_cast<std::size_t>(threadIdx.x); column < columns;
       column += blockDim.x) {
    const auto term = product(matrix[row * columns + column], vector[column]);
    local.value = add(local.value, term);
    local.sum_abs = add(local.sum_abs, absolute(term));
  }
  const auto reduced =
      block_reduce(temporary).Reduce(local, add_reference_pairs{});
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
}

template <typename Format>
__global__ void
storage_gemv_reference_kernel(const storage::storage_type_t<Format> *matrix,
                              const storage::storage_type_t<Format> *vector,
                              std::size_t rows, std::size_t columns,
                              reference_pair *result) {
  using block_reduce =
      cub::BlockReduce<reference_pair, reference_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }
  reference_pair local{};
  for (auto column = static_cast<std::size_t>(threadIdx.x); column < columns;
       column += blockDim.x) {
    const auto term =
        product(storage::decode<Format>(matrix[row * columns + column]),
                storage::decode<Format>(vector[column]));
    local.value = add(local.value, term);
    local.sum_abs = add(local.sum_abs, absolute(term));
  }
  const auto reduced =
      block_reduce(temporary).Reduce(local, add_reference_pairs{});
  if (threadIdx.x == 0) {
    result[row] = reduced;
  }
}

template <typename Format, int Lanes>
__global__ void
storage_dot_map_reduce_batched(const storage::storage_type_t<Format> *left,
                               const storage::storage_type_t<Format> *right,
                               std::size_t count, std::size_t batch_count,
                               double *partials) {
  using block_reduce =
      cub::BlockReduce<double, kernels::reduction_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto sample = static_cast<std::size_t>(blockIdx.y);
  if (sample >= batch_count) {
    return;
  }
  const auto sample_offset = sample * count;
  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto pack_count = count / Lanes;
  kernels::fp64x4 sums{};
  for (auto pack = first; pack < pack_count; pack += stride) {
    const auto offset = sample_offset + static_cast<std::size_t>(Lanes) * pack;
    const auto left_values =
        kernels::load_decoded<Format, Lanes>(left + offset);
    const auto right_values =
        kernels::load_decoded<Format, Lanes>(right + offset);
    kernels::accumulate_product<Lanes>(sums, left_values, right_values);
  }
  const auto tail_index = static_cast<std::size_t>(Lanes) * pack_count + first;
  if (tail_index < count) {
    sums.x =
        fma(storage::decode<Format>(left[sample_offset + tail_index]),
            storage::decode<Format>(right[sample_offset + tail_index]), sums.x);
  }
  const auto block_sum =
      block_reduce(temporary).Sum(kernels::combine_sums<Lanes>(sums));
  if (threadIdx.x == 0) {
    partials[sample * gridDim.x + blockIdx.x] = block_sum;
  }
}

__global__ void storage_dot_finalize_batched(const double *partials,
                                             std::size_t blocks_per_sample,
                                             std::size_t batch_count,
                                             double *result) {
  using block_reduce =
      cub::BlockReduce<double, kernels::reduction_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;
  const auto sample = static_cast<std::size_t>(blockIdx.x);
  if (sample >= batch_count) {
    return;
  }
  double sum{};
  for (auto index = static_cast<std::size_t>(threadIdx.x);
       index < blocks_per_sample; index += blockDim.x) {
    sum += partials[sample * blocks_per_sample + index];
  }
  const auto final_sum = block_reduce(temporary).Sum(sum);
  if (threadIdx.x == 0) {
    result[sample] = final_sum;
  }
}

} // namespace aut::accuracy

#endif // ACCESSOR_UNIVERSAL_TEST_ACCURACY_SIMULATION_CUH_
