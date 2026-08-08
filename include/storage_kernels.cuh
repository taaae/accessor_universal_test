#ifndef ACCESSOR_UNIVERSAL_TEST_STORAGE_KERNELS_CUH_
#define ACCESSOR_UNIVERSAL_TEST_STORAGE_KERNELS_CUH_

#include "cuda_storage_formats.cuh"

#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>

namespace aut::kernels {

inline constexpr int reduction_block_threads = 256;

struct fp64x4 {
  double x{};
  double y{};
  double z{};
  double w{};
};

template <typename Format, int Lanes>
__device__ __forceinline__ fp64x4
load_decoded(const storage::storage_type_t<Format> *values) {
  static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4);
  if constexpr (Lanes == 1) {
    return {storage::decode<Format>(values[0]), 0.0, 0.0, 0.0};
  } else if constexpr (Lanes == 2) {
    const auto decoded = storage::packed_decoder<Format>::load2(values);
    return {decoded.x, decoded.y, 0.0, 0.0};
  } else {
    const auto decoded = storage::packed_decoder<Format>::load4(values);
    return {decoded.x, decoded.y, decoded.z, decoded.w};
  }
}

template <int Lanes>
__device__ __forceinline__ void
accumulate_product(fp64x4 &sums, const fp64x4 &left, const fp64x4 &right) {
  sums.x = fma(left.x, right.x, sums.x);
  if constexpr (Lanes >= 2) {
    sums.y = fma(left.y, right.y, sums.y);
  }
  if constexpr (Lanes == 4) {
    sums.z = fma(left.z, right.z, sums.z);
    sums.w = fma(left.w, right.w, sums.w);
  }
}

template <int Lanes>
__device__ __forceinline__ double combine_sums(const fp64x4 &sums) {
  if constexpr (Lanes == 1) {
    return sums.x;
  } else if constexpr (Lanes == 2) {
    return sums.x + sums.y;
  } else {
    return (sums.x + sums.y) + (sums.z + sums.w);
  }
}

/**
 * First stage of a packed, FP64-arithmetic DOT product.
 *
 * Lanes controls only the load/decode granularity. The arrays always contain
 * scalar storage_type_t<Format> elements, so scalar/x2/x4 variants consume the
 * same bytes and can share one encoded dataset.
 */
template <typename Format, int Lanes>
__global__ void
storage_dot_map_reduce(const storage::storage_type_t<Format> *left,
                       const storage::storage_type_t<Format> *right,
                       std::size_t count, double *partials) {
  using block_reduce = cub::BlockReduce<double, reduction_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;

  const auto first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const auto pack_count = count / Lanes;

  fp64x4 sums{};
  for (auto pack = first; pack < pack_count; pack += stride) {
    const auto offset = static_cast<std::size_t>(Lanes) * pack;
    const auto left_values = load_decoded<Format, Lanes>(left + offset);
    const auto right_values = load_decoded<Format, Lanes>(right + offset);
    accumulate_product<Lanes>(sums, left_values, right_values);
  }

  // At most Lanes - 1 threads in block zero handle the scalar tail.
  const auto tail_index = static_cast<std::size_t>(Lanes) * pack_count + first;
  if (tail_index < count) {
    sums.x = fma(storage::decode<Format>(left[tail_index]),
                 storage::decode<Format>(right[tail_index]), sums.x);
  }

  const auto block_sum = block_reduce(temporary).Sum(combine_sums<Lanes>(sums));
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = block_sum;
  }
}

__global__ void storage_dot_finalize(const double *partials,
                                     std::size_t partial_count,
                                     double *result) {
  using block_reduce = cub::BlockReduce<double, reduction_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;

  double sum{};
  for (auto i = static_cast<std::size_t>(threadIdx.x); i < partial_count;
       i += blockDim.x) {
    sum += partials[i];
  }
  const auto final_sum = block_reduce(temporary).Sum(sum);
  if (threadIdx.x == 0) {
    *result = final_sum;
  }
}

/**
 * One-block-per-row row-major GEMV with packed loads along the column axis.
 * Matrix, vector, accumulation, and output semantics are:
 *
 *   FP storage -> FP64 decode -> FP64 FMA -> FP64 output.
 */
template <typename Format, int Lanes>
__global__ void storage_gemv(const storage::storage_type_t<Format> *matrix,
                             const storage::storage_type_t<Format> *vector,
                             std::size_t rows, std::size_t columns,
                             std::size_t leading_dimension, double *result) {
  using block_reduce = cub::BlockReduce<double, reduction_block_threads>;
  __shared__ typename block_reduce::TempStorage temporary;

  const auto row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const auto *row_values = matrix + row * leading_dimension;
  const auto pack_count = columns / Lanes;
  fp64x4 sums{};
  for (auto pack = static_cast<std::size_t>(threadIdx.x); pack < pack_count;
       pack += blockDim.x) {
    const auto offset = static_cast<std::size_t>(Lanes) * pack;
    const auto matrix_values = load_decoded<Format, Lanes>(row_values + offset);
    const auto vector_values = load_decoded<Format, Lanes>(vector + offset);
    accumulate_product<Lanes>(sums, matrix_values, vector_values);
  }

  const auto tail_column =
      static_cast<std::size_t>(Lanes) * pack_count + threadIdx.x;
  if (tail_column < columns) {
    sums.x = fma(storage::decode<Format>(row_values[tail_column]),
                 storage::decode<Format>(vector[tail_column]), sums.x);
  }

  const auto row_sum = block_reduce(temporary).Sum(combine_sums<Lanes>(sums));
  if (threadIdx.x == 0) {
    result[row] = row_sum;
  }
}

} // namespace aut::kernels

#endif // ACCESSOR_UNIVERSAL_TEST_STORAGE_KERNELS_CUH_
