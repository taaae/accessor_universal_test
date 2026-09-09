#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace aut::block_scale_reconstruction {

inline constexpr int grid_blocks = 512;
inline constexpr int cta_threads = 256;

template <class T>
__device__ __forceinline__ T block_sum(T value) {
  __shared__ T shared[cta_threads];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int distance = cta_threads / 2; distance; distance >>= 1) {
    if (threadIdx.x < distance) {
      shared[threadIdx.x] += shared[threadIdx.x + distance];
    }
    __syncthreads();
  }
  return shared[0];
}

template <int B, class S>
__global__ __launch_bounds__(cta_threads) void reconstruct64_kernel(
    const float* q_left, const float* q_right, const S* scale_left,
    const S* scale_right, std::size_t n, double* partials) {
  double sum = 0.0;
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
#pragma unroll 1
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const std::size_t block = i / B;
    const double left = __dmul_rn(double(q_left[i]), double(scale_left[block]));
    const double right = __dmul_rn(double(q_right[i]), double(scale_right[block]));
    sum = __fma_rn(left, right, sum);
  }
  const double total = block_sum(sum);
  if (threadIdx.x == 0) partials[blockIdx.x] = total;
}

template <int B>
__global__ __launch_bounds__(cta_threads) void reconstruct32_kernel(
    const float* q_left, const float* q_right, const float* scale_left,
    const float* scale_right, std::size_t n, double* partials) {
  double sum = 0.0;
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
#pragma unroll 1
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const std::size_t block = i / B;
    const float left = __fmul_rn(q_left[i], scale_left[block]);
    const float right = __fmul_rn(q_right[i], scale_right[block]);
    sum = __fma_rn(double(left), double(right), sum);
  }
  const double total = block_sum(sum);
  if (threadIdx.x == 0) partials[blockIdx.x] = total;
}

template <class S>
__global__ __launch_bounds__(cta_threads) void deferred_local_b128_kernel(
    const float* q_left, const float* q_right, const S* scale_left,
    const S* scale_right, std::size_t n, double* partials) {
  constexpr int values_per_lane = 4;
  constexpr int groups_per_cta = cta_threads / 32;
  const int group = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const std::size_t block_count = (n + 127) / 128;
  double accumulator = 0.0;
#pragma unroll 1
  for (std::size_t tile = std::size_t(blockIdx.x) * groups_per_cta;
       tile < block_count;
       tile += std::size_t(gridDim.x) * groups_per_cta) {
    const std::size_t block = tile + group;
    double subtotal = 0.0;
#pragma unroll 1
    for (int j = 0; j < values_per_lane; ++j) {
      const std::size_t i = block * 128 + std::size_t(j) * 32 + lane;
      if (block < block_count && i < n) {
        subtotal = __fma_rn(double(q_left[i]), double(q_right[i]), subtotal);
      }
    }
    if (block < block_count) {
      const double scale =
          __dmul_rn(double(scale_left[block]), double(scale_right[block]));
      accumulator = __fma_rn(scale, subtotal, accumulator);
    }
  }
  const double total = block_sum(accumulator);
  if (threadIdx.x == 0) partials[blockIdx.x] = total;
}

__global__ __launch_bounds__(cta_threads) void raw32_kernel(
    const float*, const float*, std::size_t, float*);
__global__ __launch_bounds__(cta_threads) void widened32_kernel(
    const float*, const float*, std::size_t, double*);
__global__ __launch_bounds__(cta_threads) void raw64_kernel(
    const double*, const double*, std::size_t, double*);
__global__ void reduce32_kernel(const float*, float*);
__global__ void reduce64_kernel(const double*, double*);

}  // namespace aut::block_scale_reconstruction
