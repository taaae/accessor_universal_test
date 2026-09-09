#pragma once
#include "instruction_cost_core.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace aut::instruction_cost {

__device__ __forceinline__ std::uint32_t ptx_add(std::uint32_t x, std::uint32_t a) {
  std::uint32_t o; asm volatile("add.u32 %0,%1,%2;" : "=r"(o) : "r"(x), "r"(a)); return o;
}
__device__ __forceinline__ std::uint32_t ptx_xor(std::uint32_t x, std::uint32_t a) {
  std::uint32_t o; asm volatile("xor.b32 %0,%1,%2;" : "=r"(o) : "r"(x), "r"(a)); return o;
}
__device__ __forceinline__ std::uint32_t ptx_rot(std::uint32_t x, std::uint32_t s) {
  std::uint32_t o; asm volatile("shf.l.wrap.b32 %0,%1,%1,%2;" : "=r"(o) : "r"(x), "r"(s)); return o;
}
__device__ __forceinline__ std::uint32_t ptx_mul(std::uint32_t x, std::uint32_t m) {
  std::uint32_t o; asm volatile("mul.lo.u32 %0,%1,%2;" : "=r"(o) : "r"(x), "r"(m)); return o;
}
__device__ __forceinline__ double ptx_cvt(std::uint32_t x) {
  double o; asm volatile("cvt.rn.f64.u32 %0,%1;" : "=d"(o) : "r"(x)); return o;
}
__device__ __forceinline__ double ptx_fma(double x, double a, double b) {
  double o; asm volatile("fma.rn.f64 %0,%1,%2,%3;" : "=d"(o) : "d"(x),"d"(a),"d"(b)); return o;
}

template <family F, int K> __device__ __forceinline__ double decode(
    std::uint32_t u, std::uint32_t operand, double fa, double fb) {
  if constexpr (F == family::fma64) {
    double v = ptx_cvt(u);
#pragma unroll
    for (int i=0;i<K;++i) v=ptx_fma(v,fa,fb);
    return v;
  } else {
#pragma unroll
    for (int i=0;i<K;++i) {
      if constexpr (F == family::add32) u=ptx_add(u,operand);
      if constexpr (F == family::xor32) u=ptx_xor(u,operand);
      if constexpr (F == family::rot32) u=ptx_rot(u,operand);
      if constexpr (F == family::mul32) u=ptx_mul(u,operand);
    }
    return ptx_cvt(u);
  }
}

template<class T> __device__ __forceinline__ T block_sum(T v) {
  __shared__ T scratch[256];
  scratch[threadIdx.x]=v; __syncthreads();
  for(int offset=128;offset;offset>>=1) { if(threadIdx.x<offset) scratch[threadIdx.x]+=scratch[threadIdx.x+offset]; __syncthreads(); }
  return scratch[0];
}

template<family F,int K> __global__ __launch_bounds__(256) void dot_timed_kernel(
 const std::uint32_t* left,const std::uint32_t* right,std::size_t n,double* partial,
 std::uint32_t operand,double fa,double fb) {
  double sum=0; const std::size_t stride=std::size_t(gridDim.x)*blockDim.x;
#pragma unroll 1
  for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=stride)
    sum=__fma_rn(decode<F,K>(left[i],operand,fa,fb),decode<F,K>(right[i],operand,fa,fb),sum);
  double total=block_sum(sum); if(threadIdx.x==0) partial[blockIdx.x]=total;
}
template<family F,int K> __global__ __launch_bounds__(256) void gemv_timed_kernel(
 const std::uint32_t* matrix,const std::uint32_t* vector,int rows,int cols,double* output,
 std::uint32_t operand,double fa,double fb) {
  int row=blockIdx.x; double sum=0;
#pragma unroll 1
  for(int col=threadIdx.x;col<cols;col+=blockDim.x)
    sum=__fma_rn(decode<F,K>(matrix[std::size_t(row)*cols+col],operand,fa,fb),decode<F,K>(vector[col],operand,fa,fb),sum);
  double total=block_sum(sum); if(threadIdx.x==0 && row<rows) output[row]=total;
}
template<family F,int K> __global__ void decoder_validation_kernel(
 const std::uint32_t* input,double* output,std::size_t n,std::uint32_t operand,double fa,double fb) {
  for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
    output[i]=decode<F,K>(input[i],operand,fa,fb);
}
__global__ void dot_raw64_kernel(const double*a,const double*b,std::size_t n,double*p);
__global__ void dot_raw32_kernel(const float*a,const float*b,std::size_t n,float*p);
__global__ void dot_fp32_to_fp64_kernel(const float*a,const float*b,std::size_t n,double*p);
__global__ void reduce64_kernel(const double*,int,double*);
__global__ void reduce32_kernel(const float*,int,float*);
__global__ void gemv_raw64_kernel(const double*,const double*,int,int,double*);
__global__ void gemv_raw32_kernel(const float*,const float*,int,int,float*);
__global__ void gemv_fp32_to_fp64_kernel(const float*,const float*,int,int,double*);

} // namespace aut::instruction_cost
