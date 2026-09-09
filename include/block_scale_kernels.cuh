#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace aut::block_scale {
inline constexpr int grid_blocks=512,cta_threads=256;
template<class T> __device__ __forceinline__ T block_sum(T v){__shared__ T s[cta_threads];s[threadIdx.x]=v;__syncthreads();for(int d=cta_threads/2;d;d>>=1){if(threadIdx.x<d)s[threadIdx.x]+=s[threadIdx.x+d];__syncthreads();}return s[0];}

template<int B,class S> __global__ __launch_bounds__(cta_threads) void per_element_kernel(const float*ql,const float*qr,const S*sl,const S*sr,std::size_t n,double*partial){
 double sum=0;const std::size_t stride=std::size_t(gridDim.x)*blockDim.x;
#pragma unroll 1
 for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=stride){const std::size_t b=i/B;double l=__dmul_rn(double(ql[i]),double(sl[b]));double r=__dmul_rn(double(qr[i]),double(sr[b]));sum=__fma_rn(l,r,sum);}
 double total=block_sum(sum);if(threadIdx.x==0)partial[blockIdx.x]=total;
}
template<int B,class S> __global__ __launch_bounds__(cta_threads) void deferred_kernel(const float*ql,const float*qr,const S*sl,const S*sr,std::size_t n,double*partial){
 constexpr int L=B<32?B:32,V=B/L,G=cta_threads/L;const int group=threadIdx.x/L,lane=threadIdx.x%L;const std::size_t nb=(n+B-1)/B;double acc=0;
#pragma unroll 1
 for(std::size_t tile=std::size_t(blockIdx.x)*G;tile<nb;tile+=std::size_t(gridDim.x)*G){const std::size_t b=tile+group;double sub=0;
#pragma unroll 1
  for(int j=0;j<V;++j){const std::size_t i=b*B+std::size_t(j)*L+lane;if(b<nb&&i<n)sub=__fma_rn(double(ql[i]),double(qr[i]),sub);}
  for(int d=L/2;d;d>>=1)sub=__dadd_rn(sub,__shfl_down_sync(0xffffffffu,sub,d,L));
  if(lane==0&&b<nb){double sb=__dmul_rn(double(sl[b]),double(sr[b]));acc=__fma_rn(sb,sub,acc);}
 }
 double total=block_sum(acc);if(threadIdx.x==0)partial[blockIdx.x]=total;
}
__global__ __launch_bounds__(cta_threads) void raw32_kernel(const float*,const float*,std::size_t,float*);
__global__ __launch_bounds__(cta_threads) void widened32_kernel(const float*,const float*,std::size_t,double*);
__global__ __launch_bounds__(cta_threads) void raw64_kernel(const double*,const double*,std::size_t,double*);
__global__ void reduce32_kernel(const float*,float*);
__global__ void reduce64_kernel(const double*,double*);
}
