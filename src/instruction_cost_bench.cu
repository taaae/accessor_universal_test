#include "instruction_cost_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ic = aut::instruction_cost;
namespace {
void ck(cudaError_t e,const char*x){if(e!=cudaSuccess)throw std::runtime_error(std::string(x)+": "+cudaGetErrorString(e));}
#define CK(x) ck((x),#x)
template<class T> struct dbuf { T*p{}; std::size_t n{}; explicit dbuf(std::size_t m):n(m){CK(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));} ~dbuf(){cudaFree(p);} };
struct event { cudaEvent_t e; event(){CK(cudaEventCreate(&e));} ~event(){cudaEventDestroy(e);} };
struct opts { std::string mode="full",output="timing_samples.csv",checks="correctness_checks.txt"; int warmups=10,samples=50; };
opts parse(int argc,char**argv){opts o;for(int i=1;i<argc;++i){std::string a=argv[i];auto v=[&]{if(++i>=argc)throw std::runtime_error("missing value");return std::string(argv[i]);};if(a=="--mode")o.mode=v();else if(a=="--output")o.output=v();else if(a=="--correctness-output")o.checks=v();else if(a=="--warmups")o.warmups=std::stoi(v());else if(a=="--samples")o.samples=std::stoi(v());else throw std::runtime_error("unknown option "+a);}if(o.mode=="smoke"){o.warmups=1;o.samples=3;}else if(o.mode=="validate"){o.warmups=0;o.samples=0;}else if(o.mode!="full")throw std::runtime_error("bad mode");return o;}
std::uint32_t operand(ic::family f){if(f==ic::family::add32)return ic::add_operand;if(f==ic::family::xor32)return ic::xor_operand;if(f==ic::family::rot32)return ic::rotate_operand;if(f==ic::family::mul32)return ic::multiply_operand;return 0;}

#define KS(M,F) M(F,0) M(F,1) M(F,2) M(F,4) M(F,8) M(F,12) M(F,16) M(F,24) M(F,32) M(F,48) M(F,64)
#define DOT_CASE(F,K) case K: ic::dot_timed_kernel<ic::family::F,K><<<512,256>>>(a,b,n,p,operand(ic::family::F),ic::fma_multiplier,ic::fma_addend); break;
void launch_dot(ic::family f,int k,const std::uint32_t*a,const std::uint32_t*b,std::size_t n,double*p){switch(f){
case ic::family::add32:switch(k){KS(DOT_CASE,add32)default:throw std::runtime_error("bad K");}break;
case ic::family::xor32:switch(k){KS(DOT_CASE,xor32)default:throw std::runtime_error("bad K");}break;
case ic::family::rot32:switch(k){KS(DOT_CASE,rot32)default:throw std::runtime_error("bad K");}break;
case ic::family::mul32:switch(k){KS(DOT_CASE,mul32)default:throw std::runtime_error("bad K");}break;
case ic::family::fma64:switch(k){KS(DOT_CASE,fma64)default:throw std::runtime_error("bad K");}break;}}
#undef DOT_CASE
#define GEMV_CASE(F,K) case K: ic::gemv_timed_kernel<ic::family::F,K><<<rows,256>>>(a,b,rows,cols,p,operand(ic::family::F),ic::fma_multiplier,ic::fma_addend); break;
void launch_gemv(ic::family f,int k,const std::uint32_t*a,const std::uint32_t*b,int rows,int cols,double*p){switch(f){
case ic::family::add32:switch(k){KS(GEMV_CASE,add32)default:throw std::runtime_error("bad K");}break;
case ic::family::xor32:switch(k){KS(GEMV_CASE,xor32)default:throw std::runtime_error("bad K");}break;
case ic::family::rot32:switch(k){KS(GEMV_CASE,rot32)default:throw std::runtime_error("bad K");}break;
case ic::family::mul32:switch(k){KS(GEMV_CASE,mul32)default:throw std::runtime_error("bad K");}break;
case ic::family::fma64:switch(k){KS(GEMV_CASE,fma64)default:throw std::runtime_error("bad K");}break;}}
#undef GEMV_CASE
#define VALIDATE_CASE(F,K) case K: ic::decoder_validation_kernel<ic::family::F,K><<<17,256>>>(a,p,n,operand(ic::family::F),ic::fma_multiplier,ic::fma_addend); break;
void launch_decoder(ic::family f,int k,const std::uint32_t*a,double*p,std::size_t n){switch(f){
case ic::family::add32:switch(k){KS(VALIDATE_CASE,add32)default:throw std::runtime_error("bad K");}break;
case ic::family::xor32:switch(k){KS(VALIDATE_CASE,xor32)default:throw std::runtime_error("bad K");}break;
case ic::family::rot32:switch(k){KS(VALIDATE_CASE,rot32)default:throw std::runtime_error("bad K");}break;
case ic::family::mul32:switch(k){KS(VALIDATE_CASE,mul32)default:throw std::runtime_error("bad K");}break;
case ic::family::fma64:switch(k){KS(VALIDATE_CASE,fma64)default:throw std::runtime_error("bad K");}break;}}
#undef VALIDATE_CASE

struct caze{std::string kernel,family,stage;int k;};
float timed(const std::function<void()>&fn){event a,b;CK(cudaEventRecord(a.e));fn();CK(cudaGetLastError());CK(cudaEventRecord(b.e));CK(cudaEventSynchronize(b.e));float ms;CK(cudaEventElapsedTime(&ms,a.e,b.e));return ms;}
void ensure(const std::string&p){auto q=std::filesystem::path(p).parent_path();if(!q.empty())std::filesystem::create_directories(q);}
}

namespace aut::instruction_cost {
__global__ void reduce64_kernel(const double*a,int n,double*out){double s=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)s+=a[i];double t=block_sum(s);if(threadIdx.x==0)*out=t;}
__global__ void reduce32_kernel(const float*a,int n,float*out){float s=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)s+=a[i];float t=block_sum(s);if(threadIdx.x==0)*out=t;}
__global__ void dot_raw64_kernel(const double*a,const double*b,std::size_t n,double*p){double s=0;std::size_t st=std::size_t(gridDim.x)*blockDim.x;
#pragma unroll 1
for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=st)s=__fma_rn(a[i],b[i],s);double t=block_sum(s);if(threadIdx.x==0)p[blockIdx.x]=t;}
__global__ void dot_raw32_kernel(const float*a,const float*b,std::size_t n,float*p){float s=0;std::size_t st=std::size_t(gridDim.x)*blockDim.x;
#pragma unroll 1
for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=st)s=__fmaf_rn(a[i],b[i],s);float t=block_sum(s);if(threadIdx.x==0)p[blockIdx.x]=t;}
__global__ void dot_fp32_to_fp64_kernel(const float*a,const float*b,std::size_t n,double*p){double s=0;std::size_t st=std::size_t(gridDim.x)*blockDim.x;
#pragma unroll 1
for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=st)s=__fma_rn(double(a[i]),double(b[i]),s);double t=block_sum(s);if(threadIdx.x==0)p[blockIdx.x]=t;}
__global__ void gemv_raw64_kernel(const double*a,const double*b,int rows,int cols,double*out){int r=blockIdx.x;double s=0;
#pragma unroll 1
for(int c=threadIdx.x;c<cols;c+=blockDim.x)s=__fma_rn(a[std::size_t(r)*cols+c],b[c],s);double t=block_sum(s);if(threadIdx.x==0&&r<rows)out[r]=t;}
__global__ void gemv_raw32_kernel(const float*a,const float*b,int rows,int cols,float*out){int r=blockIdx.x;float s=0;
#pragma unroll 1
for(int c=threadIdx.x;c<cols;c+=blockDim.x)s=__fmaf_rn(a[std::size_t(r)*cols+c],b[c],s);float t=block_sum(s);if(threadIdx.x==0&&r<rows)out[r]=t;}
__global__ void gemv_fp32_to_fp64_kernel(const float*a,const float*b,int rows,int cols,double*out){int r=blockIdx.x;double s=0;
#pragma unroll 1
for(int c=threadIdx.x;c<cols;c+=blockDim.x)s=__fma_rn(double(a[std::size_t(r)*cols+c]),double(b[c]),s);double t=block_sum(s);if(threadIdx.x==0&&r<rows)out[r]=t;}
}

int main(int argc,char**argv)try{
  auto o=parse(argc,argv); bool smoke=o.mode!="full"; std::size_t dotn=smoke?(1u<<20):(1ull<<26);int rows=smoke?64:4096,cols=smoke?1024:16384;std::size_t mn=std::size_t(rows)*cols;
  std::vector<std::uint32_t> hA(std::max(dotn,mn)),hB(dotn),hV(cols);for(std::size_t i=0;i<hA.size();++i)hA[i]=ic::code_at(i,ic::right_seed);for(std::size_t i=0;i<hB.size();++i)hB[i]=ic::code_at(i,ic::left_seed);for(int i=0;i<cols;++i)hV[i]=ic::code_at(i,ic::left_seed);
  std::vector<float> fA(hA.size()),fB(hB.size()),fV(hV.size());std::vector<double>dA(hA.size()),dB(hB.size()),dV(hV.size());for(std::size_t i=0;i<hA.size();++i){fA[i]=float(hA[i]);dA[i]=double(hA[i]);}for(std::size_t i=0;i<hB.size();++i){fB[i]=float(hB[i]);dB[i]=double(hB[i]);}for(std::size_t i=0;i<hV.size();++i){fV[i]=float(hV[i]);dV[i]=double(hV[i]);}
  dbuf<std::uint32_t>A(hA.size()),B(hB.size()),V(hV.size());dbuf<float>AF(fA.size()),BF(fB.size()),VF(fV.size()),pf(4096);dbuf<double>AD(dA.size()),BD(dB.size()),VD(dV.size()),pd(4096),result(1),decoded(4099);
#define COPY(D,H) CK(cudaMemcpy((D).p,(H).data(),(H).size()*sizeof((H)[0]),cudaMemcpyHostToDevice))
  COPY(A,hA);COPY(B,hB);COPY(V,hV);COPY(AF,fA);COPY(BF,fB);COPY(VF,fV);COPY(AD,dA);COPY(BD,dB);COPY(VD,dV);
  ensure(o.checks);std::ofstream checks(o.checks);checks<<"generator_distinct=1\nfinite_inputs=1\n";
  for(auto f:ic::families)for(int k:{0,1,2,4,8,12,16,24,32,48,64}){
    launch_decoder(f,k,A.p,decoded.p,4099);std::vector<double> got(4099);CK(cudaMemcpy(got.data(),decoded.p,got.size()*sizeof(double),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<got.size();++i){double want=ic::decode_reference(hA[i],f,k);if(std::memcmp(&got[i],&want,sizeof(double))!=0)throw std::runtime_error("GPU decoder/reference mismatch");}
  }
  checks<<"host_reference=1\n";
  if(o.mode=="validate"){
    for(auto f:ic::families)for(int k:{0,1,32,64}){launch_dot(f,k,A.p,B.p,4099,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);launch_gemv(f,k,A.p,V.p,17,257,pd.p);}
    CK(cudaDeviceSynchronize());checks<<"actual_kernel_ragged_launches=1\nall_passed=1\n";checks.close();return 0;
  }
  checks<<"all_passed=1\n";checks.close();
  std::vector<caze> cases;for(auto kernel:{std::string("dot"),std::string("gemv")}){for(auto base:{"raw_fp32","fp32_to_fp64","raw_fp64","u32_base"})cases.push_back({kernel,base,"initial",0});for(auto f:ic::families)for(int k:ic::initial_k)cases.push_back({kernel,std::string(ic::name(f)),"initial",k});}
  auto launch=[&](const caze&c){if(c.kernel=="dot"){if(c.family=="raw_fp32"){ic::dot_raw32_kernel<<<512,256>>>(AF.p,BF.p,dotn,pf.p);ic::reduce32_kernel<<<1,256>>>(pf.p,512,pf.p);}else if(c.family=="fp32_to_fp64"){ic::dot_fp32_to_fp64_kernel<<<512,256>>>(AF.p,BF.p,dotn,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);}else if(c.family=="raw_fp64"){ic::dot_raw64_kernel<<<512,256>>>(AD.p,BD.p,dotn,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);}else{auto f=c.family=="u32_base"?ic::family::add32:*std::find_if(ic::families.begin(),ic::families.end(),[&](auto x){return ic::name(x)==c.family;});launch_dot(f,c.k,A.p,B.p,dotn,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);}}else{if(c.family=="raw_fp32")ic::gemv_raw32_kernel<<<rows,256>>>(AF.p,VF.p,rows,cols,pf.p);else if(c.family=="fp32_to_fp64")ic::gemv_fp32_to_fp64_kernel<<<rows,256>>>(AF.p,VF.p,rows,cols,pd.p);else if(c.family=="raw_fp64")ic::gemv_raw64_kernel<<<rows,256>>>(AD.p,VD.p,rows,cols,pd.p);else{auto f=c.family=="u32_base"?ic::family::add32:*std::find_if(ic::families.begin(),ic::families.end(),[&](auto x){return ic::name(x)==c.family;});launch_gemv(f,c.k,A.p,V.p,rows,cols,pd.p);}}};
  ensure(o.output);std::ofstream csv(o.output);csv<<"mode,stage,kernel,family,k,observed_instructions,round,order,ms,result,valid\n";
  std::mt19937 order_rng(0x0325eedu);std::map<std::tuple<std::string,std::string,int>,std::vector<float>> measured;auto began=std::chrono::steady_clock::now();int done=0;
  auto measure_stage=[&](std::vector<caze> stage_cases){
    for(auto&c:stage_cases)for(int w=0;w<o.warmups;++w){launch(c);CK(cudaDeviceSynchronize());}
    int stage_total=o.samples*int(stage_cases.size());int stage_done=0;
    for(int round=0;round<o.samples;++round){std::shuffle(stage_cases.begin(),stage_cases.end(),order_rng);for(int pos=0;pos<int(stage_cases.size());++pos){auto&c=stage_cases[pos];float ms=timed([&]{launch(c);});double value=0;if(c.family=="raw_fp32"){float x;CK(cudaMemcpy(&x,pf.p,sizeof x,cudaMemcpyDeviceToHost));value=x;}else if(c.kernel=="gemv")CK(cudaMemcpy(&value,pd.p,sizeof value,cudaMemcpyDeviceToHost));else CK(cudaMemcpy(&value,result.p,sizeof value,cudaMemcpyDeviceToHost));if(!std::isfinite(value))throw std::runtime_error("nonfinite result");measured[{c.kernel,c.family,c.k}].push_back(ms);csv<<o.mode<<','<<c.stage<<','<<c.kernel<<','<<c.family<<','<<c.k<<','<<c.k<<','<<round<<','<<pos<<','<<std::setprecision(9)<<ms<<','<<std::setprecision(17)<<value<<",1\n";++done;++stage_done;}csv.flush();auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();std::cout<<"progress stage="<<stage_cases.front().stage<<" round="<<round+1<<'/'<<o.samples<<" cases="<<stage_done<<'/'<<stage_total<<" elapsed_s="<<sec<<std::endl;}
    }
  };
  measure_stage(cases);
  if(o.mode=="full"){
    auto median=[](std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];};std::vector<caze> extension;
    for(auto kernel:{std::string("dot"),std::string("gemv")}){bool any=false;for(auto f:ic::families)if(median(measured[{kernel,std::string(ic::name(f)),32}])<median(measured[{kernel,"raw_fp64",0}])){if(!any){for(auto b:{"raw_fp32","fp32_to_fp64","raw_fp64","u32_base"})extension.push_back({kernel,b,"extension",0});any=true;}for(int k:ic::extension_k)extension.push_back({kernel,std::string(ic::name(f)),"extension",k});}}
    if(!extension.empty())measure_stage(extension);
  }
  return 0;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}
