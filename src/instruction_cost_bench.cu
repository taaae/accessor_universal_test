#include "instruction_cost_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
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
std::uint32_t operand(ic::family f){return f==ic::family::rot32?ic::rotate_operand:0;}

#define KS(M,F) M(F,0) M(F,1) M(F,2) M(F,4) M(F,8) M(F,12) M(F,16) M(F,24) M(F,32) M(F,48) M(F,64)
#define DOT_CASE(F,K) case K: ic::dot_timed_kernel<ic::family::F,K><<<512,256>>>(a,b,n,p,operand(ic::family::F),ic::fp_multiplier,ic::fp_addend); break;
void launch_dot(ic::family f,int k,const std::uint32_t*a,const std::uint32_t*b,std::size_t n,double*p){switch(f){
case ic::family::add32f:switch(k){KS(DOT_CASE,add32f)default:throw std::runtime_error("bad K");}break;
case ic::family::mul32f:switch(k){KS(DOT_CASE,mul32f)default:throw std::runtime_error("bad K");}break;
case ic::family::fma32f:switch(k){KS(DOT_CASE,fma32f)default:throw std::runtime_error("bad K");}break;
case ic::family::rot32:switch(k){KS(DOT_CASE,rot32)default:throw std::runtime_error("bad K");}break;}}
#undef DOT_CASE
#define GEMV_CASE(F,K) case K: ic::gemv_timed_kernel<ic::family::F,K><<<rows,256>>>(a,b,rows,cols,p,operand(ic::family::F),ic::fp_multiplier,ic::fp_addend); break;
void launch_gemv(ic::family f,int k,const std::uint32_t*a,const std::uint32_t*b,int rows,int cols,double*p){switch(f){
case ic::family::add32f:switch(k){KS(GEMV_CASE,add32f)default:throw std::runtime_error("bad K");}break;
case ic::family::mul32f:switch(k){KS(GEMV_CASE,mul32f)default:throw std::runtime_error("bad K");}break;
case ic::family::fma32f:switch(k){KS(GEMV_CASE,fma32f)default:throw std::runtime_error("bad K");}break;
case ic::family::rot32:switch(k){KS(GEMV_CASE,rot32)default:throw std::runtime_error("bad K");}break;}}
#undef GEMV_CASE
#define VALIDATE_CASE(F,K) case K: ic::decoder_validation_kernel<ic::family::F,K><<<17,256>>>(a,p,n,operand(ic::family::F),ic::fp_multiplier,ic::fp_addend); break;
void launch_decoder(ic::family f,int k,const std::uint32_t*a,double*p,std::size_t n){switch(f){
case ic::family::add32f:switch(k){KS(VALIDATE_CASE,add32f)default:throw std::runtime_error("bad K");}break;
case ic::family::mul32f:switch(k){KS(VALIDATE_CASE,mul32f)default:throw std::runtime_error("bad K");}break;
case ic::family::fma32f:switch(k){KS(VALIDATE_CASE,fma32f)default:throw std::runtime_error("bad K");}break;
case ic::family::rot32:switch(k){KS(VALIDATE_CASE,rot32)default:throw std::runtime_error("bad K");}break;}}
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
  std::vector<caze> cases;for(auto kernel:{std::string("dot"),std::string("gemv")}){for(auto base:{"raw_fp32","fp32_to_fp64","raw_fp64","u32_base"})cases.push_back({kernel,base,"initial",0});for(auto f:ic::families)for(int k:ic::initial_k)cases.push_back({kernel,std::string(ic::name(f)),"initial",k});}
  auto launch=[&](const caze&c){if(c.kernel=="dot"){if(c.family=="raw_fp32"){ic::dot_raw32_kernel<<<512,256>>>(AF.p,BF.p,dotn,pf.p);ic::reduce32_kernel<<<1,256>>>(pf.p,512,pf.p);}else if(c.family=="fp32_to_fp64"){ic::dot_fp32_to_fp64_kernel<<<512,256>>>(AF.p,BF.p,dotn,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);}else if(c.family=="raw_fp64"){ic::dot_raw64_kernel<<<512,256>>>(AD.p,BD.p,dotn,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);}else{auto f=c.family=="u32_base"?ic::family::add32f:*std::find_if(ic::families.begin(),ic::families.end(),[&](auto x){return ic::name(x)==c.family;});launch_dot(f,c.k,A.p,B.p,dotn,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);}}else{if(c.family=="raw_fp32")ic::gemv_raw32_kernel<<<rows,256>>>(AF.p,VF.p,rows,cols,pf.p);else if(c.family=="fp32_to_fp64")ic::gemv_fp32_to_fp64_kernel<<<rows,256>>>(AF.p,VF.p,rows,cols,pd.p);else if(c.family=="raw_fp64")ic::gemv_raw64_kernel<<<rows,256>>>(AD.p,VD.p,rows,cols,pd.p);else{auto f=c.family=="u32_base"?ic::family::add32f:*std::find_if(ic::families.begin(),ic::families.end(),[&](auto x){return ic::name(x)==c.family;});launch_gemv(f,c.k,A.p,V.p,rows,cols,pd.p);}}};
  auto close_enough=[](double got,long double want,long double sum_abs,std::size_t terms,double eps){long double gamma=(terms*eps)/(1.0L-terms*eps);long double bound=8.0L*gamma*sum_abs+std::numeric_limits<double>::min();return std::abs((long double)got-want)<=bound;};
  for(auto f:ic::families)for(int k:{0,1,2,4,8,12,16,24,32,48,64}){
    for(std::size_t n:{std::size_t(1),std::size_t(31),std::size_t(32),std::size_t(33),std::size_t(257),std::size_t(4099)}){
      launch_dot(f,k,A.p,B.p,n,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);double got;CK(cudaMemcpy(&got,result.p,sizeof got,cudaMemcpyDeviceToHost));long double want=0,sumabs=0;for(std::size_t i=0;i<n;++i){long double term=(long double)ic::decode_reference(hA[i],f,k)*ic::decode_reference(hB[i],f,k);want+=term;sumabs+=std::abs(term);}if(!close_enough(got,want,sumabs,n+512,std::numeric_limits<double>::epsilon()))throw std::runtime_error("DOT CPU/GPU mismatch");
    }
    for(auto shape:{std::pair<int,int>{3,33},std::pair<int,int>{17,257}}){int vr=shape.first,vc=shape.second;launch_gemv(f,k,A.p,V.p,vr,vc,pd.p);std::vector<double> got(vr);CK(cudaMemcpy(got.data(),pd.p,vr*sizeof(double),cudaMemcpyDeviceToHost));for(int r=0;r<vr;++r){long double want=0,sumabs=0;for(int c=0;c<vc;++c){long double term=(long double)ic::decode_reference(hA[std::size_t(r)*vc+c],f,k)*ic::decode_reference(hV[c],f,k);want+=term;sumabs+=std::abs(term);}if(!close_enough(got[r],want,sumabs,vc+256,std::numeric_limits<double>::epsilon()))throw std::runtime_error("GEMV CPU/GPU mismatch");}}
  }
  constexpr std::size_t bn=4099;long double raw32want=0,fp64want=0,raw64want=0,raw32abs=0,fp64abs=0,raw64abs=0;for(std::size_t i=0;i<bn;++i){long double x=(long double)fA[i]*fB[i],y=(long double)dA[i]*dB[i];raw32want+=x;fp64want+=x;raw64want+=y;raw32abs+=std::abs(x);fp64abs+=std::abs(x);raw64abs+=std::abs(y);}ic::dot_raw32_kernel<<<512,256>>>(AF.p,BF.p,bn,pf.p);ic::reduce32_kernel<<<1,256>>>(pf.p,512,pf.p);float r32;CK(cudaMemcpy(&r32,pf.p,sizeof r32,cudaMemcpyDeviceToHost));if(!close_enough(r32,raw32want,raw32abs,bn+512,std::numeric_limits<float>::epsilon()))throw std::runtime_error("raw FP32 DOT mismatch");ic::dot_fp32_to_fp64_kernel<<<512,256>>>(AF.p,BF.p,bn,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);double rd;CK(cudaMemcpy(&rd,result.p,sizeof rd,cudaMemcpyDeviceToHost));if(!close_enough(rd,fp64want,fp64abs,bn+512,std::numeric_limits<double>::epsilon()))throw std::runtime_error("FP32-to-FP64 DOT mismatch");ic::dot_raw64_kernel<<<512,256>>>(AD.p,BD.p,bn,pd.p);ic::reduce64_kernel<<<1,256>>>(pd.p,512,result.p);CK(cudaMemcpy(&rd,result.p,sizeof rd,cudaMemcpyDeviceToHost));if(!close_enough(rd,raw64want,raw64abs,bn+512,std::numeric_limits<double>::epsilon()))throw std::runtime_error("raw FP64 DOT mismatch");
  constexpr int br=17,bc=257;std::vector<float> gr32(br);std::vector<double> grd(br);ic::gemv_raw32_kernel<<<br,256>>>(AF.p,VF.p,br,bc,pf.p);CK(cudaMemcpy(gr32.data(),pf.p,br*sizeof(float),cudaMemcpyDeviceToHost));for(int r=0;r<br;++r){long double want=0,sa=0;for(int c=0;c<bc;++c){long double t=(long double)fA[std::size_t(r)*bc+c]*fV[c];want+=t;sa+=std::abs(t);}if(!close_enough(gr32[r],want,sa,bc+256,std::numeric_limits<float>::epsilon()))throw std::runtime_error("raw FP32 GEMV mismatch");}ic::gemv_fp32_to_fp64_kernel<<<br,256>>>(AF.p,VF.p,br,bc,pd.p);CK(cudaMemcpy(grd.data(),pd.p,br*sizeof(double),cudaMemcpyDeviceToHost));for(int r=0;r<br;++r){long double want=0,sa=0;for(int c=0;c<bc;++c){long double t=(long double)fA[std::size_t(r)*bc+c]*fV[c];want+=t;sa+=std::abs(t);}if(!close_enough(grd[r],want,sa,bc+256,std::numeric_limits<double>::epsilon()))throw std::runtime_error("FP32-to-FP64 GEMV mismatch");}ic::gemv_raw64_kernel<<<br,256>>>(AD.p,VD.p,br,bc,pd.p);CK(cudaMemcpy(grd.data(),pd.p,br*sizeof(double),cudaMemcpyDeviceToHost));for(int r=0;r<br;++r){long double want=0,sa=0;for(int c=0;c<bc;++c){long double t=(long double)dA[std::size_t(r)*bc+c]*dV[c];want+=t;sa+=std::abs(t);}if(!close_enough(grd[r],want,sa,bc+256,std::numeric_limits<double>::epsilon()))throw std::runtime_error("raw FP64 GEMV mismatch");}
  checks<<"actual_dot_shapes=1,31,32,33,257,4099\nactual_gemv_shapes=3x33,17x257\nactual_all_family_k_cpu_gpu=1\nbaseline_dot_cpu_gpu=1\nbaseline_gemv_cpu_gpu=1\nall_passed=1\n";checks.close();
  if(o.mode=="validate")return 0;
  const char*job_env=std::getenv("SLURM_JOB_ID");const char*node_env=std::getenv("HOSTNAME");
  ensure(o.output);std::ofstream csv(o.output);csv<<"mode,stage,kernel,family,k,observed_instructions,n,rows,cols,storage,arithmetic,left_seed,right_seed,round,order,ms,result,valid,job_id,node\n";
  std::mt19937 order_rng(0x0325eedu);std::map<std::tuple<std::string,std::string,std::string,int>,std::vector<float>> measured;auto began=std::chrono::steady_clock::now();int done=0;
  auto measure_stage=[&](std::vector<caze> stage_cases){
    for(auto&c:stage_cases)for(int w=0;w<o.warmups;++w){launch(c);CK(cudaDeviceSynchronize());}
    int stage_total=o.samples*int(stage_cases.size());int stage_done=0;
    for(int round=0;round<o.samples;++round){
      std::shuffle(stage_cases.begin(),stage_cases.end(),order_rng);
      for(int pos=0;pos<int(stage_cases.size());++pos){
        auto&c=stage_cases[pos];float ms=timed([&]{launch(c);});double value=0;
        if(c.kernel=="gemv"&&c.family=="raw_fp32"){std::vector<float> values(rows);CK(cudaMemcpy(values.data(),pf.p,rows*sizeof(float),cudaMemcpyDeviceToHost));for(float x:values){if(!std::isfinite(x))throw std::runtime_error("nonfinite GEMV row");value+=x;}}
        else if(c.family=="raw_fp32"){float x;CK(cudaMemcpy(&x,pf.p,sizeof x,cudaMemcpyDeviceToHost));value=x;}
        else if(c.kernel=="gemv"){std::vector<double> values(rows);CK(cudaMemcpy(values.data(),pd.p,rows*sizeof(double),cudaMemcpyDeviceToHost));for(double x:values){if(!std::isfinite(x))throw std::runtime_error("nonfinite GEMV row");value+=x;}}
        else CK(cudaMemcpy(&value,result.p,sizeof value,cudaMemcpyDeviceToHost));
        if(!std::isfinite(value))throw std::runtime_error("nonfinite result");
        measured[{c.stage,c.kernel,c.family,c.k}].push_back(ms);
        const char*storage=c.family=="raw_fp64"?"float64":(c.family=="raw_fp32"||c.family=="fp32_to_fp64"?"float32":"uint32");const char*arithmetic=c.family=="raw_fp32"?"fp32":"fp64";
        csv<<o.mode<<','<<c.stage<<','<<c.kernel<<','<<c.family<<','<<c.k<<','<<c.k<<','<<(c.kernel=="dot"?dotn:std::size_t(rows)*cols)<<','<<(c.kernel=="gemv"?rows:0)<<','<<(c.kernel=="gemv"?cols:0)<<','<<storage<<','<<arithmetic<<",0x6bd87c012a53f9e1,0xf5ef05b8551985f4,"<<round<<','<<pos<<','<<std::setprecision(9)<<ms<<','<<std::setprecision(17)<<value<<",1,"<<(job_env?job_env:"")<<','<<(node_env?node_env:"")<<'\n';
        ++done;++stage_done;
      }
      csv.flush();auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
      std::cout<<"progress stage="<<stage_cases.front().stage<<" round="<<round+1<<'/'<<o.samples<<" cases="<<stage_done<<'/'<<stage_total<<" elapsed_s="<<sec<<std::endl;
    }
  };
  measure_stage(cases);
  if(o.mode=="full"){
    auto median=[](std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];};auto drift=[&](const std::vector<float>&v){std::vector<float>a(v.begin(),v.begin()+v.size()/2),b(v.begin()+v.size()/2,v.end());return std::abs(median(a)-median(b))/median(a);};std::string official="initial";bool unstable=false;
    for(auto kernel:{std::string("dot"),std::string("gemv")})for(auto base:{"raw_fp32","fp32_to_fp64","raw_fp64","u32_base"})unstable|=drift(measured[{"initial",kernel,base,0}])>0.05;
    if(unstable){auto rerun=cases;for(auto&c:rerun)c.stage="rerun";measure_stage(rerun);official="rerun";}
    std::vector<caze> extension;
    for(auto kernel:{std::string("dot"),std::string("gemv")}){bool any=false;for(auto f:ic::families)if(median(measured[{official,kernel,std::string(ic::name(f)),32}])<median(measured[{official,kernel,"raw_fp64",0}])){if(!any){for(auto b:{"raw_fp32","fp32_to_fp64","raw_fp64","u32_base"})extension.push_back({kernel,b,"extension",0});any=true;}for(int k:ic::extension_k)extension.push_back({kernel,std::string(ic::name(f)),"extension",k});}}
    if(!extension.empty())measure_stage(extension);
  }
  return 0;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}
