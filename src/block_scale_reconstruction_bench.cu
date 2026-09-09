#include "block_scale_reconstruction_core.hpp"
#include "block_scale_reconstruction_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace core=aut::block_scale;
namespace bs=aut::block_scale_reconstruction;
namespace {
void ck(cudaError_t e,const char*x){if(e!=cudaSuccess)throw std::runtime_error(std::string(x)+": "+cudaGetErrorString(e));}
#define CK(x) ck((x),#x)
template<class T>struct dbuf{T*p{};std::size_t n{};dbuf()=default;explicit dbuf(std::size_t m){reset(m);}dbuf(const dbuf&)=delete;dbuf&operator=(const dbuf&)=delete;dbuf(dbuf&&x)noexcept:p(x.p),n(x.n){x.p=nullptr;}~dbuf(){if(p)cudaFree(p);}void reset(std::size_t m){if(p)CK(cudaFree(p));n=std::max<std::size_t>(m,1);CK(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}};
struct event{cudaEvent_t e;event(){CK(cudaEventCreate(&e));}~event(){cudaEventDestroy(e);}};
struct options{std::string mode="full",output="timing_samples.csv",checks="correctness_checks.txt",source_commit,binary_sha,sass_sha,audit_sha;int warmups=10,samples=50;};
options parse(int ac,char**av){options o;bool explicit_warmups=false,explicit_samples=false;for(int i=1;i<ac;++i){std::string a=av[i];auto v=[&]{if(++i>=ac)throw std::runtime_error("missing option value");return std::string(av[i]);};if(a=="--mode")o.mode=v();else if(a=="--output")o.output=v();else if(a=="--correctness-output")o.checks=v();else if(a=="--warmups"){o.warmups=std::stoi(v());explicit_warmups=true;}else if(a=="--samples"){o.samples=std::stoi(v());explicit_samples=true;}else if(a=="--source-commit")o.source_commit=v();else if(a=="--binary-sha")o.binary_sha=v();else if(a=="--sass-sha")o.sass_sha=v();else if(a=="--audit-sha")o.audit_sha=v();else throw std::runtime_error("unknown option "+a);}if(o.mode=="smoke"){if((explicit_warmups&&o.warmups!=1)||(explicit_samples&&o.samples!=3))throw std::runtime_error("smoke dimensions are fixed at 1 warmup and 3 samples");o.warmups=1;o.samples=3;}else if(o.mode=="validate"){if(explicit_warmups||explicit_samples)throw std::runtime_error("validate mode does not accept timing dimensions");o.warmups=0;o.samples=0;}else if(o.mode!="full")throw std::runtime_error("invalid mode");if(o.warmups<0||o.samples<0)throw std::runtime_error("negative count");if(o.mode!="validate"&&(o.source_commit.empty()||o.binary_sha.empty()||o.sass_sha.empty()||o.audit_sha.empty()))throw std::runtime_error("timing requires provenance digests");return o;}
void parent(const std::string&p){auto q=std::filesystem::path(p).parent_path();if(!q.empty())std::filesystem::create_directories(q);}
float timed(const std::function<void()>&f){event a,b;CK(cudaEventRecord(a.e));f();CK(cudaGetLastError());CK(cudaEventRecord(b.e));CK(cudaEventSynchronize(b.e));float ms=0;CK(cudaEventElapsedTime(&ms,a.e,b.e));return ms;}
std::uint64_t bits(double x){std::uint64_t u;std::memcpy(&u,&x,8);return u;}
double median(std::vector<float>v){std::sort(v.begin(),v.end());return .5*(v[(v.size()-1)/2]+v[v.size()/2]);}
struct encoded{int b{};dbuf<float>ql,qr,sl32,sr32;dbuf<double>sl64,sr64;};
struct data{std::size_t n{};dbuf<float>fl,fr;dbuf<double>dl,dr;std::array<encoded,3> e;dbuf<float>p32,out32;dbuf<double>p64,out64;};
template<class T>void copy_at(T*d,std::size_t off,const std::vector<T>&h){if(!h.empty())CK(cudaMemcpy(d+off,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}
data prepare(std::size_t n,int fixture=0){const bool edge=fixture==1,positive=fixture==2,rounding=fixture==3;data d;d.n=n;d.fl.reset(n);d.fr.reset(n);d.dl.reset(n);d.dr.reset(n);d.p32.reset(bs::grid_blocks);d.out32.reset(1);d.p64.reset(bs::grid_blocks);d.out64.reset(1);for(int z=0;z<3;++z){auto&x=d.e[z];x.b=core::block_sizes[z];x.ql.reset(n);x.qr.reset(n);auto nb=core::ceil_div(n,x.b);x.sl32.reset(nb);x.sr32.reset(nb);x.sl64.reset(nb);x.sr64.reset(nb);}
 constexpr std::size_t chunk=1u<<20;const float one_up=std::nextafter(1.f,2.f),two_up=std::nextafter(one_up,2.f);for(std::size_t off=0;off<n;off+=chunk){std::size_t m=std::min(chunk,n-off);std::vector<float>l(m),r(m);std::vector<double>ld(m),rd(m);for(std::size_t j=0;j<m;++j){auto i=off+j;l[j]=positive?.25f:rounding?(i%5==0?two_up:one_up):core::source_value(core::value_left_seed,i);r[j]=positive?.5f:rounding?(i%7==0?two_up:one_up):core::source_value(core::value_right_seed,i);if(edge){static const float q[]{0.f,1.f,-1.f,.5f,-.5f,.25f,-.25f};l[j]=q[i%7];r[j]=q[(i*5+3)%7];}ld[j]=l[j];rd[j]=r[j];}copy_at(d.fl.p,off,l);copy_at(d.fr.p,off,r);copy_at(d.dl.p,off,ld);copy_at(d.dr.p,off,rd);
  for(auto&x:d.e){std::vector<float>ql(m),qr(m);for(std::size_t j=0;j<m;++j){auto i=off+j,b=i/x.b;float sl=edge?(b%3==0?.50000006f:b%3==1?1.00000012f:1.99999988f):positive?1.25f:rounding?(b%2?two_up:one_up):core::scale_value(core::scale_left_seed,b);float sr=edge?(b%2==0?1.99999988f:.50000006f):positive?.75f:rounding?(b%3?one_up:two_up):core::scale_value(core::scale_right_seed,b);ql[j]=(edge||positive||rounding)?l[j]:core::encode(l[j],sl);qr[j]=(edge||positive||rounding)?r[j]:core::encode(r[j],sr);}copy_at(x.ql.p,off,ql);copy_at(x.qr.p,off,qr);}
 }
 for(auto&x:d.e){auto nb=core::ceil_div(n,x.b);for(std::size_t off=0;off<nb;off+=chunk){std::size_t m=std::min(chunk,nb-off);std::vector<float>sl(m),sr(m);std::vector<double>dl(m),dr(m);for(std::size_t j=0;j<m;++j){auto b=off+j;sl[j]=edge?(b%3==0?.50000006f:b%3==1?1.00000012f:1.99999988f):positive?1.25f:rounding?(b%2?two_up:one_up):core::scale_value(core::scale_left_seed,b);sr[j]=edge?(b%2==0?1.99999988f:.50000006f):positive?.75f:rounding?(b%3?one_up:two_up):core::scale_value(core::scale_right_seed,b);dl[j]=sl[j];dr[j]=sr[j];}copy_at(x.sl32.p,off,sl);copy_at(x.sr32.p,off,sr);copy_at(x.sl64.p,off,dl);copy_at(x.sr64.p,off,dr);}}
 return d;}
struct variant{std::string id,path,scale,reconstruction,subtotal;int b;};
std::vector<variant> variants(){std::vector<variant>v;for(int b:core::block_sizes){v.push_back({bs::scaled_id("fp64",b,false),"per_element","fp32","fp64","none",b});v.push_back({bs::scaled_id("fp32",b,false),"per_element","fp32","fp32","none",b});v.push_back({bs::scaled_id("fp64",b,true),"per_element","fp64","fp64","none",b});}v.push_back({"deferred_local_b128_s32","deferred_local","fp32","fp64","thread_local_4",128});v.push_back({"deferred_local_b128_s64","deferred_local","fp64","fp64","thread_local_4",128});v.push_back({"raw_fp32","baseline","none","native_fp32","none",0});v.push_back({"fp32_to_fp64","baseline","none","widen_fp32","none",0});v.push_back({"raw_fp64","baseline","none","native_fp64","none",0});return v;}
encoded& enc(data&d,int b){for(auto&x:d.e)if(x.b==b)return x;throw std::runtime_error("bad B");}
#define LAUNCH64(B,T,FL,FR) if(v.b==B){auto&x=enc(d,B);bs::reconstruct64_kernel<B,T><<<bs::grid_blocks,bs::cta_threads>>>(x.ql.p,x.qr.p,x.FL.p,x.FR.p,d.n,d.p64.p);}
#define LAUNCH32(B) if(v.b==B){auto&x=enc(d,B);bs::reconstruct32_kernel<B><<<bs::grid_blocks,bs::cta_threads>>>(x.ql.p,x.qr.p,x.sl32.p,x.sr32.p,d.n,d.p64.p);}
void launch_first(data&d,const variant&v){if(v.path=="baseline"){if(v.id=="raw_fp32")bs::raw32_kernel<<<bs::grid_blocks,bs::cta_threads>>>(d.fl.p,d.fr.p,d.n,d.p32.p);else if(v.id=="fp32_to_fp64")bs::widened32_kernel<<<bs::grid_blocks,bs::cta_threads>>>(d.fl.p,d.fr.p,d.n,d.p64.p);else if(v.id=="raw_fp64")bs::raw64_kernel<<<bs::grid_blocks,bs::cta_threads>>>(d.dl.p,d.dr.p,d.n,d.p64.p);else throw std::runtime_error("unknown baseline");return;}if(v.path=="per_element"&&v.reconstruction=="fp32"){LAUNCH32(16)else LAUNCH32(32)else LAUNCH32(128)}else if(v.path=="per_element"&&v.scale=="fp32"){LAUNCH64(16,float,sl32,sr32)else LAUNCH64(32,float,sl32,sr32)else LAUNCH64(128,float,sl32,sr32)}else if(v.path=="per_element"&&v.scale=="fp64"){LAUNCH64(16,double,sl64,sr64)else LAUNCH64(32,double,sl64,sr64)else LAUNCH64(128,double,sl64,sr64)}else if(v.path=="deferred_local"){auto&x=enc(d,128);if(v.scale=="fp32")bs::deferred_local_b128_kernel<float><<<bs::grid_blocks,bs::cta_threads>>>(x.ql.p,x.qr.p,x.sl32.p,x.sr32.p,d.n,d.p64.p);else bs::deferred_local_b128_kernel<double><<<bs::grid_blocks,bs::cta_threads>>>(x.ql.p,x.qr.p,x.sl64.p,x.sr64.p,d.n,d.p64.p);}else throw std::runtime_error("unknown path");}
#undef LAUNCH64
#undef LAUNCH32
void launch(data&d,const variant&v){launch_first(d,v);if(v.id=="raw_fp32")bs::reduce32_kernel<<<1,bs::cta_threads>>>(d.p32.p,d.out32.p);else bs::reduce64_kernel<<<1,bs::cta_threads>>>(d.p64.p,d.out64.p);}
double result(data&d,const variant&v){if(v.id=="raw_fp32"){float x;CK(cudaMemcpy(&x,d.out32.p,4,cudaMemcpyDeviceToHost));return x;}double x;CK(cudaMemcpy(&x,d.out64.p,8,cudaMemcpyDeviceToHost));return x;}
void validate_dataset(std::size_t n,int fixture,std::ofstream&checks,long double&maxerr,long double&maxbound,int&count){const bool edge=fixture==1,positive=fixture==2,rounding=fixture==3;auto d=prepare(n,fixture);auto vs=variants();std::map<std::tuple<std::string,int>,double> width;std::map<std::pair<std::string,int>,double> reconstruction_results;
 for(const auto&v:vs){launch(d,v);CK(cudaDeviceSynchronize());double got=result(d,v);if(!std::isfinite(got))throw std::runtime_error("nonfinite validation result");long double want=0,sa=0;
  if(v.path=="baseline"){const float one_up=std::nextafter(1.f,2.f),two_up=std::nextafter(one_up,2.f);static const float q[]{0.f,1.f,-1.f,.5f,-.5f,.25f,-.25f};for(std::size_t i=0;i<n;++i){long double l=positive?.25f:rounding?(i%5==0?two_up:one_up):edge?q[i%7]:core::source_value(core::value_left_seed,i);long double r=positive?.5f:rounding?(i%7==0?two_up:one_up):edge?q[(i*5+3)%7]:core::source_value(core::value_right_seed,i);long double t=l*r;want+=t;sa+=std::abs(t);}}
  else{auto h=bs::make_fixture(n,v.b,fixture);if(v.path=="deferred_local"){want=bs::reference_deferred_local(h);sa=bs::sum_abs_deferred_elements(h);}else if(v.reconstruction=="fp32"){want=bs::reference_reconstruct32(h,v.b);sa=bs::sum_abs_reconstruct32(h,v.b);}else{want=bs::reference_reconstruct64(h,v.b);sa=bs::sum_abs_reconstruct64(h,v.b);}}
  long double eps=v.id=="raw_fp32"?std::numeric_limits<float>::epsilon():std::numeric_limits<double>::epsilon();std::size_t depth=(n+std::size_t(bs::grid_blocks*bs::cta_threads)-1)/std::size_t(bs::grid_blocks*bs::cta_threads)+24;if(v.path=="deferred_local")depth=core::ceil_div(core::ceil_div(n,128u),std::size_t(bs::grid_blocks*8))+4+24;long double gamma=(depth*eps)/(1-depth*eps);long double bound=16*gamma*sa+16*eps;long double err=std::abs((long double)got-want);if(err>bound)throw std::runtime_error("CPU/GPU validation mismatch "+v.id);maxerr=std::max(maxerr,err);maxbound=std::max(maxbound,bound);if(v.path=="deferred_local"||v.reconstruction=="fp64"){auto key=std::make_tuple(v.path,v.b);if(v.scale=="fp32")width[key]=got;else if(bits(width.at(key))!=bits(got))throw std::runtime_error("scale-width result mismatch");}if(v.path=="per_element")reconstruction_results[{v.reconstruction,v.b}]=got;++count;
 }
 if(rounding)for(int b:core::block_sizes)if(bits(reconstruction_results.at({"fp32",b}))==bits(reconstruction_results.at({"fp64",b})))throw std::runtime_error("rounding fixture did not distinguish reconstruction contracts");
 checks<<"validated_n="<<n<<",fixture="<<(edge?"edge":positive?"all_positive":rounding?"rounding_sensitive":"generated")<<",cases="<<vs.size()<<"\n";
}
}

namespace aut::block_scale_reconstruction {
__global__ void raw32_kernel(const float*a,const float*b,std::size_t n,float*p){float s=0;std::size_t st=std::size_t(gridDim.x)*blockDim.x;
#pragma unroll 1
for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=st)s=__fmaf_rn(a[i],b[i],s);float t=block_sum(s);if(threadIdx.x==0)p[blockIdx.x]=t;}
__global__ void widened32_kernel(const float*a,const float*b,std::size_t n,double*p){double s=0;std::size_t st=std::size_t(gridDim.x)*blockDim.x;
#pragma unroll 1
for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=st)s=__fma_rn(double(a[i]),double(b[i]),s);double t=block_sum(s);if(threadIdx.x==0)p[blockIdx.x]=t;}
__global__ void raw64_kernel(const double*a,const double*b,std::size_t n,double*p){double s=0;std::size_t st=std::size_t(gridDim.x)*blockDim.x;
#pragma unroll 1
for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=st)s=__fma_rn(a[i],b[i],s);double t=block_sum(s);if(threadIdx.x==0)p[blockIdx.x]=t;}
__global__ void reduce32_kernel(const float*p,float*out){float s=0;for(int i=threadIdx.x;i<grid_blocks;i+=blockDim.x)s+=p[i];float t=block_sum(s);if(threadIdx.x==0)*out=t;}
__global__ void reduce64_kernel(const double*p,double*out){double s=0;for(int i=threadIdx.x;i<grid_blocks;i+=blockDim.x)s+=p[i];double t=block_sum(s);if(threadIdx.x==0)*out=t;}
}

int main(int ac,char**av)try{auto o=parse(ac,av);parent(o.checks);std::ofstream checks(o.checks);long double maxerr=0,maxbound=0;int validated=0;if(std::numeric_limits<long double>::digits<=std::numeric_limits<double>::digits)throw std::runtime_error("long double lacks extra precision");for(std::size_t n:{0u,1u,15u,16u,17u,31u,32u,33u,127u,128u,129u,257u,4099u,1u<<20})validate_dataset(n,0,checks,maxerr,maxbound,validated);validate_dataset(4099,1,checks,maxerr,maxbound,validated);validate_dataset(4099,2,checks,maxerr,maxbound,validated);validate_dataset(257,3,checks,maxerr,maxbound,validated);checks<<std::setprecision(18)<<"max_reference_error="<<maxerr<<"\nmax_reference_bound="<<maxbound<<"\nvalidated_case_count="<<validated<<"\nrounding_contracts_distinct=1\nlong_double_bytes="<<sizeof(long double)<<"\nlong_double_digits="<<std::numeric_limits<long double>::digits<<"\nall_passed=1\n";checks.close();if(o.mode=="validate")return 0;
 std::size_t free_bytes=0,total_bytes=0;CK(cudaMemGetInfo(&free_bytes,&total_bytes));std::cout<<"device_memory_free="<<free_bytes<<" total="<<total_bytes<<std::endl;if(o.mode=="full"&&free_bytes<14ULL*1024*1024*1024)throw std::runtime_error("less than 14 GiB device memory free");
 std::vector<std::size_t> sizes=o.mode=="smoke"?std::vector<std::size_t>{1u<<14,1u<<20}:std::vector<std::size_t>(core::full_sizes.begin(),core::full_sizes.end());auto vs=variants();parent(o.output);std::ofstream csv(o.output);csv<<"mode,stage,N,B,scale_type,path,reconstruction,subtotal_mode,dot_accumulation,variant_id,payload_type,grid,threads,round,order,time_ms,result,valid,job_id,node,experiment_id,dataset_id,seed_set_id,source_commit,binary_sha,sass_sha,audit_sha\n";const char*job=std::getenv("SLURM_JOB_ID"),*node=std::getenv("HOSTNAME");std::mt19937 rng(0x0335eedu);auto started=std::chrono::steady_clock::now();std::size_t done=0,total=sizes.size()*vs.size()*o.samples;
 for(auto n:sizes){std::cout<<"preparing N="<<n<<std::endl;auto d=prepare(n);std::map<std::string,std::vector<float>> base_times;std::map<std::string,std::uint64_t> stable;
  auto stage=[&](std::string stage_name){for(auto&v:vs)for(int w=0;w<o.warmups;++w){launch(d,v);CK(cudaDeviceSynchronize());}for(int round=0;round<o.samples;++round){std::shuffle(vs.begin(),vs.end(),rng);for(int order=0;order<int(vs.size());++order){auto&v=vs[order];float ms=timed([&]{launch(d,v);});double got=result(d,v);if(!std::isfinite(got)||!(ms>0))throw std::runtime_error("invalid timed result");auto key=stage_name+v.id;if(!stable.count(key))stable[key]=bits(got);else if(stable[key]!=bits(got))throw std::runtime_error("nondeterministic result "+v.id);if(v.path=="baseline")base_times[v.id].push_back(ms);csv<<o.mode<<','<<stage_name<<','<<n<<','<<v.b<<','<<v.scale<<','<<v.path<<','<<v.reconstruction<<','<<v.subtotal<<','<<(v.id=="raw_fp32"?"fp32":"fp64")<<','<<v.id<<','<<(v.id=="raw_fp64"?"fp64":"fp32")<<",512,256,"<<round<<','<<order<<','<<std::setprecision(9)<<ms<<','<<std::setprecision(17)<<got<<",1,"<<(job?job:"")<<','<<(node?node:"")<<','<<bs::experiment_id<<','<<bs::dataset_id<<','<<core::seed_set_id<<','<<o.source_commit<<','<<o.binary_sha<<','<<o.sass_sha<<','<<o.audit_sha<<'\n';++done;}auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();std::cout<<"progress N="<<n<<" stage="<<stage_name<<" round="<<round+1<<'/'<<o.samples<<" completed="<<done<<'/'<<total<<" elapsed_s="<<sec<<std::endl;}};
  stage("initial");if(o.mode=="full"){bool drift=false;for(auto&id:{"raw_fp32","fp32_to_fp64","raw_fp64"}){auto v=base_times[id];std::vector<float>a(v.begin(),v.begin()+25),b(v.begin()+25,v.end());double ratio=std::abs(median(b)-median(a))/median(a);std::cout<<"baseline_drift N="<<n<<" stage=initial variant="<<id<<" ratio="<<ratio<<std::endl;if(ratio>.05)drift=true;}if(drift){total+=vs.size()*o.samples;base_times.clear();stage("rerun");bool persistent=false;for(auto&id:{"raw_fp32","fp32_to_fp64","raw_fp64"}){auto v=base_times[id];std::vector<float>a(v.begin(),v.begin()+25),b(v.begin()+25,v.end());double ratio=std::abs(median(b)-median(a))/median(a);std::cout<<"baseline_drift N="<<n<<" stage=rerun variant="<<id<<" ratio="<<ratio<<std::endl;if(ratio>.05)persistent=true;}std::cout<<"persistent_drift N="<<n<<" value="<<(persistent?1:0)<<std::endl;}}
 }
 csv.close();std::cout<<(o.mode=="smoke"?"SMOKE_MEASUREMENTS_COMPLETE":"GPU_MEASUREMENTS_COMPLETE")<<" rows="<<done<<std::endl;return 0;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}
