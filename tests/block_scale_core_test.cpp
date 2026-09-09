#include "block_scale_core.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace aut::block_scale;
static void require(bool x,const char*m){if(!x)throw std::runtime_error(m);}
int main(){
 require(high32(value_left_seed,0)==0xfe73d3b6u,"generator fixture left");
 require(high32(scale_left_seed,0)==0x2cb0f69fu,"generator fixture scale");
 require(source_value(value_left_seed,0)>=-1&&source_value(value_left_seed,0)<1,"source range");
 for(std::size_t i=0;i<10000;++i){float s=scale_value(scale_left_seed,i);require(std::isfinite(s)&&s>=.5f&&s<2.f,"scale range");require(double(s)==static_cast<double>(s),"scale widening");}
 for(std::size_t n:{0u,1u,15u,16u,17u,127u,128u,129u,257u})for(int b:block_sizes){auto a=make_arrays(n,b);require(a.ql.size()==n&&a.sl32.size()==ceil_div(n,b),"sizes");for(std::size_t j=0;j<a.sl32.size();++j)require(a.sl64[j]==double(a.sl32[j])&&a.sr64[j]==double(a.sr32[j]),"scale identity");require(std::isfinite(double(reference_per_element(a,b)))&&std::isfinite(double(reference_deferred(a,b))),"references");}
 require(source_value(value_left_seed,7)!=source_value(value_right_seed,7),"independent values");require(scale_value(scale_left_seed,7)!=scale_value(scale_right_seed,7),"independent scales");
 bool threw=false;try{(void)scaled_id("bad",16,false);}catch(const std::exception&){threw=true;}require(threw,"bad path accepted");
 std::cout<<"block_scale_core_test passed\n";
}
