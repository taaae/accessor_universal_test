cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

project(accessor_universal_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(dot_bench src/dot_bench.cu)
target_link_libraries(dot_bench PRIVATE CUDA::cublas CUDA::cudart)
target_compile_options(
    dot_bench
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>"
)

set_target_properties(
    dot_bench
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
