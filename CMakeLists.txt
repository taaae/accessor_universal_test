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
include(CTest)

add_executable(dot_bench src/dot_bench.cu)
target_link_libraries(dot_bench PRIVATE CUDA::cublas CUDA::cudart)
target_include_directories(dot_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(
    dot_bench
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>"
)

add_executable(dataset_inspect tools/inspect_dot_dataset.cpp)
target_include_directories(
    dataset_inspect
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(dataset_inspect PRIVATE -O3)

add_executable(accessor_dot_bench src/accessor_dot_bench.cu)
target_link_libraries(accessor_dot_bench PRIVATE CUDA::cudart)
target_include_directories(
    accessor_dot_bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    accessor_dot_bench
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

add_executable(memory_accessor_test tests/memory_accessor_test.cpp)
target_include_directories(
    memory_accessor_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(memory_accessor_test PRIVATE -O3)
add_test(NAME memory_accessor_test COMMAND memory_accessor_test)

set_target_properties(
    dot_bench
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    dataset_inspect
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    accessor_dot_bench
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    memory_accessor_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
