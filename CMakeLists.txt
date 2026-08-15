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

add_executable(storage_formats_test tests/storage_formats_test.cpp)
target_include_directories(
    storage_formats_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(storage_formats_test PRIVATE -O3)
add_test(NAME storage_formats_test COMMAND storage_formats_test)

add_executable(decoder_strategy_core_test tests/decoder_strategy_core_test.cpp)
target_include_directories(
    decoder_strategy_core_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(decoder_strategy_core_test PRIVATE -O3)
add_test(NAME decoder_strategy_core_test COMMAND decoder_strategy_core_test)

add_executable(bitwidth_benchmark_core_test
    tests/bitwidth_benchmark_core_test.cpp)
target_include_directories(
    bitwidth_benchmark_core_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(bitwidth_benchmark_core_test PRIVATE -O3)
add_test(NAME bitwidth_benchmark_core_test COMMAND bitwidth_benchmark_core_test)

function(add_bitwidth_strategy_benchmark bits)
    set(target "bitwidth_strategy_bench_${bits}")
    add_executable(${target} src/bitwidth_strategy_bench.cu)
    target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::curand)
    target_include_directories(
        ${target}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_compile_definitions(
        ${target}
        PRIVATE AUT_BITWIDTH_TOTAL_BITS=${bits}
    )
    target_compile_options(
        ${target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
    )
endfunction()

add_bitwidth_strategy_benchmark(3)
add_bitwidth_strategy_benchmark(5)
add_bitwidth_strategy_benchmark(6)
add_bitwidth_strategy_benchmark(7)
add_bitwidth_strategy_benchmark(9)
add_bitwidth_strategy_benchmark(10)
add_bitwidth_strategy_benchmark(12)
add_bitwidth_strategy_benchmark(14)
add_bitwidth_strategy_benchmark(17)

add_executable(accuracy_statistics_test tests/accuracy_statistics_test.cpp)
target_include_directories(
    accuracy_statistics_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(accuracy_statistics_test PRIVATE -O3)
add_test(NAME accuracy_statistics_test COMMAND accuracy_statistics_test)

add_executable(storage_formats_cuda_test src/storage_formats_cuda_test.cu)
target_include_directories(
    storage_formats_cuda_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    storage_formats_cuda_test
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

add_executable(storage_kernels_cuda_test src/storage_kernels_cuda_test.cu)
target_include_directories(
    storage_kernels_cuda_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    storage_kernels_cuda_test
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

add_executable(accuracy_simulation src/accuracy_simulation.cu)
target_link_libraries(accuracy_simulation PRIVATE CUDA::curand_static)
target_include_directories(
    accuracy_simulation
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    accuracy_simulation
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

add_executable(storage_performance_bench src/storage_performance_bench.cu)
target_link_libraries(storage_performance_bench PRIVATE CUDA::curand_static)
target_include_directories(
    storage_performance_bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    storage_performance_bench
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

add_executable(decoder_optimization_bench src/decoder_optimization_bench.cu)
target_link_libraries(decoder_optimization_bench PRIVATE CUDA::curand_static)
target_include_directories(
    decoder_optimization_bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    decoder_optimization_bench
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

add_executable(e2e3_strategy_smoke src/e2e3_strategy_smoke.cu)
target_link_libraries(e2e3_strategy_smoke PRIVATE CUDA::cudart_static)
target_include_directories(
    e2e3_strategy_smoke
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    e2e3_strategy_smoke
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

add_executable(e2e3_strategy_bench src/e2e3_strategy_bench.cu)
target_link_libraries(e2e3_strategy_bench PRIVATE CUDA::curand_static)
target_include_directories(
    e2e3_strategy_bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    e2e3_strategy_bench
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

add_executable(all_format_strategy_smoke src/all_format_strategy_smoke.cu)
target_link_libraries(all_format_strategy_smoke PRIVATE CUDA::cudart_static)
target_include_directories(
    all_format_strategy_smoke
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    all_format_strategy_smoke
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

foreach(expanded_bits IN ITEMS 2 4 8 16 32)
    set(expanded_target "expanded_format_strategy_smoke_${expanded_bits}")
    add_executable(${expanded_target} src/all_format_strategy_smoke.cu)
    target_link_libraries(${expanded_target} PRIVATE CUDA::cudart_static)
    target_include_directories(
        ${expanded_target}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_compile_definitions(
        ${expanded_target}
        PRIVATE
            AUT_EXPANDED_FORMAT_SMOKE=1
            AUT_EXPANDED_SMOKE_BITS=${expanded_bits}
    )
    target_compile_options(
        ${expanded_target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
    )
    set_target_properties(
        ${expanded_target}
        PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

add_executable(all_format_strategy_bench src/all_format_strategy_bench.cu)
target_link_libraries(all_format_strategy_bench PRIVATE CUDA::curand_static)
target_include_directories(
    all_format_strategy_bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    all_format_strategy_bench
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)

foreach(expanded_bits IN ITEMS 2 4 8 16 32)
    set(expanded_target "expanded_format_strategy_bench_${expanded_bits}")
    add_executable(${expanded_target} src/all_format_strategy_bench.cu)
    target_link_libraries(${expanded_target} PRIVATE CUDA::curand_static)
    target_include_directories(
        ${expanded_target}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_compile_definitions(
        ${expanded_target}
        PRIVATE
            AUT_EXPANDED_FORMAT_BENCH=1
            AUT_EXPANDED_BENCH_BITS=${expanded_bits}
    )
    target_compile_options(
        ${expanded_target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
    )
    set_target_properties(
        ${expanded_target}
        PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

set(
    precision_packing_formats
    fp16
    bf16
    fp32
    fp8_e4m3
    fp8_e5m2
    fp4_e2m1
    fp64
)
set(precision_packing_format_id 0)
foreach(precision_format IN LISTS precision_packing_formats)
    math(EXPR precision_packing_format_id "${precision_packing_format_id} + 1")
    set(precision_target "precision_packing_bench_${precision_format}")
    add_executable(${precision_target} src/precision_packing_bench.cu)
    target_link_libraries(${precision_target} PRIVATE CUDA::curand_static)
    target_include_directories(
        ${precision_target}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_compile_definitions(
        ${precision_target}
        PRIVATE
            AUT_PRECISION_FORMAT=${precision_packing_format_id}
    )
    target_compile_options(
        ${precision_target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
    )
    set_target_properties(
        ${precision_target}
        PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

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

set_target_properties(
    storage_formats_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    decoder_strategy_core_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    accuracy_statistics_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    storage_formats_cuda_test
    PROPERTIES
        # Some batch nodes expose nvcc through PATH without adding the matching
        # CUDA lib64 directory to LD_LIBRARY_PATH.  A static CUDA runtime keeps
        # this validation executable independent of that loader configuration.
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    storage_kernels_cuda_test
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    accuracy_simulation
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    storage_performance_bench
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    decoder_optimization_bench
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    e2e3_strategy_smoke
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    e2e3_strategy_bench
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    all_format_strategy_smoke
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(
    all_format_strategy_bench
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
