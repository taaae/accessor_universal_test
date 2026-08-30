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

add_executable(lns_benchmark_core_test tests/lns_benchmark_core_test.cpp)
target_include_directories(
    lns_benchmark_core_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(lns_benchmark_core_test PRIVATE -O3)
add_test(NAME lns_benchmark_core_test COMMAND lns_benchmark_core_test)

add_executable(posit_takum_core_test tests/posit_takum_core_test.cpp)
target_include_directories(
    posit_takum_core_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(posit_takum_core_test PRIVATE -O3)
add_test(NAME posit_takum_core_test COMMAND posit_takum_core_test)

add_executable(normal32_formats_test tests/normal32_formats_test.cpp)
target_include_directories(
    normal32_formats_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(normal32_formats_test PRIVATE -O3)
add_test(NAME normal32_formats_test COMMAND normal32_formats_test)


add_executable(normal32_dot_bench src/normal32_dot_bench.cu)
target_link_libraries(normal32_dot_bench PRIVATE CUDA::curand_static)
target_include_directories(
    normal32_dot_bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_options(
    normal32_dot_bench
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)
set_target_properties(
    normal32_dot_bench
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(conversion_calibration_bench
    src/conversion_calibration_bench.cu)
target_link_libraries(conversion_calibration_bench PRIVATE CUDA::cudart_static)
target_include_directories(
    conversion_calibration_bench
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/generated"
)
target_compile_options(
    conversion_calibration_bench
    PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
)
set_target_properties(
    conversion_calibration_bench
    PROPERTIES
        CUDA_RUNTIME_LIBRARY Static
        CUDA_SEPARABLE_COMPILATION OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

function(add_bitwidth_strategy_benchmark bits)
    set(target "bitwidth_strategy_bench_${bits}")
    add_executable(${target} src/bitwidth_strategy_bench.cu)
    target_link_libraries(${target} PRIVATE CUDA::curand_static)
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
    set_target_properties(
        ${target}
        PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_bitwidth_strategy_benchmark(2)
add_bitwidth_strategy_benchmark(3)
add_bitwidth_strategy_benchmark(4)
add_bitwidth_strategy_benchmark(5)
add_bitwidth_strategy_benchmark(6)
add_bitwidth_strategy_benchmark(7)
add_bitwidth_strategy_benchmark(8)
add_bitwidth_strategy_benchmark(9)
add_bitwidth_strategy_benchmark(10)
add_bitwidth_strategy_benchmark(12)
add_bitwidth_strategy_benchmark(14)
add_bitwidth_strategy_benchmark(16)
add_bitwidth_strategy_benchmark(17)
add_bitwidth_strategy_benchmark(20)
add_bitwidth_strategy_benchmark(24)
add_bitwidth_strategy_benchmark(28)
add_bitwidth_strategy_benchmark(32)

function(add_lns_strategy_benchmark name format_id enable_fp32)
    set(target "lns_strategy_bench_${name}")
    add_executable(${target} src/lns_strategy_bench.cu)
    target_link_libraries(${target} PRIVATE CUDA::curand_static)
    target_include_directories(
        ${target}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_compile_definitions(
        ${target}
        PRIVATE
            AUT_LNS_FORMAT_ID=${format_id}
            AUT_LNS_ENABLE_FP32=${enable_fp32}
    )
    target_compile_options(
        ${target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v>"
    )
    set_target_properties(
        ${target}
        PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_lns_strategy_benchmark(lns4_r1 1 1)
add_lns_strategy_benchmark(lns6_r2 2 1)
add_lns_strategy_benchmark(lns8_r2 3 1)
add_lns_strategy_benchmark(lns8_r3 4 1)
add_lns_strategy_benchmark(lns8_r4 5 1)
add_lns_strategy_benchmark(lns8_r5 6 1)
add_lns_strategy_benchmark(lns10_r4 7 1)
add_lns_strategy_benchmark(lns12_r6 8 1)
add_lns_strategy_benchmark(lns16_r4 9 1)
add_lns_strategy_benchmark(lns16_r7 10 1)
add_lns_strategy_benchmark(lns16_r10 11 1)
add_lns_strategy_benchmark(lns16_r11 12 1)
add_lns_strategy_benchmark(lns16_r12 13 1)
add_lns_strategy_benchmark(lns16_r13 14 0)
add_lns_strategy_benchmark(lns32_r20 15 0)
add_lns_strategy_benchmark(lns32_r23 16 0)
add_lns_strategy_benchmark(lns32_r28 17 0)

function(add_posit_takum_strategy_benchmark name family bits es)
    set(target "posit_takum_strategy_bench_${name}")
    add_executable(${target} src/posit_takum_strategy_bench.cu)
    target_include_directories(
        ${target}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_compile_definitions(
        ${target}
        PRIVATE
            AUT_PT_FAMILY=${family}
            AUT_PT_BITS=${bits}
            AUT_PT_ES=${es}
    )
    target_compile_options(
        ${target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v;--ftz=false>"
    )
    set_target_properties(
        ${target}
        PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_posit_takum_strategy_benchmark(posit8_es0 0 8 0)
add_posit_takum_strategy_benchmark(posit14_es1 0 14 1)
add_posit_takum_strategy_benchmark(posit16_es1 0 16 1)
add_posit_takum_strategy_benchmark(posit32_es2 0 32 2)
add_posit_takum_strategy_benchmark(takum8 1 8 0)
add_posit_takum_strategy_benchmark(takum14 1 14 0)
add_posit_takum_strategy_benchmark(takum16 1 16 0)
add_posit_takum_strategy_benchmark(takum32 1 32 0)
add_posit_takum_strategy_benchmark(takum_log8 2 8 0)
add_posit_takum_strategy_benchmark(takum_log14 2 14 0)
add_posit_takum_strategy_benchmark(takum_log16 2 16 0)
add_posit_takum_strategy_benchmark(takum_log32 2 32 0)

function(add_ieee_scalar_strategy_benchmark name format_id compute strategies q_lower q_upper)
    set(target "ieee_scalar_strategy_bench_${name}")
    add_executable(${target} src/ieee_scalar_strategy_bench.cu)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_definitions(
        ${target}
        PRIVATE
            AUT_IEEE_FORMAT_ID=${format_id}
            AUT_IEEE_COMPUTE=${compute}
            AUT_IEEE_STRATEGY_MASK=${strategies}
            AUT_IEEE_Q_LOWER=${q_lower}
            AUT_IEEE_Q_UPPER=${q_upper}
    )
    target_compile_options(
        ${target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v;--ftz=false>"
    )
    set_target_properties(
        ${target}
        PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_ieee_scalar_strategy_benchmark(fp8_e4m3_fp32 1 0 25 -8 8)
add_ieee_scalar_strategy_benchmark(fp8_e4m3_fp64 1 1 25 -8 8)
add_ieee_scalar_strategy_benchmark(fp8_e5m2_fp32 2 0 25 -15 15)
add_ieee_scalar_strategy_benchmark(fp8_e5m2_fp64 2 1 25 -15 15)
add_ieee_scalar_strategy_benchmark(e3m4_fp32 3 0 24 -5 3)
add_ieee_scalar_strategy_benchmark(e3m4_fp64 3 1 24 -5 3)
add_ieee_scalar_strategy_benchmark(e6m1_fp32 4 0 24 -28 28)
add_ieee_scalar_strategy_benchmark(e6m1_fp64 4 1 24 -28 28)
add_ieee_scalar_strategy_benchmark(e8m5_fp32 5 0 26 -120 120)
add_ieee_scalar_strategy_benchmark(e11m2_fp64 6 1 28 -1000 1000)
add_ieee_scalar_strategy_benchmark(e2m11_fp32 7 0 26 -10 1)
add_ieee_scalar_strategy_benchmark(e2m11_fp64 7 1 26 -10 1)
add_ieee_scalar_strategy_benchmark(e5m8_fp32 8 0 26 -20 14)
add_ieee_scalar_strategy_benchmark(e5m8_fp64 8 1 26 -20 14)
add_ieee_scalar_strategy_benchmark(fp16_e5m10_fp32 9 0 17 -20 14)
add_ieee_scalar_strategy_benchmark(fp16_e5m10_fp64 9 1 17 -20 14)
add_ieee_scalar_strategy_benchmark(bf16_e8m7_fp32 10 0 17 -120 120)
add_ieee_scalar_strategy_benchmark(bf16_e8m7_fp64 10 1 17 -120 120)
add_ieee_scalar_strategy_benchmark(e11m4_fp64 11 1 20 -1000 1000)
add_ieee_scalar_strategy_benchmark(e3m12_fp32 12 0 48 -12 3)
add_ieee_scalar_strategy_benchmark(e3m12_fp64 12 1 48 -12 3)
add_ieee_scalar_strategy_benchmark(e6m9_fp32 13 0 82 -36 28)
add_ieee_scalar_strategy_benchmark(e6m9_fp64 13 1 82 -36 28)
add_ieee_scalar_strategy_benchmark(fp32_e8m23_fp32 14 0 1 -120 120)
add_ieee_scalar_strategy_benchmark(fp32_e8m23_fp64 14 1 1 -120 120)
add_ieee_scalar_strategy_benchmark(e11m20_fp64 15 1 4 -1000 1000)
add_ieee_scalar_strategy_benchmark(e4m27_fp64 16 1 66 -30 7)
add_ieee_scalar_strategy_benchmark(e10m21_fp64 17 1 66 -500 500)

function(add_lut_content_control_benchmark name bits compute)
    set(target "lut_content_control_bench_${name}")
    add_executable(${target} src/lut_content_control_bench.cu)
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_definitions(
        ${target}
        PRIVATE
            AUT_LUT_CONTROL_BITS=${bits}
            AUT_LUT_CONTROL_COMPUTE=${compute}
    )
    target_compile_options(
        ${target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-O3>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v;--ftz=false>"
    )
    set_target_properties(
        ${target}
        PROPERTIES
            CUDA_RUNTIME_LIBRARY Static
            CUDA_SEPARABLE_COMPILATION OFF
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_lut_content_control_benchmark(8_fp32 8 0)
add_lut_content_control_benchmark(8_fp64 8 1)
add_lut_content_control_benchmark(14_fp32 14 0)
add_lut_content_control_benchmark(14_fp64 14 1)

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
