cmake_minimum_required(VERSION 3.20)

project(MiniTensor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(minitensor
    src/Tensor.cpp
    src/CudaBuffer.cu
)

target_include_directories(minitensor
    PUBLIC include
)

# Temporary CUDA / CudaBuffer test
add_executable(cuda_test
    src/cuda_test.cu
)

target_link_libraries(
    cuda_test
    PRIVATE
    minitensor
)

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
    minitensor_tests
    tests/test_tensor.cpp
)

target_link_libraries(
    minitensor_tests
    PRIVATE
    minitensor
    GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(minitensor_tests)