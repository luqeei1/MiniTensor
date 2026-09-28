cmake_minimum_required(VERSION 3.20)

project(MiniTensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(minitensor
    src/Tensor.cpp
)

target_include_directories(minitensor
    PUBLIC include
)

add_executable(example
    examples/basic.cpp
)

target_link_libraries(example
    PRIVATE minitensor
)