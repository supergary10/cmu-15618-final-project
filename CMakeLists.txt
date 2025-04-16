cmake_minimum_required(VERSION 3.10)
project(LeapfrogJoinBenchmark)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

find_package(OpenMP REQUIRED)

add_executable(leapfrog_benchmark
    src/main.cpp
    src/generate_data.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(leapfrog_benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()