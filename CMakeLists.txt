cmake_minimum_required(VERSION 3.10)
project(ParallelIntersectionBench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallel execution.")
else()
    message(FATAL_ERROR "OpenMP not found. This project requires OpenMP.")
endif()

include_directories(include)

add_executable(intersection_benchmark
    src/main.cpp
    src/intersection.cpp
    src/enhanced_intersection.cpp
    src/generate_data.cpp
)

target_link_libraries(intersection_benchmark PRIVATE OpenMP::OpenMP_CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel" FORCE)
  message(STATUS "Build type not specified, using Release")
endif()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)
