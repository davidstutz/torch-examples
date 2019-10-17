cmake_minimum_required(VERSION 3.2)
project(gpu)

set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS} -O3 -g")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDA REQUIRED)
# http://stackoverflow.com/questions/29121211/cuda-compilation-issue-with-cmake
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

message("CUDA: ${CUDA_LIBRARIES}")
cuda_add_library(gpu SHARED test_identity_module.cu)
target_link_libraries(gpu ${CUDA_LIBRARIES})