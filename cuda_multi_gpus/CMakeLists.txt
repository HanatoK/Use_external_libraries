cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
include(CheckIncludeFile)
project(mgpu LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(${CMAKE_PROJECT_NAME} mgpu.cpp mgpu_kernel.cuh mgpu_kernel.cu)
target_link_libraries(${CMAKE_PROJECT_NAME} CUDA::cudart CUDA::nvtx3)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY LANGUAGE CUDA)
