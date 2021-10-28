cmake_minimum_required(VERSION 3.14)

project(main LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(PROJECT_SOURCE_DIR src)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/*.h)
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/*.cpp)

find_package(Armadillo CONFIG REQUIRED)

add_executable(main ${HEADERS} ${SOURCES})
target_link_libraries(main armadillo)
