cmake_minimum_required(VERSION 3.14)

project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonLibs REQUIRED)

add_executable(${CMAKE_PROJECT_NAME} main.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME} ${PYTHON_LIBRARIES})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})

