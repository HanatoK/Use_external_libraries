cmake_minimum_required(VERSION 3.14)

project(main LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python3 REQUIRED)

if(Boost_FOUND)
  message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
  message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
endif()

set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(hello_ext MODULE hello.cpp)

target_link_libraries(hello_ext ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(hello_ext PRIVATE ${PYTHON_INCLUDE_DIRS})
