cmake_minimum_required(VERSION 3.9)
project(plugin1)

set(CMAKE_CXX_STANDARD 17)

include_directories(../../include include/)

set(plugin1_src src/plugin1.cpp include/plugin1.h ../../include/main.h)

add_library(plugin1 SHARED ${plugin1_src})
