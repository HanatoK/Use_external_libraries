cmake_minimum_required(VERSION 3.9)
project(plugin2)

set(CMAKE_CXX_STANDARD 17)

include_directories(../../include include/)

set(plugin2_src src/plugin2.cpp include/plugin2.h ../../include/main.h)

add_library(plugin2 SHARED ${plugin2_src})
