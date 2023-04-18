#!/usr/bin/env sh
TENSORFLOW_INSTALL_DIR="/home/hanatok/mambaforge"
TENSORFLOW_LIB_DIR="$TENSORFLOW_INSTALL_DIR/lib"
TENSORFLOW_INCLUDE_DIR="$TENSORFLOW_INSTALL_DIR/include"
COMPILER="g++"
EXTRA_CCFLAGS=""

$COMPILER -I$TENSORFLOW_INCLUDE_DIR main.cpp -std=c++17 -lboost_iostreams -lre2 -L$TENSORFLOW_LIB_DIR -ltensorflow_cc -ltensorflow_framework -lprotobuf -Wl,-rpath,"$TENSORFLOW_LIB_DIR" -o main -O2 
