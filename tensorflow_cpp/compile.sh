#!/usr/bin/env sh
TENSORFLOW_INSTALL_DIR="/home/hanatok/mambaforge"
TENSORFLOW_PYTHON_INSTALL_DIR="/home/hanatok/mambaforge/lib/python3.10/site-packages/tensorflow"
TENSORFLOW_LIB_DIR="$TENSORFLOW_INSTALL_DIR/lib"
TENSORFLOW_INCLUDE_DIR_1="$TENSORFLOW_INSTALL_DIR/include"
TENSORFLOW_INCLUDE_DIR_2="$TENSORFLOW_PYTHON_INSTALL_DIR/include"
COMPILER="g++"
EXTRA_CCFLAGS=""

$COMPILER -I$TENSORFLOW_INCLUDE_DIR_1 -I$TENSORFLOW_INCLUDE_DIR_2 main.cpp -std=c++17 -lboost_iostreams -lre2 -L$TENSORFLOW_LIB_DIR -ltensorflow_cc -ltensorflow_framework -lprotobuf -Wl,-rpath,"$TENSORFLOW_LIB_DIR" -o main -O2
