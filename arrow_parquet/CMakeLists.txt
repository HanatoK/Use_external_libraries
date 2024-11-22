cmake_minimum_required(VERSION 3.24)
project(MyExample)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(fmt REQUIRED)

add_executable(test_reader test_reader.cpp)
target_link_libraries(test_reader PRIVATE Arrow::arrow_shared Parquet::parquet_shared fmt::fmt)

add_executable(test_writer test_writer.cpp)
target_link_libraries(test_writer PRIVATE Arrow::arrow_shared Parquet::parquet_shared fmt::fmt)
