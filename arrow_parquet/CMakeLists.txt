project(MyExample)
cmake_minimum_required(VERSION 3.24)

find_package(Arrow REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE Arrow::arrow_shared)

# add_executable(example example.cpp)
# target_link_libraries(example PRIVATE Arrow::arrow_shared)
