cmake_minimum_required(VERSION 3.9)
project(dlopen-test)

set(CMAKE_CXX_STANDARD 17)

set(main_src src/main.cpp)
include_directories(include/)
set(headers include/main.h
            include/dlloader.h
            include/dlloader_posix.h
            include/dlloader_win32.h)

add_executable(dlopen-test ${headers} ${main_src})

target_link_libraries(dlopen-test ${CMAKE_DL_LIBS})

add_subdirectory(plugins/plugin1)
add_subdirectory(plugins/plugin2)
