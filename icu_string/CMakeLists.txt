cmake_minimum_required(VERSION 3.14)

project(main LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(PROJECT_SOURCE_DIR src)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES ${PROJECT_SOURCE_DIR}/main.cpp)

find_package(ICU COMPONENTS uc i18n REQUIRED)

# message(ICU_INCLUDE_DIRS="${ICU_INCLUDE_DIRS}")

include_directories(${ICU_INCLUDE_DIRS})

add_executable(main ${SOURCES})
target_link_libraries(main ICU::uc ICU::i18n)
