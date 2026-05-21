cmake_minimum_required(VERSION 3.24)

project(kokopop_python LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

set(KOKOPOP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KOKOPOP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(KOKOPOP_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(KOKOPOP_CORE_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
if(NOT EXISTS "${KOKOPOP_CORE_SOURCE_DIR}/include/kokopop.h")
  set(KOKOPOP_CORE_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/kokopop_core)
endif()

add_subdirectory(${KOKOPOP_CORE_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/kokopop_core)

Python3_add_library(_native MODULE WITH_SOABI src/kokopop_native.cpp)
target_link_libraries(_native PRIVATE kokopop Python3::Module)
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)

if(MSVC)
  target_compile_options(_native PRIVATE /W4)
else()
  target_compile_options(_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _native LIBRARY DESTINATION kokopop COMPONENT python)
