cmake_minimum_required(VERSION 3.10)

# Set compilers to default to Clang is not set.
if ("$ENV{CC}" STREQUAL "")
  set(ENV{CC} clang)
endif()
if ("$ENV{CXX}" STREQUAL "")
  set(ENV{CXX} clang++)
endif()


project(cube_unfoldings)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-fcolor-diagnostics -Wall)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(Nauty REQUIRED IMPORTED_TARGET nauty)

set(BUILD_TESTING OFF)
add_subdirectory(abseil-cpp)
set(BUILD_TESTING ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if( supported )
    message(STATUS "IPO / LTO enabled")
    set_property(GLOBAL PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()

enable_testing()
include(CTest)
cmake_policy(SET CMP0057 NEW)  # https://gitlab.kitware.com/cmake/cmake/issues/18198
include(GoogleTest)

add_library(
  common
  unfoldings.cc
  perfect_matchings.cc
  nauty_wrapper.cc
  unfoldings.h
  perfect_matchings.h
  nauty_wrapper.h
)

target_link_libraries(common absl::flags absl::flags_parse absl::flat_hash_set absl::synchronization PkgConfig::Nauty)

add_executable(unfoldings_test unfoldings_test.cc)
target_link_libraries(unfoldings_test gmock gtest gtest_main common)
gtest_discover_tests(unfoldings_test)

add_executable(perfect_matchings_test perfect_matchings_test.cc)
target_link_libraries(perfect_matchings_test gmock gtest gtest_main common)
gtest_discover_tests(perfect_matchings_test)

add_executable(nauty_wrapper_test nauty_wrapper_test.cc)
target_link_libraries(nauty_wrapper_test gmock gtest gtest_main common)
gtest_discover_tests(nauty_wrapper_test)

add_executable(unfoldings unfoldings_main.cc)
target_link_libraries(unfoldings common)

