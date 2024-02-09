cmake_minimum_required(VERSION 3.10)

project(elo_rater_model)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(BUILD_TESTING OFF)
add_subdirectory(abseil-cpp)
add_subdirectory(eigen)
add_subdirectory(highway)
set(BUILD_TESTING ON)

enable_testing()
include(CTest)
cmake_policy(SET CMP0057 NEW)  # https://gitlab.kitware.com/cmake/cmake/issues/18198
include(GoogleTest)

add_library(
  elo
  elo.cc
  elo.h
  elo_fast_math.cc
  elo_fast_math.h
)

target_link_libraries(
  elo
  eigen
  absl::flat_hash_map
  absl::log
  absl::log_internal_check_impl
  absl::time
  absl::random_random
  hwy
)

target_include_directories(elo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(elo_test elo_test.cc)
target_link_libraries(elo_test gmock gtest gtest_main elo)
gtest_discover_tests(elo_test)

add_executable(elo_main elo_main.cc)
target_link_libraries(elo_main elo absl::flags absl::flags_parse absl::log_initialize)
