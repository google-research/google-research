cmake_minimum_required(VERSION 3.13)

project(SGK CUDA CXX C)

cmake_policy(SET CMP0104 OLD)

##
### Handle Dependencies.
##

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)
include(cmake/TensorFlow.cmake)

##
### C++ and CUDA flags.
##

# Requires C++11.
set(CMAKE_CXX_STANDARD 11)

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")

get_tensorflow_flags(TENSORFLOW_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TENSORFLOW_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TENSORFLOW_FLAGS}")

##
### Project build.
##

include_directories("${PROJECT_SOURCE_DIR}")
add_subdirectory(sparse/ops/cc)
