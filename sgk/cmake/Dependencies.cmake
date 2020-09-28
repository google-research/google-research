include(cmake/Cuda.cmake)

# CUDA.
cuda_find_library(CUDART_LIBRARY cudart)
cuda_find_library(CUSPARSE_LIBRARY cusparse)
list(APPEND SGK_LIBS "cudart;cusparse;culibos")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# TensorFlow.
find_package(TensorFlow REQUIRED)
list(APPEND SGK_LIBS ${TENSORFLOW_LIBRARIES})
include_directories(${TENSORFLOW_INCLUDE_DIRS})

# Sputnik.
find_package(Sputnik REQUIRED)
list(APPEND SGK_LIBS ${SPUTNIK_LIBRARIES})
include_directories(${SPUTNIK_INCLUDE_DIRS})
