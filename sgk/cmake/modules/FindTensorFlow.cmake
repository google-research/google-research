include(FindPackageHandleStandardArgs)

set(TENSORFLOW_ROOT_DIR "" CACHE PATH "TensorFlow root directory")

execute_process(COMMAND
  python3 -c "import tensorflow as tf;\
    print(tf.sysconfig.get_include())"
  OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIR)

# TODO(tgale): Use python to get the library name instead of hardcoding.
execute_process(COMMAND
  python3 -c "import tensorflow as tf;\
    print(tf.sysconfig.get_lib() + '/libtensorflow_framework.so.2')"
  OUTPUT_VARIABLE TENSORFLOW_LIBRARY)

# Remove trailing whitespace.
string(REGEX REPLACE "\n$" "" TENSORFLOW_INCLUDE_DIR "${TENSORFLOW_INCLUDE_DIR}")
string(REGEX REPLACE "\n$" "" TENSORFLOW_LIBRARY "${TENSORFLOW_LIBRARY}")

find_package_handle_standard_args(
  TensorFlow DEFAULT_MSG TENSORFLOW_INCLUDE_DIR TENSORFLOW_LIBRARY)

if(TENSORFLOW_FOUND)
  set(TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIR})
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIBRARY})
  message(STATUS "Found TensorFlow (include: ${TENSORFLOW_INCLUDE_DIR},\
    library: ${TENSORFLOW_LIBRARY})")
  mark_as_advanced(TENSORFLOW_ROOT_DIR
    TENSORFLOW_LIBRARY_RELEASE
    TENSORFLOW_LIBRARY_DEBUG
    TENSORFLOW_LIBRARY
    TENSORFLOW_INCLUDE_DIR)
endif()
