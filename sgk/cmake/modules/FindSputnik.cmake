include(FindPackageHandleStandardArgs)

set(SPUTNIK_ROOT_DIR "/usr/local/sputnik" CACHE PATH "Sputnik root directory")

find_path(SPUTNIK_INCLUDE_DIR sputnik/sputnik.h PATHS "${SPUTNIK_ROOT_DIR}/include")

find_library(SPUTNIK_LIBRARY sputnik PATHS ${SPUTNIK_ROOT_DIR} PATH_SUFFIXES lib)

find_package_handle_standard_args(Sputnik DEFAULT_MSG SPUTNIK_INCLUDE_DIR SPUTNIK_LIBRARY)

if(SPUTNIK_FOUND)
  set(SPUTNIK_INCLUDE_DIRS ${SPUTNIK_INCLUDE_DIR})
  set(SPUTNIK_LIBRARIES ${SPUTNIK_LIBRARY})
  message(STATUS "Found sputnik (include: ${SPUTNIK_INCLUDE_DIR}, library: ${SPUTNIK_LIBRARY})")
  mark_as_advanced(SPUTNIK_ROOT_DIR
    SPUTNIK_LIBRARY_RELEASE
    SPUTNIK_LIBRARY_DEBUG
    SPUTNIK_LIBRARY
    SPUTNIK_INCLUDE_DIR)
endif()
