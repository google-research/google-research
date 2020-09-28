##
### Find all sources.
##

set(SGK_SRCS)
file(GLOB TMP *.h)
list(APPEND SGK_SRCS ${TMP})
file(GLOB TMP *.cc)
list(APPEND SGK_SRCS ${TMP})

# Set .cu.cc files to be compiled as CUDA.
set(SGK_CUDA_SRCS ${SGK_SRCS})
list(FILTER SGK_CUDA_SRCS INCLUDE REGEX "\.cu\.cc")
set_source_files_properties(${SGK_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

# Create libsgk.
add_library(sgk SHARED ${SGK_SRCS})
target_link_libraries(sgk ${SGK_LIBS})

# Library instllation.
install(TARGETS sgk)
