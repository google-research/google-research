# Helper to find CUDA libraries.
function(cuda_find_library out_path lib_name)
  find_library(${out_path} ${lib_name} PATHS
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    PATH_SUFFIXES lib lib64 REQUIRED)
endfunction()

# Helper to create CUDA gencode flags.
function(create_cuda_gencode_flags out archs_args)
  set(archs ${archs_args} ${ARGN})
  set(tmp "")
  foreach(arch IN LISTS archs)
    set(tmp "${tmp} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach(arch)
  set(${out} ${tmp} PARENT_SCOPE)
endfunction()
