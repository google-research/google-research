function(get_tensorflow_flags out)
  execute_process(COMMAND
    python3 -c "import tensorflow as tf;\
      print(tf.__cxx11_abi_flag__ if '__cxx11_abi_flag__'\
      in tf.__dict__ else 0)"
    OUTPUT_VARIABLE ABI_FLAG)

# Remove trailing whitespace.
string(REGEX REPLACE "\n$" "" ABI_FLAG "${ABI_FLAG}")
  set(${out} "-D_GLIBCXX_USE_CXX11_ABI=${ABI_FLAG} -DGOOGLE_CUDA" PARENT_SCOPE)
endfunction()
