load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")  # @unused

def cuda_library(copts = [], deps = [], **kwargs):
    """Wrapper over cc_library which adds default CUDA options."""
    cuda_default_copts = select({
        ":windows": ["/D__CLANG_SUPPORT_DYN_ANNOTATION__", "/DEIGEN_MPL2_ONLY", "/DEIGEN_MAX_ALIGN_BYTES=64", "/DEIGEN_HAS_TYPE_TRAITS=0", "/DTF_USE_SNAPPY", "/showIncludes", "/MD", "/O2", "/DNDEBUG", "/w", "-DWIN32_LEAN_AND_MEAN", "-DNOGDI", "/d2ReducedOptimizeHugeFunctions", "/arch:AVX", "/std:c++14", "-DTENSORFLOW_MONOLITHIC_BUILD", "/DPLATFORM_WINDOWS", "/DEIGEN_HAS_C99_MATH", "/DTENSORFLOW_USE_EIGEN_THREADPOOL", "/DEIGEN_AVOID_STL_ARRAY", "/Iexternal/gemmlowp", "/wd4018", "/wd4577", "/DNOGDI", "/UTF_COMPILE_LIBRARY"],
        "//conditions:default": ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"],
    }) + if_cuda_is_configured(["-DTENSORFLOW_USE_NVCC=1", "-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"])

    default_deps = if_cuda_is_configured([":cuda", "@local_config_cuda//cuda:cuda_headers"])

    native.cc_library(copts = cuda_default_copts + copts, deps = default_deps + deps, **kwargs)

def tf_copts():
    return ["-D_GLIBCXX_USE_CXX11_ABI=0", "-Wno-sign-compare", "-mavx"] + if_cuda_is_configured(["-DTENSORFLOW_USE_NVCC=1", "-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"])

def custom_kernel_library(name, op_def_lib, srcs, hdrs = [], deps = []):
    native.cc_library(
        name = name,
        srcs = srcs,
        hdrs = hdrs,
        copts = tf_copts(),
        deps = deps + op_def_lib + if_cuda_is_configured([":cuda", "@local_config_cuda//cuda:cuda_headers"]),
        alwayslink = 1,
    )

def gen_op_cclib(name, srcs, deps = []):
    native.cc_library(
        name = name,
        srcs = srcs,
        deps = [
            "@local_config_tf//:libtensorflow_framework",
            "@local_config_tf//:tf_header_lib",
        ] + deps,
        alwayslink = 1,
        copts = tf_copts(),
    )

def gen_op_pylib(name, cc_lib_name, srcs, kernel_deps, py_deps = [], **kwargs):
    native.cc_binary(
        name = cc_lib_name + ".so",
        deps = [cc_lib_name] + kernel_deps,
        linkshared = 1,
        copts = tf_copts(),
        **kwargs
    )

    native.py_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY3",
        data = [cc_lib_name + ".so"],
        deps = py_deps,
        **kwargs
    )
