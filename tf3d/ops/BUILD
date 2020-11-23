load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
load(
    "//:build_helper_def.bzl",
    "cuda_library",
    "custom_kernel_library",
    "gen_op_cclib",
)

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        "@local_config_cuda//cuda:darwin": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib",
        ],
        ":windows": [],
        "//conditions:default": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    }),
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "submanifold_sparse_conv_utils",
    srcs = ["submanifold_sparse_conv_utils.cc"],
    hdrs = ["submanifold_sparse_conv_utils.h"],
    deps = [
        # "@eigen3//:eigen",
        "//third_party/eigen3",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "submanifold_sparse_conv_launcher_cpu",
    srcs = ["submanifold_sparse_conv_launcher.cc"],
    hdrs = ["submanifold_sparse_conv_launcher.h"],
    deps = [
        ":submanifold_sparse_conv_utils",
        "//third_party/eigen3",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "submanifold_sparse_conv_grad_launcher_cpu",
    srcs = ["submanifold_sparse_conv_grad_launcher.cc"],
    hdrs = ["submanifold_sparse_conv_grad_launcher.h"],
    deps = [
        ":submanifold_sparse_conv_utils",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cuda_library(
    name = "submanifold_sparse_conv_launcher_gpu",
    srcs = ["submanifold_sparse_conv_launcher.cu.cc"],
    hdrs = ["submanifold_sparse_conv_launcher.h"],
    deps = [
        ":coordinates_hashmap_gpu",
        ":coordinates_hashmap_wrapper",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cuda_library(
    name = "submanifold_sparse_conv_grad_launcher_gpu",
    srcs = ["submanifold_sparse_conv_grad_launcher.cu.cc"],
    hdrs = ["submanifold_sparse_conv_grad_launcher.h"],
    deps = [
        ":coordinates_hashmap_gpu",
        ":coordinates_hashmap_wrapper",
        ":submanifold_sparse_conv_utils",
        "//third_party/eigen3",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cuda_library(
    name = "coordinates_hashmap_gpu",
    srcs = ["coordinates_hashmap_gpu.cu.cc"],
    hdrs = ["coordinates_hashmap_gpu.h"],
    deps = [
        ":submanifold_sparse_conv_utils",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cuda_library(
    name = "coordinates_hashmap_wrapper",
    srcs = ["coordinates_hashmap_wrapper.cu.cc"],
    hdrs = ["coordinates_hashmap_wrapper.h"],
    deps = [
        ":coordinates_hashmap_gpu",
        ":submanifold_sparse_conv_utils",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

custom_kernel_library(
    name = "submanifold_sparse_conv_ops",
    srcs = ["submanifold_sparse_conv_ops.cc"],
    op_def_lib = [":car_ops"],
    deps = [
        ":submanifold_sparse_conv_launcher_cpu",
        ":submanifold_sparse_conv_launcher_gpu",
        ":submanifold_sparse_conv_utils",
        "//third_party/eigen3",
    ],
)

custom_kernel_library(
    name = "submanifold_sparse_conv_grad",
    srcs = ["submanifold_sparse_conv_grad.cc"],
    op_def_lib = [":car_ops"],
    deps = [
        ":submanifold_sparse_conv_grad_launcher_cpu",
        ":submanifold_sparse_conv_grad_launcher_gpu",
        ":submanifold_sparse_conv_launcher_cpu",
        ":submanifold_sparse_conv_launcher_gpu",
        ":submanifold_sparse_conv_utils",
        "//third_party/eigen3",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

# Op definitions to be used from python.
gen_op_cclib(
    name = "car_ops",
    srcs = ["car_ops.cc"],
    deps = ["//third_party/eigen3"],
)

cc_binary(
    name = "python/_sparse_conv_ops.so",
    srcs = [],
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++14",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++11",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
        ],
    }) + if_cuda_is_configured([
        "-DTENSORFLOW_USE_NVCC=1",
        "-DGOOGLE_CUDA=1",
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
    ]),
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    linkopts = [
        "-L/usr/local/lib/python3.6/dist-packages/tensorflow",
        "-l:libtensorflow_framework.so.2",
        "-l:python/_pywrap_tensorflow_internal.so",
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        ":car_ops",
        ":submanifold_sparse_conv_grad",
        ":submanifold_sparse_conv_ops",
        ":submanifold_sparse_conv_utils",
    ] + if_cuda_is_configured([
        ":coordinates_hashmap_gpu",
    ]),
)

py_test(
    name = "sparse_conv_ops_py_test",
    srcs = [
        "python/sparse_conv_ops_test.py",
    ],
    main = "python/sparse_conv_ops_test.py",
    srcs_version = "PY2AND3",
    deps = [
        ":sparse_conv_ops_py",
    ],
)

py_library(
    name = "sparse_conv_ops_py",
    srcs = [
        "python/__init__.py",
        "python/sparse_conv_ops.py",
    ],
    data = [
        ":python/_sparse_conv_ops.so",
    ],
    srcs_version = "PY2AND3",
)
