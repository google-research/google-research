licenses(["notice"])

package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "neural_trees_helpers",
    srcs = ["neural_trees_helpers.cc"],
    hdrs = ["neural_trees_helpers.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "neural_trees_ops.so",
    srcs = [
        "neural_trees_ops.cc",
    ],
    deps = [
        ":neural_trees_helpers",
    ],
)
