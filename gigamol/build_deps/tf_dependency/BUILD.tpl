package(default_visibility = ["//visibility:public"])

DYNAMIC_COPTS = [
    "-pthread",
    "-std=c++14",
    "%{_TF_CXX11_ABI_FLAG}",
]

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    copts = DYNAMIC_COPTS, 
    includes = ["include"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "libtensorflow_framework",
    srcs = ["%{TF_SHARED_LIBRARY_NAME}"],
    copts = DYNAMIC_COPTS,
    visibility = ["//visibility:public"],
)

%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}
