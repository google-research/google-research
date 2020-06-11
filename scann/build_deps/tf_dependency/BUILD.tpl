package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "libtensorflow_framework",
    srcs = ["%{TF_SHARED_LIBRARY_NAME}"],
    visibility = ["//visibility:public"],
)

%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}