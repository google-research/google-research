# Status and related utilities for error handling, move to absl once that's
# open sourced.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "status",
    srcs = [
        "canonical_errors.cc",
        "status.cc",
        "statusor.cc",
    ],
    hdrs = [
        "canonical_errors.h",
        "status.h",
        "status_macros.h",
        "statusor.h",
    ],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/utility",
    ],
)

cc_library(
    name = "time_proto_util",
    srcs = ["time_proto_util.cc"],
    hdrs = ["time_proto_util.h"],
    deps = [
        ":status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@protobuf_archive//:protobuf",
    ],
)
