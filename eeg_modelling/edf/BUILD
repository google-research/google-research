# Status and related utilities for error handling, move to absl once that's
# open sourced.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "edf_file",
    srcs = [
        "edf_file.cc",
    ],
    hdrs = [
        "edf_file.h",
    ],
    deps = [
        "//edf/base:status",
    ],
)

cc_library(
    name = "edf_oss_file",
    srcs = [
        "edf_oss_file.cc",
    ],
    hdrs = [
        "edf_oss_file.h",
    ],
    deps = [
        ":edf_file",
        "//edf/base:status",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "edf",
    srcs = [
        "edf.cc",
    ],
    hdrs = [
        "edf.h",
    ],
    deps = [
        ":edf_file",
        ":edf_oss_file",
        "//edf/base:status",
        "//edf/proto:edf_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "edf_util",
    srcs = [
        "edf_util.cc",
    ],
    hdrs = [
        "edf_util.h",
    ],
    deps = [
        "//edf/base:status",
        "//edf/proto:edf_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "parse_edf_lib",
    srcs = [
        "parse_edf_lib.cc",
    ],
    hdrs = [
        "parse_edf_lib.h",
    ],
    deps = [
        ":edf",
        ":edf_file",
        ":edf_util",
        "//edf/base:status",
        "//edf/base:time_proto_util",
        "//edf/proto:annotation_cc_proto",
        "//edf/proto:edf_cc_proto",
        "//edf/proto:event_cc_proto",
        "//edf/proto:segment_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "edf_reader",
    srcs = [
        "edf_reader.cc",
    ],
    hdrs = [
        "edf_reader.h",
    ],
    deps = [
        ":edf",
        ":edf_file",
        ":edf_util",
        ":parse_edf_lib",
        "//edf/base:status",
        "//edf/base:time_proto_util",
        "//edf/proto:edf_cc_proto",
        "//edf/proto:event_cc_proto",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@protobuf_archive//:protobuf",
    ],
)

cc_library(
    name = "tf_example_lib",
    srcs = [
        "tf_example_lib.cc",
    ],
    hdrs = [
        "tf_example_lib.h",
    ],
    deps = [
        "//edf/base:status",
        "//edf/base:time_proto_util",
        "//edf/proto:annotation_cc_proto",
        "//edf/proto:edf_cc_proto",
        "//edf/proto:event_cc_proto",
        "//edf/proto:segment_cc_proto",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@protobuf_archive//:protobuf",
    ],
)

cc_binary(
    name = "temple_edf_to_tf_example",
    srcs = [
        "temple_edf_to_tf_example.cc",
    ],
    deps = [
        ":edf_oss_file",
        ":parse_edf_lib",
        ":temple_util",
        ":tf_example_lib",
        "//edf/proto:segment_cc_proto",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_lullaby//lullaby/util:arg_parser",
        "@org_lullaby//lullaby/util:string_view",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "temple_util",
    srcs = [
        "temple_util.cc",
    ],
    hdrs = [
        "temple_util.h",
    ],
    deps = [
        "//edf/base:status",
        "//edf/base:time_proto_util",
        "//edf/proto:annotation_cc_proto",
        "//edf/proto:event_cc_proto",
        "//edf/proto:segment_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)
