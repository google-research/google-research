licenses(["notice"])

package(
    default_visibility = [
        "//visibility:public",
    ],
)

proto_library(
    name = "edf_proto",
    srcs = ["edf.proto"],
)

cc_proto_library(
    name = "edf_cc_proto",
    deps = [":edf_proto"],
)

proto_library(
    name = "event_proto",
    srcs = ["event.proto"],
    deps = ["@protobuf_archive//:timestamp_proto"],
)

cc_proto_library(
    name = "event_cc_proto",
    deps = [":event_proto"],
)

proto_library(
    name = "annotation_proto",
    srcs = ["annotation.proto"],
    deps = [":event_proto"],
)

cc_proto_library(
    name = "annotation_cc_proto",
    deps = [":annotation_proto"],
)

proto_library(
    name = "segment_proto",
    srcs = ["segment.proto"],
    deps = ["@protobuf_archive//:timestamp_proto"],
)

cc_proto_library(
    name = "segment_cc_proto",
    deps = [":segment_proto"],
)
