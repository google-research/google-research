# TODO(rtatiefo): describe this package.

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "convert_annotated_recording_collection_to_sequence_example",
    srcs = ["convert_annotated_recording_collection_to_sequence_example.cc"],
    hdrs = [
        "convert_annotated_recording_collection_to_sequence_example.h",
    ],
    deps = [
        "//intent_recognition:annotated_recording_collection_cc_proto",
        "//intent_recognition:annotated_recording_collection_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core/example:example_protos_cc",
    ],
)

cc_binary(
    name = "convert_annotated_recording_collection_to_sequence_example_main",
    srcs = ["convert_annotated_recording_collection_to_sequence_example_main.cc"],
    deps = [
        ":convert_annotated_recording_collection_to_sequence_example",
        "//intent_recognition:annotated_recording_collection_utils",
        "@com_github_glog_glog//:glog",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core/example:example_protos_cc",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/records:record_reader",
    ],
)
