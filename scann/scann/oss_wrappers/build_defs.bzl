""".bzl file for ScaNN open source build configs."""

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

def scann_py_proto_library(
        name,
        srcs,
        py_proto_deps = [],
        proto_deps = None,
        **kwargs):
    """Generates py_proto_library for ScaNN open source version.

    Args:
      name: the name of the py_proto_library.
      srcs: the .proto files of the py_proto_library for Bazel use.
      py_proto_deps: a list of dependency labels for Bazel use; must be
        py_proto_library.
      proto_deps: a list of dependency labels for internal use.
      **kwargs: other keyword arguments that are passed to py_proto_library.
    """
    _ignore = [proto_deps]  # buildifier: disable=unused-variable
    py_proto_library(
        name = name,
        srcs = srcs,
        deps = py_proto_deps + ["@com_google_protobuf//:protobuf_python"],
        default_runtime = "@com_google_protobuf//:protobuf_python",
        **kwargs
    )
