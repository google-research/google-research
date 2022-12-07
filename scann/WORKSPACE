load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//build_deps/py:python_configure.bzl", "python_configure")
load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")

# Needed for highway's config_setting_group
http_archive(
    name = "bazel_skylib",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz"],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "platforms",
    sha256 = "5308fc1d8865406a49427ba24a9ab53087f17f5266a7aabbfc28823f3916e1ca",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz",
    ],
)

tf_configure(
    name = "local_config_tf",
)

python_configure(name = "local_config_python")

git_repository(
    name = "pybind11_bazel",
    commit = "9a24c33cbdc510fa60ab7f5ffb7d80ab89272799",
    remote = "https://github.com/pybind/pybind11_bazel.git",
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-2.10.1",
    urls = [
        "https://github.com/pybind/pybind11/archive/v2.10.1.tar.gz",
    ],
)

http_archive(
    name = "com_google_absl",
    sha256 = "94aef187f688665dc299d09286bfa0d22c4ecb86a80b156dff6aabadc5a5c26d",
    strip_prefix = "abseil-cpp-273292d1cfc0a94a65082ee350509af1d113344d",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/273292d1cfc0a94a65082ee350509af1d113344d.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/273292d1cfc0a94a65082ee350509af1d113344d.tar.gz",
    ],
)

# rules_proto defines abstract rules for building Protocol Buffers.
http_archive(
    name = "rules_proto",
    sha256 = "ba66b430aaed1b6d2530154fd5ff968fb810ee0936edfe5f48e1f1c36ff2ea65",
    strip_prefix = "rules_proto-e507ccded37c389186afaeb2b836ec576dc875dc",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/e507ccded37c389186afaeb2b836ec576dc875dc.zip",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)

http_archive(
    name = "six_archive",
    build_file = "//build_deps/patches:six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    urls = ["https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55"],
)

git_repository(
    name = "com_google_protobuf",
    patches = ["//build_deps/patches:protobuf.patch"],
    remote = "https://github.com/protocolbuffers/protobuf.git",
    tag = "v3.9.2",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

git_repository(
    name = "com_google_googletest",
    remote = "https://github.com/google/googletest.git",
    tag = "release-1.10.0",
)

git_repository(
    name = "com_google_highway",
    remote = "https://github.com/google/highway.git",
    tag = "1.0.1",
)

git_repository(
    name = "cnpy",
    commit = "57184ee0db37cac383fc29175950747a46a8b512",
    remote = "https://github.com/sammymax/cnpy.git",
)

# rules_cc defines rules for generating C++ code from Protocol Buffers.
http_archive(
    name = "rules_cc",
    sha256 = "56ac9633c13d74cb71e0546f103ce1c58810e4a76aa8325da593ca4277908d72",
    strip_prefix = "rules_cc-40548a2974f1aea06215272d9c2b47a14a24e556",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/40548a2974f1aea06215272d9c2b47a14a24e556.zip",
    ],
)

load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")

rules_cc_dependencies()
