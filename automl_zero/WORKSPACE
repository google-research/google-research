load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.zip"],
)

bind(
    name   = "gflags",
    actual = "@com_github_gflags_gflags//:gflags",
)

http_archive(
  name = "com_google_absl",
  sha256 = "6622893ab117501fc23268a2936e0d46ee6cb0319dcf2275e33a708cd9634ea6",
  urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20200923.3.zip"],
  strip_prefix = "abseil-cpp-20200923.3",
)

http_archive(
    name = "com_google_googletest",
    sha256 = "24564e3b712d3eb30ac9a85d92f7d720f60cc0173730ac166f27dda7fed76cb2",
    strip_prefix = "googletest-release-1.12.1",
    urls = ["https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip"],
)

http_archive(
    name = "rules_cc",
    sha256 = "d9f4686206d20d7c5513a39933aa1148d21d6ce16134ae4c4567c40bbac359bd",
    strip_prefix = "rules_cc-0.0.1",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.0.1.zip"],
)

http_archive(
    name = "rules_proto",
    strip_prefix = "rules_proto-4.0.0-3.20.0",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0-3.20.0.zip",
    ],
)

http_archive(
        name = "bazel_skylib",
        sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        ],
)

http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = [
        "https://mirror.bazel.build/zlib.net/zlib-1.2.11.tar.gz",
        "https://zlib.net/zlib-1.2.11.tar.gz",
    ],
)

http_archive(
    name = "com_google_protobuf",
    sha256 = "1e622ce4b84b88b6d2cdf1db38d1a634fe2392d74f0b7b74ff98f3a51838ee53",
    strip_prefix = "protobuf-3.8.0",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v3.8.0.zip",  # 2019-05-24
    ],
)

http_archive(
    name = "eigen_archive",
    build_file = "//:eigen.BUILD",
    sha256 = "9a01fed6311df359f3f9af119fcf298a3353aef7d1b1bc86f6c8ae0ca6a2f842",
    strip_prefix = "/eigen-eigen-5d5dd50b2eb6",
    urls = [
        "https://mirror.bazel.build/bitbucket.org/eigen/eigen/get/5d5dd50b2eb6.zip",
        "https://bitbucket.org/eigen/eigen/get/5d5dd50b2eb6.zip",
    ],
)

http_archive(
    name = "com_google_glog",
    sha256 = "122fb6b712808ef43fbf80f75c52a21c9760683dae470154f02bddfc61135022",
    strip_prefix = "glog-0.6.0",
    urls = ["https://github.com/google/glog/archive/refs/tags/v0.6.0.zip"],
)
