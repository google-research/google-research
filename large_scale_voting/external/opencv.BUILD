cc_library(
    name = "core",
    srcs = ["lib/x86_64-linux-gnu/libopencv_core.so"],
    hdrs = glob([
        "include/opencv4/opencv2/core/**",
        "include/opencv4/opencv2/*.hpp",
    ]),
    includes = ["include/opencv4"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "highgui",
    srcs = ["lib/x86_64-linux-gnu/libopencv_highgui.so"],
    hdrs = glob([
        "include/opencv4/opencv2/highgui/**",
        "include/opencv4/opencv2/*.hpp",
    ]),
    includes = ["include/opencv4"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "imgcodecs",
    srcs = ["lib/x86_64-linux-gnu/libopencv_imgcodecs.so"],
    hdrs = glob([
        "include/opencv4/opencv2/imgcodecs/**",
        "include/opencv4/opencv2/*.hpp",
    ]),
    includes = ["include/opencv4"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "imgproc",
    srcs = ["lib/x86_64-linux-gnu/libopencv_imgproc.so"],
    hdrs = glob([
        "include/opencv4/opencv2/imgproc/**",
        "include/opencv4/opencv2/*.hpp",
    ]),
    includes = ["include/opencv4"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
