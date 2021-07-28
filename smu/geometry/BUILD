cc_binary(
    name = "smu",
    srcs = [
        "smu_main.cc",
    ],
    deps = [
        ":smu_lib",
        "@lillymol//Foundational/cmdline:iwcmdline",
        "@lillymol//Molecule_Lib:moleculeio",
        "@lillymol//Molecule_Lib:iwmolecule",
    ],
    linkopts = [
        "-lz",
    ],
)

cc_library(
    name = "smu_lib",
    srcs = [
        "smu.cc",
        "smu_results.cc",
    ],
    hdrs = [
        "smu.h",
    ],
    deps = [
        "@lillymol//Molecule_Lib:iwmolecule",
    ]
)
