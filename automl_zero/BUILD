# A C++ implementation of AutoML-Zero.
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_proto_library")

licenses(["notice"])  # Apache 2.0

proto_library(
    name = "algorithm_proto",
    srcs = ["algorithm.proto"],
    deps = [
        ":instruction_proto",
    ],
)

cc_proto_library(
    name = "checkpointing_cc_proto",
    deps = [":algorithm_proto"],
)

cc_library(
    name = "compute_cost",
    srcs = ["compute_cost.cc"],
    hdrs = ["compute_cost.h"],
    deps = [
        ":algorithm",
        ":instruction",
        ":instruction_cc_proto",
    ],
)

proto_library(
    name = "task_proto",
    srcs = ["task.proto"],
)

cc_proto_library(
    name = "datasets_cc_proto",
    deps = [":task_proto"],
)

cc_library(
    name = "dataset",
    hdrs = ["task.h"],
    deps = [
        "datasets_cc_proto",
        ":definitions",
        "@com_google_googletest//:gtest_prod",
    ],
)

cc_test(
    name = "dataset_test",
    srcs = ["task_test.cc"],
    deps = [
        ":dataset",
        ":dataset_util",
        ":definitions",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "dataset_util",
    srcs = ["task_util.cc"],
    hdrs = ["task_util.h"],
    deps = [
        ":algorithm",
        ":dataset",
        ":datasets_cc_proto",
        ":definitions",
        ":executor",
        ":generator",
        ":memory",
        ":random_generator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "dataset_util_test",
    srcs = ["task_util_test.cc"],
    deps = [
        ":algorithm",
        ":dataset",
        ":dataset_util",
        ":datasets_cc_proto",
        ":definitions",
        ":executor",
        ":generator",
        ":generator_test_util",
        ":memory",
        ":random_generator",
        ":random_generator_test_util",
        ":test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen",
    ],
)

cc_library(
    name = "algorithm",
    srcs = ["algorithm.cc"],
    hdrs = ["algorithm.h"],
    deps = [
        ":checkpointing_cc_proto",
        ":definitions",
        ":instruction",
        ":random_generator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_test(
    name = "algorithm_test",
    srcs = ["algorithm_test.cc"],
    deps = [
        ":algorithm",
        ":dataset",
        ":dataset_util",
        ":definitions",
        ":executor",
        ":generator_test_util",
        ":memory",
        ":random_generator",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "algorithm_test_util",
    testonly = 1,
    srcs = ["algorithm_test_util.cc"],
    hdrs = ["algorithm_test_util.h"],
    deps = [
        ":algorithm",
        ":checkpointing_cc_proto",
        ":definitions",
        ":generator_test_util",
        ":instruction",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "definitions",
    hdrs = ["definitions.h"],
    deps = [
        ":instruction_cc_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_glog//:glog",
        "@com_google_protobuf//:protobuf",
        "@eigen_archive//:eigen",
    ],
)

cc_test(
    name = "definitions_test",
    srcs = ["definitions_test.cc"],
    deps = [
        ":definitions",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "evaluator",
    srcs = ["evaluator.cc"],
    hdrs = ["evaluator.h"],
    deps = [
        ":algorithm",
        ":dataset",
        ":dataset_util",
        ":datasets_cc_proto",
        ":definitions",
        ":executor",
        ":experiment_cc_proto",
        ":fec_cache",
        ":random_generator",
        ":train_budget",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "evaluator_test",
    srcs = ["evaluator_test.cc"],
    deps = [
        ":algorithm",
        ":dataset",
        ":dataset_util",
        ":datasets_cc_proto",
        ":definitions",
        ":evaluator",
        ":executor",
        ":generator",
        ":generator_test_util",
        ":random_generator",
        ":random_generator_test_util",
        ":test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "executor",
    hdrs = ["executor.h"],
    deps = [
        ":algorithm",
        ":dataset",
        ":datasets_cc_proto",
        ":definitions",
        ":instruction",
        ":instruction_cc_proto",
        ":memory",
        ":random_generator",
        "@com_google_googletest//:gtest_prod",
    ],
)

cc_test(
    name = "executor_test",
    srcs = ["executor_test.cc"],
    deps = [
        ":algorithm",
        ":dataset",
        ":dataset_util",
        ":datasets_cc_proto",
        ":definitions",
        ":executor",
        ":generator",
        ":generator_test_util",
        ":instruction",
        ":instruction_cc_proto",
        ":memory",
        ":random_generator",
        ":random_generator_test_util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen",
    ],
)

proto_library(
    name = "experiment_proto",
    srcs = ["experiment.proto"],
    deps = [
        ":fec_cache_proto",
        ":generator_proto",
        ":instruction_proto",
        ":mutator_proto",
        ":task_proto",
        ":train_budget_proto",
    ],
)

cc_proto_library(
    name = "experiment_cc_proto",
    deps = [":experiment_proto"],
)

cc_library(
    name = "experiment_util",
    srcs = ["experiment_util.cc"],
    hdrs = ["experiment_util.h"],
    deps = [
        ":datasets_cc_proto",
        ":definitions",
        ":instruction_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/memory",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "fec_hashing",
    srcs = ["fec_hashing.cc"],
    hdrs = ["fec_hashing.h"],
    deps = [
        ":definitions",
        ":executor",
    ],
)

cc_test(
    name = "fec_hashing_test",
    srcs = ["fec_hashing_test.cc"],
    deps = [
        ":definitions",
        ":fec_hashing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "generator",
    srcs = ["generator.cc"],
    hdrs = ["generator.h"],
    deps = [
        ":algorithm",
        ":definitions",
        ":generator_cc_proto",
        ":instruction",
        ":instruction_cc_proto",
        ":random_generator",
        ":randomizer",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "generator_test",
    srcs = ["generator_test.cc"],
    deps = [
        ":algorithm_test_util",
        ":dataset",
        ":dataset_util",
        ":definitions",
        ":evaluator",
        ":executor",
        ":generator",
        ":generator_test_util",
        ":instruction_cc_proto",
        ":random_generator",
        ":random_generator_test_util",
        ":test_util",
        ":util",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "generator_test_util",
    testonly = 1,
    srcs = ["generator_test_util.cc"],
    hdrs = ["generator_test_util.h"],
    deps = [
        ":algorithm",
        ":generator",
        ":random_generator",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "train_budget",
    srcs = ["train_budget.cc"],
    hdrs = ["train_budget.h"],
    deps = [
        ":algorithm",
        ":compute_cost",
        ":generator",
        ":instruction",
        ":train_budget_cc_proto",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "train_budget_test",
    srcs = ["train_budget_test.cc"],
    deps = [
        ":algorithm",
        ":generator_test_util",
        ":train_budget",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "instruction",
    srcs = ["instruction.cc"],
    hdrs = ["instruction.h"],
    deps = [
        ":checkpointing_cc_proto",
        ":definitions",
        ":instruction_cc_proto",
        ":random_generator",
        "@com_google_absl//absl/strings",
    ],
)

proto_library(
    name = "instruction_proto",
    srcs = ["instruction.proto"],
)

cc_proto_library(
    name = "instruction_cc_proto",
    deps = [":instruction_proto"],
)

cc_test(
    name = "instruction_test",
    srcs = ["instruction_test.cc"],
    deps = [
        ":definitions",
        ":instruction",
        ":instruction_cc_proto",
        ":random_generator",
        ":random_generator_test_util",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fec_cache",
    srcs = ["fec_cache.cc"],
    hdrs = ["fec_cache.h"],
    deps = [
        ":definitions",
        ":executor",
        ":fec_cache_cc_proto",
        ":fec_hashing",
    ],
)

proto_library(
    name = "fec_cache_proto",
    srcs = ["fec_cache.proto"],
)

cc_proto_library(
    name = "fec_cache_cc_proto",
    deps = [":fec_cache_proto"],
)

cc_test(
    name = "fec_cache_test",
    srcs = ["fec_cache_test.cc"],
    deps = [
        ":definitions",
        ":fec_cache",
        ":fec_cache_cc_proto",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "memory",
    hdrs = ["memory.h"],
    deps = [
        ":definitions",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_test(
    name = "memory_test",
    srcs = ["memory_test.cc"],
    deps = [
        ":definitions",
        ":memory",
        ":util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mutator",
    srcs = ["mutator.cc"],
    hdrs = ["mutator.h"],
    deps = [
        ":algorithm",
        ":definitions",
        ":instruction_cc_proto",
        ":mutator_cc_proto",
        ":random_generator",
        ":randomizer",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_prod",
    ],
)

cc_test(
    name = "mutator_test",
    srcs = ["mutator_test.cc"],
    deps = [
        ":algorithm",
        ":algorithm_test_util",
        ":definitions",
        ":generator",
        ":generator_test_util",
        ":instruction_cc_proto",
        ":mutator",
        ":mutator_cc_proto",
        ":mutator_test_util",
        ":random_generator",
        ":random_generator_test_util",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mutator_test_util",
    testonly = 1,
    srcs = ["mutator_test_util.cc"],
    hdrs = ["mutator_test_util.h"],
    deps = [
        ":mutator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "random_generator",
    srcs = ["random_generator.cc"],
    hdrs = ["random_generator.h"],
    deps = [
        ":definitions",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "random_generator_test",
    srcs = ["random_generator_test.cc"],
    deps = [
        ":definitions",
        ":random_generator",
        ":test_util",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "random_generator_test_util",
    testonly = 1,
    srcs = ["random_generator_test_util.cc"],
    hdrs = ["random_generator_test_util.h"],
    deps = [
        ":random_generator",
    ],
)

cc_library(
    name = "randomizer",
    srcs = ["randomizer.cc"],
    hdrs = ["randomizer.h"],
    deps = [
        ":algorithm",
        ":definitions",
        ":instruction_cc_proto",
        ":random_generator",
    ],
)

cc_library(
    name = "regularized_evolution",
    srcs = ["regularized_evolution.cc"],
    hdrs = ["regularized_evolution.h"],
    deps = [
        ":algorithm",
        ":checkpointing_cc_proto",
        ":dataset_util",
        ":definitions",
        ":evaluator",
        ":executor",
        ":generator",
        ":instruction",
        ":mutator",
        ":random_generator",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_prod",
    ],
)

cc_test(
    name = "regularized_evolution_test",
    srcs = ["regularized_evolution_test.cc"],
    deps = [
        ":algorithm",
        ":algorithm_test_util",
        ":dataset_util",
        ":definitions",
        ":experiment_cc_proto",
        ":generator_test_util",
        ":instruction_cc_proto",
        ":mutator",
        ":mutator_test_util",
        ":random_generator",
        ":regularized_evolution",
        ":test_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "run_search_experiment",
    srcs = ["run_search_experiment.cc"],
    deps = [
        ":algorithm",
        ":dataset_util",
        ":datasets_cc_proto",
        ":definitions",
        ":evaluator",
        ":experiment_cc_proto",
        ":experiment_util",
        ":fec_cache",
        ":generator",
        ":instruction_cc_proto",
        ":mutator",
        ":random_generator",
        ":regularized_evolution",
        ":train_budget",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    hdrs = ["test_util.h"],
    deps = [
        ":definitions",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "test_util_test",
    srcs = ["test_util_test.cc"],
    deps = [
        ":definitions",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "train_budget_proto",
    srcs = ["train_budget.proto"],
    deps = [":generator_proto"],
)

proto_library(
    name = "generator_proto",
    srcs = ["generator.proto"],
)

proto_library(
    name = "mutator_proto",
    srcs = ["mutator.proto"],
)

cc_proto_library(
    name = "mutator_cc_proto",
    deps = [":mutator_proto"],
)

cc_proto_library(
    name = "generator_cc_proto",
    deps = [":generator_proto"],
)

cc_proto_library(
    name = "train_budget_cc_proto",
    deps = [":train_budget_proto"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        ":definitions",
        "@eigen_archive//:eigen",
    ],
)

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "@com_google_googletest//:gtest_main",
    ],
)
