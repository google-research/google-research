// Copyright 2022 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "executor.h"

#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "task.pb.h"
#include "task.h"
#include "task_util.h"
#include "definitions.h"
#include "instruction.pb.h"
#include "algorithm.h"
#include "generator.h"
#include "generator_test_util.h"
#include "instruction.h"
#include "memory.h"
#include "random_generator.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "Eigen/Core"

namespace automl_zero {

using ::absl::StrCat;  // NOLINT
using ::std::abs;  // NOLINT
using ::std::isinf;  // NOLINT
using ::std::isnan;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::vector;  // NOLINT
using ::testing::ElementsAre;
using ::testing::Test;
using test_only::GenerateTask;

constexpr double kTestTolerance = 0.0001;
constexpr IntegerT kNumTrainExamples = 1000;
constexpr IntegerT kNumValidExamples = 100;
constexpr double kMaxAbsError = 100.0;
constexpr double kLargeMaxAbsError = 1000000000.0;

bool VectorEq(const Vector<4>& vector1,
              const vector<double>& vector2) {
  Eigen::Map<const Vector<4>> vector2_eigen(vector2.data());
  return vector1.isApprox(vector2_eigen);
}
bool VectorEq(const Vector<16>& vector1,
              const vector<double>& vector2) {
  Eigen::Map<const Vector<16>> vector2_eigen(vector2.data());
  return vector1.isApprox(vector2_eigen);
}

TEST(ExecutorTest, PredictComponentFunctionRuns) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that counts the examples in the
  // kPredictionsScalarAddress.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  constexpr AddressT temp_scalar_address = 2;
  algorithm.predict_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP, temp_scalar_address, ActivationDataSetter(1.0));
  algorithm.predict_[2] = make_shared<const Instruction>(
      SCALAR_SUM_OP,
      temp_scalar_address, kPredictionsScalarAddress,
      kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  executor.Execute();
  Memory<4> memory;
  executor.GetMemory(&memory);
  EXPECT_FLOAT_EQ(
      memory.scalar_[kPredictionsScalarAddress],
      static_cast<double>(kNumTrainExamples + kNumValidExamples));
}

TEST(ExecutorTest, LearnComponentFunctionRuns) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that counts the examples in the
  // kPredictionsScalarAddress.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  constexpr AddressT temp_scalar_address = 2;
  algorithm.learn_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP, temp_scalar_address, ActivationDataSetter(1.0));
  algorithm.learn_[2] = make_shared<const Instruction>(
      SCALAR_SUM_OP,
      temp_scalar_address, kPredictionsScalarAddress,
      kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  executor.Execute();
  Memory<4> memory;
  executor.GetMemory(&memory);
  EXPECT_FLOAT_EQ(
      memory.scalar_[kPredictionsScalarAddress],
      static_cast<double>(kNumTrainExamples));
}

TEST(ExecutorTest, ComputesLossCorrectly) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm in which the error is always 0.1.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  algorithm.predict_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kPredictionsScalarAddress,
      ActivationDataSetter(0.9));

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  double fitness = executor.Execute();
  EXPECT_FLOAT_EQ(fitness, FlipAndSquash(0.1));
}

TEST(ExecutorTest, ProbAccuracyComputesLossCorrectly) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: ACCURACY "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm in which the accuracy is always 0.0.
  Algorithm algorithm_0 = SimpleNoOpAlgorithm();
  algorithm_0.predict_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kPredictionsScalarAddress,
      ActivationDataSetter(-3.0));

  RandomGenerator rand_gen;
  Executor<4> executor_0(
      algorithm_0, dataset, kNumTrainExamples, kNumValidExamples,
      &rand_gen, kLargeMaxAbsError);
  double fitness_0 = executor_0.Execute();
  EXPECT_FLOAT_EQ(fitness_0, 0.0);

  // Create a Algorithm in which the accuracy is always 1.0.
  Algorithm algorithm_1 = SimpleNoOpAlgorithm();
  algorithm_1.predict_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kPredictionsScalarAddress,
      ActivationDataSetter(3.0));

  Executor<4> executor_1(
      algorithm_1, dataset, kNumTrainExamples, kNumValidExamples,
      &rand_gen, kLargeMaxAbsError);
  double fitness_1 = executor_1.Execute();
  EXPECT_FLOAT_EQ(fitness_1, 1.0);

  // Create a Algorithm, whose logit is infinity.
  Algorithm algorithm_inf = SimpleNoOpAlgorithm();
  algorithm_inf.predict_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kPredictionsScalarAddress,
      ActivationDataSetter(std::numeric_limits<double>::infinity()));

  Executor<4> executor_inf(algorithm_inf, dataset, kNumTrainExamples,
                           kNumValidExamples, &rand_gen, kLargeMaxAbsError);
  double fitness_inf = executor_inf.Execute();
  EXPECT_FLOAT_EQ(fitness_inf, 1.0);

  // Create a Algorithm, whose logit is negative infinity.
  Algorithm algorithm_ninf = SimpleNoOpAlgorithm();
  algorithm_ninf.predict_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kPredictionsScalarAddress,
      ActivationDataSetter(-std::numeric_limits<double>::infinity()));

  Executor<4> executor_ninf(algorithm_ninf, dataset, kNumTrainExamples,
                            kNumValidExamples, &rand_gen, kLargeMaxAbsError);
  double fitness_ninf = executor_ninf.Execute();
  EXPECT_FLOAT_EQ(fitness_ninf, kMinFitness);
}

TEST(ExecutorTest, ReportsErrors) {
  const IntegerT num_train_examples = 11;
  const IntegerT num_valid_examples = 9;
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_increment_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                num_train_examples,
                                " "
                                "num_valid_examples: ",
                                num_valid_examples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));
  Algorithm algorithm = SimpleNoOpAlgorithm();
  RandomGenerator rand_gen;
  Executor<4> executor(
      algorithm, dataset, num_train_examples, num_valid_examples,
      &rand_gen, kLargeMaxAbsError);
  vector<double> train_errors;
  vector<double> valid_errors;
  executor.Execute(&train_errors, &valid_errors);
  EXPECT_EQ(train_errors.size(), num_train_examples);
  EXPECT_EQ(valid_errors.size(), num_valid_examples);
  EXPECT_THAT(
      train_errors,
      ElementsAre(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0));
  EXPECT_THAT(
      valid_errors,
      ElementsAre(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0));
}

TEST(ExecutorTest, ItereatesThroughFeatures) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_increment_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that aggretates the mean value of the features.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  constexpr AddressT temp_scalar_address = 2;
  algorithm.predict_[0] = make_shared<const Instruction>(
      VECTOR_MEAN_OP, kFeaturesVectorAddress, temp_scalar_address);
  algorithm.predict_[2] = make_shared<const Instruction>(
      SCALAR_SUM_OP,
      temp_scalar_address, kPredictionsScalarAddress,
      kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  executor.Execute();
  Memory<4> memory;
  executor.GetMemory(&memory);
  EXPECT_FLOAT_EQ(memory.scalar_[kPredictionsScalarAddress], 504450.0);
}

TEST(ExecutorTest, ItereatesThroughLabelsDuringTraining) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_increment_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that aggretates the mean value of the labels.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  algorithm.learn_[0] = make_shared<const Instruction>(
      SCALAR_SUM_OP,
      kLabelsScalarAddress, kPredictionsScalarAddress,
      kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  executor.Execute();
  Memory<4> memory;
  executor.GetMemory(&memory);
  EXPECT_FLOAT_EQ(memory.scalar_[kPredictionsScalarAddress], 499500.0);
}

TEST(ExecutorTest, ItereatesThroughLabelsDuringValidation) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_increment_task {increment: 0.1} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  Algorithm algorithm = SimpleNoOpAlgorithm();

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  double fitness = executor.Execute();
  EXPECT_TRUE(abs(fitness - FlipAndSquash(5.7301812)) < kTestTolerance);
}

TEST(ExecutorTest, ValidationDoesNotSeeLabels) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_increment_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that aggretates the mean value of the labels.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  algorithm.predict_[0] = make_shared<const Instruction>(
      SCALAR_SUM_OP,
      kLabelsScalarAddress, kPredictionsScalarAddress,
      kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  executor.Execute();
  Memory<4> memory;
  executor.GetMemory(&memory);
  EXPECT_FLOAT_EQ(memory.scalar_[kPredictionsScalarAddress], 0.0);
}

TEST(ExecutorTest, StopsEarlyIfLargeErrorInSetupComponentFunction) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_increment_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that creates a NaN in the predict component function.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  algorithm.setup_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kPredictionsScalarAddress,
      ActivationDataSetter(kMaxAbsError + 10.0));

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfLargeErrorInPredictComponentFunction) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_increment_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that creates a NaN in the predict component function.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  algorithm.predict_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kPredictionsScalarAddress,
      ActivationDataSetter(kMaxAbsError + 10.0));

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfLargeErrorInLearnComponentFunction) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that creates a NaN in the predict component function.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  algorithm.learn_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP,
      kPredictionsScalarAddress,
      ActivationDataSetter(kMaxAbsError + 10.0));

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfInfinityInSetupComponentFunction) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that creates a NaN in the predict component function.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  AddressT one_address = 2;
  AddressT zero_address = 3;
  algorithm.setup_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP, one_address, ActivationDataSetter(1.0));
  algorithm.setup_[1] = make_shared<const Instruction>(
      SCALAR_DIVISION_OP,
      one_address, zero_address, kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfInfinityInPredictComponentFunction) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that creates a NaN in the predict component function.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  AddressT one_address = 2;
  AddressT zero_address = 3;
  algorithm.setup_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP, one_address, ActivationDataSetter(1.0));
  algorithm.predict_[0] = make_shared<const Instruction>(
      SCALAR_DIVISION_OP,
      one_address, zero_address, kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfInfinityInLearnComponentFunction) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that creates a NaN in the predict component function.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  AddressT one_address = 2;
  AddressT zero_address = 3;
  algorithm.setup_[0] = make_shared<const Instruction>(
      SCALAR_CONST_SET_OP, one_address, ActivationDataSetter(1.0));
  algorithm.learn_[0] = make_shared<const Instruction>(
      SCALAR_DIVISION_OP,
      one_address, zero_address, kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfNanInSetupComponentFunction) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that creates a NaN in the predict component function.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  AddressT zero_address = 2;
  algorithm.setup_[1] = make_shared<const Instruction>(
      SCALAR_DIVISION_OP,
      zero_address, zero_address, kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfNanInPredictComponentFunction) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that creates a NaN in the predict component function.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  AddressT zero_address = 2;
  algorithm.predict_[0] = make_shared<const Instruction>(
      SCALAR_DIVISION_OP,
      zero_address, zero_address, kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfNanInLearnComponentFunction) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_ones_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  // Create a Algorithm that creates a NaN in the predict component function.
  Algorithm algorithm = SimpleNoOpAlgorithm();
  AddressT zero_address = 2;
  algorithm.learn_[0] = make_shared<const Instruction>(
      SCALAR_DIVISION_OP,
      zero_address, zero_address, kPredictionsScalarAddress);

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfErrorTooLarge) {
  auto dataset = GenerateTask<4>(
      "unit_test_fixed_task { "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [1000.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "} "
      "eval_type: RMS_ERROR "
      "num_train_examples: 10 "
      "num_valid_examples: 10 "
      "num_tasks: 1 "
      "features_size: 4 ");

  Algorithm algorithm = SimpleNoOpAlgorithm();

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);

  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, DoesNotStopIfErrorNotLargeEnough) {
  auto dataset = GenerateTask<4>(
      "unit_test_fixed_task { "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "} "
      "eval_type: RMS_ERROR "
      "num_train_examples: 10 "
      "num_valid_examples: 10 "
      "num_tasks: 1 "
      "features_size: 4 ");

  Algorithm algorithm = SimpleNoOpAlgorithm();

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_TRUE(fitness != kMinFitness);
}

TEST(ExecutorTest, StopsEarlyIfProblemDuringValidation) {
  auto dataset = GenerateTask<4>(
      "unit_test_fixed_task { "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  train_labels {elements: [0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_features {elements: [0.0, 0.0, 0.0, 0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [1000.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "  valid_labels {elements: [0.0]} "
      "} "
      "eval_type: RMS_ERROR "
      "num_train_examples: 10 "
      "num_valid_examples: 10 "
      "num_tasks: 1 "
      "features_size: 4 ");

  Algorithm algorithm = SimpleNoOpAlgorithm();

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_DOUBLE_EQ(fitness, kMinFitness);
}

TEST(ExecutorTest, DoesNotStopsEarlyIfEverythingIsFine) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_zeros_task {} "
                                "eval_type: RMS_ERROR "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  Algorithm algorithm = SimpleNoOpAlgorithm();

  RandomGenerator rand_gen;
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kMaxAbsError);
  double fitness = executor.Execute();
  Memory<4> memory;
  EXPECT_FLOAT_EQ(fitness, FlipAndSquash(0.0));
}

TEST(ExecutorTest, TrainOptimizationsAreCorrect) {
  Task<4> dataset =
      GenerateTask<4>(StrCat("scalar_2layer_nn_regression_task {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: ",
                                kFirstParamSeedForTest,
                                " "
                                "data_seeds: ",
                                kFirstDataSeedForTest));
  RandomGenerator rand_gen;
  Algorithm algorithm = SimpleGz();
  double fitness_no_opt;
  { // Without optimization.
    Executor<4> executor(
        algorithm, dataset, kNumTrainExamples, kNumValidExamples,
        &rand_gen, kLargeMaxAbsError);
    // Iterators that tracks the progresss of training.
    TaskIterator<4> train_it = executor.dataset_.TrainIterator();
    EXPECT_TRUE(executor.TrainNoOptImpl(kNumTrainExamples, nullptr, &train_it));
    fitness_no_opt = executor.Validate(nullptr);
  }
  double fitness_opt;
  { // With optimization.
    Executor<4> executor(
        algorithm, dataset, kNumTrainExamples, kNumValidExamples,
        &rand_gen, kLargeMaxAbsError);
    // Iterators that tracks the progresss of training.
    TaskIterator<4> train_it = executor.dataset_.TrainIterator();
    EXPECT_TRUE(
        executor.TrainOptImpl<10>(kNumTrainExamples, nullptr, &train_it));
    fitness_opt = executor.Validate(nullptr);
  }
  EXPECT_FLOAT_EQ(fitness_no_opt, fitness_opt);
}

// TODO(crazydonkey): the number of examples passed to the executor is not
// correct, it should be multiplied by the number of epochs, so right now the
// executor is only training one epoch. This means this test cannot be testing
// the multi-epoch training. Please fix and uncomment.
// TEST(ExecutorTest, MultiEpochTrainingWorksCorrectly) {
//   Task<4> dataset = GenerateTask<4>(StrCat(
//       "scalar_2layer_nn_regression_task {} "
//       "num_train_examples: ", kNumTrainExamples, " "
//       "num_valid_examples: ", kNumValidExamples, " "
//       "eval_type: RMS_ERROR "
//       "param_seeds: ", kFirstParamSeedForTest, " "
//       "data_seeds: ", kFirstDataSeedForTest));
//   RandomGenerator rand_gen;
//   Algorithm algorithm = SimpleGz();
//   // Check that multiple epoch training works correctly. For example,
//   // training for another epoch will improve the validation error.
//   {
//     Task<4> dataset = GenerateTask<4>(StrCat(
//         "scalar_2layer_nn_regression_task {} "
//         "num_train_examples: ", kNumTrainExamples, " "
//         "num_valid_examples: ", kNumValidExamples, " "
//         "eval_type: RMS_ERROR "
//         "param_seeds: ", kFirstParamSeedForTest, " "
//         "num_train_epochs: 2 "
//         "data_seeds: ", kFirstDataSeedForTest));
//     // With optimization.
//     Executor<4> executor(
//         algorithm, dataset, kNumTrainExamples, kNumValidExamples, &rand_gen,
//         kLargeMaxAbsError);
//     // Iterators that tracks the progresss of training.
//     typename std::vector<Vector<4>>::const_iterator train_feature_it =
//         executor.dataset_.train_features_.begin();
//     std::vector<Scalar>::const_iterator train_label_it =
//         executor.dataset_.train_labels_.begin();
//     EXPECT_TRUE(
//         executor.TrainOptImpl<10>(
//             kNumTrainExamples, nullptr, &train_feature_it, &train_label_it));
//     double fitness_0 = executor.Validate(nullptr);
//     EXPECT_TRUE(
//         executor.TrainOptImpl<10>(
//             kNumTrainExamples, nullptr, &train_feature_it, &train_label_it));
//     EXPECT_TRUE(train_feature_it == executor.dataset_.train_features_.end());
//     double fitness_1 = executor.Validate(nullptr);
//     EXPECT_GT(fitness_1, fitness_0);
//   }
// }

class ExecuteInstructionTest : public Test {
 protected:
  ExecuteInstructionTest()
      : in1_(1),
        in2_(0),
        out_(1),
        bit_gen_(100000),
        train_rand_gen_(&bit_gen_) {
    memory_.Wipe();
  }
  template<class Setter0>
  Instruction MakeZeroInputsInstruction(
      const Op op, Setter0 setter0) {
    return Instruction(op, out_, setter0);
  }
  template<class Setter0, class Setter1>
  Instruction MakeZeroInputsInstruction(
      const Op op, Setter0 setter0, Setter1 setter1) {
    return Instruction(op, out_, setter0, setter1);
  }
  template<class Setter0, class Setter1, class Setter2>
  Instruction MakeZeroInputsInstruction(
      const Op op, Setter0 setter0, Setter1 setter1, Setter2 setter2) {
    return Instruction(op, out_, setter0, setter1, setter2);
  }
  Instruction MakeOneInputInstruction(const Op op) {
    return Instruction(op, in1_, out_);
  }
  Instruction MakeTwoInputsInstruction(const Op op) {
    return Instruction(op, in1_, in2_, out_);
  }
  void VerifyNothingToScalarEquals(
      const Instruction& instruction, const double expected_out) {
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_LE(abs(memory_.scalar_[out_] - expected_out), kTestTolerance);
  }
  void VerifyNothingToScalarIsRandomized(const Instruction& instruction) {
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    const double out1 = memory_.scalar_[out_];
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    const double out2 = memory_.scalar_[out_];
    EXPECT_GT(abs(out1 - out2), kTestTolerance);
  }
  void VerifyNothingToVectorEquals(
      const Instruction& instruction,
      const vector<double>& expected_out) {
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Vector<4> expected_out_vector(expected_out.data());
    EXPECT_LE((memory_.vector_[out_] - expected_out_vector).norm(),
              kTestTolerance);
  }
  void VerifyNothingToVectorIsRandomized(const Instruction& instruction) {
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Vector<4> out1 = memory_.vector_[out_];
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Vector<4> out2 = memory_.vector_[out_];
    EXPECT_GT((out1 - out2).norm(), kTestTolerance);
  }
  void VerifyNothingToMatrixEquals(
      const Instruction& instruction,
      const vector<double>& expected_out) {
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Matrix<4> expected_out_matrix(expected_out.data());
    EXPECT_LE((memory_.matrix_[out_] - expected_out_matrix).norm(),
              kTestTolerance);
  }
  void VerifyNothingToMatrixIsRandomized(const Instruction& instruction) {
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Matrix<4> out1 = memory_.matrix_[out_];
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Matrix<4> out2 = memory_.matrix_[out_];
    EXPECT_GT((out1 - out2).norm(), kTestTolerance);
  }
  void VerifyScalarToScalarEquals(
      const Instruction& instruction,
      const double in1,
      const double expected_out) {
    memory_.scalar_[in1_] = in1;
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_LE(abs(memory_.scalar_[out_] - expected_out), kTestTolerance);
  }
  void VerifyScalarToScalarIsGreater(
      const Instruction& instruction,
      const double in1,
      const double than) {
    memory_.scalar_[in1_] = in1;
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_GT(memory_.scalar_[out_], than);
  }
  void VerifyScalarToScalarIsLess(
      const Instruction& instruction,
      const double in1,
      const double than) {
    memory_.scalar_[in1_] = in1;
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_LT(memory_.scalar_[out_], than);
  }
  void VerifyScalarToScalarIsNan(
      const Instruction& instruction, const double in1) {
    memory_.scalar_[in1_] = in1;
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_TRUE(isnan(memory_.scalar_[out_]));
  }
  void VerifyScalarToScalarIsInf(
      const Instruction& instruction, const double in1) {
    memory_.scalar_[in1_] = in1;
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_TRUE(isinf(memory_.scalar_[out_]));
  }
  void VerifyVectorToScalarEquals(
      const Instruction& instruction,
      const vector<double>& in1,
      const double expected_out) {
    memory_.vector_[in1_] = Vector<4>(in1.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_LE(abs(memory_.scalar_[out_] - expected_out), kTestTolerance);
  }
  void VerifyVectorToVectorEquals(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& expected_out) {
    memory_.vector_[in1_] = Vector<4>(in1.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Vector<4> expected_out_vector(expected_out.data());
    EXPECT_LE((memory_.vector_[out_] - expected_out_vector).norm(),
              kTestTolerance);
  }
  void VerifyMatrixToScalarEquals(
      const Instruction& instruction,
      const vector<double>& in1,
      const double expected_out) {
    memory_.matrix_[in1_] = Matrix<4>(in1.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_LE(abs(memory_.scalar_[out_] - expected_out), kTestTolerance);
  }
  void VerifyMatrixToVectorEquals(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& expected_out) {
    memory_.matrix_[in1_] = Matrix<4>(in1.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Vector<4> expected_out_vector(expected_out.data());
    EXPECT_LE((memory_.vector_[out_] - expected_out_vector).norm(),
              kTestTolerance);
  }
  void VerifyScalarScalarToScalarEquals(
      const Instruction& instruction,
      const double in1,
      const double in2,
      const double expected_out) {
    memory_.scalar_[in1_] = in1;
    memory_.scalar_[in2_] = in2;
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_LE(abs(memory_.scalar_[out_] - expected_out), kTestTolerance);
  }
  void VerifyScalarScalarToScalarIsNan(
      const Instruction& instruction,
      const double in1,
      const double in2) {
    memory_.scalar_[in1_] = in1;
    memory_.scalar_[in2_] = in2;
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_TRUE(isnan(memory_.scalar_[out_]));
  }
  void VerifyScalarScalarToScalarInstructionIsInf(
      const Instruction& instruction,
      const double in1,
      const double in2) {
    memory_.scalar_[in1_] = in1;
    memory_.scalar_[in2_] = in2;
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_TRUE(isinf(memory_.scalar_[out_]));
  }
  void VerifyScalarVectorToVectorEquals(
      const Instruction& instruction,
      const double in1,
      const vector<double>& in2,
      const vector<double>& expected_out) {
    memory_.scalar_[in1_] = in1;
    memory_.vector_[in2_] = Vector<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Vector<4> expected_out_vector(expected_out.data());
    EXPECT_LE((memory_.vector_[out_] - expected_out_vector).norm(),
              kTestTolerance);
  }
  void VerifyScalarMatrixToMatrixEquals(
      const Instruction& instruction,
      const double in1,
      const vector<double>& in2,
      const vector<double>& expected_out) {
    memory_.scalar_[in1_] = in1;
    memory_.matrix_[in2_] = Matrix<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Matrix<4> expected_out_matrix(expected_out.data());
    EXPECT_LE((memory_.matrix_[out_] - expected_out_matrix).norm(),
              kTestTolerance);
  }
  void VerifyVectorVectorToScalarEquals(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& in2,
      const double expected_out) {
    memory_.vector_[in1_] = Vector<4>(in1.data());
    memory_.vector_[in2_] = Vector<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_LE(abs(memory_.scalar_[out_] - expected_out), kTestTolerance);
  }
  void VerifyVectorVectorToVectorInstructionWorksCorrectly(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& in2,
      const vector<double>& expected_out) {
    memory_.vector_[in1_] = Vector<4>(in1.data());
    memory_.vector_[in2_] = Vector<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Vector<4> expected_out_vector(expected_out.data());
    EXPECT_LE((memory_.vector_[out_] - expected_out_vector).norm(),
              kTestTolerance);
  }
  void VerifyVectorVectorToVectorInstructionIsNan(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& in2) {
    memory_.vector_[in1_] = Vector<4>(in1.data());
    memory_.vector_[in2_] = Vector<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_TRUE(isnan(memory_.vector_[out_].norm()));
  }
  void VerifyVectorVectorToVectorInstructionIsInf(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& in2) {
    memory_.vector_[in1_] = Vector<4>(in1.data());
    memory_.vector_[in2_] = Vector<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_TRUE(isinf(memory_.vector_[out_].norm()));
  }
  void VerifyVectorVectorToMatrixEquals(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& in2,
      const vector<double>& expected_out) {
    memory_.vector_[in1_] = Vector<4>(in1.data());
    memory_.vector_[in2_] = Vector<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Matrix<4> expected_out_matrix(expected_out.data());
    EXPECT_LE((memory_.matrix_[out_] - expected_out_matrix).norm(),
              kTestTolerance);
  }
  void VerifyMatrixVectorToVectorEquals(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& in2,
      const vector<double>& expected_out) {
    memory_.matrix_[in1_] = Matrix<4>(in1.data());
    memory_.vector_[in2_] = Vector<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Vector<4> expected_out_vector(expected_out.data());
    EXPECT_LE((memory_.vector_[out_] - expected_out_vector).norm(),
              kTestTolerance);
  }
  void VerifyMatrixMatrixToMatrixInstructionWorksCorrectly(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& in2,
      const vector<double>& expected_out) {
    memory_.matrix_[in1_] = Matrix<4>(in1.data());
    memory_.matrix_[in2_] = Matrix<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Matrix<4> expected_out_matrix(expected_out.data());
    EXPECT_LE((memory_.matrix_[out_] - expected_out_matrix).norm(),
              kTestTolerance);
  }
  void VerifyMatrixMatrixToMatrixInstructionIsNan(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& in2) {
    memory_.matrix_[in1_] = Matrix<4>(in1.data());
    memory_.matrix_[in2_] = Matrix<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_TRUE(isnan(memory_.vector_[out_].norm()));
  }
  void VerifyMatrixMatrixToMatrixInstructionIsInf(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& in2) {
    memory_.matrix_[in1_] = Matrix<4>(in1.data());
    memory_.matrix_[in2_] = Matrix<4>(in2.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    EXPECT_TRUE(isnan(memory_.vector_[out_].norm()));
  }
  void VerifyMatrixToMatrixInstructionWorksCorrectly(
      const Instruction& instruction,
      const vector<double>& in1,
      const vector<double>& expected_out) {
    memory_.matrix_[in1_] = Matrix<4>(in1.data());
    ExecuteInstruction(instruction, &train_rand_gen_, &memory_);
    Matrix<4> expected_out_matrix(expected_out.data());
    EXPECT_LE((memory_.matrix_[out_] - expected_out_matrix).norm(),
              kTestTolerance);
  }

  const AddressT in1_;
  const AddressT in2_;
  const AddressT out_;
  Memory<4> memory_;
  mt19937 bit_gen_;
  RandomGenerator train_rand_gen_;
};

TEST_F(ExecuteInstructionTest, ScalarArithmeticRelated_ScalarSumOp) {
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_SUM_OP), 0.5, 2.0, 2.5);
}

TEST_F(ExecuteInstructionTest, ScalarArithmeticRelated_ScalarDiffOp) {
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_DIFF_OP), -0.2, 2.3, -2.5);
}

TEST_F(ExecuteInstructionTest, ScalarArithmeticRelated_ScalarProductOp) {
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_PRODUCT_OP), -0.2, 2.3, -0.46);
}

TEST_F(ExecuteInstructionTest, ScalarArithmeticRelated_ScalarDivisionOp) {
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_DIVISION_OP), 8.8, -2.0, -4.4);
  VerifyScalarScalarToScalarIsNan(
      MakeTwoInputsInstruction(SCALAR_DIVISION_OP), 0.0, 0.0);
  VerifyScalarScalarToScalarInstructionIsInf(
      MakeTwoInputsInstruction(SCALAR_DIVISION_OP), 1.0, 0.0);
}

TEST_F(ExecuteInstructionTest, ScalarArithmeticRelated_ScalarMinOp) {
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_MIN_OP), 0.5, 2.2, 0.5);
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_MIN_OP), 2.2, 0.5, 0.5);
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_MIN_OP), -2.2, -0.5, -2.2);
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_MIN_OP), 2.2, -0.5, -0.5);
}

TEST_F(ExecuteInstructionTest, ScalarArithmeticRelated_ScalarMaxOp) {
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_MAX_OP), 0.5, 2.2, 2.2);
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_MAX_OP), 2.2, 0.5, 2.2);
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_MAX_OP), -2.2, -0.5, -0.5);
  VerifyScalarScalarToScalarEquals(
      MakeTwoInputsInstruction(SCALAR_MAX_OP), 0.5, -2.2, 0.5);
}

TEST_F(ExecuteInstructionTest, ScalarArithmeticRelated_ScalarAbsOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ABS_OP), 2.5, 2.5);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ABS_OP), -2.5, 2.5);
}

TEST_F(ExecuteInstructionTest, ScalarArithmeticRelated_ScalarHeavisideOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_HEAVYSIDE_OP), -2.5, 0.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_HEAVYSIDE_OP), -0.5, 0.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_HEAVYSIDE_OP), 0.5, 1.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_HEAVYSIDE_OP), 2.5, 1.0);
}

TEST_F(ExecuteInstructionTest, ScalarArithmeticRelated_ScalarConstSetOp) {
  VerifyNothingToScalarEquals(
      MakeZeroInputsInstruction(
          SCALAR_CONST_SET_OP, ActivationDataSetter(-0.5)), -0.5);
}

TEST_F(ExecuteInstructionTest, TrigonometryRelated_ScalarSinOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_SIN_OP), 0.0, 0.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_SIN_OP), kPi / 6.0, 0.5);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_SIN_OP), kPi / 2.0, 1.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_SIN_OP), 3 * kPi / 2, -1.0);
}

TEST_F(ExecuteInstructionTest, TrigonometryRelated_ScalarCosOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_COS_OP), 0.0, 1.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_COS_OP), kPi / 3.0, 0.5);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_COS_OP), kPi / 2.0, 0.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_COS_OP), kPi, -1.0);
}

TEST_F(ExecuteInstructionTest, TrigonometryRelated_ScalarTanOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_TAN_OP), 0.0, 0.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_TAN_OP), kPi / 4.0, 1.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_TAN_OP), 3 * kPi / 4.0, -1.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_TAN_OP), 5 * kPi / 4.0, 1.0);
  VerifyScalarToScalarIsGreater(
      MakeOneInputInstruction(SCALAR_TAN_OP),
      kPi / 2.0 - 0.000000001, 1000000.0);
  VerifyScalarToScalarIsLess(
      MakeOneInputInstruction(SCALAR_TAN_OP),
      kPi / 2.0 + 0.000000001, -1000000.0);
}

TEST_F(ExecuteInstructionTest, TrigonometryRelated_ScalarArcSinOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCSIN_OP), 0.0, 0.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCSIN_OP), 0.5, kPi / 6.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCSIN_OP), 1.0, kPi / 2.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCSIN_OP), -1.0, -kPi / 2);
}

TEST_F(ExecuteInstructionTest, TrigonometryRelated_ScalarArcCosOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCCOS_OP), 1.0, 0.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCCOS_OP), 0.5, kPi / 3.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCCOS_OP), 0.0, kPi / 2.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCCOS_OP), -1.0, kPi);
}

TEST_F(ExecuteInstructionTest, TrigonometryRelated_ScalarArcTanOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCTAN_OP), 0.0, 0.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCTAN_OP), 1.0, kPi / 4.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCTAN_OP), -1.0, -kPi / 4.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCTAN_OP),
      1000000000.0, kPi / 2.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_ARCTAN_OP),
      -1000000000.0, -kPi / 2.0);
}

TEST_F(ExecuteInstructionTest, CalculusRelated_ScalarExpOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_EXP_OP), 0.0, 1.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_EXP_OP), 1.0, kE);
}

TEST_F(ExecuteInstructionTest, CalculusRelated_ScalarLogOp) {
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_LOG_OP), 1.0, 0.0);
  VerifyScalarToScalarEquals(
      MakeOneInputInstruction(SCALAR_LOG_OP), kE, 1.0);
  VerifyScalarToScalarIsInf(
      MakeOneInputInstruction(SCALAR_LOG_OP), 0.0);
  VerifyScalarToScalarIsNan(
      MakeOneInputInstruction(SCALAR_LOG_OP), -0.1);
}

TEST_F(ExecuteInstructionTest, VectorArithmeticRelated_VectorSumOp) {
  VerifyVectorVectorToVectorInstructionWorksCorrectly(
      MakeTwoInputsInstruction(VECTOR_SUM_OP),
      {0.1, -0.1, 1.2, -10.0}, {2.3, -0.3, 0.5, -0.001},
      {2.4, -0.4, 1.7, -10.001});
}

TEST_F(ExecuteInstructionTest, VectorArithmeticRelated_VectorDiffOp) {
  VerifyVectorVectorToVectorInstructionWorksCorrectly(
      MakeTwoInputsInstruction(VECTOR_DIFF_OP),
      {0.1, -0.1, 1.2, -10.0}, {2.3, -0.3, 0.5, -0.001},
      {-2.2, 0.2, 0.7, -9.999});
}

TEST_F(ExecuteInstructionTest, VectorArithmeticRelated_VectorProductOp) {
  VerifyVectorVectorToVectorInstructionWorksCorrectly(
      MakeTwoInputsInstruction(VECTOR_PRODUCT_OP),
      {0.1, -0.1, 1.2, -10.0}, {2.3, -0.3, 0.5, -0.001},
      {0.23, 0.03, 0.6, 0.01});
}

TEST_F(ExecuteInstructionTest, VectorArithmeticRelated_VectorDivisionOp) {
  VerifyVectorVectorToVectorInstructionWorksCorrectly(
      MakeTwoInputsInstruction(VECTOR_DIVISION_OP),
      {7.0, -18.18, 1.0, 0.0}, {2.0, 3.0, 0.5, -0.5},
      {3.5, -6.06, 2.0, 0.0});
  VerifyVectorVectorToVectorInstructionIsNan(
      MakeTwoInputsInstruction(VECTOR_DIVISION_OP),
      {7.0, -18.18, 0.0, -10.0}, {2.0, 3.0, 0.0, -0.5});
  VerifyVectorVectorToVectorInstructionIsInf(
      MakeTwoInputsInstruction(VECTOR_DIVISION_OP),
      {7.0, -18.18, 1.0, -10.0}, {2.0, 3.0, 0.0, -0.5});
}

TEST_F(ExecuteInstructionTest, VectorArithmeticRelated_VectorMinOp) {
  VerifyVectorVectorToVectorInstructionWorksCorrectly(
      MakeTwoInputsInstruction(VECTOR_MIN_OP),
      {0.5, 2.2, -2.2, 2.2}, {2.2, 0.5, -0.5, -0.5},
      {0.5, 0.5, -2.2, -0.5});
}

TEST_F(ExecuteInstructionTest, VectorArithmeticRelated_VectorMaxOp) {
  VerifyVectorVectorToVectorInstructionWorksCorrectly(
      MakeTwoInputsInstruction(VECTOR_MAX_OP),
      {0.5, 2.2, -2.2, 0.5}, {2.2, 0.5, -0.5, -2.2},
      {2.2, 2.2, -0.5, 0.5});
}

TEST_F(ExecuteInstructionTest, VectorArithmeticRelated_VectorAbsOp) {
  VerifyVectorToVectorEquals(
      MakeOneInputInstruction(VECTOR_ABS_OP),
      {0.5, 0.0, -2.2, 100.5}, {0.5, 0.0, 2.2, 100.5});
}

TEST_F(ExecuteInstructionTest, VectorArithmeticRelated_VectorHeavisideOp) {
  VerifyVectorToVectorEquals(
      MakeOneInputInstruction(VECTOR_HEAVYSIDE_OP),
      {-0.01, 0.001, 1.3, -0.001}, {0.0, 1.0, 1.0, 0.0});
}

TEST_F(ExecuteInstructionTest, VectorArithmeticRelated_VectorConstSetOp) {
  VerifyNothingToVectorEquals(
      MakeZeroInputsInstruction(VECTOR_CONST_SET_OP,
                                FloatDataSetter(IndexToFloat(2, 4)),
                                FloatDataSetter(-1.5)),
      {0.0, 0.0, -1.5, 0.0});
}

TEST_F(ExecuteInstructionTest, MatrixArithmeticRelated_MatrixSumOp) {
  VerifyMatrixMatrixToMatrixInstructionWorksCorrectly(
      MakeTwoInputsInstruction(MATRIX_SUM_OP),
      {-2.0, 10.0, 0.3, 0.0,
       0.0, 8.0, -0.1, 0.0,
       20.0, 0.0, 20.0, 50.0,
       -0.01, -1.0, 25.0, -32.0},
      {-0.2, 1.0, 0.03, 0.0,
       0.0, 0.8, -0.01, 0.0,
       2.0, 0.0, 2.0, 5.0,
       -0.001, -0.1, 2.5, -3.2},
      {-2.2, 11.0, 0.33, 0.0,
       0.0, 8.8, -0.11, 0.0,
       22.0, 0.0, 22.0, 55.0,
       -0.011, -1.1, 27.5, -35.2});
}

TEST_F(ExecuteInstructionTest, MatrixArithmeticRelated_MatrixDiffOp) {
  VerifyMatrixMatrixToMatrixInstructionWorksCorrectly(
      MakeTwoInputsInstruction(MATRIX_DIFF_OP),
      {-2.0, 10.0, 0.3, 0.0,
       0.0, 8.0, -0.1, 0.0,
       20.0, 0.0, 20.0, 50.0,
       -0.01, -1.0, 25.0, -32.0},
      {-0.2, 1.0, 0.03, 0.0,
       0.0, 0.8, -0.01, 0.0,
       2.0, 0.0, 2.0, 5.0,
       -0.001, -0.1, 2.5, -3.2},
      {-1.8, 9.0, 0.27, 0.0,
       0.0, 7.2, -0.09, 0.0,
       18.0, 0.0, 18.0, 45.0,
       -0.009, -0.9, 22.5, -28.8});
}

TEST_F(ExecuteInstructionTest, MatrixArithmeticRelated_MatrixProductOp) {
  VerifyMatrixMatrixToMatrixInstructionWorksCorrectly(
      MakeTwoInputsInstruction(MATRIX_PRODUCT_OP),
      {0.1, -0.1, 1.2, -10.0,
       0.1, 1.2, -0.1, -10.0,
       1.0, -1.0, 12.0, -100.0,
       0.01, -0.01, 0.12, -1.00},
      {2.3, -0.3, 0.5, -0.001,
       2.3, 0.5, -0.3, -0.001,
       23, -3.0, 5.0, -0.01,
       0.23, -0.03, 0.05, -0.0001},
      {0.23, 0.03, 0.6, 0.01,
       0.23, 0.6, 0.03, 0.01,
       23.0, 3.0, 60.0, 1.0,
       0.0023, 0.0003, 0.006, 0.0001});
}

TEST_F(ExecuteInstructionTest, MatrixArithmeticRelated_MatrixDivisionOp) {
  VerifyMatrixMatrixToMatrixInstructionWorksCorrectly(
      MakeTwoInputsInstruction(MATRIX_DIVISION_OP),
      {7.0, -18.18, 1.0, 0.0,
       7.0, 1.0, -18.18, 0.0,
       70.0, -181.8, 10.0, 0.0,
       70.0, -181.8, 0.0, 10.0},
      {2.0, 3.0, 0.5, -0.5,
       2.0, 0.5, 3.0, -0.5,
       20.0, 30.0, 5.0, -5.0,
       2.0, 3.0, -0.5, 0.5},
      {3.5, -6.06, 2.0, 0.0,
       3.5, 2.0, -6.06, 0.0,
       3.5, -6.06, 2.0, 0.0,
       35.0, -60.6, 0.0, 20.0});
  VerifyMatrixMatrixToMatrixInstructionIsNan(
      MakeTwoInputsInstruction(VECTOR_DIVISION_OP),
      {7.0, -18.18, 1.0, 0.0,
       7.0, 1.0, 0.0, 0.0,
       70.0, -181.8, 10.0, 0.0,
       70.0, -181.8, 0.0, 10.0},
      {2.0, 3.0, 0.5, -0.5,
       2.0, 0.5, 0.0, -0.5,
       20.0, 30.0, 5.0, -5.0,
       2.0, 3.0, -0.5, 0.5});
  VerifyMatrixMatrixToMatrixInstructionIsInf(
      MakeTwoInputsInstruction(VECTOR_DIVISION_OP),
      {7.0, -18.18, 1.0, 0.0,
       7.0, 1.0, 1.0, 0.0,
       70.0, -181.8, 10.0, 0.0,
       70.0, -181.8, 0.0, 10.0},
      {2.0, 3.0, 0.5, -0.5,
       2.0, 0.5, 0.0, -0.5,
       20.0, 30.0, 5.0, -5.0,
       2.0, 3.0, -0.5, 0.5});
}

TEST_F(ExecuteInstructionTest, MatrixArithmeticRelated_MatrixMinOp) {
  VerifyMatrixMatrixToMatrixInstructionWorksCorrectly(
      MakeTwoInputsInstruction(MATRIX_MIN_OP),
      {0.5, 2.2, -2.2, 2.2,
       0.5, -2.2, 2.2, 2.2,
       5.0, 22.0, -22.0, 22.0,
       0.05, 0.22, -0.22, 0.22},
      {2.2, 0.5, -0.5, -0.5,
       2.2, -0.5, 0.5, -0.5,
       22.0, 5.0, -5.0, -5.0,
       0.22, 0.05, -0.05, -0.05},
      {0.5, 0.5, -2.2, -0.5,
       0.5, -2.2, 0.5, -0.5,
       5.0, 5.0, -22.0, -5.0,
       0.05, 0.05, -0.22, -0.05});
}

TEST_F(ExecuteInstructionTest, MatrixArithmeticRelated_MatrixMaxOp) {
  VerifyMatrixMatrixToMatrixInstructionWorksCorrectly(
      MakeTwoInputsInstruction(MATRIX_MAX_OP),
      {0.5, 2.2, -2.2, 0.5,
       0.5, -2.2, 2.2, 0.5,
       5.0, 22.0, -22.0, 5.0,
       0.05, 0.22, -0.22, 0.05},
      {2.2, 0.5, -0.5, -2.2,
       2.2, -0.5, 0.5, -2.2,
       22.0, 5.0, -5.0, -22.0,
       0.22, 0.05, -0.05, -0.22},
      {2.2, 2.2, -0.5, 0.5,
       2.2, -0.5, 2.2, 0.5,
       22.0, 22.0, -5.0, 5.0,
       0.22, 0.22, -0.05, 0.05});
}

TEST_F(ExecuteInstructionTest, MatrixArithmeticRelated_MatrixAbsOp) {
  VerifyMatrixToMatrixInstructionWorksCorrectly(
      MakeOneInputInstruction(MATRIX_ABS_OP),
      {0.5, 0.0, -2.2, 100.5,
       0.5, -2.2, 0.0, 100.5,
       5.0, 0.0, -22.0, 1005.0,
       0.05, 0.0, -0.22, 10.05},
      {0.5, 0.0, 2.2, 100.5,
       0.5, 2.2, 0.0, 100.5,
       5.0, 0.0, 22.0, 1005.0,
       0.05, 0.0, 0.22, 10.05});
}

TEST_F(ExecuteInstructionTest, MatrixArithmeticRelated_MatrixHeavisideOp) {
  VerifyMatrixToMatrixInstructionWorksCorrectly(
      MakeOneInputInstruction(MATRIX_HEAVYSIDE_OP),
      {0.5, 0.1, -2.2, 100.5,
       0.5, -2.2, 0.0, 100.5,
       5.0, 1.0, -22.0, 1005.0,
       0.05, 0.01, -0.22, 10.05},
      {1.0, 1.0, 0.0, 1.0,
       1.0, 0.0, 0.0, 1.0,
       1.0, 1.0, 0.0, 1.0,
       1.0, 1.0, 0.0, 1.0});
}

TEST_F(ExecuteInstructionTest, MatrixArithmeticRelated_MatrixConstSetOp) {
  VerifyNothingToMatrixEquals(
      MakeZeroInputsInstruction(MATRIX_CONST_SET_OP,
                                FloatDataSetter(IndexToFloat(2, 4)),
                                FloatDataSetter(IndexToFloat(1, 4)),
                                FloatDataSetter(-1.5)),
      {0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0,
       0.0, -1.5, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0});
}

TEST_F(ExecuteInstructionTest, LinearAlgebraRelated_ScalarVectorProductOp) {
  VerifyScalarVectorToVectorEquals(
      MakeTwoInputsInstruction(SCALAR_VECTOR_PRODUCT_OP),
      -0.2, {2.3, -0.3, 0.5, -0.001},
      {-0.46, 0.06, -0.1, 0.0002});
}

TEST_F(ExecuteInstructionTest, LinearAlgebraRelated_VectorInnerProductOp) {
  VerifyVectorVectorToScalarEquals(
      MakeTwoInputsInstruction(VECTOR_INNER_PRODUCT_OP),
      {-0.01, 0.0, 1.3, -0.001}, {2.3, 0.3, 0.5, -0.001},
      0.627001);
}

TEST_F(ExecuteInstructionTest, LinearAlgebraRelated_VectorOuterProductOp) {
  VerifyVectorVectorToMatrixEquals(
      MakeTwoInputsInstruction(VECTOR_OUTER_PRODUCT_OP),
      {0.1, -0.1, 1.2, -10.0}, {2.3, -0.3, 0.5, -0.001},
      {0.23, -0.03, 0.05, -0.0001,
       -0.23, 0.03, -0.05, 0.0001,
       2.76, -0.36, 0.6, -0.0012,
       -23, 3.0, -5.0, 0.01});
}

TEST_F(ExecuteInstructionTest, LinearAlgebraRelated_ScalarMatrixProductOp) {
  VerifyScalarMatrixToMatrixEquals(
      MakeTwoInputsInstruction(SCALAR_MATRIX_PRODUCT_OP),
      0.5,
      {2.2, 0.5, -0.5, -2.2,
       2.2, -0.5, 0.5, -2.2,
       22.0, 5.0, -5.0, -22.0,
       0.22, 0.05, -0.05, -0.22},
      {1.1, 0.25, -0.25, -1.1,
       1.1, -0.25, 0.25, -1.1,
       11.0, 2.5, -2.5, -11.0,
       0.11, 0.025, -0.025, -0.11});
}

TEST_F(ExecuteInstructionTest, LinearAlgebraRelated_MatrixVectorProductOp) {
  VerifyMatrixVectorToVectorEquals(
      MakeTwoInputsInstruction(MATRIX_VECTOR_PRODUCT_OP),
      {-0.2, 1.0, 0.03, 0.0,
       0.0, 0.8, -0.01, 0.0,
       2.0, 0.0, 2.0, 5.0,
       -0.001, -0.1, 2.5, -3.2},
      {0.1, -2.2, 10.0, 0.0},
      {-1.92, -1.86, 20.2, 25.2199});
}

TEST_F(ExecuteInstructionTest, LinearAlgebraRelated_VectorNormOp) {
  VerifyVectorToScalarEquals(
      MakeOneInputInstruction(VECTOR_NORM_OP),
      {2.2, -0.5, 0.0, 0.01},
      2.25612499654);
}

TEST_F(ExecuteInstructionTest, LinearAlgebraRelated_MatrixNormOp) {
  VerifyMatrixToScalarEquals(
      MakeOneInputInstruction(MATRIX_NORM_OP),
      {0.0, 0.5, -0.5, -2.2,
       2.2, -0.5, 0.5, -2.2,
       22.0, 5.0, -5.0, -22.0,
       0.22, 0.05, -0.05, -0.22},
      32.149989113528484);
}

TEST_F(ExecuteInstructionTest, LinearAlgebraRelated_MatrixTransposeOp) {
  VerifyMatrixToMatrixInstructionWorksCorrectly(
      MakeOneInputInstruction(MATRIX_TRANSPOSE_OP),
      {0.0, 0.5, -0.5, -2.2,
       2.2, -0.5, 0.5, -2.2,
       22.0, 5.0, -5.0, -22.0,
       0.22, 0.05, -0.05, -0.22},
      {0.0, 2.2, 22.0, 0.22,
       0.5, -0.5, 5.0, 0.05,
       -0.5, 0.5, -5.0, -0.05,
       -2.2, -2.2, -22.0, -0.22});
}

TEST_F(ExecuteInstructionTest, LinearAlgebraRelated_MatrixMatrixProductOp) {
  VerifyMatrixMatrixToMatrixInstructionWorksCorrectly(
      MakeTwoInputsInstruction(MATRIX_MATRIX_PRODUCT_OP),
      {0.1, 2.5, -0.5, -2.2,
       10.3, -0.06, 0.7, -2.1,
       22.0, 5.0, -5.0, -22.0,
       0.4, 19.05, -0.05, -0.22},
      {10.0, 2.0, -0.5, 0.003,
       10.3, 0.06, -7.3, -8.0,
       -28.0, 0.076, 3.0, -32.0,
       0.4, -2.0, 0.08, -0.7},
      {39.87, 4.712, -19.976, -2.4597,
       81.942, 24.8496, -2.78, -20.4191,
       402.7, 87.92, -64.26, 135.466,
       201.527, 2.3792, -139.4326, -150.6448});
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_VectorMeanOp) {
  VerifyVectorToScalarEquals(
      MakeOneInputInstruction(VECTOR_MEAN_OP),
      {-0.01, -0.2, 1.3, -0.001}, 0.27225);
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_VectorStDevOp) {
  VerifyVectorToScalarEquals(
      MakeOneInputInstruction(VECTOR_ST_DEV_OP),
      {2.2, -0.5, 0.0, 0.01},
      1.04391989635);
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_MatrixMeanOp) {
  VerifyMatrixToScalarEquals(
      MakeOneInputInstruction(MATRIX_MEAN_OP),
      {0.1, 2.5, -0.5, -2.2,
       10.3, -0.06, 0.7, -2.1,
       22.0, 5.0, -5.0, -22.0,
       0.4, 19.05, -0.05, -0.22},
      1.745);
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_MatrixStDevOp) {
  VerifyMatrixToScalarEquals(
      MakeOneInputInstruction(MATRIX_ST_DEV_OP),
      {0.1, 2.5, -0.5, -2.2,
       10.3, -0.06, 0.7, -2.1,
       22.0, 5.0, -5.0, -22.0,
       0.4, 19.05, -0.05, -0.22},
      9.5352523563878488);
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_MatrixRowMeanOp) {
  VerifyMatrixToVectorEquals(
      MakeOneInputInstruction(MATRIX_ROW_MEAN_OP),
      {0.1, 2.5, -0.5, -2.2,
       10.3, -0.06, 0.7, -2.1,
       22.0, 5.0, -5.0, -22.0,
       0.4, 19.05, -0.05, -0.22},
      {-0.025, 2.21, 0.0, 4.795});
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_MatrixRowStDevOp) {
  VerifyMatrixToVectorEquals(
      MakeOneInputInstruction(MATRIX_ROW_ST_DEV_OP),
      {0.1, 2.5, -0.5, -2.2,
       10.3, -0.06, 0.7, -2.1,
       22.0, 5.0, -5.0, -22.0,
       0.4, 19.05, -0.05, -0.22},
      {1.68430252627, 4.78166289067, 15.9530561335, 8.23324510749});
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_ScalarGaussianSetOp) {
  VerifyNothingToScalarEquals(
      MakeZeroInputsInstruction(SCALAR_GAUSSIAN_SET_OP,
                                FloatDataSetter(20.0),
                                FloatDataSetter(10.0)),
      28.042686);
  VerifyNothingToScalarIsRandomized(
      MakeZeroInputsInstruction(SCALAR_GAUSSIAN_SET_OP,
                                FloatDataSetter(20.0),
                                FloatDataSetter(10.0)));
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_VectorGaussianSetOp) {
  VerifyNothingToVectorEquals(
      MakeZeroInputsInstruction(VECTOR_GAUSSIAN_SET_OP,
                                FloatDataSetter(20.0),
                                FloatDataSetter(10.0)),
      {28.0427, 19.8492, 37.8303, 24.1638});
  VerifyNothingToVectorIsRandomized(
      MakeZeroInputsInstruction(VECTOR_GAUSSIAN_SET_OP,
                                FloatDataSetter(20.0),
                                FloatDataSetter(10.0)));
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_MatrixGaussianSetOp) {
  VerifyNothingToMatrixEquals(
      MakeZeroInputsInstruction(MATRIX_GAUSSIAN_SET_OP,
                                FloatDataSetter(0.0),
                                FloatDataSetter(0.1)),
      {0.0804269, -0.00150771, 0.178303, 0.0416377,
       0.126852, 0.0111104, -0.0138105, 0.0856213,
       0.0580157, -0.0448301, 0.164389, 0.0162463,
       -0.0230088, 0.268562, 0.0362391, -0.0849112});
  VerifyNothingToMatrixIsRandomized(
      MakeZeroInputsInstruction(MATRIX_GAUSSIAN_SET_OP,
                                FloatDataSetter(0.0),
                                FloatDataSetter(0.1)));
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_ScalarUniformSetOp) {
  VerifyNothingToScalarEquals(
      MakeZeroInputsInstruction(SCALAR_UNIFORM_SET_OP,
                                FloatDataSetter(-2.5),
                                FloatDataSetter(-2.0)),
      -2.3562196);
  VerifyNothingToScalarIsRandomized(
      MakeZeroInputsInstruction(SCALAR_UNIFORM_SET_OP,
                                FloatDataSetter(-2.5),
                                FloatDataSetter(-2.0)));
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_VectorUniformSetOp) {
  VerifyNothingToVectorEquals(
      MakeZeroInputsInstruction(VECTOR_UNIFORM_SET_OP,
                                FloatDataSetter(-2.5),
                                FloatDataSetter(-2.0)),
      {-2.35622, -2.24588, -2.25098, -2.10917});
  VerifyNothingToVectorIsRandomized(
      MakeZeroInputsInstruction(VECTOR_UNIFORM_SET_OP,
                                FloatDataSetter(-2.5),
                                FloatDataSetter(-2.0)));
}

TEST_F(ExecuteInstructionTest, ProbabilityRelated_MatrixUniformSetOp) {
  VerifyNothingToMatrixEquals(
      MakeZeroInputsInstruction(MATRIX_UNIFORM_SET_OP,
                                FloatDataSetter(-2.5),
                                FloatDataSetter(-2.0)),
      {-2.35622, -2.24588, -2.25098, -2.10917,
       -2.4388, -2.35507, -2.48641, -2.21651,
       -2.2658, -2.2554, -2.04414, -2.17942,
       -2.29889, -2.4564, -2.14704, -2.28224});
  VerifyNothingToMatrixIsRandomized(
      MakeZeroInputsInstruction(MATRIX_UNIFORM_SET_OP,
                                FloatDataSetter(-2.5),
                                FloatDataSetter(-2.0)));
}

TEST(SquashTest, MapsEndpointsCorrectly) {
  EXPECT_DOUBLE_EQ(FlipAndSquash(0.0), kMaxFitness);
  EXPECT_DOUBLE_EQ(FlipAndSquash(std::numeric_limits<double>::infinity()),
                   kMinFitness);
}

TEST(FitnessTest, StaysWithinBounds) {
  EXPECT_LT(FlipAndSquash(0.001), kMaxFitness);
  EXPECT_LT(FlipAndSquash(1.0), kMaxFitness);
  EXPECT_LT(FlipAndSquash(1000.0), kMaxFitness);
  EXPECT_GT(FlipAndSquash(0.001), kMinFitness);
  EXPECT_GT(FlipAndSquash(1.0), kMinFitness);
  EXPECT_GT(FlipAndSquash(1000.0), kMinFitness);
}

TEST(SquashTest, PreservesOrder) {
  EXPECT_LE(FlipAndSquash(0.000001), FlipAndSquash(0.0));
  EXPECT_LE(FlipAndSquash(0.000002), FlipAndSquash(0.000001));
  EXPECT_LE(FlipAndSquash(0.001), FlipAndSquash(0.000002));
  EXPECT_LE(FlipAndSquash(0.001000001), FlipAndSquash(0.001));
  EXPECT_LE(FlipAndSquash(0.1), FlipAndSquash(0.001000001));
  EXPECT_LE(FlipAndSquash(0.100001), FlipAndSquash(0.1));
  EXPECT_LE(FlipAndSquash(1.0), FlipAndSquash(0.100001));
  EXPECT_LE(FlipAndSquash(1.000001), FlipAndSquash(1.0));
  EXPECT_LE(FlipAndSquash(10.0), FlipAndSquash(1.000001));
  EXPECT_LE(FlipAndSquash(1000.0), FlipAndSquash(10.0));
  EXPECT_LE(FlipAndSquash(1000000.0), FlipAndSquash(1000.0));
  EXPECT_LE(FlipAndSquash(1000000000.0), FlipAndSquash(1000000.0));
}

TEST(SquashTest, HandlesErrors) {
  EXPECT_DOUBLE_EQ(
      FlipAndSquash(std::numeric_limits<double>::quiet_NaN()),
      kMinFitness);
}

namespace internal {

TEST(SoftmaxTest, TruncatesCorrectly) {
  Vector<16> logits;
  logits << 0.4, 0.0, -0.5, 0.1, 2.1, -0.002, -1.0, 0.0,
            1.1, 1.2, 0.003, -1.8, -0.05, -1.0, 0.0, 0.3;
  Vector<16> predictions = TruncatingSoftmax<16>(logits);
  for (IntegerT i = 0; i < 10; ++i) {
    EXPECT_GT(predictions(i), 0.0);
  }
  for (IntegerT i = 10; i < 16; ++i) {
    EXPECT_EQ(predictions(i), 0.0);
  }
}

TEST(SoftmaxTest, NormalizesCorrectly) {
  Vector<16> logits;
  logits << 0.4, 0.0, -0.5, 0.1, 2.1, -0.002, -1.0, 0.0,
            1.1, 1.2, 0.003, -1.8, -0.05, -1.0, 0.0, 0.3;
  Vector<16> predictions = TruncatingSoftmax<16>(logits);
  const double total = predictions.sum();
  EXPECT_FLOAT_EQ(total, 1.0);
}

TEST(ArgmaxTest, ComputesCorrectly) {
  Vector<4> input;
  input << 0.4, 0.0, 0.5, 0.1;
  EXPECT_EQ(Argmax<4>(input), 2);
}

}  // namespace internal

}  // namespace automl_zero
