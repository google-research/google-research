// Copyright 2020 The Google Research Authors.
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

#include "generator.h"

#include <functional>
#include <limits>
#include <random>
#include <sstream>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "algorithm_test_util.h"
#include "dataset.h"
#include "dataset_util.h"
#include "definitions.h"
#include "definitions.proto.h"
#include "evaluator.h"
#include "executor.h"
#include "generator_test_util.h"
#include "random_generator.h"
#include "random_generator_test_util.h"
#include "test_util.h"
#include "util.h"
#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::absl::StrCat;
using ::std::function;
using ::std::mt19937;
using test_only::GenerateDataset;

constexpr IntegerT kNumTrainExamples = 1000;
constexpr IntegerT kNumValidExamples = 100;
constexpr double kLargeMaxAbsError = 1000000000.0;

TEST(GeneratorTest, NoOpHasNoOpInstructions) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      10,  // setup_size_init
      12,  // predict_size_init
      13,  // learn_size_init
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  const InstructionIndexT setup_instruction_index = 2;
  const InstructionIndexT predict_instruction_index = 1;
  const InstructionIndexT learn_instruction_index = 3;
  Algorithm algorithm = generator.NoOp();
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->op_, NO_OP);
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->in1_, 0);
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->in2_, 0);
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->out_, 0);
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->GetActivationData(),
            0.0);
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->GetIndexData0(), 0);
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->GetFloatData0(), 0.0);
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->GetFloatData1(), 0.0);
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->GetFloatData2(), 0.0);
  EXPECT_EQ(algorithm.setup_[setup_instruction_index]->GetVectorData().norm(),
            0.0);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]->op_, NO_OP);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]->in1_, 0);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]->in2_, 0);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]->out_, 0);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]->GetActivationData(),
            0.0);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]->GetIndexData0(), 0);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]->GetFloatData0(),
            0.0);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]->GetFloatData1(),
            0.0);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]->GetFloatData2(),
            0.0);
  EXPECT_EQ(algorithm.predict_[predict_instruction_index]
                ->GetVectorData().norm(),
            0.0);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->op_, NO_OP);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->in1_, 0);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->in2_, 0);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->out_, 0);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->GetActivationData(),
            0.0);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->GetIndexData0(), 0);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->GetFloatData0(), 0.0);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->GetFloatData1(), 0.0);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->GetFloatData2(), 0.0);
  EXPECT_EQ(algorithm.learn_[learn_instruction_index]->GetVectorData().norm(),
            0.0);
}

TEST(GeneratorTest, NoOpProducesCorrectComponentFunctionSize) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      10,  // setup_size_init
      12,  // predict_size_init
      13,  // learn_size_init
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  Algorithm algorithm = generator.NoOp();
  EXPECT_EQ(algorithm.setup_.size(), 10);
  EXPECT_EQ(algorithm.predict_.size(), 12);
  EXPECT_EQ(algorithm.learn_.size(), 13);
}

TEST(GeneratorTest, Gz_Learns) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      10,  // setup_size_init, irrelevant
      12,  // predict_size_init, irrelevant
      13,  // learn_size_init, irrelevant
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  Dataset<4> dataset =
      GenerateDataset<4>(StrCat("scalar_linear_regression_dataset {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 100 "
                                "data_seeds: 1000 "));
  Algorithm algorithm = generator.LinearModel(kDefaultLearningRate);
  mt19937 bit_gen(10000);
  RandomGenerator rand_gen(&bit_gen);
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  double fitness = executor.Execute();
  std::cout << "Gz_Learns fitness = " << fitness << std::endl;
  EXPECT_GE(fitness, 0.0);
  EXPECT_LE(fitness, 1.0);
  EXPECT_GT(fitness, 0.999);
}

TEST(GeneratorTest, LinearModel_Learns) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      10,  // setup_size_init, irrelevant
      12,  // predict_size_init, irrelevant
      13,  // learn_size_init, irrelevant
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  Dataset<4> dataset =
      GenerateDataset<4>(StrCat("scalar_linear_regression_dataset {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 100 "
                                "data_seeds: 1000 "));
  Algorithm algorithm = generator.LinearModel(kDefaultLearningRate);
  mt19937 bit_gen(10000);
  RandomGenerator rand_gen(&bit_gen);
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  double fitness = executor.Execute();
  std::cout << "Gz_Learns fitness = " << fitness << std::endl;
  EXPECT_GE(fitness, 0.0);
  EXPECT_LE(fitness, 1.0);
  EXPECT_GT(fitness, 0.999);
}

TEST(GeneratorTest, GrTildeGrWithBias_PermanenceTest) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      0,  // setup_size_init, irrelevant.
      0,  // predict_size_init, irrelevant.
      0,  // learn_size_init, irrelevant.
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  Dataset<4> dataset = GenerateDataset<4>(StrCat(
      "scalar_2layer_nn_regression_dataset {} "
      "num_train_examples: ", kNumTrainExamples, " "
      "num_valid_examples: ", kNumValidExamples, " "
      "num_datasets: 1 "
      "eval_type: RMS_ERROR "
      "param_seeds: 1000 "
      "data_seeds: 10000 "));
  Algorithm algorithm = generator.NeuralNet(
      kDefaultLearningRate, kDefaultInitScale, kDefaultInitScale);
  mt19937 bit_gen(10000);
  RandomGenerator rand_gen(&bit_gen);
  Executor<4> executor(algorithm, dataset, kNumTrainExamples, kNumValidExamples,
                       &rand_gen, kLargeMaxAbsError);
  double fitness = executor.Execute();
  std::cout << "GrTildeGrWithBias_PermanenceTest fitness = " << fitness
            << std::endl;
  EXPECT_FLOAT_EQ(fitness, 0.80256736);
}

TEST(GeneratorTest, RandomInstructions) {
  mt19937 bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Generator generator(
      kNoOpModel,  // Irrelevant.
      2,           // setup_size_init
      4,           // predict_size_init
      5,           // learn_size_init
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      &bit_gen,  // bit_gen
      &rand_gen);  // rand_gen
  const Algorithm no_op_algorithm = generator.NoOp();
  const IntegerT total_instructions = 2 + 4 + 5;
  EXPECT_TRUE(IsEventually(
      function<IntegerT(void)>([&](){
        Algorithm random_algorithm = generator.Random();
        return CountDifferentInstructions(random_algorithm, no_op_algorithm);
      }),
      Range<IntegerT>(0, total_instructions + 1), {total_instructions}));
}

TEST(GeneratorTest, RandomInstructionsProducesCorrectComponentFunctionSizes) {
  mt19937 bit_gen;
  RandomGenerator rand_gen(&bit_gen);
  Generator generator(
      kNoOpModel,  // Irrelevant.
      2,           // setup_size_init
      4,           // predict_size_init
      5,           // learn_size_init
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP},
      &bit_gen,  // bit_gen
      &rand_gen);  // rand_gen
  Algorithm algorithm = generator.Random();
  EXPECT_EQ(algorithm.setup_.size(), 2);
  EXPECT_EQ(algorithm.predict_.size(), 4);
  EXPECT_EQ(algorithm.learn_.size(), 5);
}

TEST(GeneratorTest, GzHasCorrectComponentFunctionSizes) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      0,  // setup_size_init, no padding.
      0,  // predict_size_init, no padding.
      0,  // learn_size_init, no padding.
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  Algorithm algorithm = generator.LinearModel(kDefaultLearningRate);
  EXPECT_EQ(algorithm.setup_.size(), 1);
  EXPECT_EQ(algorithm.predict_.size(), 1);
  EXPECT_EQ(algorithm.learn_.size(), 4);
}

TEST(GeneratorTest, GzTildeGzHasCorrectComponentFunctionSizes) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      0,  // setup_size_init, no padding.
      0,  // predict_size_init, no padding.
      0,  // learn_size_init, no padding.
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  Algorithm algorithm =
      generator.UnitTestNeuralNetNoBiasNoGradient(kDefaultLearningRate);
  EXPECT_EQ(algorithm.setup_.size(), 1);
  EXPECT_EQ(algorithm.predict_.size(), 3);
  EXPECT_EQ(algorithm.learn_.size(), 9);
}

TEST(GeneratorTest, GzTildeGzPadsComponentFunctionSizesCorrectly) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      10,  // setup_size_init
      12,  // predict_size_init
      13,  // learn_size_init
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  Algorithm algorithm =
      generator.UnitTestNeuralNetNoBiasNoGradient(kDefaultLearningRate);
  EXPECT_EQ(algorithm.setup_.size(), 10);
  EXPECT_EQ(algorithm.predict_.size(), 12);
  EXPECT_EQ(algorithm.learn_.size(), 13);
}

TEST(GeneratorTest, GrTildeGrPadsComponentFunctionSizesCorrectly) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      16,  // setup_size_init
      18,  // predict_size_init
      19,  // learn_size_init
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  Algorithm algorithm = generator.NeuralNet(
      kDefaultLearningRate, kDefaultInitScale, kDefaultInitScale);
  EXPECT_EQ(algorithm.setup_.size(), 16);
  EXPECT_EQ(algorithm.predict_.size(), 18);
  EXPECT_EQ(algorithm.learn_.size(), 19);
}

TEST(GeneratorTest, GzPadsComponentFunctionSizesCorrectly) {
  Generator generator(
      kNoOpModel,  // Irrelevant.
      10,  // setup_size_init
      12,  // predict_size_init
      13,  // learn_size_init
      {},  // allowed_setup_ops, irrelevant.
      {},  // allowed_predict_ops, irrelevant.
      {},  // allowed_learn_ops, irrelevant.
      nullptr,  // bit_gen, irrelevant.
      nullptr);  // rand_gen, irrelevant.
  Algorithm algorithm = generator.LinearModel(kDefaultLearningRate);
  EXPECT_EQ(algorithm.setup_.size(), 10);
  EXPECT_EQ(algorithm.predict_.size(), 12);
  EXPECT_EQ(algorithm.learn_.size(), 13);
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
