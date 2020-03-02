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

#include "evaluator.h"

#include <functional>

#include "algorithm.h"
#include "dataset.h"
#include "dataset_util.h"
#include "datasets.proto.h"
#include "definitions.h"
#include "executor.h"
#include "generator.h"
#include "generator_test_util.h"
#include "random_generator.h"
#include "random_generator_test_util.h"
#include "test_util.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "util/random/mt_random.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::absl::StrCat;  // NOLINT
using ::std::function;  // NOLINT
using ::std::min;  // NOLINT
using test_only::GenerateDataset;

constexpr IntegerT kNumTrainExamples = 1000;
constexpr IntegerT kNumValidExamples = 100;
constexpr IntegerT kNumDatasets = 2;
constexpr double kNumericTolerance = 0.0000001;
constexpr double kMaxAbsError = 100.0;

TEST(EvaluatorTest, AveragesOverDatasets) {
  Dataset<4> dataset_one =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1001 "
                                "data_seeds: 11001 "));
  Dataset<4> dataset_two =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1012 "
                                "data_seeds: 11012 "));

  Algorithm algorithm = SimpleGz();

  // Run with datasets independently.
  MTRandom bit_gen_sep(100000);
  RandomGenerator rand_gen_sep(&bit_gen_sep);
  Executor<4> executor0(
      algorithm, dataset_one, kNumTrainExamples, kNumValidExamples,
      &rand_gen_sep, kMaxAbsError);
  const double fitness0 = executor0.Execute();
  Executor<4> executor1(
      algorithm, dataset_two, kNumTrainExamples, kNumValidExamples,
      &rand_gen_sep, kMaxAbsError);
  const double fitness1 = executor1.Execute();
  const double expected_fitness = (fitness0 + fitness1) / 2.0;

  MTRandom bit_gen(100000);
  RandomGenerator rand_gen(&bit_gen);
  const auto dataset_collection = ParseTextFormat<DatasetCollection>(
      StrCat("datasets { "
             "  scalar_2layer_nn_regression_dataset {} "
             "  features_size: 4 "
             "  num_train_examples: ",
             kNumTrainExamples,
             " "
             "  num_valid_examples: ",
             kNumValidExamples,
             " "
             "  num_datasets: ",
             kNumDatasets,
             " "
             "  eval_type: RMS_ERROR "
             "} "));
  Evaluator evaluator(MEAN_FITNESS_COMBINATION, dataset_collection, &rand_gen,
                      nullptr,  // functional_cache
                      nullptr,  // train_budget
                      kMaxAbsError,
                      false);  // verbose
  const double fitness = evaluator.Evaluate(algorithm);

  EXPECT_FLOAT_EQ(fitness, expected_fitness);
}

TEST(EvaluatorTest, GrTildeGrWithBiasHasHighFitness) {
  MTRandom bit_gen(100000);
  RandomGenerator rand_gen(&bit_gen);
  Generator generator = SimpleGenerator();
  Algorithm algorithm = generator.NeuralNet(0.036210, 0.180920, 0.145231);

  const auto dataset_collection = ParseTextFormat<DatasetCollection>(
      StrCat("datasets { "
             "  scalar_2layer_nn_regression_dataset {} "
             "  features_size: 4 "
             "  num_train_examples: 10000 "
             "  num_valid_examples: ",
             kNumValidExamples,
             " "
             "  num_datasets: 1 "
             "  eval_type: RMS_ERROR "
             "} "));

  Evaluator evaluator(MEAN_FITNESS_COMBINATION, dataset_collection,
                      &rand_gen,  // random_seed
                      nullptr,    // functional_cache
                      nullptr,  // train_budget
                      kMaxAbsError,
                      false);  // verbose
  const double fitness = evaluator.Evaluate(algorithm);
  EXPECT_FLOAT_EQ(fitness, 0.99069703);
}

namespace internal {

TEST(CombineFitnessesTest, MeanWorksCorrectly) {
  EXPECT_EQ(
      CombineFitnesses({0.4, 0.2, 0.6, 0.8}, MEAN_FITNESS_COMBINATION), 0.5);
}

TEST(CombineFitnessesTest, MedianWorksCorrectly) {
  EXPECT_EQ(
      CombineFitnesses({0.9, 0.12, 0.13, 0.1, 0.11},
                       MEDIAN_FITNESS_COMBINATION),
      0.12);
}

}  // namespace internal

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
