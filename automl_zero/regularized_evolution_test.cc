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

#include "regularized_evolution.h"

#include <limits>
#include <random>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "algorithm.h"
#include "algorithm_test_util.h"
#include "dataset_util.h"
#include "definitions.h"
#include "definitions.proto.h"
#include "experiment.proto.h"
#include "generator_test_util.h"
#include "mutator.h"
#include "mutator_test_util.h"
#include "random_generator.h"
#include "test_util.h"
#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/memory/memory.h"
#include "absl/time/time.h"

namespace brain {
namespace evolution {
namespace amlz {

using ::absl::GetCurrentTimeNanos;  // NOLINT
using ::absl::make_unique;  // NOLINT
using ::absl::StrCat;  // NOLINT
using ::std::function;  // NOLINT
using ::std::make_shared;  // NOLINT
using ::std::mt19937;  // NOLINT
using ::std::numeric_limits;  // NOLINT
using ::std::shared_ptr;  // NOLINT
using ::std::string;  // NOLINT
using ::std::vector;  // NOLINT
using ::testing::Test;

constexpr RandomSeedT kEvolutionSeed = 100000;
constexpr double kFitnessTolerance = 0.0001;
constexpr IntegerT kInitMutations = 0;
constexpr IntegerT kNanosPerMilli = 1000000;
constexpr IntegerT kNumDatasetsForSearch = 2;
constexpr IntegerT kNumTrainExamplesForSearch = 1000;
constexpr IntegerT kNumValidExamplesForSearch = 100;
constexpr double kLargeMaxAbsError = 1000000000.0;

bool PopulationsEq(
    const RegularizedEvolution& regularized_evolution_1,
    const RegularizedEvolution& regularized_evolution_2);

// Required for the fitness.
TEST(doubleTest, Requirement) {
  ASSERT_TRUE(numeric_limits<double>::has_infinity);
}

TEST(RegularizedEvolutionTest, Runs) {
  mt19937 bit_gen(kEvolutionSeed);
  RandomGenerator rand_gen(&bit_gen);
  Generator generator = SimpleGenerator();
  const auto dataset_collection = ParseTextFormat<DatasetCollection>(
      StrCat("datasets { "
             "  scalar_2layer_nn_regression_dataset {} "
             "  features_size: 4 "
             "  num_train_examples: ",
             kNumTrainExamplesForSearch,
             " "
             "  num_valid_examples: ",
             kNumValidExamplesForSearch,
             " "
             "  num_datasets: ",
             kNumDatasetsForSearch,
             " "
             "  eval_type: RMS_ERROR "
             "} "));
  Evaluator evaluator(MEAN_FITNESS_COMBINATION, dataset_collection, &rand_gen,
                      nullptr,  // functional_cache
                      nullptr,  // train_budget
                      kLargeMaxAbsError,
                      false);  // verbose
  Mutator mutator = SimpleMutator();
  RegularizedEvolution regularized_evolution(
      &rand_gen,
      5,  // population_size
      2,  // tournament_size
      kInitMutations,
      kUnlimitedIndividuals,  // progress_every
      false,  // progress_every_by_time
      &generator,
      &evaluator,
      &mutator);
  regularized_evolution.Init();
  regularized_evolution.Run(20, kUnlimitedTime);
}

TEST(RegularizedEvolutionTest, TimesCorrectly) {
  mt19937 bit_gen(kEvolutionSeed);
  RandomGenerator rand_gen(&bit_gen);
  Generator generator = SimpleGenerator();
  const auto dataset_collection = ParseTextFormat<DatasetCollection>(
      StrCat("datasets { "
             "  scalar_2layer_nn_regression_dataset {} "
             "  features_size: 4 "
             "  num_train_examples: ",
             kNumTrainExamplesForSearch,
             " "
             "  num_valid_examples: ",
             kNumValidExamplesForSearch,
             " "
             "  num_datasets: ",
             kNumDatasetsForSearch,
             " "
             "  eval_type: RMS_ERROR "
             "} "));
  Evaluator evaluator(MEAN_FITNESS_COMBINATION, dataset_collection, &rand_gen,
                      nullptr,  // functional_cache
                      nullptr,  // train_budget
                      kLargeMaxAbsError,
                      false);  // verbose
  Mutator mutator = SimpleMutator();
  RegularizedEvolution regularized_evolution(
      &rand_gen,
      5,  // population_size
      2,  // tournament_size
      kInitMutations,
      kUnlimitedIndividuals,  // progress_every
      false,  // progress_every_by_time
      &generator,
      &evaluator,
      &mutator);
  regularized_evolution.Init();
  const IntegerT one_second = 1000000000;
  IntegerT start_nanos = GetCurrentTimeNanos();
  regularized_evolution.Run(kUnlimitedIndividuals, one_second);
  IntegerT elapsed_time = GetCurrentTimeNanos() - start_nanos;
  EXPECT_GE(elapsed_time, one_second);
  EXPECT_LT(elapsed_time, 2 * one_second);
}

TEST(RegularizedEvolutionTest, CountsCorrectly) {
  mt19937 bit_gen(kEvolutionSeed);
  RandomGenerator rand_gen(&bit_gen);
  Generator generator = SimpleGenerator();
  const auto dataset_collection = ParseTextFormat<DatasetCollection>(
      StrCat("datasets { "
             "  scalar_2layer_nn_regression_dataset {} "
             "  features_size: 4 "
             "  num_train_examples: ",
             kNumTrainExamplesForSearch,
             " "
             "  num_valid_examples: ",
             kNumValidExamplesForSearch,
             " "
             "  num_datasets: ",
             kNumDatasetsForSearch,
             " "
             "  eval_type: RMS_ERROR "
             "} "));

  Evaluator evaluator(MEAN_FITNESS_COMBINATION, dataset_collection, &rand_gen,
                      nullptr,  // functional_cache
                      nullptr,  // train_budget
                      kLargeMaxAbsError,
                      false);  // verbose
  Mutator mutator = SimpleMutator();
  RegularizedEvolution regularized_evolution(
      &rand_gen,
      5,  // population_size
      2,  // tournament_size
      kInitMutations,
      kUnlimitedIndividuals,  // progress_every
      false,  // progress_every_by_time
      &generator,
      &evaluator,
      &mutator);
  EXPECT_EQ(regularized_evolution.NumIndividuals(), 0);
  EXPECT_EQ(regularized_evolution.NumIndividuals(), 0);
  EXPECT_EQ(regularized_evolution.Init(), 5);
  EXPECT_EQ(regularized_evolution.NumIndividuals(), 5);
  EXPECT_EQ(regularized_evolution.NumIndividuals(), 5);
  EXPECT_EQ(regularized_evolution.Run(10, kUnlimitedTime), 10);
  EXPECT_EQ(regularized_evolution.NumIndividuals(), 15);
  EXPECT_EQ(regularized_evolution.NumIndividuals(), 15);
  EXPECT_EQ(regularized_evolution.Run(10, kUnlimitedTime), 10);
  EXPECT_EQ(regularized_evolution.NumIndividuals(), 25);
  EXPECT_EQ(regularized_evolution.NumIndividuals(), 25);
}

bool PopulationsEq(
    const RegularizedEvolution& regularized_evolution_1,
    const RegularizedEvolution& regularized_evolution_2) {
  if (regularized_evolution_1.population_size_ !=
      regularized_evolution_2.population_size_) {
    return false;
  }

  if (regularized_evolution_1.num_individuals_ !=
      regularized_evolution_2.num_individuals_) {
    return false;
  }

  CHECK_EQ(regularized_evolution_1.algorithms_.size(),
           regularized_evolution_1.fitnesses_.size());
  CHECK_EQ(regularized_evolution_2.algorithms_.size(),
           regularized_evolution_2.fitnesses_.size());
  if (regularized_evolution_1.algorithms_.size() !=
      regularized_evolution_2.algorithms_.size()) {
    return false;
  }

  vector<shared_ptr<const Algorithm>>::const_iterator algorithms1_it =
      regularized_evolution_1.algorithms_.begin();
  for (shared_ptr<const Algorithm> algorithm2 :
       regularized_evolution_2.algorithms_) {
    if (**algorithms1_it != *algorithm2) {
      return false;
    }
    ++algorithms1_it;
  }
  CHECK(algorithms1_it == regularized_evolution_1.algorithms_.end());

  std::vector<double>::const_iterator fitnesses1_it =
      regularized_evolution_1.fitnesses_.begin();
  for (const double& fitness2 : regularized_evolution_2.fitnesses_) {
    if (std::abs(*fitnesses1_it - fitness2) > kFitnessTolerance) {
      return false;
    }
    ++fitnesses1_it;
  }
  CHECK(fitnesses1_it == regularized_evolution_1.fitnesses_.end());

  return true;
}

IntegerT GetsFromPosition(
    RegularizedEvolution* regularized_evolution) {
  double fitness;
  shared_ptr<const Algorithm> algorithm = regularized_evolution->Get(&fitness);
  return static_cast<IntegerT>(algorithm->predict_[0]->GetIntegerData());
}

}  // namespace amlz
}  // namespace evolution
}  // namespace brain
