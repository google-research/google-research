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

#include "task_util.h"

#include <cmath>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "algorithm.h"
#include "task.h"
#include "task.proto.h"
#include "definitions.h"
#include "executor.h"
#include "generator.h"
#include "generator_test_util.h"
#include "memory.h"
#include "random_generator.h"
#include "random_generator_test_util.h"
#include "test_util.h"
#include "testing/base/public/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "third_party/eigen3/Eigen/Core"

namespace automl_zero {

using ::absl::StrCat;  // NOLINT
using ::Eigen::Map;
using ::std::abs;  // NOLINT
using ::std::function;  // NOLINT
using ::std::make_pair;  // NOLINT
using ::std::pair;  // NOLINT
using ::std::vector;  // NOLINT
using ::std::unique_ptr;  // NOLINT
using ::std::unordered_set;  // NOLINT
using ::testing::Test;
using test_only::GenerateTask;

constexpr IntegerT kNumTrainExamples = 1000;
constexpr IntegerT kNumValidExamples = 100;
constexpr double kLargeMaxAbsError = 1000000000.0;
constexpr IntegerT kNumAllTrainExamples = 1000;

bool ScalarEq(const Scalar& scalar1, const Scalar scalar2) {
  return abs(scalar1 - scalar2) < kDataTolerance;
}

template <FeatureIndexT F>
bool VectorEq(const Vector<F>& vector1,
              const ::std::vector<double>& vector2) {
  Map<const Vector<F>> vector2_eigen(vector2.data());
  return (vector1 - vector2_eigen).norm() < kDataTolerance;
}

// Scalar and Vector are trivially destructible.
const Vector<4> kZeroVector = Vector<4>::Zero(4, 1);
const Vector<4> kOnesVector = Vector<4>::Ones(4, 1);

TEST(FillTasksTest, WorksCorrectly) {
  Task<4> expected_task_0 =
      GenerateTask<4>(StrCat("scalar_2layer_nn_regression_task {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1000 "
                                "data_seeds: 10000 "));
  Task<4> expected_task_1 =
      GenerateTask<4>(StrCat("scalar_2layer_nn_regression_task {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1001 "
                                "data_seeds: 10001 "));

  TaskCollection task_collection;
  TaskSpec* task = task_collection.add_tasks();
  task->set_features_size(4);
  task->set_num_train_examples(kNumTrainExamples);
  task->set_num_valid_examples(kNumValidExamples);
  task->set_num_tasks(2);
  task->set_eval_type(RMS_ERROR);
  task->add_data_seeds(10000);
  task->add_data_seeds(10001);
  task->add_param_seeds(1000);
  task->add_param_seeds(1001);
  task->mutable_scalar_2layer_nn_regression_task();
  vector<unique_ptr<TaskInterface>> owned_tasks;
  FillTasks(task_collection, &owned_tasks);
  vector<Task<4>*> tasks;
  for (const unique_ptr<TaskInterface>& owned_dataset : owned_tasks) {
    tasks.push_back(SafeDowncast<4>(owned_dataset.get()));
  }

  EXPECT_EQ(tasks[0]->MaxTrainExamples(), kNumTrainExamples);

  EXPECT_LT(
      (tasks[0]->train_features_[478] -
       expected_task_0.train_features_[478]).norm(),
      kDataTolerance);
  EXPECT_LT(
      (tasks[1]->train_features_[478] -
       expected_task_1.train_features_[478]).norm(),
      kDataTolerance);
  EXPECT_LT(
      (tasks[0]->valid_features_[94] -
       expected_task_0.valid_features_[94]).norm(),
      kDataTolerance);
  EXPECT_LT(
      (tasks[1]->valid_features_[94] -
       expected_task_1.valid_features_[94]).norm(),
      kDataTolerance);

  EXPECT_LT(
      abs(tasks[0]->train_labels_[478] -
          expected_task_0.train_labels_[478]),
      kDataTolerance);
  EXPECT_LT(
      abs(tasks[1]->train_labels_[478] -
          expected_task_1.train_labels_[478]),
      kDataTolerance);
  EXPECT_LT(
      abs(tasks[0]->valid_labels_[94] -
          expected_task_0.valid_labels_[94]),
      kDataTolerance);
  EXPECT_LT(
      abs(tasks[1]->valid_labels_[94] -
          expected_task_1.valid_labels_[94]),
      kDataTolerance);

  EXPECT_EQ(tasks[0]->index_, 0);
  EXPECT_EQ(tasks[1]->index_, 1);
}

TEST(FillTaskTest, FillsEvalType) {
  std::string task_spec_string =
      StrCat("scalar_linear_regression_task {} "
             "num_train_examples: 1000 "
             "num_valid_examples: 100 "
             "param_seeds: 1000 "
             "data_seeds: 1000 "
             "eval_type: RMS_ERROR");
  Task<4> dataset = GenerateTask<4>(task_spec_string);
  TaskSpec task_spec =
      ParseTextFormat<TaskSpec>(task_spec_string);
  EXPECT_EQ(dataset.eval_type_, task_spec.eval_type());
}

TEST(FillTaskWithZerosTest, WorksCorrectly) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_zeros_task {} "
                                "eval_type: ACCURACY "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));
  for (const Scalar& label : dataset.train_labels_) {
    EXPECT_FLOAT_EQ(label, 0.0);
  }
  for (const Vector<4>& feature : dataset.valid_features_) {
    EXPECT_TRUE(feature.isApprox(kZeroVector));
  }
  for (const Scalar& label : dataset.valid_labels_) {
    EXPECT_FLOAT_EQ(label, 0.0);
  }
}

TEST(FillTaskWithOnesTest, WorksCorrectly) {
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
  for (const Vector<4>& feature : dataset.train_features_) {
    EXPECT_TRUE(feature.isApprox(kOnesVector));
  }
  for (const Scalar& label : dataset.train_labels_) {
    EXPECT_FLOAT_EQ(label, 1.0);
  }
  for (const Vector<4>& feature : dataset.valid_features_) {
    EXPECT_TRUE(feature.isApprox(kOnesVector));
  }
  for (const Scalar& label : dataset.valid_labels_) {
    EXPECT_FLOAT_EQ(label, 1.0);
  }
}

TEST(FillTaskWithIncrementingIntegersTest, WorksCorrectly) {
  auto dataset =
      GenerateTask<4>(StrCat("unit_test_increment_task {} "
                                "eval_type: ACCURACY "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_tasks: 1 "
                                "features_size: 4 "));

  EXPECT_TRUE(dataset.train_features_[0].isApprox(kZeroVector));
  EXPECT_TRUE(
      dataset.train_features_[kNumTrainExamples - 1].isApprox(
          kOnesVector * static_cast<double>(kNumTrainExamples - 1)));

  EXPECT_FLOAT_EQ(dataset.train_labels_[0], 0.0);
  EXPECT_FLOAT_EQ(
      dataset.train_labels_[kNumTrainExamples - 1],
      static_cast<double>(kNumTrainExamples - 1));

  EXPECT_TRUE(dataset.valid_features_[0].isApprox(kZeroVector));
  EXPECT_TRUE(
      dataset.valid_features_[kNumValidExamples - 1].isApprox(
          kOnesVector * static_cast<double>(kNumValidExamples - 1)));

  EXPECT_FLOAT_EQ(dataset.valid_labels_[0], 0.0);
  EXPECT_FLOAT_EQ(
      dataset.valid_labels_[kNumValidExamples - 1],
      static_cast<double>(kNumValidExamples - 1));
}

TEST(FillTaskWithNonlinearDataTest, DifferentForDifferentSeeds) {
  Task<4> dataset_1000_10000 =
      GenerateTask<4>(StrCat("scalar_2layer_nn_regression_task {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1000 "
                                "data_seeds: 10000 "));
  Task<4> dataset_1001_10000 =
      GenerateTask<4>(StrCat("scalar_2layer_nn_regression_task {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1001 "
                                "data_seeds: 10000 "));
  Task<4> dataset_1000_10001 =
      GenerateTask<4>(StrCat("scalar_2layer_nn_regression_task {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1000 "
                                "data_seeds: 10001 "));
  EXPECT_NE(dataset_1000_10000, dataset_1001_10000);
  EXPECT_NE(dataset_1000_10000, dataset_1000_10001);
  EXPECT_NE(dataset_1001_10000, dataset_1000_10001);
}

TEST(FillTaskWithNonlinearDataTest, SameForSameSeed) {
  Task<4> dataset_1000_10000_a =
      GenerateTask<4>(StrCat("scalar_2layer_nn_regression_task {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1000 "
                                "data_seeds: 10000 "));
  Task<4> dataset_1000_10000_b =
      GenerateTask<4>(StrCat("scalar_2layer_nn_regression_task {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1000 "
                                "data_seeds: 10000 "));
  EXPECT_EQ(dataset_1000_10000_a, dataset_1000_10000_b);
}

TEST(FillTaskWithNonlinearDataTest, PermanenceTest) {
  Task<4> dataset =
      GenerateTask<4>(StrCat("scalar_2layer_nn_regression_task {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1000 "
                                "data_seeds: 10000 "));
  EXPECT_TRUE(VectorEq<4>(
      dataset.train_features_[0],
      {1.30836, -0.192507, 0.549877, -0.667065}));
  EXPECT_TRUE(VectorEq<4>(
      dataset.train_features_[994],
      {-0.265714, 1.38325, 0.775253, 1.78923}));
  EXPECT_TRUE(VectorEq<4>(
      dataset.valid_features_[0],
      {1.39658, 0.293097, -0.504938, -1.09144}));
  EXPECT_TRUE(VectorEq<4>(
      dataset.valid_features_[94],
      {-0.224309, 1.78054, 1.24783, 0.54083}));
  EXPECT_TRUE(ScalarEq(dataset.train_labels_[0], 1.508635));
  EXPECT_TRUE(ScalarEq(dataset.train_labels_[994], -2.8410525));
  EXPECT_TRUE(ScalarEq(dataset.valid_labels_[0], 0.0));
  EXPECT_TRUE(ScalarEq(dataset.valid_labels_[98], -0.66133333));
}

void ClearSeeds(TaskCollection* task_collection) {
  for (TaskSpec& dataset : *task_collection->mutable_tasks()) {
    dataset.clear_param_seeds();
    dataset.clear_data_seeds();
  }
}

TEST(RandomizeTaskSeedsTest, FillsCorrectNumberOfRandomSeeds) {
  auto task_collection = ParseTextFormat<TaskCollection>(
      "tasks {num_tasks: 8} "
      "tasks {num_tasks: 3} ");

  RandomizeTaskSeeds(&task_collection, GenerateRandomSeed());
  EXPECT_EQ(task_collection.tasks_size(), 2);
  EXPECT_EQ(task_collection.tasks(0).param_seeds_size(), 8);
  EXPECT_EQ(task_collection.tasks(0).data_seeds_size(), 8);
  EXPECT_EQ(task_collection.tasks(1).param_seeds_size(), 3);
  EXPECT_EQ(task_collection.tasks(1).data_seeds_size(), 3);
}

TEST(RandomizeTaskSeedsTest, SameForSameSeed) {
  const RandomSeedT seed = GenerateRandomSeed();
  auto task_collection_1 = ParseTextFormat<TaskCollection>(
      "tasks {num_tasks: 8} "
      "tasks {num_tasks: 3} ");
  TaskCollection task_collection_2 = task_collection_1;
  RandomizeTaskSeeds(&task_collection_1, seed);
  RandomizeTaskSeeds(&task_collection_2, seed);
  EXPECT_EQ(task_collection_1.tasks(0).param_seeds(5),
            task_collection_2.tasks(0).param_seeds(5));
  EXPECT_EQ(task_collection_1.tasks(0).data_seeds(5),
            task_collection_2.tasks(0).data_seeds(5));
  EXPECT_EQ(task_collection_1.tasks(1).param_seeds(1),
            task_collection_2.tasks(1).param_seeds(1));
  EXPECT_EQ(task_collection_1.tasks(1).data_seeds(1),
            task_collection_2.tasks(1).data_seeds(1));
}

TEST(RandomizeTaskSeedsTest, DifferentForDifferentSeeds) {
  const RandomSeedT seed1 = 519801251;
  const RandomSeedT seed2 = 208594758;
  auto task_collection_1 = ParseTextFormat<TaskCollection>(
      "tasks {num_tasks: 8} "
      "tasks {num_tasks: 3} ");
  TaskCollection task_collection_2 = task_collection_1;
  RandomizeTaskSeeds(&task_collection_1, seed1);
  RandomizeTaskSeeds(&task_collection_2, seed2);
  EXPECT_NE(task_collection_1.tasks(0).param_seeds(5),
            task_collection_2.tasks(0).param_seeds(5));
  EXPECT_NE(task_collection_1.tasks(0).data_seeds(5),
            task_collection_2.tasks(0).data_seeds(5));
  EXPECT_NE(task_collection_1.tasks(1).param_seeds(1),
            task_collection_2.tasks(1).param_seeds(1));
  EXPECT_NE(task_collection_1.tasks(1).data_seeds(1),
            task_collection_2.tasks(1).data_seeds(1));
}

TEST(RandomizeTaskSeedsTest, CoversParamSeeds) {
  IntegerT num_tasks = 0;
  auto task_collection = ParseTextFormat<TaskCollection>("tasks {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<RandomSeedT(void)>([&task_collection, &num_tasks, seed]() {
        // We need to keep increasing the number of tasks in order to
        // generate new seeds because the RandomizeTaskSeeds function is
        // deterministic.
        ++num_tasks;
        task_collection.mutable_tasks(0)->set_num_tasks(num_tasks);

        ClearSeeds(&task_collection);
        RandomizeTaskSeeds(&task_collection, seed);
        const RandomSeedT param_seed =
            *task_collection.tasks(0).param_seeds().rbegin();
        return (param_seed % 5);
      }),
      Range<RandomSeedT>(0, 5), Range<RandomSeedT>(0, 5)));
}

TEST(RandomizeTaskSeedsTest, CoversDataSeeds) {
  IntegerT num_tasks = 0;
  auto task_collection = ParseTextFormat<TaskCollection>("tasks {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<RandomSeedT(void)>([&task_collection, &num_tasks, seed]() {
        ++num_tasks;
        task_collection.mutable_tasks(0)->set_num_tasks(num_tasks);
        ClearSeeds(&task_collection);

        // Return the last seed.
        RandomizeTaskSeeds(&task_collection, seed);
        const RandomSeedT data_seed =
            *task_collection.tasks(0).data_seeds().rbegin();
        return (data_seed % 5);
      }),
      Range<RandomSeedT>(0, 5), Range<RandomSeedT>(0, 5)));
}

TEST(RandomizeTaskSeedsTest, ParamAndDataSeedsAreIndependent) {
  IntegerT num_tasks = 0;
  auto task_collection = ParseTextFormat<TaskCollection>("tasks {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&task_collection,
                                                      &num_tasks, seed]() {
        ++num_tasks;
        task_collection.mutable_tasks(0)->set_num_tasks(num_tasks);
        ClearSeeds(&task_collection);
        RandomizeTaskSeeds(&task_collection, seed);

        // Return the last data seed and the last param seed.
        const RandomSeedT param_seed =
            *task_collection.tasks(0).param_seeds().rbegin();
        const RandomSeedT data_seed =
            *task_collection.tasks(0).data_seeds().rbegin();
        return (make_pair(param_seed % 3, data_seed % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

TEST(RandomizeTaskSeedsTest, ParamSeedsAreIndepdendentWithinTaskSpec) {
  IntegerT num_tasks = 1;
  auto task_collection = ParseTextFormat<TaskCollection>("tasks {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&task_collection,
                                                      &num_tasks, seed]() {
        ++num_tasks;
        task_collection.mutable_tasks(0)->set_num_tasks(num_tasks);
        ClearSeeds(&task_collection);
        RandomizeTaskSeeds(&task_collection, seed);

        // Return the last two seeds.
        auto param_seed_it =
            task_collection.tasks(0).param_seeds().rbegin();
        const RandomSeedT param_seed_1 = *param_seed_it;
        ++param_seed_it;
        const RandomSeedT param_seed_2 = *param_seed_it;
        return (make_pair(param_seed_1 % 3, param_seed_2 % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

TEST(RandomizeTaskSeedsTest, DataSeedsAreIndepdendentWithinTaskSpec) {
  IntegerT num_tasks = 1;
  auto task_collection = ParseTextFormat<TaskCollection>("tasks {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&task_collection,
                                                      &num_tasks, seed]() {
        ++num_tasks;
        task_collection.mutable_tasks(0)->set_num_tasks(num_tasks);
        ClearSeeds(&task_collection);
        RandomizeTaskSeeds(&task_collection, seed);

        // Return the last two seeds.
        auto data_seed_it =
            task_collection.tasks(0).data_seeds().rbegin();
        const RandomSeedT data_seed_1 = *data_seed_it;
        ++data_seed_it;
        const RandomSeedT data_seed_2 = *data_seed_it;
        return (make_pair(data_seed_1 % 3, data_seed_2 % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

TEST(RandomizeTaskSeedsTest, ParamSeedsAreIndepdendentAcrossTaskSpecs) {
  IntegerT num_tasks = 1;
  auto task_collection = ParseTextFormat<TaskCollection>(
      "tasks {} "
      "tasks {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&task_collection,
                                                      &num_tasks, seed]() {
        ++num_tasks;
        task_collection.mutable_tasks(0)->set_num_tasks(num_tasks);
        task_collection.mutable_tasks(1)->set_num_tasks(num_tasks);
        ClearSeeds(&task_collection);
        RandomizeTaskSeeds(&task_collection, seed);

        // Return the last seed of each TaskSpec.
        const RandomSeedT param_seed_1 =
            *task_collection.tasks(0).param_seeds().rbegin();
        const RandomSeedT param_seed_2 =
            *task_collection.tasks(1).param_seeds().rbegin();
        return (make_pair(param_seed_1 % 3, param_seed_2 % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

TEST(RandomizeTaskSeedsTest, DataSeedsAreIndepdendentAcrossTaskSpecs) {
  IntegerT num_tasks = 1;
  auto task_collection = ParseTextFormat<TaskCollection>(
      "tasks {} "
      "tasks {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&task_collection,
                                                      &num_tasks, seed]() {
        ++num_tasks;
        task_collection.mutable_tasks(0)->set_num_tasks(num_tasks);
        task_collection.mutable_tasks(1)->set_num_tasks(num_tasks);
        ClearSeeds(&task_collection);
        RandomizeTaskSeeds(&task_collection, seed);

        // Return the last seed of each TaskSpec.
        const RandomSeedT data_seed_1 =
            *task_collection.tasks(0).data_seeds().rbegin();
        const RandomSeedT data_seed_2 =
            *task_collection.tasks(1).data_seeds().rbegin();
        return (make_pair(data_seed_1 % 3, data_seed_2 % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

}  // namespace automl_zero
