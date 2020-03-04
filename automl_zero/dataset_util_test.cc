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

#include "dataset_util.h"

#include <cmath>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "algorithm.h"
#include "dataset.h"
#include "datasets.proto.h"
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
using test_only::GenerateDataset;

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

TEST(FillDatasetsTest, WorksCorrectly) {
  Dataset<4> expected_dataset_0 =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1000 "
                                "data_seeds: 10000 "));
  Dataset<4> expected_dataset_1 =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1001 "
                                "data_seeds: 10001 "));

  DatasetCollection dataset_collection;
  DatasetSpec* dataset = dataset_collection.add_datasets();
  dataset->set_features_size(4);
  dataset->set_num_train_examples(kNumTrainExamples);
  dataset->set_num_valid_examples(kNumValidExamples);
  dataset->set_num_datasets(2);
  dataset->set_eval_type(RMS_ERROR);
  dataset->add_data_seeds(10000);
  dataset->add_data_seeds(10001);
  dataset->add_param_seeds(1000);
  dataset->add_param_seeds(1001);
  dataset->mutable_scalar_2layer_nn_regression_dataset();
  vector<unique_ptr<DatasetInterface>> owned_datasets;
  FillDatasets(dataset_collection, &owned_datasets);
  vector<Dataset<4>*> datasets;
  for (const unique_ptr<DatasetInterface>& owned_dataset : owned_datasets) {
    datasets.push_back(SafeDowncast<4>(owned_dataset.get()));
  }

  EXPECT_EQ(datasets[0]->MaxTrainExamples(), kNumTrainExamples);

  EXPECT_LT(
      (datasets[0]->train_features_[478] -
       expected_dataset_0.train_features_[478]).norm(),
      kDataTolerance);
  EXPECT_LT(
      (datasets[1]->train_features_[478] -
       expected_dataset_1.train_features_[478]).norm(),
      kDataTolerance);
  EXPECT_LT(
      (datasets[0]->valid_features_[94] -
       expected_dataset_0.valid_features_[94]).norm(),
      kDataTolerance);
  EXPECT_LT(
      (datasets[1]->valid_features_[94] -
       expected_dataset_1.valid_features_[94]).norm(),
      kDataTolerance);

  EXPECT_LT(
      abs(datasets[0]->train_labels_[478] -
          expected_dataset_0.train_labels_[478]),
      kDataTolerance);
  EXPECT_LT(
      abs(datasets[1]->train_labels_[478] -
          expected_dataset_1.train_labels_[478]),
      kDataTolerance);
  EXPECT_LT(
      abs(datasets[0]->valid_labels_[94] -
          expected_dataset_0.valid_labels_[94]),
      kDataTolerance);
  EXPECT_LT(
      abs(datasets[1]->valid_labels_[94] -
          expected_dataset_1.valid_labels_[94]),
      kDataTolerance);

  EXPECT_EQ(datasets[0]->index_, 0);
  EXPECT_EQ(datasets[1]->index_, 1);
}

TEST(FillDatasetTest, FillsEvalType) {
  std::string dataset_spec_string =
      StrCat("scalar_linear_regression_dataset {} "
             "num_train_examples: 1000 "
             "num_valid_examples: 100 "
             "param_seeds: 1000 "
             "data_seeds: 1000 "
             "eval_type: RMS_ERROR");
  Dataset<4> dataset = GenerateDataset<4>(dataset_spec_string);
  DatasetSpec dataset_spec =
      ParseTextFormat<DatasetSpec>(dataset_spec_string);
  EXPECT_EQ(dataset.eval_type_, dataset_spec.eval_type());
}

TEST(FillDatasetWithZerosTest, WorksCorrectly) {
  auto dataset =
      GenerateDataset<4>(StrCat("unit_test_zeros_dataset {} "
                                "eval_type: ACCURACY "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_datasets: 1 "
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

TEST(FillDatasetWithOnesTest, WorksCorrectly) {
  auto dataset =
      GenerateDataset<4>(StrCat("unit_test_ones_dataset {} "
                                "eval_type: ACCURACY "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_datasets: 1 "
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

TEST(FillDatasetWithIncrementingIntegersTest, WorksCorrectly) {
  auto dataset =
      GenerateDataset<4>(StrCat("unit_test_increment_dataset {} "
                                "eval_type: ACCURACY "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "num_datasets: 1 "
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

TEST(FillDatasetWithNonlinearDataTest, DifferentForDifferentSeeds) {
  Dataset<4> dataset_1000_10000 =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1000 "
                                "data_seeds: 10000 "));
  Dataset<4> dataset_1001_10000 =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1001 "
                                "data_seeds: 10000 "));
  Dataset<4> dataset_1000_10001 =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
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

TEST(FillDatasetWithNonlinearDataTest, SameForSameSeed) {
  Dataset<4> dataset_1000_10000_a =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
                                "num_train_examples: ",
                                kNumTrainExamples,
                                " "
                                "num_valid_examples: ",
                                kNumValidExamples,
                                " "
                                "eval_type: RMS_ERROR "
                                "param_seeds: 1000 "
                                "data_seeds: 10000 "));
  Dataset<4> dataset_1000_10000_b =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
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

TEST(FillDatasetWithNonlinearDataTest, PermanenceTest) {
  Dataset<4> dataset =
      GenerateDataset<4>(StrCat("scalar_2layer_nn_regression_dataset {} "
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

void ClearSeeds(DatasetCollection* dataset_collection) {
  for (DatasetSpec& dataset : *dataset_collection->mutable_datasets()) {
    dataset.clear_param_seeds();
    dataset.clear_data_seeds();
  }
}

TEST(RandomizeDatasetSeedsTest, FillsCorrectNumberOfRandomSeeds) {
  auto dataset_collection = ParseTextFormat<DatasetCollection>(
      "datasets {num_datasets: 8} "
      "datasets {num_datasets: 3} ");

  RandomizeDatasetSeeds(&dataset_collection, GenerateRandomSeed());
  EXPECT_EQ(dataset_collection.datasets_size(), 2);
  EXPECT_EQ(dataset_collection.datasets(0).param_seeds_size(), 8);
  EXPECT_EQ(dataset_collection.datasets(0).data_seeds_size(), 8);
  EXPECT_EQ(dataset_collection.datasets(1).param_seeds_size(), 3);
  EXPECT_EQ(dataset_collection.datasets(1).data_seeds_size(), 3);
}

TEST(RandomizeDatasetSeedsTest, SameForSameSeed) {
  const RandomSeedT seed = GenerateRandomSeed();
  auto dataset_collection_1 = ParseTextFormat<DatasetCollection>(
      "datasets {num_datasets: 8} "
      "datasets {num_datasets: 3} ");
  DatasetCollection dataset_collection_2 = dataset_collection_1;
  RandomizeDatasetSeeds(&dataset_collection_1, seed);
  RandomizeDatasetSeeds(&dataset_collection_2, seed);
  EXPECT_EQ(dataset_collection_1.datasets(0).param_seeds(5),
            dataset_collection_2.datasets(0).param_seeds(5));
  EXPECT_EQ(dataset_collection_1.datasets(0).data_seeds(5),
            dataset_collection_2.datasets(0).data_seeds(5));
  EXPECT_EQ(dataset_collection_1.datasets(1).param_seeds(1),
            dataset_collection_2.datasets(1).param_seeds(1));
  EXPECT_EQ(dataset_collection_1.datasets(1).data_seeds(1),
            dataset_collection_2.datasets(1).data_seeds(1));
}

TEST(RandomizeDatasetSeedsTest, DifferentForDifferentSeeds) {
  const RandomSeedT seed1 = 519801251;
  const RandomSeedT seed2 = 208594758;
  auto dataset_collection_1 = ParseTextFormat<DatasetCollection>(
      "datasets {num_datasets: 8} "
      "datasets {num_datasets: 3} ");
  DatasetCollection dataset_collection_2 = dataset_collection_1;
  RandomizeDatasetSeeds(&dataset_collection_1, seed1);
  RandomizeDatasetSeeds(&dataset_collection_2, seed2);
  EXPECT_NE(dataset_collection_1.datasets(0).param_seeds(5),
            dataset_collection_2.datasets(0).param_seeds(5));
  EXPECT_NE(dataset_collection_1.datasets(0).data_seeds(5),
            dataset_collection_2.datasets(0).data_seeds(5));
  EXPECT_NE(dataset_collection_1.datasets(1).param_seeds(1),
            dataset_collection_2.datasets(1).param_seeds(1));
  EXPECT_NE(dataset_collection_1.datasets(1).data_seeds(1),
            dataset_collection_2.datasets(1).data_seeds(1));
}

TEST(RandomizeDatasetSeedsTest, CoversParamSeeds) {
  IntegerT num_datasets = 0;
  auto dataset_collection = ParseTextFormat<DatasetCollection>("datasets {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<RandomSeedT(void)>([&dataset_collection, &num_datasets, seed]() {
        // We need to keep increasing the number of datasets in order to
        // generate new seeds because the RandomizeDatasetSeeds function is
        // deterministic.
        ++num_datasets;
        dataset_collection.mutable_datasets(0)->set_num_datasets(num_datasets);

        ClearSeeds(&dataset_collection);
        RandomizeDatasetSeeds(&dataset_collection, seed);
        const RandomSeedT param_seed =
            *dataset_collection.datasets(0).param_seeds().rbegin();
        return (param_seed % 5);
      }),
      Range<RandomSeedT>(0, 5), Range<RandomSeedT>(0, 5)));
}

TEST(RandomizeDatasetSeedsTest, CoversDataSeeds) {
  IntegerT num_datasets = 0;
  auto dataset_collection = ParseTextFormat<DatasetCollection>("datasets {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<RandomSeedT(void)>([&dataset_collection, &num_datasets, seed]() {
        ++num_datasets;
        dataset_collection.mutable_datasets(0)->set_num_datasets(num_datasets);
        ClearSeeds(&dataset_collection);

        // Return the last seed.
        RandomizeDatasetSeeds(&dataset_collection, seed);
        const RandomSeedT data_seed =
            *dataset_collection.datasets(0).data_seeds().rbegin();
        return (data_seed % 5);
      }),
      Range<RandomSeedT>(0, 5), Range<RandomSeedT>(0, 5)));
}

TEST(RandomizeDatasetSeedsTest, ParamAndDataSeedsAreIndependent) {
  IntegerT num_datasets = 0;
  auto dataset_collection = ParseTextFormat<DatasetCollection>("datasets {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&dataset_collection,
                                                      &num_datasets, seed]() {
        ++num_datasets;
        dataset_collection.mutable_datasets(0)->set_num_datasets(num_datasets);
        ClearSeeds(&dataset_collection);
        RandomizeDatasetSeeds(&dataset_collection, seed);

        // Return the last data seed and the last param seed.
        const RandomSeedT param_seed =
            *dataset_collection.datasets(0).param_seeds().rbegin();
        const RandomSeedT data_seed =
            *dataset_collection.datasets(0).data_seeds().rbegin();
        return (make_pair(param_seed % 3, data_seed % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

TEST(RandomizeDatasetSeedsTest, ParamSeedsAreIndepdendentWithinDatasetSpec) {
  IntegerT num_datasets = 1;
  auto dataset_collection = ParseTextFormat<DatasetCollection>("datasets {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&dataset_collection,
                                                      &num_datasets, seed]() {
        ++num_datasets;
        dataset_collection.mutable_datasets(0)->set_num_datasets(num_datasets);
        ClearSeeds(&dataset_collection);
        RandomizeDatasetSeeds(&dataset_collection, seed);

        // Return the last two seeds.
        auto param_seed_it =
            dataset_collection.datasets(0).param_seeds().rbegin();
        const RandomSeedT param_seed_1 = *param_seed_it;
        ++param_seed_it;
        const RandomSeedT param_seed_2 = *param_seed_it;
        return (make_pair(param_seed_1 % 3, param_seed_2 % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

TEST(RandomizeDatasetSeedsTest, DataSeedsAreIndepdendentWithinDatasetSpec) {
  IntegerT num_datasets = 1;
  auto dataset_collection = ParseTextFormat<DatasetCollection>("datasets {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&dataset_collection,
                                                      &num_datasets, seed]() {
        ++num_datasets;
        dataset_collection.mutable_datasets(0)->set_num_datasets(num_datasets);
        ClearSeeds(&dataset_collection);
        RandomizeDatasetSeeds(&dataset_collection, seed);

        // Return the last two seeds.
        auto data_seed_it =
            dataset_collection.datasets(0).data_seeds().rbegin();
        const RandomSeedT data_seed_1 = *data_seed_it;
        ++data_seed_it;
        const RandomSeedT data_seed_2 = *data_seed_it;
        return (make_pair(data_seed_1 % 3, data_seed_2 % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

TEST(RandomizeDatasetSeedsTest, ParamSeedsAreIndepdendentAcrossDatasetSpecs) {
  IntegerT num_datasets = 1;
  auto dataset_collection = ParseTextFormat<DatasetCollection>(
      "datasets {} "
      "datasets {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&dataset_collection,
                                                      &num_datasets, seed]() {
        ++num_datasets;
        dataset_collection.mutable_datasets(0)->set_num_datasets(num_datasets);
        dataset_collection.mutable_datasets(1)->set_num_datasets(num_datasets);
        ClearSeeds(&dataset_collection);
        RandomizeDatasetSeeds(&dataset_collection, seed);

        // Return the last seed of each DatasetSpec.
        const RandomSeedT param_seed_1 =
            *dataset_collection.datasets(0).param_seeds().rbegin();
        const RandomSeedT param_seed_2 =
            *dataset_collection.datasets(1).param_seeds().rbegin();
        return (make_pair(param_seed_1 % 3, param_seed_2 % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

TEST(RandomizeDatasetSeedsTest, DataSeedsAreIndepdendentAcrossDatasetSpecs) {
  IntegerT num_datasets = 1;
  auto dataset_collection = ParseTextFormat<DatasetCollection>(
      "datasets {} "
      "datasets {} ");
  const RandomSeedT seed = GenerateRandomSeed();
  EXPECT_TRUE(IsEventually(
      function<pair<RandomSeedT, RandomSeedT>(void)>([&dataset_collection,
                                                      &num_datasets, seed]() {
        ++num_datasets;
        dataset_collection.mutable_datasets(0)->set_num_datasets(num_datasets);
        dataset_collection.mutable_datasets(1)->set_num_datasets(num_datasets);
        ClearSeeds(&dataset_collection);
        RandomizeDatasetSeeds(&dataset_collection, seed);

        // Return the last seed of each DatasetSpec.
        const RandomSeedT data_seed_1 =
            *dataset_collection.datasets(0).data_seeds().rbegin();
        const RandomSeedT data_seed_2 =
            *dataset_collection.datasets(1).data_seeds().rbegin();
        return (make_pair(data_seed_1 % 3, data_seed_2 % 3));
      }),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3)),
      CartesianProduct(Range<RandomSeedT>(0, 3), Range<RandomSeedT>(0, 3))));
}

}  // namespace automl_zero
