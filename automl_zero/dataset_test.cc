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

#include "dataset.h"

#include "dataset_util.h"
#include "definitions.h"
#include "gtest/gtest.h"

namespace automl_zero {

using test_only::GenerateDataset;

template<FeatureIndexT F>
IntegerT CountOccurrences(
    const vector<double>& element, const vector<Vector<F>>& container) {
  IntegerT count = 0;
  CHECK_EQ(element.size(), F);
  Vector<F> eigen_element(element.data());
  for (const Vector<F>& current_element : container) {
    if ((current_element - eigen_element).norm() < kDataTolerance) {
      ++count;
    }
  }
  return count;
}

// Tests that all the examples appear the correct number of times and that the
// features and labels remain matched correctly after generating the epochs.
TEST(DatasetTest, EpochsContainCorrectTrainExamples) {
  auto dataset = GenerateDataset<4>(
      "unit_test_fixed_dataset { "
      "  train_features {elements: [0.41, 0.42, 0.43, 0.44]} "
      "  train_features {elements: [0.51, 0.52, 0.53, 0.54]} "
      "  train_features {elements: [0.61, 0.62, 0.63, 0.64]} "
      "  train_labels {elements: [4.1]} "
      "  train_labels {elements: [5.1]} "
      "  train_labels {elements: [6.1]} "
      "  valid_features {elements: [0.71, 0.72, 0.73, 0.74]} "
      "  valid_features {elements: [0.81, 0.82, 0.83, 0.84]} "
      "  valid_labels {elements: [7.1]} "
      "  valid_labels {elements: [8.1]} "
      "} "
      "eval_type: RMS_ERROR "
      "num_train_examples: 3 "
      "num_train_epochs: 8 "
      "num_valid_examples: 2 "
      "num_datasets: 1 "
      "features_size: 4 ");

  DatasetIterator<4> train_it = dataset.TrainIterator();
  vector<Vector<5>> features_and_labels;
  while (!train_it.Done()) {
    Vector<4> features = train_it.GetFeatures();
    Scalar label = train_it.GetLabel();
    Vector<5> curr_features_and_labels;
    curr_features_and_labels << features, label;
    features_and_labels.push_back(curr_features_and_labels);
    train_it.Next();
  }
  EXPECT_EQ(features_and_labels.size(), 24);
  EXPECT_EQ(
      CountOccurrences<5>({0.41, 0.42, 0.43, 0.44, 4.1}, features_and_labels),
      8);
  EXPECT_EQ(
      CountOccurrences<5>({0.51, 0.52, 0.53, 0.54, 5.1}, features_and_labels),
      8);
  EXPECT_EQ(
      CountOccurrences<5>({0.61, 0.62, 0.63, 0.64, 6.1}, features_and_labels),
      8);
}

TEST(DatasetTest, EpochsContainShuffledTrainExamples) {
  auto dataset = GenerateDataset<4>(
      "unit_test_fixed_dataset { "
      "  train_features {elements: [0.41, 0.42, 0.43, 0.44]} "
      "  train_features {elements: [0.51, 0.52, 0.53, 0.54]} "
      "  train_features {elements: [0.61, 0.62, 0.63, 0.64]} "
      "  train_labels {elements: [4.1]} "
      "  train_labels {elements: [5.1]} "
      "  train_labels {elements: [6.1]} "
      "  valid_features {elements: [0.71, 0.72, 0.73, 0.74]} "
      "  valid_features {elements: [0.81, 0.82, 0.83, 0.84]} "
      "  valid_labels {elements: [7.1]} "
      "  valid_labels {elements: [8.1]} "
      "} "
      "eval_type: RMS_ERROR "
      "num_train_examples: 3 "
      "num_train_epochs: 8 "
      "num_valid_examples: 2 "
      "num_datasets: 1 "
      "features_size: 4 ");

  DatasetIterator<4> train_it = dataset.TrainIterator();
  vector<Vector<5>> features_and_labels;
  while (!train_it.Done()) {
    Vector<4> features = train_it.GetFeatures();
    Scalar label = train_it.GetLabel();
    Vector<5> curr_features_and_labels;
    curr_features_and_labels << features, label;
    features_and_labels.push_back(curr_features_and_labels);
    train_it.Next();
  }
  EXPECT_TRUE(
      (features_and_labels[7] - features_and_labels[1]).norm() >
      kDataTolerance);
}

}  // namespace automl_zero
