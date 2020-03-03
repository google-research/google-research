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

// Dataset generation.

// These are called once-per-worker, so they can be slow.

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_DATASET_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_DATASET_H_

#include <algorithm>
#include <random>
#include <vector>

#include "datasets.proto.h"
#include "definitions.h"

namespace brain {
namespace evolution {
namespace amlz {

constexpr IntegerT kNumTrainExamplesNotSet = -963487122;
constexpr double kDataTolerance = 0.00001;

// Holds data temporarily while it is being created, so that later it can be
// moved to a Dataset. This allows the Dataset to store const data.
template <FeatureIndexT F>
class DatasetBuffer {
 public:
  DatasetBuffer() : consumed_(false) {}
  bool IsConsumed() {return consumed_;}
  void Consume() {consumed_ = true;}

  // How the datasets are filled is up to each dataset Creator struct. By the
  // end of dataset creation, the train/valid features/labels should be
  // assigned correctly.
  std::vector<Vector<F>> train_features_;
  std::vector<Vector<F>> valid_features_;
  EvalType eval_type_;
  std::vector<Scalar> train_labels_;
  std::vector<Scalar> valid_labels_;

 private:
  // Whether this object has already been consumed by moving the data into
  // a dataset. A consumed DatasetBuffer has no further use.
  bool consumed_;
};

// We define this base class so we can put datasets of different sizes into the
// same container of datasets.
class DatasetInterface {
 public:
  // Always have at least one virtual method in this class. Because it is meant
  // to be downcasted, we need to keep it polymorphic.
  virtual ~DatasetInterface() {}

  // Returns the size of the feature vectors in this dataset.
  virtual FeatureIndexT FeaturesSize() const = 0;

  // Returns the eval type.
  virtual EvalType GetEvalType() const = 0;

  // Returns the number of examples in the dataset. These can only be called
  // after the dataset creation is complete.
  virtual IntegerT TrainExamplesPerEpoch() const = 0;
  virtual IntegerT NumTrainEpochs() const = 0;
  virtual IntegerT MaxTrainExamples() const = 0;
  virtual IntegerT ValidSteps() const = 0;
};

template <FeatureIndexT F>
class DatasetIterator;

template<typename RankT>
bool ItemEquals(const RankT& data1, const RankT& data2) {
  return (data1 - data2).norm() < kDataTolerance;
}
template<>
inline bool ItemEquals<Scalar>(const Scalar& data1, const Scalar& data2) {
  return abs(data1 - data2) < kDataTolerance;
}

template <typename RankT>
bool DataEquals(const std::vector<RankT>& data1,
                const std::vector<RankT>& data2) {
  if (data1.size() != data2.size()) return false;
  for (IntegerT index = 0; index < data1.size(); ++index) {
    if (!ItemEquals(data1[index], data2[index])) {
      return false;
    }
  }
  return true;
}

inline std::vector<std::vector<IntegerT>> GenerateEpochs(
    const IntegerT num_examples, const IntegerT num_epochs,
    std::mt19937* bit_gen) {
  std::vector<IntegerT> indexes;
  for (IntegerT i = 0; i < num_examples; ++i) indexes.push_back(i);
  std::vector<std::vector<IntegerT>> epochs(num_epochs);
  for (std::vector<IntegerT>& epoch : epochs) {
    epoch.insert(epoch.begin(), indexes.begin(), indexes.end());
    std::shuffle(indexes.begin(), indexes.end(), *bit_gen);
  }
  return epochs;
}

template <
    // The dimensionality of activations.
    FeatureIndexT F>
class Dataset : public DatasetInterface {
 public:
  explicit Dataset(const size_t index, const EvalType eval_type,
                   const IntegerT num_train_epochs, std::mt19937* bit_gen,
                   DatasetBuffer<F>* buffer)
      : index_(index),
        eval_type_(eval_type),
        train_features_(std::move(buffer->train_features_)),
        train_labels_(std::move(buffer->train_labels_)),
        train_epochs_(
            GenerateEpochs(train_features_.size(), num_train_epochs, bit_gen)),
        valid_features_(std::move(buffer->valid_features_)),
        valid_labels_(std::move(buffer->valid_labels_)),
        valid_epochs_(GenerateEpochs(valid_features_.size(), 1, bit_gen)) {
    CHECK(!buffer->IsConsumed());
    buffer->Consume();
    CHECK_EQ(train_features_.size(), train_labels_.size());
    CHECK_EQ(valid_features_.size(), valid_labels_.size());
  }

  Dataset(const Dataset&) = delete;
  Dataset& operator=(const Dataset&) = delete;

  Dataset(Dataset&& other)
      : index_(other.index_),
        eval_type_(other.eval_type_),
        train_features_(std::move(other.train_features_)),
        train_labels_(std::move(other.train_labels_)),
        train_epochs_(std::move(other.train_epochs_)),
        valid_features_(std::move(other.valid_features_)),
        valid_labels_(std::move(other.valid_labels_)),
        valid_epochs_(std::move(other.valid_epochs_)) {}

  Dataset& operator=(Dataset&& other) {
    this->index_ = other.index_;
    this->eval_type_ = other.eval_type_;
    this->train_features_ = std::move(other.train_features_);
    this->train_labels_ = std::move(other.train_labels_);
    this->train_epochs_ = std::move(other.train_epochs_);
    this->valid_features_ = std::move(other.valid_features_);
    this->valid_labels_ = std::move(other.valid_labels_);
    this->valid_epochs_ = std::move(other.valid_epochs_);
    return *this;
  }

  bool operator==(const Dataset<F>& other) const {
    CHECK_EQ(train_features_.size(), train_labels_.size());
    CHECK_EQ(other.train_features_.size(), other.train_labels_.size());
    if (!DataEquals(train_features_, other.train_features_)) {
      return false;
    }
    if (!DataEquals(train_labels_, other.train_labels_)) {
      return false;
    }
    if (train_epochs_ != other.train_epochs_) {
      return false;
    }

    CHECK_EQ(valid_features_.size(), valid_labels_.size());
    CHECK_EQ(other.valid_features_.size(), other.valid_labels_.size());
    if (!DataEquals(valid_features_, other.valid_features_)) {
      return false;
    }
    if (!DataEquals(valid_labels_, other.valid_labels_)) {
      return false;
    }
    CHECK_EQ(valid_epochs_.size(), 1);
    CHECK_EQ(other.valid_epochs_.size(), 1);
    return true;
  }

  bool operator!=(const Dataset<F>& other) const { return !(*this == other); }

  FeatureIndexT FeaturesSize() const override {return F;}
  EvalType GetEvalType() const override {return eval_type_;}
  IntegerT TrainExamplesPerEpoch() const override {
    return train_features_.size();
  }
  IntegerT NumTrainEpochs() const override {
    return train_epochs_.size();
  }
  IntegerT MaxTrainExamples() const override {
    return TrainExamplesPerEpoch() * NumTrainEpochs();
  }
  IntegerT ValidSteps() const override {
    return valid_features_.size();
  }

  // Iterate.
  DatasetIterator<F> TrainIterator() const {
    return DatasetIterator<F>(&train_features_, &train_labels_, &train_epochs_);
  }
  DatasetIterator<F> ValidIterator() const {
    return DatasetIterator<F>(&valid_features_, &valid_labels_, &valid_epochs_);
  }

  // ***IMPORTANT***: if you add a member variable below, you *must* also add it
  // to the move constructor. Or else it may just disappear in the middle of
  // your experiment.

  // Dataset index. Used to distinguish between different dataset caches.
  const size_t index_;

  const EvalType eval_type_;

 private:
  FRIEND_TEST(FillDatasetsTest, WorksCorrectly);
  FRIEND_TEST(FillDatasetWithZerosTest, WorksCorrectly);
  FRIEND_TEST(FillDatasetWithOnesTest, WorksCorrectly);
  FRIEND_TEST(FillDatasetWithIncrementingIntegersTest, WorksCorrectly);
  FRIEND_TEST(FillDatasetWithNonlinearDataTest, PermanenceTest);
  FRIEND_TEST(FillDatasetWithProjectedBinaryClassificationDatasetTest,
              WorksCorrectly);
  FRIEND_TEST(FillDatasetWithProjectedBinaryClassificationDatasetTest,
              BalancedClass);
  FRIEND_TEST(FillDatasetWithDownsampledBinaryClassificationDatasetTest,
              WorksCorrectly);
  FRIEND_TEST(FillDatasetWithDownsampledBinaryClassificationDatasetTest,
              BalancedClass);
  FRIEND_TEST(FillDatasetWithProjectedMulticlassClassificationDatasetTest,
              WorksCorrectly);
  FRIEND_TEST(FillDatasetWithProjectedMulticlassClassificationDatasetTest,
              BalancedClass);
  FRIEND_TEST(FillDatasetWithProjectedMulticlassClassificationDatasetTest,
              SoftensLabels);
  FRIEND_TEST(FillDatasetWithCustomNNClassificationDataTest, BalancedClass);
  FRIEND_TEST(FillDatasetWithCustomNNDistillationDataTest, PermanenceTest);
  FRIEND_TEST(CreateDatasetWithPolynomialRegressionDataTest, LabelsAreCorrect);
  FRIEND_TEST(CreateDatasetWithRandomPolynomialDataTest,
              DifferentForDifferentSeeds);
  FRIEND_TEST(CreateDatasetWithRationalDataTest,
              LabelsAreCorrect);
  FRIEND_TEST(CreateDatasetWithRandomRationalDataTest,
              DifferentForDifferentSeeds);
  FRIEND_TEST(UnitTestFixedDatasetCreatorTest, GeneratesScalarDataset);
  FRIEND_TEST(UnitTestFixedDatasetCreatorTest, GeneratesVectorDataset);
  FRIEND_TEST(FillWithDynamicMatrix, FillWithDynamicMatrixPermanenceTest);
  FRIEND_TEST(DatasetTest, HasCorrectSizes);
  FRIEND_TEST(CreateDatasetWithRandomMulticlassRationalDataTest,
              DifferentParamSeedsCoverAllLabelIndexes);
  FRIEND_TEST(CreateDatasetWithRandomMulticlassRationalDataTest,
              SameParamSeedsUsesOnlyTwoLabelIndexes);

  // ***IMPORTANT***: if you add a member variable below, you *must* also add it
  // to the move constructor. Or else it may just disappear in the middle of
  // your experiment.

  // The xx_features_ and xx_labels_ only contain one epoch worth of examples.
  // The xx_epochs_ is a list of lists where the outer index is the epoch number
  // and the inner list is the order of the examples in that epoch.
  const std::vector<Vector<F>> train_features_;
  const std::vector<Scalar> train_labels_;
  const std::vector<std::vector<IntegerT>> train_epochs_;
  const std::vector<Vector<F>> valid_features_;
  const std::vector<Scalar> valid_labels_;
  const std::vector<std::vector<IntegerT>> valid_epochs_;
};

template <FeatureIndexT F>
class DatasetIterator {
 public:
  DatasetIterator(const std::vector<Vector<F>>* features,
                  const std::vector<Scalar>* labels,
                  const std::vector<std::vector<IntegerT>>* epochs)
      : features_(features),
        labels_(labels),
        epochs_(epochs),
        current_example_(0),
        current_epoch_(0) {}

  DatasetIterator(const DatasetIterator&) = delete;
  DatasetIterator& operator=(const DatasetIterator&) = delete;

  DatasetIterator(DatasetIterator&& other)
      : features_(other.features_),
        labels_(other.labels_),
        epochs_(other.epochs_),
        current_example_(other.current_example_),
        current_epoch_(other.current_epoch_) {}

  DatasetIterator& operator=(DatasetIterator&& other) {
    this->features_ = other.features_;
    this->labels_ = other.labels_;
    this->epochs_ = other.epochs_;
    this->current_example_ = other.current_example_;
    this->current_epoch_ = other.current_epoch_;
    return *this;
  }

  bool Done() const {
    return current_epoch_ >= epochs_->size();
  }

  void Next() {
    CHECK_LE(current_epoch_, epochs_->size());
    ++current_example_;
    if (current_example_ >= features_->size()) {
      current_example_ = 0;
      ++current_epoch_;
    }
  }

  inline const Vector<F>& GetFeatures() const {
    return features_->at(epochs_->at(current_epoch_).at(current_example_));
  }

  inline const Scalar& GetLabel() const {
    return labels_->at(epochs_->at(current_epoch_).at(current_example_));
  }

 private:
  const std::vector<Vector<F>>* features_;
  const std::vector<Scalar>* labels_;
  const std::vector<std::vector<IntegerT>>* epochs_;
  IntegerT current_example_;
  IntegerT current_epoch_;
};

}  // namespace amlz
}  // namespace evolution
}  // namespace brain

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_DATASET_H_
