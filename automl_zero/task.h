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

// Taskl generation.

// These are called once-per-worker, so they can be slow.

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_TASK_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_TASK_H_

#include <algorithm>
#include <random>
#include <vector>

#include "task.proto.h"
#include "definitions.h"

namespace automl_zero {

constexpr IntegerT kNumTrainExamplesNotSet = -963487122;
constexpr double kDataTolerance = 0.00001;

// Holds data temporarily while it is being created, so that later it can be
// moved to a Task. This allows the Task to store const data.
template <FeatureIndexT F>
class TaskBuffer {
 public:
  TaskBuffer() : consumed_(false) {}
  bool IsConsumed() {return consumed_;}
  void Consume() {consumed_ = true;}

  // How the tasks are filled is up to each task Creator struct. By the
  // end of task creation, the train/valid features/labels should be
  // assigned correctly.
  std::vector<Vector<F>> train_features_;
  std::vector<Vector<F>> valid_features_;
  EvalType eval_type_;
  std::vector<Scalar> train_labels_;
  std::vector<Scalar> valid_labels_;

 private:
  // Whether this object has already been consumed by moving the data into
  // a task. A consumed TaskBuffer has no further use.
  bool consumed_;
};

// We define this base class so we can put tasks of different sizes into the
// same container of tasks.
class TaskInterface {
 public:
  // Always have at least one virtual method in this class. Because it is meant
  // to be downcasted, we need to keep it polymorphic.
  virtual ~TaskInterface() {}

  // Returns the size of the feature vectors in this task.
  virtual FeatureIndexT FeaturesSize() const = 0;

  // Returns the eval type.
  virtual EvalType GetEvalType() const = 0;

  // Returns the number of examples in the task. These can only be called
  // after the task creation is complete.
  virtual IntegerT TrainExamplesPerEpoch() const = 0;
  virtual IntegerT NumTrainEpochs() const = 0;
  virtual IntegerT MaxTrainExamples() const = 0;
  virtual IntegerT ValidSteps() const = 0;
};

template <FeatureIndexT F>
class TaskIterator;

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
class Task : public TaskInterface {
 public:
  explicit Task(const size_t index, const EvalType eval_type,
                const IntegerT num_train_epochs, std::mt19937* bit_gen,
                TaskBuffer<F>* buffer)
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

  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;

  Task(Task&& other)
      : index_(other.index_),
        eval_type_(other.eval_type_),
        train_features_(std::move(other.train_features_)),
        train_labels_(std::move(other.train_labels_)),
        train_epochs_(std::move(other.train_epochs_)),
        valid_features_(std::move(other.valid_features_)),
        valid_labels_(std::move(other.valid_labels_)),
        valid_epochs_(std::move(other.valid_epochs_)) {}

  Task& operator=(Task&& other) {
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

  bool operator==(const Task<F>& other) const {
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

  bool operator!=(const Task<F>& other) const { return !(*this == other); }

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
  TaskIterator<F> TrainIterator() const {
    return TaskIterator<F>(&train_features_, &train_labels_, &train_epochs_);
  }
  TaskIterator<F> ValidIterator() const {
    return TaskIterator<F>(&valid_features_, &valid_labels_, &valid_epochs_);
  }

  // ***IMPORTANT***: if you add a member variable below, you *must* also add it
  // to the move constructor. Or else it may just disappear in the middle of
  // your experiment.

  // Task index. Used to distinguish between different task caches.
  const size_t index_;

  const EvalType eval_type_;

 private:
  FRIEND_TEST(FillTasksTest, WorksCorrectly);
  FRIEND_TEST(FillTaskWithZerosTest, WorksCorrectly);
  FRIEND_TEST(FillTaskWithOnesTest, WorksCorrectly);
  FRIEND_TEST(FillTaskWithIncrementingIntegersTest, WorksCorrectly);
  FRIEND_TEST(FillTaskWithNonlinearDataTest, PermanenceTest);
  FRIEND_TEST(FillTaskWithProjectedBinaryClassificationTaskTest,
              WorksCorrectly);
  FRIEND_TEST(FillTaskWithProjectedBinaryClassificationTaskTest,
              BalancedClass);
  FRIEND_TEST(FillTaskWithDownsampledBinaryClassificationTaskTest,
              WorksCorrectly);
  FRIEND_TEST(FillTaskWithDownsampledBinaryClassificationTaskTest,
              BalancedClass);
  FRIEND_TEST(FillTaskWithProjectedMulticlassClassificationTaskTest,
              WorksCorrectly);
  FRIEND_TEST(FillTaskWithProjectedMulticlassClassificationTaskTest,
              BalancedClass);
  FRIEND_TEST(FillTaskWithProjectedMulticlassClassificationTaskTest,
              SoftensLabels);
  FRIEND_TEST(FillTaskWithCustomNNClassificationDataTest, BalancedClass);
  FRIEND_TEST(FillTaskWithCustomNNDistillationDataTest, PermanenceTest);
  FRIEND_TEST(CreateTaskWithPolynomialRegressionDataTest, LabelsAreCorrect);
  FRIEND_TEST(CreateTaskWithRandomPolynomialDataTest,
              DifferentForDifferentSeeds);
  FRIEND_TEST(CreateTaskWithRationalDataTest,
              LabelsAreCorrect);
  FRIEND_TEST(CreateTaskWithRandomRationalDataTest,
              DifferentForDifferentSeeds);
  FRIEND_TEST(UnitTestFixedTaskCreatorTest, GeneratesScalarTask);
  FRIEND_TEST(UnitTestFixedTaskCreatorTest, GeneratesVectorTask);
  FRIEND_TEST(FillWithDynamicMatrix, FillWithDynamicMatrixPermanenceTest);
  FRIEND_TEST(TaskTest, HasCorrectSizes);
  FRIEND_TEST(CreateTaskWithRandomMulticlassRationalDataTest,
              DifferentParamSeedsCoverAllLabelIndexes);
  FRIEND_TEST(CreateTaskWithRandomMulticlassRationalDataTest,
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
class TaskIterator {
 public:
  TaskIterator(const std::vector<Vector<F>>* features,
                  const std::vector<Scalar>* labels,
                  const std::vector<std::vector<IntegerT>>* epochs)
      : features_(features),
        labels_(labels),
        epochs_(epochs),
        current_example_(0),
        current_epoch_(0) {}

  TaskIterator(const TaskIterator&) = delete;
  TaskIterator& operator=(const TaskIterator&) = delete;

  TaskIterator(TaskIterator&& other)
      : features_(other.features_),
        labels_(other.labels_),
        epochs_(other.epochs_),
        current_example_(other.current_example_),
        current_epoch_(other.current_epoch_) {}

  TaskIterator& operator=(TaskIterator&& other) {
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

}  // namespace automl_zero

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_TASK_H_
