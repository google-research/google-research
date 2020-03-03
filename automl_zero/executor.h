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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_EXECUTOR_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_EXECUTOR_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>

#include "glog/logging.h"
#include "datasets.proto.h"
#include "dataset.h"
#include "definitions.h"
#include "definitions.proto.h"
#include "algorithm.h"
#include "instruction.h"
#include "memory.h"
#include "random_generator.h"
#include "testing/production_stub/public/gunit_prod.h"

namespace automl_zero {

constexpr FeatureIndexT kNumClasses = 10;
constexpr double kPadLabel = 0.0;

template <FeatureIndexT F>
class Executor {
 public:
  // Constructs a standard executor. Uses a clean memory and automatically
  // executes the setup component_function. All arguments are stored by
  // reference, so they must out-live the Executor instance.
  Executor(const Algorithm& algorithm, const Dataset<F>& dataset,
           // Includes the examples in all the training epochs.
           IntegerT num_all_train_examples, IntegerT num_valid_examples,
           RandomGenerator* rand_gen,
           // Errors larger than this trigger early stopping, as they signal
           // models that likely have runnaway behavior. Early stopping can also
           // be triggered if the loss for an example is infinite, nan, or too
           // large. If early stopping is triggered, the fitness for the
           // execution will be set to the minimum value.
           double max_abs_error);
  Executor(const Executor& other) = delete;
  Executor& operator=(const Executor& other) = delete;

  // Most code should use only the Execute method. Other methods below provide
  // lower-level access and can be used by tests and dataset generators. Returns
  // the fitness, according to the EvalType enum for the relevant dataset.
  double Execute(
      std::vector<double>* train_errors = nullptr,
      std::vector<double>* valid_errors = nullptr);

  // Use only from unit tests.
  inline Memory<F>& MemoryRef() {return memory_;}

 private:
  FRIEND_TEST(ExecutorTest, PredictComponentFunctionRuns);
  FRIEND_TEST(ExecutorTest, LearnComponentFunctionRuns);
  FRIEND_TEST(ExecutorTest, ItereatesThroughFeatures);
  FRIEND_TEST(ExecutorTest, ItereatesThroughLabelsDuringTraining);
  FRIEND_TEST(ExecutorTest, ValidationDoesNotSeeLabels);
  FRIEND_TEST(ExecutorTest, TrainOptimizationsAreCorrect);
  FRIEND_TEST(ExecutorTest, MultiEpochTrainingWorksCorrectly);

  // Performs training until the end. Returns whether successful. If not, it
  // means training stopped early.
  bool Train(std::vector<double>* errors);

  // Performs training for a given number of steps. Returns whether successful.
  // If not, it means training stopped early.
  bool Train(IntegerT max_steps, std::vector<double>* errors,
             // The iterators are used to track the training progress.
             // They should point to dataset_.train_features_.begin(),
             // dataset_.train_labels_.begin() and
             // dataset_.vector_train_labels_.begin() initially, and will be
             // updated after each training step.
             DatasetIterator<F>* train_it);

  // Implementations of the train component_function, with different
  // optimizations.
  bool TrainNoOptImpl(IntegerT max_steps, std::vector<double>* errors,
                      // See `Train` for more details about the following args.
                      DatasetIterator<F>* train_it);

  template <size_t max_component_function_size>
  bool TrainOptImpl(IntegerT max_steps, std::vector<double>* errors,
                    // See `Train` for more details about the following args.
                    DatasetIterator<F>* train_it);

  // Performs validation and returns the loss.
  double Validate(std::vector<double>* errors);

  // Copies memory_ into *memory. Useful for tests.
  void GetMemory(Memory<F>* memory);

  // The Algorithm being trained.
  const Algorithm& algorithm_;

  // The dataset used for training.
  const Dataset<F>& dataset_;

  const IntegerT num_all_train_examples_;
  const IntegerT num_valid_examples_;
  RandomGenerator* rand_gen_;
  Memory<F> memory_;

  const double max_abs_error_;
};

// Fills the training and validation labels, using the given Algorithm and
// memory. Can alter this memory, but only if the predict component_function of
// the Algorithm does so--only runs the predict component_function. Useful for
// dataset generators to generate labels.
template <FeatureIndexT F>
void ExecuteAndFillLabels(const Algorithm& algorithm, Memory<F>* memory,
                          DatasetBuffer<F>* buffer,
                          RandomGenerator* rand_gen);

// Maps the interval [0.0, inf] to [0.0, 1.0]. The squashing is done by an
// arctan, so that losses in [0.0, 0.5] approximately undergo an affine
// transformation.
double FlipAndSquash(double value);

inline double Sigmoid(double x) {
  return static_cast<double>(1.0) /
      (static_cast<double>(1.0) + std::exp(-x));
}

namespace internal {

template<FeatureIndexT F>
inline Vector<F> TruncatingSoftmax(const Vector<F>& input);

template<FeatureIndexT F>
inline FeatureIndexT Argmax(const Vector<F>& input);

}  // namespace internal


////////////////////////////////////////////////////////////////////////////////
// Scalar arithmetic-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteScalarSumOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] + memory->scalar_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteScalarDiffOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] - memory->scalar_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteScalarProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] * memory->scalar_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteScalarDivisionOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] /
      memory->scalar_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteScalarMinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::min(
      memory->scalar_[instruction.in1_],
      memory->scalar_[instruction.in2_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarMaxOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::max(
      memory->scalar_[instruction.in1_],
      memory->scalar_[instruction.in2_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarAbsOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::abs(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarHeavisideOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->scalar_[instruction.in1_] >= 0.0 ? 1.0 : 0.0;
}

template<FeatureIndexT F>
inline void ExecuteScalarConstSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = instruction.GetActivationData();
}

template<FeatureIndexT F>
inline void ExecuteScalarReciprocalOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      static_cast<double>(1.0) / memory->scalar_[instruction.in1_];
}


////////////////////////////////////////////////////////////////////////////////
// Trigonometry-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteScalarSinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::sin(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarCosOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::cos(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarTanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::tan(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarArcSinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::asin(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarArcCosOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::acos(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarArcTanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::atan(
      memory->scalar_[instruction.in1_]);
}


////////////////////////////////////////////////////////////////////////////////
// Calculus-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteScalarExpOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::exp(
      memory->scalar_[instruction.in1_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarLogOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] = std::log(
      memory->scalar_[instruction.in1_]);
}


////////////////////////////////////////////////////////////////////////////////
// Vector arithmetic-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteVectorSumOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      memory->vector_[instruction.in1_] + memory->vector_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteVectorDiffOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      memory->vector_[instruction.in1_] - memory->vector_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteVectorProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      (memory->vector_[instruction.in1_].array() *
       memory->vector_[instruction.in2_].array()).matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorDvisionOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      (memory->vector_[instruction.in1_].array() /
       memory->vector_[instruction.in2_].array()).matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorMinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      (memory->vector_[instruction.in1_].array().min(
          memory->vector_[instruction.in2_].array())).matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorMaxOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      (memory->vector_[instruction.in1_].array().max(
          memory->vector_[instruction.in2_].array())).matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorAbsOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      memory->vector_[instruction.in1_].array().abs().matrix();
}

template<FeatureIndexT F>
inline void ExecuteVectorHeavisideOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  const double* in = memory->vector_[instruction.in1_].data();
  const double* in_end = in + F;
  double* out = memory->vector_[instruction.out_].data();
  while (in != in_end) {
    *out = *in > 0.0 ? 1.0 : 0.0;
    ++out;
    ++in;
  }
}

template<FeatureIndexT F>
inline void ExecuteVectorReluOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  const double* in = memory->vector_[instruction.in1_].data();
  const double* in_end = in + F;
  double* out = memory->vector_[instruction.out_].data();
  while (in != in_end) {
    *out = *in > 0.0 ? *in : 0.0;
    ++out;
    ++in;
  }
}

template<FeatureIndexT F>
inline void ExecuteVectorConstSetOldOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  CHECK_EQ(F, 4);
  memory->vector_[instruction.out_].block(0, 0, 4, 1) =
      instruction.GetVectorData();
}

template<FeatureIndexT F>
inline void ExecuteVectorConstSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  const FeatureIndexT index =
      FloatToIndex(instruction.GetFloatData0(), F);
  memory->vector_[instruction.out_](index) = instruction.GetFloatData1();
}

template<FeatureIndexT F>
inline void ExecuteVectorReciprocalOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      (static_cast<double>(1.0) /
       memory->vector_[instruction.in1_].array())
          .matrix();
}


////////////////////////////////////////////////////////////////////////////////
// Matrix arithmetic-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteMatrixSumOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in1_] + memory->matrix_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteMatrixDiffOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in1_] - memory->matrix_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteMatrixProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      (memory->matrix_[instruction.in1_].array() *
       memory->matrix_[instruction.in2_].array()).matrix();
}

template<FeatureIndexT F>
inline void ExecuteMatrixDivisionOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      (memory->matrix_[instruction.in1_].array() /
       memory->matrix_[instruction.in2_].array()).matrix();
}

template<FeatureIndexT F>
inline void ExecuteMatrixMinOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  const double* in1 = memory->matrix_[instruction.in1_].data();
  const double* in2 = memory->matrix_[instruction.in2_].data();
  const double* in1_end = in1 + F * F;
  double* out = memory->matrix_[instruction.out_].data();
  while (in1 != in1_end) {
    const double in1v = *in1;
    const double in2v = *in2;
    *out = in1v < in2v ? in1v : in2v;
    ++out;
    ++in1;
    ++in2;
  }
}

template<FeatureIndexT F>
inline void ExecuteMatrixMaxOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  const double* in1 = memory->matrix_[instruction.in1_].data();
  const double* in2 = memory->matrix_[instruction.in2_].data();
  const double* in1_end = in1 + F * F;
  double* out = memory->matrix_[instruction.out_].data();
  while (in1 != in1_end) {
    const double in1v = *in1;
    const double in2v = *in2;
    *out = in1v > in2v ? in1v : in2v;
    ++out;
    ++in1;
    ++in2;
  }
}

template<FeatureIndexT F>
inline void ExecuteMatrixAbsOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in1_].array().abs().matrix();
}

template<FeatureIndexT F>
inline void ExecuteMatrixHeavisideOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  const double* in = memory->matrix_[instruction.in1_].data();
  const double* in_end = in + F * F;
  double* out = memory->matrix_[instruction.out_].data();
  while (in != in_end) {
    *out = *in > 0.0 ? 1.0 : 0.0;
    ++out;
    ++in;
  }
}

template<FeatureIndexT F>
inline void ExecuteMatrixRowConstSetOldOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  CHECK_EQ(F, 4);
  memory->matrix_[instruction.out_]
      .block(instruction.GetIndexData0(), 0, 1, 4) =
          instruction.GetVectorData().transpose();
}

template<FeatureIndexT F>
inline void ExecuteMatrixConstSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_](
      FloatToIndex(instruction.GetFloatData0(), F),
      FloatToIndex(instruction.GetFloatData1(), F)) =
          instruction.GetFloatData2();
}

template<FeatureIndexT F>
inline void ExecuteMatrixReciprocalOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      (static_cast<double>(1.0) /
       memory->matrix_[instruction.in1_].array())
          .matrix();
}


////////////////////////////////////////////////////////////////////////////
// Linear algebra-related instructions.
////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteScalarVectorProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      memory->vector_[instruction.in2_] * memory->scalar_[instruction.in1_];
}

template<FeatureIndexT F>
inline void ExecuteVectorInnerProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->vector_[instruction.in1_].dot(
          memory->vector_[instruction.in2_]);
}

template<FeatureIndexT F>
inline void ExecuteVectorOuterProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      memory->vector_[instruction.in1_] *
          memory->vector_[instruction.in2_].transpose();
}

template<FeatureIndexT F>
inline void ExecuteScalarMatrixProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in2_] * memory->scalar_[instruction.in1_];
}

template<FeatureIndexT F>
inline void ExecuteMatrixVectorProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      memory->matrix_[instruction.in1_] * memory->vector_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteVectorNormOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->vector_[instruction.in1_].norm();
}

template<FeatureIndexT F>
inline void ExecuteMatrixNormOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->matrix_[instruction.in1_].norm();
}

template<FeatureIndexT F>
inline void ExecuteMatrixRowNormOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      memory->matrix_[instruction.in1_].rowwise().norm();
}

template<FeatureIndexT F>
inline void ExecuteMatrixColumnNormOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      memory->matrix_[instruction.in1_].colwise().norm();
}

template<FeatureIndexT F>
inline void ExecuteMatrixTransposeOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  if (instruction.out_ == instruction.in1_) {
    memory->matrix_[instruction.in1_].transposeInPlace();
  } else {
    memory->matrix_[instruction.out_] =
        memory->matrix_[instruction.in1_].transpose();
  }
}

template<FeatureIndexT F>
inline void ExecuteMatrixMatrixProductOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      memory->matrix_[instruction.in1_] * memory->matrix_[instruction.in2_];
}

template<FeatureIndexT F>
inline void ExecuteScalarBroadcastOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      memory->scalar_[instruction.in1_] * Vector<F>::Ones(F);
}

template<FeatureIndexT F>
inline void ExecuteVectorColumnBroadcastOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      memory->vector_[instruction.in1_].replicate(1, F);
}

template<FeatureIndexT F>
inline void ExecuteVectorRowBroadcastOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->matrix_[instruction.out_] =
      memory->vector_[instruction.in1_].replicate(1, F).transpose();
}


////////////////////////////////////////////////////////////////////////////////
// Probability-related instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteVectorMeanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->vector_[instruction.in1_].mean();
}

template<FeatureIndexT F>
inline void ExecuteVectorStDevOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  const Vector<F>& values = memory->vector_[instruction.in1_];
  const double mean = values.mean();
  memory->scalar_[instruction.out_] =
      sqrt(values.dot(values) / static_cast<double>(F) -
           mean * mean);
}

template<FeatureIndexT F>
inline void ExecuteMatrixMeanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      memory->matrix_[instruction.in1_].mean();
}

template<FeatureIndexT F>
inline void ExecuteMatrixStDevOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  const Matrix<F>& values = memory->matrix_[instruction.in1_];
  const double mean = values.mean();
  memory->scalar_[instruction.out_] =
      sqrt((values.array() * values.array()).sum() /
           static_cast<double>(F * F) -
           mean * mean);
}

template<FeatureIndexT F>
inline void ExecuteMatrixRowMeanOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->vector_[instruction.out_] =
      memory->matrix_[instruction.in1_].rowwise().mean();
}

template<FeatureIndexT F>
inline void ExecuteMatrixRowStDevOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  for (IntegerT row = 0; row < F; ++row) {
    const Vector<F>& values =
        memory->matrix_[instruction.in1_].row(row);
    const double mean = values.mean();
    const double stdev =
        sqrt((values.array() * values.array()).sum() /
             static_cast<double>(F) -
             mean * mean);
    memory->vector_[instruction.out_](row) = stdev;
  }
}

template<FeatureIndexT F>
inline void ExecuteScalarGaussianSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      rand_gen->GaussianActivation(
          instruction.GetFloatData0(), instruction.GetFloatData1());
}

template<FeatureIndexT F>
inline void ExecuteVectorGaussianSetOldOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  rand_gen->FillGaussian<F>(
      0.0, instruction.GetActivationData(),
      &memory->vector_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteVectorGaussianSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  rand_gen->FillGaussian<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->vector_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteMatrixGaussianSetOldOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  rand_gen->FillGaussian<F>(
      0.0, instruction.GetActivationData(),
      &memory->matrix_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteMatrixGaussianSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  rand_gen->FillGaussian<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->matrix_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarUniformSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      rand_gen->UniformActivation(
          instruction.GetFloatData0(), instruction.GetFloatData1());
}

template<FeatureIndexT F>
inline void ExecuteVectorUniformSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  rand_gen->FillUniform<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->vector_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteMatrixUniformSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  rand_gen->FillUniform<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->matrix_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteScalarBetaSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  memory->scalar_[instruction.out_] =
      rand_gen->BetaActivation(
          instruction.GetFloatData0(), instruction.GetFloatData1());
}

template<FeatureIndexT F>
inline void ExecuteVectorBetaSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  rand_gen->FillBeta<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->vector_[instruction.out_]);
}

template<FeatureIndexT F>
inline void ExecuteMatrixBetaSetOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  rand_gen->FillBeta<F>(
      instruction.GetFloatData0(), instruction.GetFloatData1(),
      &memory->matrix_[instruction.out_]);
}


////////////////////////////////////////////////////////////////////////////////
// Debug instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteScalarPrintOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  Print() << "identity" << instruction.GetIntegerData() << ", "
          << "value=" << memory->scalar_[instruction.out_] << Flush();
}

template<FeatureIndexT F>
inline void ExecuteVectorPrintOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  Print() << "identity" << instruction.GetIntegerData() << ", "
          << "value=" << ToString<F>(memory->vector_[instruction.out_])
          << Flush();
}

template<FeatureIndexT F>
inline void ExecuteMatrixPrintOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  Print() << "identity" << instruction.GetIntegerData() << ", "
          << "value=" << ToString<F>(memory->matrix_[instruction.out_])
          << Flush();
}


////////////////////////////////////////////////////////////////////////////////
// Other instructions.
////////////////////////////////////////////////////////////////////////////////

template<FeatureIndexT F>
inline void ExecuteNoOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
}

template<FeatureIndexT F>
inline void ExecuteUnsupportedOp(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  LOG(FATAL) << "Unsupported op." << std::endl;
}


template<FeatureIndexT F>
static constexpr std::array<
    void(*)(const Instruction&, RandomGenerator*, Memory<F>*),
    128> kOpIndexToExecuteFunction = {
        &ExecuteNoOp<F>,                   // NO_OP = 0
        &ExecuteScalarSumOp<F>,            // SCALAR_SUM_OP = 1
        &ExecuteMatrixVectorProductOp<F>,  // MATRIX_VECTOR_PRODUCT_OP = 2
        &ExecuteVectorMeanOp<F>,           // VECTOR_MEAN_OP = 3
        &ExecuteVectorGaussianSetOldOp,    // VECTOR_GAUSSIAN_SET_OLD_OP = 4
        &ExecuteMatrixGaussianSetOldOp,    // MATRIX_GAUSSIAN_SET_OLD_OP = 5
        &ExecuteScalarConstSetOp,        // SCALAR_CONST_SET_OP = 6
        &ExecuteVectorReluOp,              // VECTOR_RELU_OP = 7
        &ExecuteVectorInnerProductOp,      // VECTOR_INNER_PRODUCT_OP = 8
        &ExecuteScalarDiffOp,              // SCALAR_DIFF_OP = 9
        &ExecuteScalarProductOp,           // SCALAR_PRODUCT_OP = 10
        &ExecuteScalarVectorProductOp,     // SCALAR_VECTOR_PRODUCT_OP = 11
        &ExecuteVectorSumOp,               // VECTOR_SUM_OP = 12
        &ExecuteVectorHeavisideOp,         // VECTOR_HEAVYSIDE_OP = 13
        &ExecuteVectorProductOp,           // VECTOR_PRODUCT_OP = 14
        &ExecuteVectorOuterProductOp,      // VECTOR_OUTER_PRODUCT_OP = 15
        &ExecuteMatrixSumOp,               // MATRIX_SUM_OP = 16
        &ExecuteVectorConstSetOldOp,       // VECTOR_CONST_SET_OLD_OP = 17
        &ExecuteMatrixRowConstSetOldOp,    // MATRIX_ROW_CONST_SET_OLD_OP = 18
        &ExecuteScalarDivisionOp,          // SCALAR_DIVISION_OP = 19
        &ExecuteScalarMinOp,               // SCALAR_MIN_OP = 20
        &ExecuteScalarMaxOp,               // SCALAR_MAX_OP = 21
        &ExecuteScalarAbsOp,               // SCALAR_ABS_OP = 22
        &ExecuteScalarHeavisideOp,         // SCALAR_HEAVYSIDE_OP = 23
        &ExecuteScalarSinOp,               // SCALAR_SIN_OP = 24
        &ExecuteScalarCosOp,               // SCALAR_COS_OP = 25
        &ExecuteScalarTanOp,               // SCALAR_TAN_OP = 26
        &ExecuteScalarArcSinOp,            // SCALAR_ARCSIN_OP = 27
        &ExecuteScalarArcCosOp,            // SCALAR_ARCCOS_OP = 28
        &ExecuteScalarArcTanOp,            // SCALAR_ARCTAN_OP = 29
        &ExecuteScalarExpOp,               // SCALAR_EXP_OP = 30
        &ExecuteScalarLogOp,               // SCALAR_LOG_OP = 31
        &ExecuteVectorDiffOp,              // VECTOR_DIFF_OP = 32
        &ExecuteVectorDvisionOp,           // VECTOR_DIVISION_OP = 33
        &ExecuteVectorMinOp,               // VECTOR_MIN_OP = 34
        &ExecuteVectorMaxOp,               // VECTOR_MAX_OP = 35
        &ExecuteVectorAbsOp,               // VECTOR_ABS_OP = 36
        &ExecuteMatrixDiffOp,              // MATRIX_DIFF_OP = 37
        &ExecuteMatrixProductOp,           // MATRIX_PRODUCT_OP = 38
        &ExecuteMatrixDivisionOp,          // MATRIX_DIVISION_OP = 39
        &ExecuteMatrixMinOp,               // MATRIX_MIN_OP = 40
        &ExecuteMatrixMaxOp,               // MATRIX_MAX_OP = 41
        &ExecuteMatrixAbsOp,               // MATRIX_ABS_OP = 42
        &ExecuteMatrixHeavisideOp,         // MATRIX_HEAVYSIDE_OP = 43
        &ExecuteMatrixConstSetOp,          // MATRIX_CONST_SET_OP = 44
        &ExecuteVectorConstSetOp,          // VECTOR_CONST_SET_OP = 45
        &ExecuteScalarMatrixProductOp,     // SCALAR_MATRIX_PRODUCT_OP = 46
        &ExecuteVectorNormOp,              // VECTOR_NORM_OP = 47
        &ExecuteMatrixNormOp,              // MATRIX_NORM_OP = 48
        &ExecuteVectorStDevOp,             // VECTOR_ST_DEV_OP = 49
        &ExecuteMatrixMeanOp,              // MATRIX_MEAN_OP = 50
        &ExecuteMatrixStDevOp,             // MATRIX_ST_DEV_OP = 51
        &ExecuteMatrixRowMeanOp,           // MATRIX_ROW_MEAN_OP = 52
        &ExecuteMatrixRowStDevOp,          // MATRIX_ROW_ST_DEV_OP = 53
        &ExecuteScalarGaussianSetOp,       // SCALAR_GAUSSIAN_SET_OP = 54
        &ExecuteScalarUniformSetOp,        // SCALAR_UNIFORM_SET_OP = 55
        &ExecuteVectorUniformSetOp,        // VECTOR_UNIFORM_SET_OP = 56
        &ExecuteMatrixUniformSetOp,        // MATRIX_UNIFORM_SET_OP = 57
        &ExecuteScalarBetaSetOp,           // SCALAR_BETA_SET_OP = 58
        &ExecuteVectorBetaSetOp,           // VECTOR_BETA_SET_OP = 59
        &ExecuteMatrixBetaSetOp,           // MATRIX_BETA_SET_OP = 60
        &ExecuteMatrixTransposeOp,         // MATRIX_TRANSPOSE_OP = 61
        &ExecuteMatrixMatrixProductOp,     // MATRIX_MATRIX_PRODUCT_OP = 62
        &ExecuteVectorGaussianSetOp,       // VECTOR_GAUSSIAN_SET_OP = 63
        &ExecuteMatrixGaussianSetOp,       // MATRIX_GAUSSIAN_SET_OP = 64
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 65
        &ExecuteScalarReciprocalOp,        // SCALAR_RECIPROCAL_OP = 66
        &ExecuteVectorReciprocalOp,        // VECTOR_RECIPROCAL_OP = 67
        &ExecuteMatrixReciprocalOp,        // MATRIX_RECIPROCAL_OP = 68
        &ExecuteMatrixRowNormOp,           // MATRIX_ROW_NORM_OP = 69
        &ExecuteMatrixColumnNormOp,        // MATRIX_COLUMN_NORM_OP = 70
        &ExecuteScalarBroadcastOp,         // SCALAR_BROADCAST_OP = 71
        &ExecuteVectorColumnBroadcastOp,   // VECTOR_COLUMN_BROADCAST_OP = 72
        &ExecuteVectorRowBroadcastOp,      // VECTOR_ROW_BROADCAST_OP = 73
        &ExecuteScalarPrintOp,             // SCALAR_PRINT_OP = 74
        &ExecuteVectorPrintOp,             // VECTOR_PRINT_OP = 75
        &ExecuteMatrixPrintOp,             // MATRIX_PRINT_OP = 76
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 77
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 78
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 79
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 80
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 81
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 82
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 83
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 84
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 85
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 86
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 87
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 88
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 89
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 90
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 91
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 92
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 93
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 94
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 95
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 96
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 97
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 98
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 99
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 100
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 101
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 102
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 103
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 104
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 105
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 106
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 107
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 108
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 109
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 110
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 111
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 112
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 113
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 114
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 115
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 116
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 117
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 118
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 119
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 120
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 121
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 122
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 123
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 124
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 125
        &ExecuteUnsupportedOp,             // UNSUPPORTED_OP = 126
        &ExecuteUnsupportedOp              // UNSUPPORTED_OP = 127
    };

template<FeatureIndexT F>
inline void ExecuteInstruction(
    const Instruction& instruction, RandomGenerator* rand_gen,
    Memory<F>* memory) {
  (*kOpIndexToExecuteFunction<F>[instruction.op_])(
      instruction, rand_gen, memory);
}

template <FeatureIndexT F>
struct ZeroLabelAssigner {
  inline static void Assign(Memory<F>* memory) {
    memory->scalar_[kLabelsScalarAddress] = 0.0;
  }
};

template <FeatureIndexT F>
struct LabelAssigner {
  inline static void Assign(const Scalar& label, Memory<F>* memory) {
    memory->scalar_[kLabelsScalarAddress] = label;
  }
};

template<FeatureIndexT F>
struct PredictionGetter {
  inline static Scalar Get(Memory<F>* memory) {
    return memory->scalar_[kPredictionsScalarAddress];
  }
};

template <FeatureIndexT F>
struct ErrorComputer {
  inline static double Compute(const Memory<F>& memory, const Scalar& label) {
    return std::abs(label - memory.scalar_[kPredictionsScalarAddress]);
  }
};

template <FeatureIndexT F>
struct ProbabilityConverter {
  inline static void Convert(Memory<F>* memory) {
    memory->scalar_[kPredictionsScalarAddress] =
        Sigmoid(memory->scalar_[kPredictionsScalarAddress]);
  }
};

template <FeatureIndexT F>
Executor<F>::Executor(const Algorithm& algorithm, const Dataset<F>& dataset,
                      const IntegerT num_all_train_examples,
                      const IntegerT num_valid_examples,
                      RandomGenerator* rand_gen,
                      const double max_abs_error)
    : algorithm_(algorithm),
      dataset_(dataset),
      num_all_train_examples_(num_all_train_examples),
      num_valid_examples_(num_valid_examples),
      rand_gen_(rand_gen),
      max_abs_error_(max_abs_error) {
  memory_.Wipe();
  for (const std::shared_ptr<const Instruction>& instruction :
       algorithm_.setup_) {
    ExecuteInstruction(*instruction, rand_gen_, &memory_);
  }
}

template <FeatureIndexT F>
double Executor<F>::Execute(std::vector<double>* train_errors,
                            std::vector<double>* valid_errors) {
  CHECK_GE(dataset_.NumTrainEpochs(), 1);

  // Iterators that track the progresss of training.
  DatasetIterator<F> train_it = dataset_.TrainIterator();

  // Train for multiple epochs, evaluate on validation set
  // after each epoch and take the best validation result as fitness.
  const IntegerT num_all_train_examples =
      std::min(num_all_train_examples_,
               static_cast<IntegerT>(dataset_.MaxTrainExamples()));
  const IntegerT num_examples_per_epoch =
      dataset_.TrainExamplesPerEpoch() == kNumTrainExamplesNotSet ?
      num_all_train_examples : dataset_.TrainExamplesPerEpoch();
  IntegerT num_remaining = num_all_train_examples;
  double best_fitness = kMinFitness;
  while (num_remaining > 0) {
    if (!Train(
            std::min(num_examples_per_epoch, num_remaining),
            train_errors, &train_it)) {
      if (num_remaining == num_all_train_examples) {
        return kMinFitness;
      } else {
        break;
      }
    }
    num_remaining -= num_examples_per_epoch;
    const double current_fitness = Validate(valid_errors);
    best_fitness = std::max(current_fitness, best_fitness);
    // Only save the errors of the first epoch.
    if (train_errors != nullptr) {
      train_errors = nullptr;
      valid_errors = nullptr;
    }
  }

  return best_fitness;
}

template <FeatureIndexT F>
bool Executor<F>::Train(std::vector<double>* errors) {
  // Iterators that tracks the progresss of training.
  typename std::vector<Vector<F>>::const_iterator train_feature_it =
      dataset_.train_features_.begin();
  typename std::vector<Scalar>::const_iterator train_label_it =
      dataset_.train_labels_.begin();
  const IntegerT num_all_train_examples =
      std::min(num_all_train_examples_,
               static_cast<IntegerT>(dataset_.train_features_.size()));
  return Train(num_all_train_examples, errors, &train_feature_it,
               &train_label_it);
}

// At or above these many steps, we optimize the train component_function.
constexpr IntegerT kTrainStepsOptThreshold = 1000;

template <FeatureIndexT F>
bool Executor<F>::Train(const IntegerT max_steps, std::vector<double>* errors,
                        DatasetIterator<F>* train_it) {
  CHECK(errors == nullptr || max_steps <= 100) <<
      "You should only record the training errors for few training steps."
      << std::endl;
  if (max_steps < kTrainStepsOptThreshold) {
    return TrainNoOptImpl(max_steps, errors, train_it);
  } else {
    if (algorithm_.predict_.size() <= 10 && algorithm_.learn_.size() <= 10) {
      return TrainOptImpl<10>(max_steps, errors, train_it);
    } else if (algorithm_.predict_.size() <= 100 &&
               algorithm_.learn_.size() <= 100) {
      return TrainOptImpl<100>(max_steps, errors, train_it);
    } else if (algorithm_.predict_.size() <= 1000 &&
               algorithm_.learn_.size() <= 1000) {
      return TrainOptImpl<1000>(max_steps, errors, train_it);
    } else {
      LOG(FATAL) << "ComponentFunction size not yet supported." << std::endl;
    }
  }
}

// We don't care that this function is inline. We just want to keep it here,
// next to TrainOptImpl (below).
template <FeatureIndexT F>
inline bool Executor<F>::TrainNoOptImpl(const IntegerT max_steps,
                                        std::vector<double>* errors,
                                        DatasetIterator<F>* train_it) {
  if (errors != nullptr) {
    errors->reserve(max_steps);
  }
  for (IntegerT step = 0; step < max_steps; ++step) {
    // Run predict component_function for this example.
    const Vector<F>& features = train_it->GetFeatures();
    memory_.vector_[kFeaturesVectorAddress] = features;
    ZeroLabelAssigner<F>::Assign(&memory_);
    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm_.predict_) {
      ExecuteInstruction(*instruction, rand_gen_, &memory_);
    }

    if (dataset_.eval_type_ == ACCURACY) {
      ProbabilityConverter<F>::Convert(&memory_);
    }

    // Check whether we should stop early.
    const Scalar& label = train_it->GetLabel();
    const double abs_error = ErrorComputer<F>::Compute(memory_, label);
    if (isnan(abs_error) || abs_error > max_abs_error_) {
      return false;
    }
    if (errors != nullptr) {
      errors->push_back(abs_error);
    }

    // Run learn component_function for this example.
    memory_.vector_[kFeaturesVectorAddress] = features;
    LabelAssigner<F>::Assign(label, &memory_);
    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm_.learn_) {
      ExecuteInstruction(*instruction, rand_gen_, &memory_);
    }

    // Check whether we are done.
    train_it->Next();
    if (train_it->Done()) {
      break;  // Reached the end of the dataset.
    }
  }
  return true;
}

template <FeatureIndexT F>
template <size_t max_component_function_size>
bool Executor<F>::TrainOptImpl(const IntegerT max_steps,
                               std::vector<double>* errors,
                               DatasetIterator<F>* train_it) {
  if (errors != nullptr) {
    errors->reserve(max_steps);
  }

  std::array<Instruction, max_component_function_size>
      optimized_predict_component_function;
  typename std::array<Instruction, max_component_function_size>::iterator
      optimized_predict_instr_it = optimized_predict_component_function.begin();
  for (const std::shared_ptr<const Instruction>& predict_instr :
       algorithm_.predict_) {
    *optimized_predict_instr_it = *predict_instr;
    ++optimized_predict_instr_it;
  }
  const IntegerT num_predict_instr = algorithm_.predict_.size();

  std::array<Instruction, max_component_function_size>
      optimized_learn_component_function;
  typename std::array<Instruction, max_component_function_size>::iterator
      optimized_learn_instr_it = optimized_learn_component_function.begin();
  for (const std::shared_ptr<const Instruction>& learn_instr :
       algorithm_.learn_) {
    *optimized_learn_instr_it = *learn_instr;
    ++optimized_learn_instr_it;
  }
  const IntegerT num_learn_instr = algorithm_.learn_.size();

  for (IntegerT step = 0; step < max_steps; ++step) {
    // Run predict component_function for this example.
    const Vector<F>& features = train_it->GetFeatures();
    memory_.vector_[kFeaturesVectorAddress] = features;
    ZeroLabelAssigner<F>::Assign(&memory_);
    IntegerT predict_instr_num = 0;
    for (const Instruction& instruction :
         optimized_predict_component_function) {
      if (predict_instr_num == num_predict_instr) {
        break;
      }
      ExecuteInstruction(instruction, rand_gen_, &memory_);
      ++predict_instr_num;
    }

    if (dataset_.eval_type_ == ACCURACY) {
      ProbabilityConverter<F>::Convert(&memory_);
    }

    // Check whether we should stop early.
    const Scalar& label = train_it->GetLabel();
    const double abs_error = ErrorComputer<F>::Compute(memory_, label);
    if (isnan(abs_error) || abs_error > max_abs_error_) {
      return false;
    }
    if (errors != nullptr) {
      errors->push_back(abs_error);
    }

    // Run learn component_function for this example.
    memory_.vector_[kFeaturesVectorAddress] = features;
    LabelAssigner<F>::Assign(label, &memory_);
    IntegerT learn_instr_num = 0;
    for (const Instruction& instruction : optimized_learn_component_function) {
      if (learn_instr_num == num_learn_instr) {
        break;
      }
      ExecuteInstruction(instruction, rand_gen_, &memory_);
      ++learn_instr_num;
    }

    // Check whether we are done.
    train_it->Next();
    if (train_it->Done()) {
      break;  // Reached the end of the dataset.
    }
  }
  return true;
}

// Minimum negative error tolerated to account for numerical issue around zero.
constexpr double kNegativeErrorTolerance = -1e-6;

template <FeatureIndexT F>
struct SquashedRmseLossAccumulator {
  inline static void Accumulate(
      const Memory<F>& memory, const Scalar& label,
      double* error, double* loss) {
    *error = label - memory.scalar_[kPredictionsScalarAddress];
    *loss += *error * *error;
  }
};

template <FeatureIndexT F>
struct ProbAccuracyLossAccumulator {
  inline static void Accumulate(
      const Memory<F>& memory, const Scalar& label,
      double* error, double* loss) {
    double logit = memory.scalar_[kPredictionsScalarAddress];
    double pred_prob = Sigmoid(logit);
    if ((pred_prob > 1.0) || (pred_prob < 0.0)) {
      *error = std::numeric_limits<double>::infinity();
    } else {
      bool is_correct = ((label > 0.5) == (pred_prob > 0.5));
      *error = is_correct ? 0.0 : 1.0;
    }
    *loss += *error;
  }
};

template <FeatureIndexT F>
double Executor<F>::Validate(std::vector<double>* errors) {
  double loss = 0.0;
  if (errors != nullptr) {
    errors->reserve(dataset_.ValidSteps());
  }
  const IntegerT num_steps =
      std::min(num_valid_examples_,
               static_cast<IntegerT>(dataset_.ValidSteps()));

  CHECK(errors == nullptr || num_steps <= 100) <<
      "You should only record the validation errors for few validation steps."
      << std::endl;

  DatasetIterator<F> valid_it = dataset_.ValidIterator();
  for (IntegerT step = 0; step < num_steps; ++step) {
    // Run predict component_function for this example.
    const Vector<F>& features = valid_it.GetFeatures();
    memory_.vector_[kFeaturesVectorAddress] = features;
    ZeroLabelAssigner<F>::Assign(&memory_);
    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm_.predict_) {
      ExecuteInstruction(*instruction, rand_gen_, &memory_);
    }

    // Accumulate the loss.
    double error = 0.0;
    const Scalar& label = valid_it.GetLabel();
    switch (dataset_.eval_type_) {
      case RMS_ERROR: {
        SquashedRmseLossAccumulator<F>::Accumulate(memory_, label, &error,
                                                   &loss);
        break;
      }
      case ACCURACY: {
        ProbAccuracyLossAccumulator<F>::Accumulate(memory_, label, &error,
                                                   &loss);
        break;
      }
      case INVALID_EVAL_TYPE:
        LOG(FATAL) << "Invalid eval type." << std::endl;
      // Do not add default case here. All enum values should be supported.
    }

    const double abs_error = std::abs(error);
    if (isnan(abs_error) || abs_error > max_abs_error_) {
      // Stop early. Return infinite loss.
      return kMinFitness;
    }
    if (errors != nullptr) {
      errors->push_back(std::abs(error));
    }

    valid_it.Next();
    if (valid_it.Done()) {
      break;  // Reached the end of the dataset.
    }
  }

  // Convert to fitness.
  double fitness;
  switch (dataset_.eval_type_) {
    case INVALID_EVAL_TYPE:
      LOG(FATAL) << "Invalid eval type." << std::endl;
    case RMS_ERROR:
      loss /= static_cast<double>(dataset_.ValidSteps());
      fitness = FlipAndSquash(sqrt(loss));
      break;
    case ACCURACY:
      loss /= static_cast<double>(dataset_.ValidSteps());
      fitness = 1.0 - loss;
      break;
  }

  return fitness;
}

template <FeatureIndexT F>
void Executor<F>::GetMemory(Memory<F>* memory) {
  memory->scalar_ = memory_.scalar_;
  memory->vector_ = memory_.vector_;
  memory->matrix_ = memory_.matrix_;
}

template <FeatureIndexT F>
void ExecuteAndFillLabels(const Algorithm& algorithm, Memory<F>* memory,
                          DatasetBuffer<F>* buffer,
                          RandomGenerator* rand_gen) {
  // Fill training labels.
  typename std::vector<Scalar>::iterator train_label_it =
      buffer->train_labels_.begin();
  for (const Vector<F>& train_features : buffer->train_features_) {
    // Run predict component_function for this example.
    memory->vector_[kFeaturesVectorAddress] = train_features;
    ZeroLabelAssigner<F>::Assign(memory);
    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm.predict_) {
      ExecuteInstruction(*instruction, rand_gen, memory);
    }
    *train_label_it = PredictionGetter<F>::Get(memory);
    ++train_label_it;
  }

  // Fill validation labels.
  std::vector<Scalar>::iterator valid_label_it =
      buffer->valid_labels_.begin();
  for (const Vector<F>& valid_features : buffer->valid_features_) {
    // Run predict component_function for this example.
    memory->vector_[kFeaturesVectorAddress] = valid_features;
    ZeroLabelAssigner<F>::Assign(memory);
    for (const std::shared_ptr<const Instruction>& instruction :
         algorithm.predict_) {
      ExecuteInstruction(*instruction, rand_gen, memory);
    }
    *valid_label_it = PredictionGetter<F>::Get(memory);
    ++valid_label_it;
  }
}

constexpr double kMinusTwoOverPi = -0.63661977236758138243;

inline double FlipAndSquash(const double value) {
  if (isnan(value) || isinf(value)) {
    return 0.0;
  }
  CHECK_GE(value, 0.0);
  return static_cast<double>(1.0) + kMinusTwoOverPi * atan(value);
}

namespace internal {

template<FeatureIndexT F>
inline Vector<F> TruncatingSoftmax(const Vector<F>& input) {
  // TODO(ereal): rewrite using Eigen's block<>() method.
  // TODO(ereal): consider reusing vectors.
  Vector<kNumClasses> truncated;
  truncated.resize(kNumClasses, 1);
  for (FeatureIndexT i = 0; i < kNumClasses; ++i) {
    truncated(i) = input(i);
  }
  const Vector<kNumClasses> shifted =
      truncated - Vector<kNumClasses>::Ones(kNumClasses) * truncated.maxCoeff();
  const Vector<kNumClasses> exponentiated = shifted.array().exp().matrix();
  const double total = exponentiated.sum();
  const Vector<kNumClasses> normalized = exponentiated / total;
  Vector<F> padded;
  padded.resize(F, 1);
  for (FeatureIndexT i = 0; i < kNumClasses; ++i) {
    padded(i) = normalized(i);
  }
  for (FeatureIndexT i = kNumClasses; i < F; ++i) {
    padded(i) = kPadLabel;
  }
  return padded;
}

template<FeatureIndexT F>
inline FeatureIndexT Argmax(const Vector<F>& input) {
  FeatureIndexT max_index = 0;
  double max_element = std::numeric_limits<double>::lowest();
  for (FeatureIndexT index = 0; index < F; ++index) {
    if (input(index) >= max_element) {
      max_index = index;
      max_element = input(index);
    }
  }
  return max_index;
}

// Computes -x * log(y). Assumes 0.0 <= x,y <= 1.0.
inline double MinusXLogY(const double x, const double y) {
  constexpr double zero = 0.0;
  if (y > zero) {
    return - x * log(y);
  } else {
    if (x > zero) {
      return std::numeric_limits<double>::infinity();
    } else {
      return zero;
    }
  }
}

}  // namespace internal

}  // namespace automl_zero

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_EXECUTOR_H_
