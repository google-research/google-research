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

#ifndef AUTOML_ZERO_TRAIN_BUDGET_H_
#define AUTOML_ZERO_TRAIN_BUDGET_H_

#include "generator.h"
#include "instruction.h"
#include "train_budget.pb.h"

namespace automl_zero {

// Class to determine how long a given algorithm should be trained for.
class TrainBudget {
 public:
  explicit TrainBudget(
      // A Algorithm to use as a reference. See `TrainExamples`.
      const Algorithm& baseline_algorithm,
      // Fraction of the training time of the baseline above which the
      // Algorithm will be discarded.
      double threshold_factor);

  TrainBudget(const TrainBudget& other) = delete;
  TrainBudget& operator=(const TrainBudget& other) = delete;

  // Returns the number of training examples to use for a given Algorithm.
  // In particular, returns 0 if the algorithm should not be evaluated
  // altogether (because it is too costly).
  // The return value is either `budget` if `algorithm` is less than
  //     threshold_factor * training time of `baseline_algorithm`
  // Otherwise, returns 0, which will cause the algorithm to be assigned
  // the minimum fitness.
  IntegerT TrainExamples(
      // The Algorithm we will be training.
      const Algorithm& algorithm,
      // The compute budget, measured in training examples.
      IntegerT budget) const;

 private:
  // Cost for running each component function once. Measured in compute-units.
  const double baseline_setup_cost_;
  const double baseline_train_cost_;
  const double threshold_factor_;
};

std::unique_ptr<TrainBudget> BuildTrainBudget(
    TrainBudgetSpec train_budget_spec,
    // Generator to create baseline.
    Generator* generator);

}  // namespace automl_zero

#endif  // AUTOML_ZERO_TRAIN_BUDGET_H_
