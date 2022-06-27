// Copyright 2022 The Google Research Authors.
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

#include "train_budget.h"

#include "algorithm.h"
#include "compute_cost.h"
#include "absl/memory/memory.h"

namespace automl_zero {

using ::absl::make_unique;  // NOLINT
using ::std::endl;  // NOLINT
using ::std::unique_ptr;  // NOLINT

TrainBudget::TrainBudget(
    const Algorithm& baseline_algorithm, const double threshold_factor)
    : baseline_setup_cost_(
          ComputeCost(baseline_algorithm.setup_)),
      baseline_train_cost_(
          ComputeCost(baseline_algorithm.predict_) +
          ComputeCost(baseline_algorithm.learn_)),
      threshold_factor_(threshold_factor) {}

IntegerT TrainBudget::TrainExamples(
    const Algorithm& algorithm, const IntegerT budget) const {
  const double setup_cost = ComputeCost(algorithm.setup_);
  const double train_cost =
      ComputeCost(algorithm.predict_) + ComputeCost(algorithm.learn_);
  CHECK_GT(train_cost, 0.0);
  const double suggested_train_examples = static_cast<double>(budget);
  const double baseline_cost =
      baseline_setup_cost_ + suggested_train_examples * baseline_train_cost_;
  const double suggested_cost =
      setup_cost + suggested_train_examples * train_cost;
  if (suggested_cost <= baseline_cost * threshold_factor_) {
    return budget;
  } else {
    return 0;
  }
}

unique_ptr<TrainBudget> BuildTrainBudget(
    TrainBudgetSpec train_budget_spec, Generator* generator) {
  const HardcodedAlgorithmID baseline_id =
      static_cast<HardcodedAlgorithmID>(
          train_budget_spec.train_budget_baseline());
  const Algorithm baseline_algorithm = generator->ModelByID(baseline_id);
  return make_unique<TrainBudget>(
      baseline_algorithm, train_budget_spec.train_budget_threshold_factor());
}

}  // namespace automl_zero
