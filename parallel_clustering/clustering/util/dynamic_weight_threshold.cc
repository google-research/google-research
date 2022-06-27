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

#include "clustering/util/dynamic_weight_threshold.h"

#include <cmath>

#include "clustering/util/dynamic_weight_threshold.pb.h"
#include "absl/status/status.h"

namespace research_graph {

absl::StatusOr<double> DynamicWeightThreshold(
    const DynamicWeightThresholdConfig& config, int num_iterations,
    int iteration) {
  if (num_iterations < 1)
    return absl::InvalidArgumentError("num_iterations must be >= 1");

  if (iteration < 0 || iteration >= num_iterations)
    return absl::InvalidArgumentError(
        "iteration must be between 0 and num_iterations-1 inclusive.");

  if (num_iterations == 1) {
    if (config.upper_bound() != config.lower_bound()) {
      return absl::InvalidArgumentError(
          "If num_iterations=1, upper and lower bounds must match.");
    }
    return config.upper_bound();
  }
  const double upper_bound = config.upper_bound();
  const double lower_bound = config.lower_bound();
  double dynamic_threshold;
  switch (config.weight_decay_function()) {
    case DynamicWeightThresholdConfig::LINEAR_DECAY:
      dynamic_threshold =
          upper_bound -
          ((upper_bound - lower_bound) / (num_iterations - 1)) * iteration;
      return dynamic_threshold;
    case DynamicWeightThresholdConfig::EXPONENTIAL_DECAY:
      if (lower_bound <= 0 || upper_bound <= 0)
        return absl::InvalidArgumentError(
            "lower and upper bounds need to positive, if EXPONENTIAL_DECAY is "
            "used");

      dynamic_threshold =
          upper_bound * std::pow(lower_bound / upper_bound,
                                 static_cast<double>(iteration) /
                                     static_cast<double>(num_iterations - 1));
      return dynamic_threshold;
    default:
      return absl::InvalidArgumentError(
          "Unsupported weight decay function provided");
  }
}

}  // namespace research_graph
