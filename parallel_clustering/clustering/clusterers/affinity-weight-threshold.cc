// Copyright 2021 The Google Research Authors.
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

#include "clustering/clusterers/affinity-weight-threshold.h"

#include <algorithm>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "clustering/config.pb.h"
#include "clustering/util/dynamic_weight_threshold.h"

namespace research_graph::in_memory {

absl::StatusOr<double> AffinityWeightThreshold(
    const AffinityClustererConfig& config, int iteration) {
  if (iteration < 0)
    return absl::InvalidArgumentError(
        "Affinity clustering iteration number must be nonnegative");
  switch (config.weight_threshold_config_case()) {
    case AffinityClustererConfig::kWeightThreshold:
      return config.weight_threshold();

    case AffinityClustererConfig::kPerIterationWeightThresholds: {
      int num_thresholds =
          config.per_iteration_weight_thresholds().thresholds_size();
      if (num_thresholds == 0) return 0.0;
      return config.per_iteration_weight_thresholds().thresholds(
          std::min(iteration, num_thresholds - 1));
    }

    case AffinityClustererConfig::kDynamicWeightThresholdConfig:
      return DynamicWeightThreshold(config.dynamic_weight_threshold_config(),
                                    config.num_iterations(), iteration);

    case AffinityClustererConfig::WEIGHT_THRESHOLD_CONFIG_NOT_SET:
      return 0.0;

    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown weight_threshold_config setting: ",
                       config.weight_threshold_config_case()));
  }
}

}  // namespace research_graph::in_memory
