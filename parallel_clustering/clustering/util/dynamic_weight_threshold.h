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

#ifndef PARALLEL_CLUSTERING_CLUSTERING_UTIL_DYNAMIC_WEIGHT_THRESHOLD_H_
#define PARALLEL_CLUSTERING_CLUSTERING_UTIL_DYNAMIC_WEIGHT_THRESHOLD_H_

#include "clustering/util/dynamic_weight_threshold.pb.h"
#include "absl/status/statusor.h"

namespace research_graph {

// Computes weight threshold for the given iteration of affinity clustering,
// based on the provided DynamicWeightThresholdConfig.
absl::StatusOr<double> DynamicWeightThreshold(
    const DynamicWeightThresholdConfig& config, int num_iteration,
    int iteration);

}  // namespace research_graph

#endif  // PARALLEL_CLUSTERING_CLUSTERING_UTIL_DYNAMIC_WEIGHT_THRESHOLD_H_
