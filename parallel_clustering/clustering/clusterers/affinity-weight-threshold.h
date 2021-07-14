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

#ifndef PARALLEL_CLUSTERING_CLUSTERING_CLUSTERERS_AFFINITY_WEIGHT_THRESHOLD_H_
#define PARALLEL_CLUSTERING_CLUSTERING_CLUSTERERS_AFFINITY_WEIGHT_THRESHOLD_H_

#include "absl/status/statusor.h"
#include "clustering/config.pb.h"

namespace research_graph::in_memory {

// Gives the edge weight threshold used in the provided iteration of affinity
// clustering, depending on the provided config.
// If none of the weight_threshold_config fields are set, returns 0.0.
// Returns an error in case of invalid arguments.
absl::StatusOr<double> AffinityWeightThreshold(
    const AffinityClustererConfig& config, int iteration);

}  // namespace research_graph::in_memory

#endif  // PARALLEL_CLUSTERING_CLUSTERING_CLUSTERERS_AFFINITY_WEIGHT_THRESHOLD_H_
