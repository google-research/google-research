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

#include "belief_propagation_utils.h"

#include <cmath>

std::vector<double> WeightFunction(const std::vector<double>& x,
                                   const int num_clusters,
                                   const double num_in_cluster_edges,
                                   const double num_between_cluster_edges) {
  std::vector<double> weights(num_clusters, 0);
  // Same as i, j variables in the paper.
  // j-> dimension and i-> summation.
  for (int j = 0; j < num_clusters; j++) {
    for (int i = 0; i < num_clusters; i++) {
      if (i != j) {
        weights[j] += (exp(x[i]) * num_between_cluster_edges);
      }
    }
    weights[j] += (exp(x[j]) * num_in_cluster_edges);
    weights[j] = log(weights[j]);
  }
  return weights;
}
