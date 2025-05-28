// Copyright 2025 The Google Research Authors.
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

#ifndef ONLINE_CORRELATION_CLUSTERING_UTILS_H_
#define ONLINE_CORRELATION_CLUSTERING_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

class ComputeCost {
 public:
  // The i-th integer in the input `clustering` is the cluster id of the i-th
  // node of the graph. Computes the correlation clustering objective for the
  // provided clustering and graph.
  static uint64_t ComputeClusteringCost(
      const std::vector<std::vector<int>> neighbors,
      const std::vector<int>& clustering);
};

class RecourseCalculator {
 public:
  int RecourseCostUsingMaxOverlap(std::vector<int> clustering_x,
                                  std::vector<int> clustering_y);

 private:
  std::vector<int> recourse_per_node_seen_;
};

class AgreementReconcileClustering {
 public:
  // Given a new clustering computed by running the agreement algorithms, this
  // method applies the rules for maintaining a low-recourse solution over
  // multiple updates.
  std::vector<int> AgreeementClusteringTransformCost(
      std::vector<int> clustering_old, std::vector<int> clustering_new);
  // The clustering that is maintained throughtout the updates.
  std::vector<int> maintained_clustering_;

 private:
  // The size of the origin cluster of a node.
  std::vector<int> origin_cluster_size_;
};

#endif  // ONLINE_CORRELATION_CLUSTERING_UTILS_H_
