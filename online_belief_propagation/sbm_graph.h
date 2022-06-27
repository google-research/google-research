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

#ifndef ONLINE_BELIEF_PROPAGATION_SBM_GRAPH_H_
#define ONLINE_BELIEF_PROPAGATION_SBM_GRAPH_H_

#include "graph.h"

// Class for generating graphs according to the stochastic block model
// distribution.
class SBMGraph : public Graph {
 public:
  // Generates graph according to the stochastic block model distribution with
  // 'n' vertices and 'k' blocks. Each vertex is assigned uniformly at random
  // to a block. Edges within blocks appear with probability a/n, while edges
  // between blocks appear with probability b/n. All parameters (other than
  // 'seed') should be non-negative, and 'a' should be greater than or equal to
  // 'b'.
  SBMGraph(int64_t n, int k, double a, double b, int seed);
  const std::vector<int>& GetGroundTruthCommunities() const;

 protected:
  // The true underlying clustering
  std::vector<int> ground_truth_communities_;
};

#endif  // ONLINE_BELIEF_PROPAGATION_SBM_GRAPH_H_
