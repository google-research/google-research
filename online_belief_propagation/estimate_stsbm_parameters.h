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

#ifndef ONLINE_BELIEF_PROPAGATION_ESTIMATE_STSBM_PARAMETERS_H_
#define ONLINE_BELIEF_PROPAGATION_ESTIMATE_STSBM_PARAMETERS_H_

#include "graph.h"

// Parameters for generating a random graph according to the Symmetric
// Stochastic Block Model (StSBM).
struct StSBMParameters {
  // Number of vertices.
  int n;
  // Number of communities.
  int k;
  // n x (probability of two vertices beloging to the same community being
  // connected)
  double a;
  // n x (probability of two vertices beloging to different communities being
  // connected)
  double b;
};

// Given a graph and the ground-truth communities of its vertices, estimates the
// StSBM parameters, as described in Section 6.3 of the paper. For each index i,
// ground_truth_communities[i] indicates the community to which vertex i
// belongs. This function assumes that the set of elements of
// 'ground_truth_communities', after removing duplicates, is of the form {0, 1,
// ..., k - 1} for some value of k.
StSBMParameters EstimateStSBMParameters(
    const Graph& graph, const std::vector<int> ground_truth_communities);

#endif  // ONLINE_BELIEF_PROPAGATION_ESTIMATE_STSBM_PARAMETERS_H_
