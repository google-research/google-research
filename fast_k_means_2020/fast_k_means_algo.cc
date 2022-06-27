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

#include "fast_k_means_algo.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace fast_k_means {

void FastKMeansAlgo::RunAlgorithm(const vector<vector<double>>& input, int k,
                                  int number_of_trees, double scaling_factor,
                                  int number_greedy_rounds) {
  multi_trees_.InitializeTree(input, number_of_trees, scaling_factor);
  for (int i = 0; i < k; i++) {
    pair<int, uint64_t> best_center_and_improvement (0, 0);
    for (int j = 0; j < number_greedy_rounds; j++) {
      int next_center = multi_trees_.SampleAPoint();
      if (next_center == -1) break;
      uint64_t improvement =
          multi_trees_.ComputeCostAndOpen(next_center, false);
      // For the case of i = 0, it is important to have equality here.
      if (improvement >= best_center_and_improvement.second) {
        best_center_and_improvement.first = next_center;
        best_center_and_improvement.second = improvement;
      }
    }
    centers.push_back(best_center_and_improvement.first);
    multi_trees_.ComputeCostAndOpen(best_center_and_improvement.first, true);
  }
}

vector<int> FastKMeansAlgo::GetAssignment() {
  return multi_trees_.closets_open_center;
}
}  // namespace fast_k_means
