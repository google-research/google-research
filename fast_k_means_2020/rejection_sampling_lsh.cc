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

#include "rejection_sampling_lsh.h"

#include <algorithm>

#include "compute_cost.h"
#include "random_handler.h"

namespace fast_k_means {

void RejectionSamplingLSH::RunAlgorithm(const vector<vector<double>>& input,
                                        int k, int number_of_trees,
                                        double scaling_factor,
                                        int number_greedy_rounds,
                                        double boosting_prob_factor) {
  // Initializing LSH.
  // Size of lsh
  int size_lsh = 15;
  LSHDataStructure lsh_(10, size_lsh, input[0].size());
  // Running clustering algorithm.
  multi_trees_.InitializeTree(input, number_of_trees, scaling_factor);
  double max_prob = 0.0;
  while (centers.size() < k) {
    pair<int, unsigned long long_t> best_center_and_improvement(0, 0);
    // Number of the times that we successfully sample.
    int number_sampled = 0;
    while (number_sampled < number_greedy_rounds) {
      int next_center = multi_trees_.SampleAPoint();
      double prob = 1.0;
      if (!centers.empty()) {
        prob =
            lsh_.QueryPoint(input[next_center], size_lsh) /
            ((multi_trees_.distance_to_center[next_center] * input[0].size()) /
             (scaling_factor * scaling_factor));
        prob *= boosting_prob_factor;
        max_prob = std::max(prob, max_prob);
      }
      if (static_cast<double>(RandomHandler::eng() /
                              std::numeric_limits<unsigned long long_t>::max()) > prob)
        continue;
      unsigned long long_t improvement =
          multi_trees_.ComputeCostAndOpen(next_center, false);
      if (improvement >= best_center_and_improvement.second) {
        best_center_and_improvement.first = next_center;
        best_center_and_improvement.second = improvement;
      }
      number_sampled++;
    }
    multi_trees_.ComputeCostAndOpen(best_center_and_improvement.first, true);
    centers.push_back(best_center_and_improvement.first);
    lsh_.InsertPoint(best_center_and_improvement.first,
                     input[best_center_and_improvement.first]);
  }
}

vector<int> RejectionSamplingLSH::GetAssignment() {
  return multi_trees_.closets_open_center;
}

}  // namespace fast_k_means
