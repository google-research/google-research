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

#ifndef FAST_K_MEANS_2020_REJECTION_SAMPLING_LSH_H_
#define FAST_K_MEANS_2020_REJECTION_SAMPLING_LSH_H_

#include "lsh.h"
#include "multi_tree_clustering.h"
#include "random_handler.h"

namespace fast_k_means {

class RejectionSamplingLSH {
 public:
  // Runs the rejection sampling algorithm and stores the results in centers_.
  void RunAlgorithm(const vector<vector<double>>& input, int k,
                    int number_of_trees, double scaling_factor,
                    int number_greedy_rounds, double boosting_prob_factor);

  // Return the assignment of the points to the centers.
  vector<int> GetAssignment();
  vector<int> centers;

 private:
  MultiTreeClustering multi_trees_;
};

}  // namespace fast_k_means

#endif  // FAST_K_MEANS_2020_REJECTION_SAMPLING_LSH_H_
