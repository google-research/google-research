// Copyright 2020 The Google Research Authors.
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

#ifndef FAST_K_MEANS_2020_KMEANSPP_SEEDING_H_
#define FAST_K_MEANS_2020_KMEANSPP_SEEDING_H_

#include "compute_cost.h"
#include "random_handler.h"

namespace fast_k_means {

class KMeansPPSeeding {
 public:
  int ReturnD2Sample(const std::vector<std::vector<double>>& input);
  void RunAlgorithm(const std::vector<std::vector<double>>& input, int k,
                    int number_greedy_rounds);
  void UpdateDistance(const std::vector<std::vector<double>>& input);
  double ComputeImprovement(const std::vector<std::vector<double>>& input,
                            int center);
  int ReturnBestCenter(const std::vector<std::vector<double>>& input,
                       bool first_round, int number_greedy_rounds);
  std::vector<double> distance;
  std::vector<int> centers_;
};

}  //  namespace fast_k_means

#endif  // FAST_K_MEANS_2020_KMEANSPP_SEEDING_H_
