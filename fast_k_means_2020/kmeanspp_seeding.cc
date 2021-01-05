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

#include "kmeanspp_seeding.h"

#include <cstdint>

#include "compute_cost.h"
#include "random_handler.h"

namespace fast_k_means {

using std::pair;
using std::vector;

int KMeansPPSeeding::ReturnD2Sample(const vector<vector<double>>& input) {
  double total_prob = 0;
  for (int i = 0; i < input.size(); i++) total_prob += distance[i];
  unsigned long long_t rand_ll_int =
      RandomHandler::eng() % static_cast<unsigned long long_t>(total_prob);
  double rand_ll = static_cast<double>(rand_ll_int);
  int picked_center = input.size() - 1;
  for (int i = 0; i < input.size(); i++) {
    if (rand_ll < distance[i]) {
      picked_center = i;
      break;
    }
    rand_ll -= distance[i];
    }
    return picked_center;
}

int KMeansPPSeeding::ReturnBestCenter(const vector<vector<double>>& input,
                                      bool first_round,
                                      int number_greedy_rounds) {
  pair<int, double> best_center_and_improvement(0, 0.0);
  for (int i = 0; i < number_greedy_rounds; i++) {
    int picked_center = ReturnD2Sample(input);
    if (first_round) return picked_center;
    double improvement = ComputeImprovement(input, picked_center);
    if (improvement > best_center_and_improvement.second) {
      best_center_and_improvement.first = picked_center;
      best_center_and_improvement.second = improvement;
    }
  }
  return best_center_and_improvement.first;
}

void KMeansPPSeeding::RunAlgorithm(const vector<vector<double>>& input, int k,
                                   int number_greedy_rounds) {
  for (int i = 0; i < input.size(); i++) distance.push_back(1);
  for (int i = 0; i < k; i++) {
    int picked_center = ReturnBestCenter(input, i == 0, number_greedy_rounds);
    centers_.push_back(picked_center);
    UpdateDistance(input);
  }
}

void KMeansPPSeeding::UpdateDistance(const vector<vector<double>>& input) {
  if (centers_.size() == 1) {
    for (int i = 0; i < input.size(); i++)
      distance[i] = ComputeCost::CompDis(input, centers_[0], i);
    return;
  }
  for (int i = 0; i < input.size(); i++)
    distance[i] =
        std::min(distance[i],
                 ComputeCost::CompDis(input, centers_[centers_.size() - 1], i));
}

double KMeansPPSeeding::ComputeImprovement(const vector<vector<double>>& input,
                                           int center) {
  double improvement = 0.0;
  for (int i = 0; i < input.size(); i++)
    improvement += distance[i] - std::min(distance[i], ComputeCost::CompDis(
                                                           input, center, i));
  return improvement;
}

}  //  namespace fast_k_means
