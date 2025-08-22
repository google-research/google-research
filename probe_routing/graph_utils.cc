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

#include "graph_utils.h"

#include "absl/log/check.h"

namespace geo_algorithms {

double WeightedCost(const std::vector<double>& cost_weights,
                    const std::vector<double>& cost_vector) {
  CHECK_EQ(cost_weights.size(), cost_vector.size());
  double result = 0.0;
  for (int i = 0; i < cost_weights.size(); i++) {
    result += cost_weights[i] * cost_vector[i];
  }
  return result;
}

std::vector<double> AddCostVectors(const std::vector<double>& a,
                                   const std::vector<double>& b) {
  CHECK_EQ(a.size(), b.size());
  std::vector<double> sum;
  for (int i = 0; i < a.size(); i++) {
    sum.push_back(a[i] + b[i]);
  }
  return sum;
}

std::vector<double> SubtractCostVectors(const std::vector<double>& a,
                                        const std::vector<double>& b) {
  CHECK_EQ(a.size(), b.size());
  std::vector<double> diff;
  for (int i = 0; i < a.size(); i++) {
    diff.push_back(a[i] - b[i]);
  }
  return diff;
}

}  // namespace geo_algorithms
