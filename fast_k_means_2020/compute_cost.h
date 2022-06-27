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

// Two utility functions useful for computing the distance and clustering.

#ifndef FAST_K_MEANS_2020_COMPUTE_COST_H_
#define FAST_K_MEANS_2020_COMPUTE_COST_H_

#include <vector>

namespace fast_k_means {

class ComputeCost {
 public:
  // Returns the cost of the solution using the centers.
  // Expects ID of the centers, not the coordinates.
  static double GetCost(const std::vector<std::vector<double>>& input_point,
                        const std::vector<int>& centers);

  // Returns the cost of the solution using the centers.
  // Expects the coordinates.
  static double GetCost(const std::vector<std::vector<double>>& input_point,
                        const std::vector<std::vector<double>>& centers);

  // Computes the D^2 distance between two points.
  static double CompDis(const std::vector<std::vector<double>>& input_point,
                        int point_x, int point_y);
  // Computes the D^2 distance between a point and a center.
  static double CompDis(const std::vector<std::vector<double>>& input_point,
                        const std::vector<std::vector<double>>& centers,
                        int point_x, int center_y);
};

}  // namespace fast_k_means

#endif  // FAST_K_MEANS_2020_COMPUTE_COST_H_
