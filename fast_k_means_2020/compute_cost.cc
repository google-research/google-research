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

#include "compute_cost.h"

#include <iostream>

namespace fast_k_means {

double ComputeCost::CompDis(const std::vector<std::vector<double>>& input_point,
                            const std::vector<std::vector<double>>& centers,
                            int point_x, int center_y) {
  double dist = 0;
  for (int i = 0; i < input_point[0].size(); i++)
    dist += (input_point[point_x][i] - centers[center_y][i]) *
            (input_point[point_x][i] - centers[center_y][i]);
  return dist;
}

double ComputeCost::CompDis(const std::vector<std::vector<double>>& input_point,
                            int point_x, int point_y) {
  double dist = 0;
  for (int i = 0; i < input_point[0].size(); i++)
    dist += (input_point[point_x][i] - input_point[point_y][i]) *
            (input_point[point_x][i] - input_point[point_y][i]);
  return dist;
}

double ComputeCost::GetCost(const std::vector<std::vector<double>>& input_point,
                            const std::vector<int>& centers) {
  double total_cost = 0;
  for (int i = 0; i < input_point.size(); i++) {
    double closest = 0.0;
    for (int j = 0; j < centers.size(); j++)
      if (j == 0)
        closest = CompDis(input_point, centers[j], i);
      else
        closest = std::min(closest, CompDis(input_point, centers[j], i));
    total_cost += closest;
  }
  return total_cost;
}

double ComputeCost::GetCost(const std::vector<std::vector<double>>& input_point,
                            const std::vector<std::vector<double>>& centers) {
  double total_cost = 0;
  for (int i = 0; i < input_point.size(); i++) {
    double closest = 0.0;
    for (int j = 0; j < centers.size(); j++)
      if (j == 0)
        closest = CompDis(input_point, centers, i, j);
      else
        closest = std::min(closest, CompDis(input_point, centers, i, j));
    total_cost += closest;
  }
  return total_cost;
}

}  // namespace fast_k_means
