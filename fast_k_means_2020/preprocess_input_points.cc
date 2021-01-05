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

#include "preprocess_input_points.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace fast_k_means {

using std::max;
using std::min;
using std::vector;

void PreProcessInputPoints::ShiftToDimensionsZero(
    vector<vector<int>>* input_points) {
  for (int j = 0; j < (*input_points)[0].size(); j++) {
    int min_coordinate = std::numeric_limits<int>::max();
    for (int i = 0; i < input_points->size(); i++)
      min_coordinate = min(min_coordinate, (*input_points)[i][j]);
    for (int i = 0; i < input_points->size(); i++)
      (*input_points)[i][j] -= min_coordinate;
  }
}

void PreProcessInputPoints::RandomShiftSpace(
    vector<vector<int>>* input_points) {
  for (int j = 0; j < (*input_points)[0].size(); j++) {
    int max_coordinate = 0;
    for (int i = 0; i < input_points->size(); i++)
      max_coordinate = max(max_coordinate, (*input_points)[i][j]);
    unsigned long long_t shift =
        fast_k_means::RandomHandler::eng() % max(1, max_coordinate);
    for (int i = 0; i < input_points->size(); i++)
      (*input_points)[i][j] += shift;
  }
}

vector<vector<int>> PreProcessInputPoints::ScaleToIntSpace(
    const vector<vector<double>>& input_point_double, double scaling_factor) {
  vector<vector<int>> input_points;
  input_points.reserve(input_point_double.size());
  for (int i = 0; i < input_point_double.size(); i++)
    input_points.push_back(vector<int>(input_point_double[0].size()));
  for (int j = 0; j < input_point_double[0].size(); j++)
    for (int i = 0; i < input_point_double.size(); i++)
      input_points[i][j] =
          static_cast<int>(input_point_double[i][j] * scaling_factor);
  return input_points;
}

}  // namespace fast_k_means
