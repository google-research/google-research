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

#include "clustering.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace fair_secretary {

using std::string;
using std::vector;

// The distance between points.
double Distance(const vector<int>& a, const vector<int>& b) {
  double dist = 0;
  for (int i = 0; i < a.size(); i++) {
    double differ = a[i] - b[i];
    dist += differ * differ;
  }
  return sqrt(dist);
}

// Cost of opening the center.
double Cost(const vector<vector<int>>& input, const vector<int>& center) {
  double cost = 0;
  for (const auto& point : input) {
    cost += Distance(point, center);
  }
  return cost;
}

vector<SecretaryInstance> ClusteringOracle::GetSecretaryInput(
    int num_elements) {
  // Path to the input dataset.
  string input_path = "";
  std::ifstream in(input_path);
  string input;
  // Ignoring the first line.
  vector<SecretaryInstance> instance;
  int n, d;
  in >> n >> d;
  vector<vector<int>> cor(n, vector<int>(d));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      in >> cor[i][j];
    }
  }
  instance.reserve(n);
  for (int i = 0; i < n; i++) {
    instance.push_back(SecretaryInstance(Cost(cor, cor[i]), cor[i][8]));
  }
  num_colors = 0;
  for (int i = 0; i < instance.size(); i++) {
    num_colors = std::max(num_colors, instance[i].color);
  }
  num_colors++;
  return instance;
}

}  //  namespace fair_secretary
