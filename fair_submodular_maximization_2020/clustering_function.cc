// Copyright 2020 The Authors.
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

#include "clustering_function.h"

#include <cstdio>

void ClusteringFunction::Reset() {
  solution_.clear();
  max_value_ = 0;
}

std::vector<std::pair<int, int>> ClusteringFunction::Init(
    std::string experiment_name) {
  // Input format:
  // - in the first line we expect the number of points followed by the
  //   number of dimensions.
  // - each line after that is the dimension of point number 'i', space
  //   separated.
  // For example:
  // 3 2
  // 1.5 2.0
  // 11 12.32
  // 3 4

  // The path that the input is stored.
  char input_path[] = "";
  FILE* file = fopen(input_path, "r");
  int n, d;
  fscanf(file, "%d %d", &n, &d);
  for (int i = 0; i < n; i++) {
    input_.push_back(std::vector<double>(d));
    for (int j = 0; j < d; j++) fscanf(file, "%lf", &input_[i][j]);
  }
  for (int i = 1; i < n; i++) max_value_ += distance(0, i) * 2;
  std::cerr << "number of nodes, number of dimensions, max value: " << n << " "
            << d << " " << max_value_ << std::endl;
  if (experiment_name == "bank") return BankPrep();
  return std::vector<std::pair<int, int>>();
}

const std::vector<std::pair<int, int>>& ClusteringFunction::GetUniverse()
    const {
  return universe_;
}

std::string ClusteringFunction::GetName() const { return "ClusteringFunction"; }

std::unique_ptr<SubmodularFunction> ClusteringFunction::Clone() const {
  return std::make_unique<ClusteringFunction>(*this);
}

double ClusteringFunction::GetMaxValue() { return max_value_; }

void ClusteringFunction::Add(std::pair<int, int> element) {
  solution_.push_back(element);
}

double ClusteringFunction::Delta(std::pair<int, int> element) {
  double answer = Objective(solution_);
  solution_.push_back(element);
  answer = Objective(solution_) - answer;
  solution_.pop_back();
  return answer;
}

double ClusteringFunction::Objective(
    const std::vector<std::pair<int, int>>& elements) const {
  if (elements.empty()) {
    return 0;
  }

  double cost = max_value_;
  for (int i = 0; i < input_.size(); i++) {
    double min_dist = distance(i, elements[0].first);
    for (int j = 1; j < elements.size(); j++) {
      min_dist = std::min(min_dist, distance(i, elements[j].first));
    }
    cost -= min_dist;
  }
  return cost;
}

double ClusteringFunction::distance(int x, int y) const {
  double answer = 0;
  for (int i = 0; i < input_[x].size(); i++) {
    answer += (input_[x][i] - input_[y][i]) * (input_[x][i] - input_[y][i]);
  }
  return answer;
}

std::vector<std::pair<int, int>> ClusteringFunction::BankPrep() {
  std::vector<int> occurance(6, 0);
  for (int i = 0; i < input_.size(); i++) {
    // Assigning values between 0-5 to the nodes
    int color = input_[i][0] / 10 - 2;
    color = std::max(color, 0);
    color = std::min(color, 5);
    universe_.push_back(std::pair<int, int>(i, color));
    occurance[color]++;
  }
  std::vector<std::pair<int, int>> bounds;
  std::cout << "occurrences: ";
  for (int i = 0; i < occurance.size(); i++) {
    std::cout << occurance[i] << " ";
    bounds.push_back(std::pair<int, int>(6, 8));
  }
  std::cout << std::endl;
  return bounds;
}
