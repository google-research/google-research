// Copyright 2023 The Authors.
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

#include "bank_data.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <vector>

#include "absl/container/flat_hash_map.h"

std::vector<std::vector<double>> ReadData(const char input_path[]) {
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

  FILE* file = fopen(input_path, "r");
  int n, d;
  std::vector<std::vector<double>> input;
  fscanf(file, "%d %d", &n, &d);
  for (int i = 0; i < n; i++) {
    input.push_back(std::vector<double>(d));
    for (int j = 0; j < d; j++) fscanf(file, "%lf", &input[i][j]);
  }
  std::cout << "number of nodes, number of dimensions:" << n << " " << d
            << std::endl;
  return input;
}

BankData::BankData(const char input_path[]) {
  input_ = ReadData(input_path);
  age_grpcards_ = std::vector<int>(6, 0);
  balance_grpcards_ = std::vector<int>(5, 0);
  age_map_.clear();
  balance_map_.clear();
  double minn = 100000;
  double maxx = 0;
  for (int i = 0; i < input_.size(); i++) {
    // Assigning age group between 0-5 and balance group beween 0-10 to the
    // nodes
    int age_group = input_[i][0] / 10 - 2;
    age_group = std::max(age_group, 0);
    age_group = std::min(age_group, 5);
    age_map_[i] = age_group;
    age_grpcards_[age_group]++;

    minn = std::min(minn, input_[i][1]);
    maxx = std::max(maxx, input_[i][1]);
    int balance_group = input_[i][1] / 2000 + 1;
    balance_group = std::max(balance_group, 0);
    balance_group = std::min(balance_group, (int)balance_grpcards_.size());
    balance_map_[i] = balance_group;
    balance_grpcards_[balance_group]++;
  }
  std::cout << "min max: " << minn << " " << maxx << std::endl;
  std::cout << "Age groups cardinalities: ";
  for (int i = 0; i < age_grpcards_.size(); i++) {
    std::cout << age_grpcards_[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Balance groups cardinalities: ";
  for (int i = 0; i < balance_grpcards_.size(); i++) {
    std::cout << balance_grpcards_[i] << " ";
  }
  std::cout << std::endl;
}
