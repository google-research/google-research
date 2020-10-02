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

//
//  Greedy Algorithm
//

// First version, simple: stores basically nothing (just the available
// universe), and lazily runs greedy whenever getSolutionValue() is called.

#include "simple_greedy.h"

using std::pair;
using std::vector;

void SimpleGreedy::Init(const SubmodularFunction& sub_func_f,
                        int cardinality_k) {
  cardinality_k_ = cardinality_k;
  sub_func_f_ = sub_func_f.Clone();
  available_elements_.clear();
}

void SimpleGreedy::Insert(int element) { available_elements_.insert(element); }

void SimpleGreedy::Erase(int element) {
  if (!available_elements_.count(element)) return;
  available_elements_.erase(element);
}

double SimpleGreedy::GetSolutionValue() {
  sub_func_f_->Reset();
  double obj_val = 0.0;
  for (int i = 0; i < cardinality_k_; ++i) {
    // select best element to add
    pair<double, int> best(-1, -1);
    for (int x : available_elements_) {
      best = max(best,
                 std::make_pair(sub_func_f_->DeltaAndIncreaseOracleCall(x), x));
    }
    if (best.first < 1e-11) {
      // Nothing to add.
      break;
    } else {
      sub_func_f_->AddAndIncreaseOracleCall(best.second);
      obj_val += best.first;
    }
  }
  return obj_val;
}

vector<int> SimpleGreedy::GetSolutionVector() {
  // Untested.
  sub_func_f_->Reset();
  vector<int> solution;
  for (int i = 0; i < cardinality_k_; ++i) {
    // Select best element to add.
    pair<double, int> best(-1, -1);
    for (int x : available_elements_) {
      best = max(best,
                 std::make_pair(sub_func_f_->DeltaAndIncreaseOracleCall(x), x));
    }
    if (best.first < 1e-11) {
      // Nothing to add.
      break;
    } else {
      sub_func_f_->AddAndIncreaseOracleCall(best.second);
      solution.push_back(best.second);
    }
  }
  return solution;
}

std::string SimpleGreedy::GetAlgorithmName() const { return "greedy (simple)"; }
