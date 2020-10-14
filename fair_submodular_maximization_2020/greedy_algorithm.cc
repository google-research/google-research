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

// A fancier version of Greedy:
// Keeps a solution always ready, recomputes it when elements arrive/leave
// faster, but keeps k "prefix" copies of function f, so uses more memory
// should be equivalent to SimpleGreedy in what it outputs.

#include "greedy_algorithm.h"

void Greedy::Init(SubmodularFunction& sub_func_f,
                  std::vector<std::pair<int, int>> bounds, int cardinality_k) {
  cardinality_k_ = cardinality_k;
  sub_func_f_ = sub_func_f.Clone();
  sub_func_f_->Reset();
  solution_.clear();
  elements_.clear();
}

void Greedy::Insert(std::pair<int, int> element, bool non_monotone) {
  elements_.insert(element);
}

double Greedy::GetSolutionValue() {
  while (solution_.size() < cardinality_k_) {
    std::pair<double, std::pair<int, int>> best(-1, std::make_pair(-1, -1));
    solution_.push_back(std::make_pair(-1, -1));
    for (auto& element : elements_) {
      solution_[solution_.size() - 1] = element;
      best =
          max(best,
              make_pair(sub_func_f_->ObjectiveAndIncreaseOracleCall(solution_),
                        element));
    }
    if (best.first < 0) break;
    solution_[solution_.size() - 1] = best.second;
  }
  return sub_func_f_->ObjectiveAndIncreaseOracleCall(solution_);
}

std::vector<std::pair<int, int>> Greedy::GetSolutionVector() {
  return solution_;
}

std::string Greedy::GetAlgorithmName() const { return "greedy (optimized)"; }
