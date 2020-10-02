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

void Greedy::InsertIntoSolution(int element) {
  partial_F_.emplace_back(partial_F_.back()->Clone());
  double delta_e = partial_F_.back()->AddAndIncreaseOracleCall(element, -1);
  solution_.push_back(element);
  obj_vals_.push_back(obj_vals_.back() + delta_e);
}

// Remove the last element from the solution.
void Greedy::RemoveLastElement() {
  if (solution_.empty()) {
    Fail("trying to remove element from empty solution");
  }
  obj_vals_.pop_back();
  solution_.pop_back();
  partial_F_.pop_back();
}

// Complete the solution to k elements.
void Greedy::Complete() {
  while (static_cast<int>(solution_.size()) < cardinality_k_) {
    // select best element to add
    std::pair<double, int> best(-1, -1);
    for (int x : available_elements_) {
      best = max(
          best,
          std::make_pair(partial_F_.back()->DeltaAndIncreaseOracleCall(x), x));
    }
    if (best.first < 1e-11) {
      // nothing to add
      break;
    } else {
      InsertIntoSolution(best.second);
    }
  }
}

void Greedy::Init(const SubmodularFunction& sub_func_f, int cardinality_k) {
  cardinality_k_ = cardinality_k;
  partial_F_.clear();
  partial_F_.emplace_back(sub_func_f.Clone());
  solution_.clear();
  obj_vals_ = {0.0};
  available_elements_.clear();
}

void Greedy::Insert(int element) {
  available_elements_.insert(element);
  // Check if e should have been inserted at some point (if we had run greedy
  // with e present).
  for (int i = 0;
       i < std::min(cardinality_k_, static_cast<int>(partial_F_.size())); ++i) {
    // If full: partial_F[0], ..., partial_F[k], we go 0..k-1 (solution is
    // 0..k-1) not full: partial_F[0], ..., partial_F[l], we go 0..l (solution
    // is 0..l-1) should we add it as the (i+1)-th element?
    double old_delta = (i == static_cast<int>(solution_.size()))
                           ? 0
                           : (obj_vals_[i + 1] - obj_vals_[i]);
    if (partial_F_[i]->DeltaAndIncreaseOracleCall(element) > old_delta) {
      // Yes.
      // Remove last element until only i remain.
      while (static_cast<int>(solution_.size()) > i) {
        RemoveLastElement();
      }
      InsertIntoSolution(element);
      Complete();
      break;
    }
  }
}

void Greedy::Erase(int element) {
  if (!available_elements_.count(element)) return;
  available_elements_.erase(element);
  // Check if it was part of the solution.
  for (int i = 0; i < static_cast<int>(solution_.size()); ++i) {
    if (element == solution_[i]) {
      while (static_cast<int>(solution_.size()) > i) {
        RemoveLastElement();
      }
      Complete();
      break;
    }
  }
}

double Greedy::Greedy::GetSolutionValue() { return obj_vals_.back(); }

std::vector<int> Greedy::GetSolutionVector() { return solution_; }

std::string Greedy::GetAlgorithmName() const { return "greedy (optimized)"; }
