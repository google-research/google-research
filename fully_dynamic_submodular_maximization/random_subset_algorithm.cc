// Copyright 2020 The Google Research Authors.
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

#include "random_subset_algorithm.h"

#include "utilities.h"

using std::vector;

void RandomSubsetAlgorithm::Initialization(const SubmodularFunction& sub_func_f,
                                           int cardinality_k) {
  cardinality_k_ = cardinality_k;
  sub_func_f_ = sub_func_f.Clone();
  universe_elements_.clear();
  solution_.clear();
}

void RandomSubsetAlgorithm::Insert(int element) {
  if (static_cast<int>(solution_.size()) < cardinality_k_) {
    solution_.push_back(element);
  } else {
    universe_elements_.push_back(element);
    const double probability_in_random_solution =
        static_cast<double>(cardinality_k_) /
        (cardinality_k_ + static_cast<int>(universe_elements_.size()));
    const double random_value =
        static_cast<double>(RandomHandler::generator_()) /
        RandomHandler::generator_.max();
    if (random_value < probability_in_random_solution) {
      // Swap e with a random element from the solution.
      std::swap(universe_elements_.back(),
                solution_[RandomHandler::generator_() % cardinality_k_]);
    }
  }
}

void RandomSubsetAlgorithm::Erase(int element) {
  auto it_element = find(solution_.begin(), solution_.end(), element);
  if (it_element != solution_.end()) {
    solution_.erase(it_element);
    // Find a replacement in other_elements.
    if (!universe_elements_.empty()) {
      size_t random_index =
          RandomHandler::generator_() % universe_elements_.size();
      solution_.push_back(universe_elements_[random_index]);
      universe_elements_.erase(universe_elements_.begin() + random_index);
    }
    return;
  }

  it_element =
      find(universe_elements_.begin(), universe_elements_.end(), element);
  if (it_element != universe_elements_.end()) {
    universe_elements_.erase(it_element);
  }
}

double RandomSubsetAlgorithm::GetSolutionValue() {
  sub_func_f_->Reset();
  double obj_val = 0.0;
  for (int element_in_solution : solution_) {
    obj_val += sub_func_f_->AddAndIncreaseOracleCall(element_in_solution, -1);
  }
  SubmodularFunction::oracle_calls_ -=
      2 * static_cast<int>(solution_.size()) - 1;
  // Should be just one call in our model.
  return obj_val;
}

vector<int> RandomSubsetAlgorithm::GetSolutionVector() { return solution_; }

std::string RandomSubsetAlgorithm::GetAlgorithmName() const {
  return "totally random algorithm";
}
