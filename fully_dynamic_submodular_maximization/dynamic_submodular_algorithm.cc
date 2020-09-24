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

//
//             Our algorithm
//
// The simple algorithm from Section 3 of the paper.
// Reconstructs starting from level l_begin.

#include "dynamic_submodular_algorithm.h"

using std::vector;

void OurSimpleAlgorithm::OurSimpleAlgorithmSingleThreshold::LevelConstruct(
    int l_begin) {
  // For levels lower than l_begin, just condition f on the S-sets of those
  // levels.

  // When we are at l, this is the size of S_1 u S_2 u ... S_l.
  size_of_S_ = 0;
  sub_func_f_.Reset();
  for (int l = 0; l < l_begin; ++l) {
    for (int e : solutions_S_[l]) {
      sub_func_f_.AddAndIncreaseOracleCall(e);
      // We do not count these in our model.
      sub_func_f_.oracle_calls_--;
      ++size_of_S_;
    }
  }

  // For levels beginning from l_begin, remake A, B and S.
  // currentA = union of B[l_begin] and A[l_begin].
  vector<int> currentA(buffer_B_[l_begin].begin(), buffer_B_[l_begin].end());
  currentA.insert(currentA.end(), levels_A_[l_begin].begin(),
                  levels_A_[l_begin].end());
  RandomHandler::Shuffle(currentA);
  for (int l = l_begin; l <= num_T_; ++l) {
    levels_A_[l].clear();
    solutions_S_[l].clear();
    buffer_B_[l].clear();
  }
  Filter(currentA, [&](int e) {
    return sub_func_f_.DeltaAndIncreaseOracleCall(e) >=
           gamma_ / (2 * cardinality_k_);
  });
  for (int l = l_begin; l <= num_T_ && size_of_S_ < cardinality_k_; ++l) {
    levels_A_[l].insert(currentA.begin(), currentA.end());
    if (currentA.empty()) {
      return;
    }
    while (currentA.size() >= 1ULL << (num_T_ - l)) {
      for (int e : currentA) {
        if (sub_func_f_.DeltaAndIncreaseOracleCall(e) >
            gamma_ / (2 * cardinality_k_)) {
          solutions_S_[l].insert(e);
          sub_func_f_.AddAndIncreaseOracleCall(e);
          size_of_S_++;
          if (size_of_S_ == cardinality_k_) return;
          break;
        }
      }
      Filter(currentA, [&](int e) {
        return sub_func_f_.DeltaAndIncreaseOracleCall(e) >=
               gamma_ / (2 * cardinality_k_);
      });
    }
  }
}

// Preprocessing the variables as explained in the paper.
OurSimpleAlgorithm::OurSimpleAlgorithmSingleThreshold::
    OurSimpleAlgorithmSingleThreshold(int num_T, SubmodularFunction& sub_func_f,
                                      int cardinality_k, double gamma,
                                      double eps)
    : num_T_(num_T),
      buffer_B_(num_T + 1),
      levels_A_(num_T + 1),
      solutions_S_(num_T + 1),
      size_of_S_(0),
      sub_func_f_(sub_func_f),
      lowest_level_(-1),
      cardinality_k_(cardinality_k),
      gamma_(gamma),
      eps_(eps) {}

// Take any k elements (or all if fewer).
double
OurSimpleAlgorithm::OurSimpleAlgorithmSingleThreshold::GetSolutionValue() {
  sub_func_f_.Reset();
  int count = 0;
  double obj_val = 0.0;
  for (int l = 0; l <= num_T_ && count < cardinality_k_; ++l) {
    for (int e : solutions_S_[l]) {
      obj_val += sub_func_f_.AddAndIncreaseOracleCall(e, -1);
      ++count;
      if (count == cardinality_k_) {
        break;
      }
    }
  }
  // In our oracle model it is just one call.
  SubmodularFunction::oracle_calls_ -= 2 * count - 1;
  return obj_val;
}

// Take any k elements (or all if fewer).
vector<int>
OurSimpleAlgorithm::OurSimpleAlgorithmSingleThreshold::GetSolutionVector() {
  vector<int> solution;
  for (int l = 0; l <= num_T_; ++l) {
    for (int e : solutions_S_[l]) {
      solution.push_back(e);
      if (static_cast<int>(solution.size()) == cardinality_k_) {
        return solution;
      }
    }
  }
  return solution;
}

void OurSimpleAlgorithm::OurSimpleAlgorithmSingleThreshold::Insert(
    int element) {
  sub_func_f_.Reset();
  if (sub_func_f_.DeltaAndIncreaseOracleCall(element) <
      gamma_ / (2 * cardinality_k_)) {
    return;
  }
  for (int l = 0; l <= num_T_; ++l) {
    buffer_B_[l].insert(element);
  }
  for (int l = 0; l <= num_T_; ++l) {
    if (buffer_B_[l].size() >= (1ULL << (num_T_ - l)) &&
        size_of_S_ < cardinality_k_) {
      LevelConstruct(l);
      return;
    }
  }
}

void OurSimpleAlgorithm::OurSimpleAlgorithmSingleThreshold::Erase(int element) {
  // Erase from A and B:
  for (int l = 0; l <= num_T_; ++l) {
    buffer_B_[l].erase(element);
    levels_A_[l].erase(element);
  }
  // Erase from S:
  for (int l = 0; l <= num_T_; ++l) {
    if (solutions_S_[l].find(element) != solutions_S_[l].end()) {
      sub_func_f_.Reset();
      solutions_S_[l].erase(element);
      size_of_S_--;
      // Recompute objective function.
      double objective = 0.0;
      for (int i = 0; i <= num_T_; ++i) {
        for (int w : solutions_S_[i]) {
          objective += sub_func_f_.AddAndIncreaseOracleCall(w, -1);
          SubmodularFunction::oracle_calls_ -= 2;
        }
      }
      SubmodularFunction::oracle_calls_++;
      // Update lowest_level.
      if (lowest_level_ == -1) {
        lowest_level_ = l;
      }
      lowest_level_ = std::min(lowest_level_, l);
      // If objective function decreased by too much, then recompute
      // starting from lowest_level.
      if (objective < (1 - eps_) * (gamma_ / 2)) {
        LevelConstruct(lowest_level_);
        lowest_level_ = -1;
      }
      // An element can be in at most one S-set, so we can return.
      return;
    }
  }
}

OurSimpleAlgorithm::OurSimpleAlgorithm(double eps) : eps_(eps) {}

void OurSimpleAlgorithm::Init(const SubmodularFunction& sub_func_f,
                              int cardinality_k) {
  // Create a copy of f to be shared among the "singles".
  sub_func_f_ = sub_func_f.Clone();
  singles_.clear();

  // To make sure we do not make error because of double computations.
  constexpr double double_error = 1e-6;
  // num_T = log_2(n)
  int num_T = static_cast<int>(
      ceil(log2(sub_func_f_->GetUniverse().size()) - double_error));
  for (double gamma : sub_func_f_->GetOptEstimates(cardinality_k)) {
    singles_.emplace_back(num_T, *sub_func_f_, cardinality_k, gamma, eps_);
  }
}

double OurSimpleAlgorithm::GetSolutionValue() {
  double best = 0.0;
  for (auto& single : singles_) {
    best = std::max(best, single.GetSolutionValue());
  }
  return best;
}

vector<int> OurSimpleAlgorithm::GetSolutionVector() {
  vector<int> solution;
  double best = 0.0;
  for (auto& single : singles_) {
    double val = single.GetSolutionValue();
    if (val > best) {
      best = val;
      solution = single.GetSolutionVector();
    }
  }
  return solution;
}

void OurSimpleAlgorithm::Insert(int element) {
  for (auto& single : singles_) {
    single.Insert(element);
  }
}

void OurSimpleAlgorithm::Erase(int element) {
  for (auto& single : singles_) {
    single.Erase(element);
  }
}

std::string OurSimpleAlgorithm::GetAlgorithmName() const {
  return std::string("our simple algorithm (eps = ") + std::to_string(eps_) +
         std::string(")");
}
