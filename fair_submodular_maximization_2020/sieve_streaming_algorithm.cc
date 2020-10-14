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

#include "sieve_streaming_algorithm.h"

using std::vector;

SieveStreaming::SingleThresholdSieve::SingleThresholdSieve(
    const SubmodularFunction& sub_func_f, int cardinality_k, double gamma)
    : sub_func_f_(sub_func_f.Clone()),
      cardinality_k_(cardinality_k),
      obj_vals_({0.0}),
      gamma_(gamma) {}

// Process the arrival of a new element e on the stream.
void SieveStreaming::SingleThresholdSieve::Process(
    std::pair<int, int> element) {
  if (static_cast<int>(solution_.size()) < cardinality_k_) {
    double delta_e = sub_func_f_->AddAndIncreaseOracleCall(
        element, gamma_ / (2 * cardinality_k_));
    // A constant that is used to controll the value of the elements added.
    // Values less than this are basically zero (to avoid double errors).
    static const double epsilon_to_add = 1e-11;
    if (delta_e > epsilon_to_add) {
      solution_.push_back(element);
      obj_vals_.push_back(obj_vals_.back() + delta_e);
    }
  } else {
    // Do nothing, solution already full.
  }
}

void SieveStreaming::SingleThresholdSieve::Reset() {
  sub_func_f_->Reset();
  solution_.clear();
  obj_vals_ = {0.0};
}

bool SieveStreaming::SingleThresholdSieve::IsInSolution(
    std::pair<int, int> element) const {
  return std::find(solution_.begin(), solution_.end(), element) !=
         solution_.end();
}

double SieveStreaming::SingleThresholdSieve::GetSolutionValue() const {
  return obj_vals_.back();
}

std::vector<std::pair<int, int>>
SieveStreaming::SingleThresholdSieve::GetSolutionVector() const {
  return solution_;
}

void SieveStreaming::Init(SubmodularFunction& sub_func_f,
                          std::vector<std::pair<int, int>> bounds,
                          int cardinality_k) {
  sieves_.clear();
  for (double gamma : sub_func_f.GetOptEstimates(cardinality_k)) {
    sieves_.emplace_back(sub_func_f, cardinality_k, gamma);
  }
}

double SieveStreaming::GetSolutionValue() {
  double best = 0.0;
  for (auto& sieve : sieves_) {
    best = std::max(best, sieve.GetSolutionValue());
  }
  return best;
}

vector<std::pair<int, int>> SieveStreaming::GetSolutionVector() {
  vector<std::pair<int, int>> solution;
  double best = 0.0;
  for (auto& sieve : sieves_) {
    double tmp = sieve.GetSolutionValue();
    if (tmp > best) {
      best = tmp;
      solution = sieve.GetSolutionVector();
    }
  }
  return solution;
}

void SieveStreaming::Insert(std::pair<int, int> element, bool non_monotone) {
  for (auto& sieve : sieves_) {
    sieve.Process(element);
  }
}

std::string SieveStreaming::GetAlgorithmName() const {
  return "Sieve-Streaming";
}
