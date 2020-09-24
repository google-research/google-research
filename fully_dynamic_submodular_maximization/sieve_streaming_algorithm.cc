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
//  Sieve Streaming Algorithm
//
// The Sieve-Streaming algorithm of Badanidiyuru et al. a faithful simulation:
// Whenever an element is deleted from the solution, the sieve is rerun.

#include "sieve_streaming_algorithm.h"

using std::vector;

SieveStreaming::SingleThresholdSieve::SingleThresholdSieve(
    const SubmodularFunction& sub_func_f, int cardinality_k, double gamma)
    : sub_func_f_(sub_func_f.Clone()),
      cardinality_k_(cardinality_k),
      obj_vals_({0.0}),
      gamma_(gamma) {}

// Process the arrival of a new element e on the stream.
void SieveStreaming::SingleThresholdSieve::Process(int element) {
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

bool SieveStreaming::SingleThresholdSieve::IsInSolution(int element) const {
  return std::find(solution_.begin(), solution_.end(), element) !=
         solution_.end();
}

double SieveStreaming::SingleThresholdSieve::GetSolutionValue() const {
  return obj_vals_.back();
}

vector<int> SieveStreaming::SingleThresholdSieve::GetSolutionVector() const {
  return solution_;
}

void SieveStreaming::Init(const SubmodularFunction& sub_func_f,
                          int cardinality_k) {
  sieves_.clear();
  for (double gamma : sub_func_f.GetOptEstimates(cardinality_k)) {
    sieves_.emplace_back(sub_func_f, cardinality_k, gamma);
  }
  stream_.clear();
  position_on_stream_.clear();
}

double SieveStreaming::GetSolutionValue() {
  double best = 0.0;
  for (auto& sieve : sieves_) {
    best = std::max(best, sieve.GetSolutionValue());
  }
  return best;
}

vector<int> SieveStreaming::GetSolutionVector() {
  vector<int> solution;
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

void SieveStreaming::Insert(int element) {
  if (position_on_stream_.count(element)) {
    Fail("element is being inserted again despite not being deleted");
  }
  position_on_stream_[element] = static_cast<int>(stream_.size());
  stream_.push_back(element);

  for (auto& sieve : sieves_) {
    sieve.Process(element);
  }
}

void SieveStreaming::Erase(int element) {
  if (!position_on_stream_.count(element)) {
    Fail("element is being deleted but was never inserted");
  }
  int pos = position_on_stream_[element];
  position_on_stream_.erase(element);
  stream_[pos] = deletedElement;
  // Alternatively we could actually delete it from the vector.

  for (auto& sieve : sieves_) {
    if (sieve.IsInSolution(element)) {
      // rebuild from scratch
      sieve.Reset();
      for (int el : stream_) {
        if (el != deletedElement) {
          sieve.Process(el);
        }
      }
    }
  }
}

std::string SieveStreaming::GetAlgorithmName() const {
  return "Sieve-Streaming";
}
