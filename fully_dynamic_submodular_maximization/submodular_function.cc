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
// Submodular Function
//

// Inherit from this to define your own submodular function
// the function should keep a current set S as its state.

#include "submodular_function.h"

// Add a new element to set S.
void SubmodularFunction::AddAndIncreaseOracleCall(int element) {
  ++oracle_calls_;
  Add(element);
}

double SubmodularFunction::DeltaAndIncreaseOracleCall(int element) const {
  ++oracle_calls_;
  return Delta(element);
}

// Adds element if and only if its contribution is >= thre.
// Return the contribution increase.
double SubmodularFunction::AddAndIncreaseOracleCall(int element, double thre) {
  double delta_e = Delta(element);
  if (delta_e >= thre) {
    Add(element);
    return delta_e;
  } else {
    return 0.0;
  }
}

std::vector<double> SubmodularFunction::GetOptEstimates(
    int cardinality_k) const {
  // Gets geometrically increasing sequence of estimates for OPT.
  // Should be always run on an empty function.
  // Used for Sieve-Streaming and for our algorithm.
  // Could optimize: this gets re-estimated every time an algorithm is run.
  static constexpr double epsilon_for_opt_estimates = 0.3;

  // Elements with delta less than this wont be considered. For the experiments,
  // mostly the delta values are integer so values smaller way than 1 are
  // simply double errors.
  static const double min_considered_delta = 1e-11;

  double lb_opt = std::numeric_limits<double>::max();
  double ub_opt = 0.0;
  for (int e : GetUniverse()) {
    double delta_e = Delta(e);
    if (delta_e > min_considered_delta) lb_opt = std::min(lb_opt, delta_e);
    ub_opt = std::max(ub_opt, delta_e);
  }
  ub_opt *= cardinality_k;
  // lbopt = smallest value of a single element (lower bound on OPT).
  // ubopt = largest value of a single element, times k (upper bound on OPT).
  return LogSpace(lb_opt, ub_opt, 1 + epsilon_for_opt_estimates);
}
int64_t SubmodularFunction::oracle_calls_ = 0;
