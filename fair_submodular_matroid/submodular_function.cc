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

#include "submodular_function.h"

#include <stdint.h>

#include <algorithm>
#include <vector>

#include "utilities.h"

void SubmodularFunction::Swap(int element, int swap) {
  Remove(swap);
  Add(element);
}

double SubmodularFunction::ObjectiveAndIncreaseOracleCall(
    const std::vector<int>& elements) const {
  ++oracle_calls_;
  return Objective(elements);
}

double SubmodularFunction::DeltaAndIncreaseOracleCall(int element) {
  ++oracle_calls_;
  return Delta(element);
}

double SubmodularFunction::RemovalDeltaAndIncreaseOracleCall(int element) {
  ++oracle_calls_;
  return RemovalDelta(element);
}

double SubmodularFunction::AddAndIncreaseOracleCall(int element, double thre) {
  // default implementation, can be overloaded by something more efficient
  double delta_e = DeltaAndIncreaseOracleCall(element);
  if (delta_e >= thre) {
    Add(element);
    return delta_e;
  } else {
    return 0.0;
  }
}

double SubmodularFunction::RemoveAndIncreaseOracleCall(int element) {
  // default implementation, can be overloaded by something more efficient
  double res = RemovalDeltaAndIncreaseOracleCall(element);
  Remove(element);
  return res;
}

std::vector<double> SubmodularFunction::GetOptEstimates(
    int upper_bound_on_size_of_any_feasible_set) {
  // Gets geometrically increasing sequence of estimates for OPT.
  // Should be always run on an empty function.
  // Used e.g. for Sieve-Streaming.
  // Could optimize: this gets re-estimated every time an algorithm is run.
  static constexpr double epsilon_for_opt_estimates = 0.3;
  double ub_opt = 0.0;
  for (const auto& e : GetUniverse()) {
    double delta_e = Delta(e);
    ub_opt = std::max(ub_opt, delta_e);
  }
  ub_opt *= upper_bound_on_size_of_any_feasible_set;
  // lbopt = smallest value of a single element (lower bound on OPT).
  // ubopt = largest value of a single element, times k (upper bound on OPT).
  return LogSpace(ub_opt / (upper_bound_on_size_of_any_feasible_set), ub_opt,
                  1 + epsilon_for_opt_estimates);
}

int64_t SubmodularFunction::oracle_calls_ = 0;
