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

#include "submodular_function.h"

double SubmodularFunction::ObjectiveAndIncreaseOracleCall(
    const std::vector<std::pair<int, int>>& elements) {
  ++oracle_calls_;
  return Objective(elements);
}

void SubmodularFunction::AddAndIncreaseOracleCall(std::pair<int, int> element) {
  ++oracle_calls_;
  Add(element);
}

double SubmodularFunction::DeltaAndIncreaseOracleCall(
    std::pair<int, int> element) {
  ++oracle_calls_;
  return Delta(element);
}

double SubmodularFunction::AddAndIncreaseOracleCall(std::pair<int, int> element,
                                                    double thre) {
  ++oracle_calls_;
  double delta_e = Delta(element);
  if (delta_e >= thre) {
    Add(element);
    return delta_e;
  } else {
    return 0.0;
  }
}

std::vector<double> SubmodularFunction::GetOptEstimates(int cardinality_k) {
  // Gets geometrically increasing sequence of estimates for OPT.
  // Should be always run on an empty function.
  // Used for Sieve-Streaming and for our algorithm.
  // Could optimize: this gets re-estimated every time an algorithm is run.
  static constexpr double epsilon_for_opt_estimates = 0.3;
  double ub_opt = 0.0;
  for (const auto& e : GetUniverse()) {
    double delta_e = Delta(e);
    ub_opt = std::max(ub_opt, delta_e);
  }
  ub_opt *= cardinality_k;
  // lbopt = smallest value of a single element (lower bound on OPT).
  // ubopt = largest value of a single element, times k (upper bound on OPT).
  return LogSpace(ub_opt / (cardinality_k), ub_opt,
                  1 + epsilon_for_opt_estimates);
}

int64_t SubmodularFunction::oracle_calls_ = 0;
