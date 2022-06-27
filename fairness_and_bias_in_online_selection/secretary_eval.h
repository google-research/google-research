// Copyright 2022 The Google Research Authors.
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

#ifndef FAIRNESS_AND_BIAS_SELECTION_SECRETARY_EVAL_H_
#define FAIRNESS_AND_BIAS_SELECTION_SECRETARY_EVAL_H_

#include <vector>

#include "utils.h"

namespace fair_secretary {

// Evaluates the quality of the solution, and provide stats to the STDOUT.
class SecretaryEval {
 public:
  static void Eval(const std::vector<SecretaryInstance>& instance,
                   const std::vector<SecretaryInstance>& answer, int num_color);
  static void ThEval(const std::vector<SecretaryInstance>& instance,
                     const std::vector<std::vector<SecretaryInstance>>& answers,
                     int num_colors);
  static void InnerUnbalanced(const std::vector<SecretaryInstance>& instance,
                              const SecretaryInstance& ans,
                              std::vector<int>& correct_answer,
                              std::vector<int>& num_answer,
                              std::vector<int>& max_dist, const int num_colors,
                              int& not_picked, int& total_correct_answer);
};

}  // namespace fair_secretary

#endif  // FAIRNESS_AND_BIAS_SELECTION_SECRETARY_EVAL_H_
