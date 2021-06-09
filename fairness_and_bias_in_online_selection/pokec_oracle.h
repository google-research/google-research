// Copyright 2021 The Google Research Authors.
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

#ifndef FAIRNESS_AND_BIAS_SELECTION_POKEC_ORACLE_H_
#define FAIRNESS_AND_BIAS_SELECTION_POKEC_ORACLE_H_

#include "utils.h"

namespace fair_secretary {

// An oracle that is used for Influence Maximization experiments. Please make
// sure that the paths to the inputs are set in pokec_oracle.cc.
class PokecOracle {
 public:
  // If num_elements is indicated it returns that many elements otherwise it
  // returns all of them.
  std::vector<SecretaryInstance> GetSecretaryInput();

  // Indicates the number of colors.
  int num_colors;
};

}  // namespace fair_secretary

#endif  // FAIRNESS_AND_BIAS_SELECTION_POKEC_ORACLE_H_
