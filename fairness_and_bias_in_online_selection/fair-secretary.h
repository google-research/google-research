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

#ifndef FAIRNESS_AND_BIAS_SELECTION_FAIR_SECRETARY_H_
#define FAIRNESS_AND_BIAS_SELECTION_FAIR_SECRETARY_H_

#include <vector>

#include "utils.h"

namespace fair_secretary {

// The fair secretaty algorithm.
class FairSecretaryAlgorithm {
 public:
  FairSecretaryAlgorithm(const std::vector<int>& thre, int num_colors);
  SecretaryInstance ComputeSolution(
      const std::vector<SecretaryInstance>& elements);

 private:
  // Indicates the number of colors.
  int num_colors_;
  // Indicades a threshold for each color.
  std::vector<int> thre_;
};

}  // namespace fair_secretary

#endif  // FAIRNESS_AND_BIAS_SELECTION_FAIR_SECRETARY_H_
