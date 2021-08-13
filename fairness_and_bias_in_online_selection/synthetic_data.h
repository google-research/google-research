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

#ifndef FAIRNESS_AND_BIAS_SELECTION_SYNTHETIC_DATA_H_
#define FAIRNESS_AND_BIAS_SELECTION_SYNTHETIC_DATA_H_

#include "distributions.h"
#include "utils.h"

namespace fair_secretary {

// Creates synthetic dataset based on the sizes and probabilities given.
class SyntheticData {
 public:
  std::vector<SecretaryInstance> GetSecretaryInput(
      const std::vector<int>& sizes, const std::vector<double>& prob = {});

  std::vector<SecretaryInstance> GetProphetInput(
      const int size,
      const std::vector<std::reference_wrapper<RandomDistribution>>& dist);

  // Indicates the number of colors.
  int num_colors;
};

}  // namespace fair_secretary

#endif  // FAIRNESS_AND_BIAS_SELECTION_SYNTHETIC_DATA_H_
