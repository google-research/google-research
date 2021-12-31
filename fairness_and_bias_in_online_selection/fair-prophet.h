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

#ifndef FAIRNESS_AND_BIAS_SELECTION_FAIR_PROPHET_H_
#define FAIRNESS_AND_BIAS_SELECTION_FAIR_PROPHET_H_

#include <vector>

#include "distributions.h"
#include "utils.h"

namespace fair_secretary {

// Fair Prophet algorithm introduced in this work. It support both IID and
// non-IID algorithms.
class FairProphetAlgorithm {
 public:
  SecretaryInstance ComputeSolution(
      const std::vector<SecretaryInstance>& elements,
      const std::vector<std::reference_wrapper<RandomDistribution>>
          distributions,
      const std::vector<double>& q);
  SecretaryInstance ComputeSolutionIID(
      const std::vector<SecretaryInstance>& elements,
      const std::vector<std::reference_wrapper<RandomDistribution>>
          distributions,
      const std::vector<double>& q);
};

}  // namespace fair_secretary

#endif  // FAIRNESS_AND_BIAS_SELECTION_FAIR_PROPHET_H_
