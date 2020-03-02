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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_COMPUTE_COST_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_COMPUTE_COST_H_

#include "instruction.h"

namespace brain {
namespace evolution {
namespace amlz {

// Returns the cost of train a model in "compute-units". Compute-units are an
// arbitrary unit of compute cost, to compare across instructions and
// component functions. The only requirement on compute units is that they must
// be (roughly) proportional to time.

double ComputeCost(
    const std::vector<std::shared_ptr<const Instruction>>& component_function);

double ComputeCost(const Instruction& instruction);

}  // namespace amlz
}  // namespace evolution
}  // namespace brain

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_COMPUTE_COST_H_
