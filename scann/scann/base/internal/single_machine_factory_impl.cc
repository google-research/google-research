// Copyright 2025 The Google Research Authors.
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

#include "scann/base/internal/single_machine_factory_impl.h"

#include <cstddef>
#include <vector>

#include "scann/utils/common.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"

namespace research_scann {
namespace internal {

std::vector<float> InverseMultiplier(PreQuantizedFixedPoint* fixed_point) {
  std::vector<float> inverse_multipliers;
  inverse_multipliers.resize(fixed_point->multiplier_by_dimension->size());

  for (size_t i : Seq(inverse_multipliers.size())) {
    inverse_multipliers[i] = 1.0f / fixed_point->multiplier_by_dimension->at(i);
  }
  return inverse_multipliers;
}

}  // namespace internal
}  // namespace research_scann
