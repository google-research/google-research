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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/distance_measures/one_to_one/hamming_distance.h"

namespace tensorflow {
namespace scann_ops {

SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS(GeneralHammingDistance, 32)
SCANN_REGISTER_DISTANCE_MEASURE(GeneralHammingDistance)

double BinaryHammingDistance::GetDistanceDense(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) const {
  return static_cast<double>(DenseBinaryHammingDistance(a, b));
}

SCANN_REGISTER_DISTANCE_MEASURE(BinaryHammingDistance);

}  // namespace scann_ops
}  // namespace tensorflow
