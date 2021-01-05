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

#include "scann/distance_measures/one_to_one/l2_distance.h"

namespace tensorflow {
namespace scann_ops {

SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS(SquaredL2Distance,
                                              kEarlyStoppingNotSupported);
SCANN_REGISTER_DISTANCE_MEASURE(SquaredL2Distance);

SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS(NegatedSquaredL2Distance,
                                              kEarlyStoppingNotSupported);
SCANN_REGISTER_DISTANCE_MEASURE(NegatedSquaredL2Distance);

}  // namespace scann_ops
}  // namespace tensorflow
