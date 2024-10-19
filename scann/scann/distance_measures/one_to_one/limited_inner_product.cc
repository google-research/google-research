// Copyright 2024 The Google Research Authors.
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

#include "scann/distance_measures/one_to_one/limited_inner_product.h"

#include "scann/distance_measures/one_to_one/common.h"

namespace research_scann {

SCANN_DEFINE_DISTANCE_MEASURE_VIRTUAL_METHODS(LimitedInnerProductDistance,
                                              kEarlyStoppingNotSupported);
SCANN_REGISTER_DISTANCE_MEASURE(LimitedInnerProductDistance);

}  // namespace research_scann
