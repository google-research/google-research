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



#ifndef SCANN_UTILS_FACTORY_HELPERS_H_
#define SCANN_UTILS_FACTORY_HELPERS_H_

#include <cstdint>

#include "scann/distance_measures/distance_measures.h"
#include "scann/proto/scann.pb.h"

namespace research_scann {

struct GenericSearchParameters {
  Status PopulateValuesFromScannConfig(const ScannConfig& config);

  shared_ptr<const DistanceMeasure> pre_reordering_dist;

  int32_t pre_reordering_num_neighbors = -1;

  float pre_reordering_epsilon = numeric_limits<float>::quiet_NaN();

  shared_ptr<const DistanceMeasure> reordering_dist;

  int32_t post_reordering_num_neighbors = -1;

  float post_reordering_epsilon = numeric_limits<float>::quiet_NaN();
};

}  // namespace research_scann

#endif
