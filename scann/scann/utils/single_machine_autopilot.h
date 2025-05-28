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

#ifndef SCANN_UTILS_SINGLE_MACHINE_AUTOPILOT_H_
#define SCANN_UTILS_SINGLE_MACHINE_AUTOPILOT_H_

#include <memory>

#include "scann/data_format/dataset.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/types.h"

namespace research_scann {

StatusOr<ScannConfig> Autopilot(const ScannConfig& config,
                                shared_ptr<const Dataset> dataset,
                                DatapointIndex n = kInvalidDatapointIndex,
                                DimensionIndex dim = kInvalidDimension);

}

#endif
