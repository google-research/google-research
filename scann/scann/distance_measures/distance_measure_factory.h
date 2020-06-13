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

#ifndef SCANN__DISTANCE_MEASURES_DISTANCE_MEASURE_FACTORY_H_
#define SCANN__DISTANCE_MEASURES_DISTANCE_MEASURE_FACTORY_H_

#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/distance_measures/distance_measures.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {

StatusOr<shared_ptr<DistanceMeasure>> GetDistanceMeasure(string_view name);
StatusOr<shared_ptr<DistanceMeasure>> GetDistanceMeasure(
    const DistanceMeasureConfig& config);

}  // namespace scann_ops
}  // namespace tensorflow

#endif
