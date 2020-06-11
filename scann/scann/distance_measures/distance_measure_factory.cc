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

#include "scann/distance_measures/distance_measure_factory.h"

namespace tensorflow {
namespace scann_ops {

StatusOr<shared_ptr<DistanceMeasure>> GetDistanceMeasure(
    const DistanceMeasureConfig& config) {
  if (config.distance_measure().empty()) {
    return InvalidArgumentError(
        "Empty DistanceMeasureConfig proto! Must specify distance_measure.");
  }
  return GetDistanceMeasure(config.distance_measure());
}

StatusOr<shared_ptr<DistanceMeasure>> GetDistanceMeasure(string_view name) {
  if (name == "DotProductDistance")
    return shared_ptr<DistanceMeasure>(new DotProductDistance());
  if (name == "BinaryDotProductDistance")
    return shared_ptr<DistanceMeasure>(new BinaryDotProductDistance());
  if (name == "AbsDotProductDistance")
    return shared_ptr<DistanceMeasure>(new AbsDotProductDistance());
  if (name == "L2Distance")
    return shared_ptr<DistanceMeasure>(new L2Distance());
  if (name == "SquaredL2Distance")
    return shared_ptr<DistanceMeasure>(new SquaredL2Distance());
  if (name == "NegatedSquaredL2Distance")
    return shared_ptr<DistanceMeasure>(new NegatedSquaredL2Distance());
  if (name == "L1Distance")
    return shared_ptr<DistanceMeasure>(new L1Distance());
  if (name == "CosineDistance")
    return shared_ptr<DistanceMeasure>(new CosineDistance());
  if (name == "BinaryCosineDistance")
    return shared_ptr<DistanceMeasure>(new BinaryCosineDistance());
  if (name == "GeneralJaccardDistance")
    return shared_ptr<DistanceMeasure>(new GeneralJaccardDistance());
  if (name == "BinaryJaccardDistance")
    return shared_ptr<DistanceMeasure>(new BinaryJaccardDistance());
  if (name == "LimitedInnerProductDistance")
    return shared_ptr<DistanceMeasure>(new LimitedInnerProductDistance());
  if (name == "GeneralHammingDistance")
    return shared_ptr<DistanceMeasure>(new GeneralHammingDistance());
  if (name == "BinaryHammingDistance")
    return shared_ptr<DistanceMeasure>(new BinaryHammingDistance());
  if (name == "NonzeroIntersectDistance")
    return shared_ptr<DistanceMeasure>(new NonzeroIntersectDistance());
  return InvalidArgumentError("Invalid distance_measure: '%s'", name);
}

}  // namespace scann_ops
}  // namespace tensorflow
