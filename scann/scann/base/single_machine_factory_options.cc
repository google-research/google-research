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

#include "scann/base/single_machine_factory_options.h"

#include "scann/utils/input_data_utils.h"

namespace research_scann {

StatusOr<DimensionIndex>
SingleMachineFactoryOptions::ComputeConsistentDimensionality(
    const ScannConfig& config, const Dataset* dataset) const {
  return ComputeConsistentDimensionalityFromIndex(
      config, dataset, hashed_dataset.get(), pre_quantized_fixed_point.get(),
      bfloat16_dataset.get());
}

}  // namespace research_scann
