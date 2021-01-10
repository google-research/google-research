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

#ifndef SCANN_UTILS_INPUT_DATA_UTILS_H_
#define SCANN_UTILS_INPUT_DATA_UTILS_H_

#include "scann/data_format/dataset.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/fixed_point/pre_quantized_fixed_point.h"

namespace research_scann {

StatusOr<DatapointIndex> ComputeConsistentNumPointsFromIndex(
    const Dataset* dataset, const DenseDataset<uint8_t>* hashed_dataset,
    const PreQuantizedFixedPoint* pre_quantized_fixed_point,
    const DenseDataset<uint8_t>* compressed_dataset,
    const vector<int64_t>* crowding_attributes);

StatusOr<DimensionIndex> ComputeConsistentDimensionalityFromIndex(
    const HashConfig& config, const Dataset* dataset,
    const DenseDataset<uint8_t>* hashed_dataset,
    const PreQuantizedFixedPoint* pre_quantized_fixed_point,
    const DenseDataset<uint8_t>* compressed_dataset);

}  // namespace research_scann

#endif
