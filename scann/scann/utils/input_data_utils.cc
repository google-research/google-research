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

#include "scann/utils/input_data_utils.h"

#include <cstdint>

#include "scann/proto/partitioning.pb.h"
#include "scann/proto/projection.pb.h"

namespace research_scann {

StatusOr<DatapointIndex> ComputeConsistentNumPointsFromIndex(
    const Dataset* dataset, const DenseDataset<uint8_t>* hashed_dataset,
    const PreQuantizedFixedPoint* pre_quantized_fixed_point,
    const DenseDataset<int16_t>* bfloat16_dataset,
    const vector<int64_t>* crowding_attributes,
    const vector<std::string>* crowding_dimension_names) {
  if (!dataset && !hashed_dataset && !pre_quantized_fixed_point &&
      !bfloat16_dataset) {
    return InvalidArgumentError(
        "dataset, hashed_dataset, pre_quantized_fixed_point, and "
        "bfloat16_dataset are all null.");
  }

  DatapointIndex sz = kInvalidDatapointIndex;
  if (dataset) sz = dataset->size();

  if (hashed_dataset) {
    if (sz == kInvalidDatapointIndex) {
      sz = hashed_dataset->size();
    } else {
      SCANN_RET_CHECK_EQ(sz, hashed_dataset->size())
              .SetCode(absl::StatusCode::kInvalidArgument)
          << "Mismatch between original and hashed database sizes.";
    }
  }

  if (pre_quantized_fixed_point) {
    SCANN_RET_CHECK(pre_quantized_fixed_point->fixed_point_dataset);
    if (sz == kInvalidDatapointIndex) {
      sz = pre_quantized_fixed_point->fixed_point_dataset->size();
    } else {
      SCANN_RET_CHECK_EQ(sz,
                         pre_quantized_fixed_point->fixed_point_dataset->size())
              .SetCode(absl::StatusCode::kInvalidArgument)
          << "Mismatch between original/hashed database and fixed-point "
             "database sizes.";
    }
  }

  if (bfloat16_dataset) {
    if (sz == kInvalidDatapointIndex) {
      sz = bfloat16_dataset->size();
    } else {
      SCANN_RET_CHECK_EQ(sz, bfloat16_dataset->size())
              .SetCode(absl::StatusCode::kInvalidArgument)
          << "Mismatch between original/hashed/int8 database and bfloat16 "
             "database sizes.";
    }
  }

  if (crowding_attributes && !crowding_attributes->empty() &&
      sz != kInvalidDatapointIndex) {
    if (crowding_dimension_names && !crowding_dimension_names->empty()) {
      SCANN_RET_CHECK_EQ(crowding_attributes->size(),
                         crowding_dimension_names->size() * sz);
    } else {
      SCANN_RET_CHECK_EQ(crowding_attributes->size(), sz);
    }
  }

  if (sz == kInvalidDatapointIndex)
    return InvalidArgumentError("Dataset size could not be determined.");
  return sz;
}

StatusOr<DimensionIndex> ComputeConsistentDimensionalityFromIndex(
    const ScannConfig& config, const Dataset* dataset,
    const DenseDataset<uint8_t>* hashed_dataset,
    const PreQuantizedFixedPoint* pre_quantized_fixed_point,
    const DenseDataset<int16_t>* bfloat16_dataset) {
  if (!dataset && !hashed_dataset && !pre_quantized_fixed_point &&
      !bfloat16_dataset) {
    return InvalidArgumentError(
        "dataset, hashed_dataset, pre_quantized_fixed_point, and "
        "bfloat16_dataset are all null.");
  }

  DimensionIndex dims = kInvalidDimension;
  if (dataset) dims = dataset->dimensionality();

  if (pre_quantized_fixed_point) {
    DimensionIndex d =
        pre_quantized_fixed_point->fixed_point_dataset->dimensionality();
    if (dims == kInvalidDimension) {
      dims = d;
    } else {
      SCANN_RET_CHECK_EQ(dims, d).SetCode(absl::StatusCode::kInvalidArgument)
          << "Mismatch between original and fixed-point database "
             "dimensionalities.";
    }
  }

  if (bfloat16_dataset) {
    DimensionIndex d = bfloat16_dataset->dimensionality();
    if (dims == kInvalidDimension) {
      dims = d;
    } else {
      SCANN_RET_CHECK_EQ(dims, d).SetCode(absl::StatusCode::kInvalidArgument)
          << "Mismatch between original/fixed-point database and bfloat16 "
             "database dimensionalities.";
    }
  }

  auto projection_check = [&dims](const ProjectionConfig& proj) -> Status {
    if (proj.has_input_dim()) {
      DimensionIndex d = proj.input_dim();
      if (dims == kInvalidDimension) {
        dims = d;
      } else {
        SCANN_RET_CHECK_EQ(dims, d).SetCode(absl::StatusCode::kInvalidArgument)
            << "Mismatch between original/fixed-point/bfloat16 and hash "
               "projection dimensionalities.";
      }
    }
    return OkStatus();
  };
  if (config.partitioning().has_projection())
    SCANN_RETURN_IF_ERROR(projection_check(config.partitioning().projection()));
  const HashConfig& hash_config = config.hash();
  if (hash_config.has_projection() &&
      hash_config.asymmetric_hash().has_projection())
    return InvalidArgumentError(
        "Both hash and its asymmetric_hash subfield have projection configs.");
  if (hash_config.has_projection())
    SCANN_RETURN_IF_ERROR(projection_check(hash_config.projection()));
  if (hash_config.asymmetric_hash().has_projection())
    SCANN_RETURN_IF_ERROR(
        projection_check(hash_config.asymmetric_hash().projection()));

  if (dims == kInvalidDimension)
    return InvalidArgumentError(
        "Dataset dimensionality could not be determined.");
  return dims;
}

}  // namespace research_scann
