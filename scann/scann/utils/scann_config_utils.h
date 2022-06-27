// Copyright 2022 The Google Research Authors.
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

#ifndef SCANN_UTILS_SCANN_CONFIG_UTILS_H_
#define SCANN_UTILS_SCANN_CONFIG_UTILS_H_

#include <cstdint>

#include "scann/data_format/datapoint.h"
#include "scann/data_format/features.pb.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/proto/exact_reordering.pb.h"
#include "scann/proto/input_output.pb.h"
#include "scann/proto/scann.pb.h"
#include "scann/utils/types.h"

namespace research_scann {

Status CanonicalizeScannConfigForRetrieval(ScannConfig* config);

StatusOr<InputOutputConfig::InMemoryTypes> DetectInMemoryTypeFromDisk(
    const ScannConfig& config, const size_t shard = 0);

StatusOr<DimensionIndex> DetectInMemoryDimensionFromDisk(
    string_view database_wildcard, const size_t shard = 0);

StatusOr<InputOutputConfig::InMemoryTypes> TagFromGFVFeatureType(
    const GenericFeatureVector::FeatureType& feature_type);

StatusOr<InputOutputConfig::InMemoryTypes> DetectInMemoryTypeFromGfv(
    const GenericFeatureVector& gfv);

template <typename T = float>
StatusOr<DimensionIndex> DetectInMemoryDimensionFromGfv(
    const GenericFeatureVector& gfv) {
  Datapoint<T> dp;
  SCANN_RETURN_IF_ERROR(dp.FromGfv(gfv));
  return dp.dimensionality();
}

Status EnsureCorrectNormalizationForDistanceMeasure(ScannConfig* config);

std::string GetPossiblyPartitionedWildcard(const ScannConfig& config);

int GetNumPartitionedShards(const std::string& partitioner_prefix,
                            int32_t n_epochs);

}  // namespace research_scann

#endif
