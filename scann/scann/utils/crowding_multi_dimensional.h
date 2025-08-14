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

#ifndef SCANN_UTILS_CROWDING_MULTI_DIMENSIONAL_H_
#define SCANN_UTILS_CROWDING_MULTI_DIMENSIONAL_H_

#include <bitset>
#include <cstdint>
#include <optional>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "scann/data_format/features.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

class CrowdingMultiDimensional {
 public:
  static constexpr absl::string_view kQuotaWeightDimensionName = "quota_weight";

  CrowdingMultiDimensional(
      ConstSpan<int64_t> datapoint_index_to_crowding_attribute,
      ConstSpan<std::string> crowding_dimension_names);

  void SetQuota(absl::string_view dimension, int quota);

  void SetQuota(absl::string_view dimension, int64_t crowding_attribute,
                int quota);

  [[nodiscard]] bool Add(DatapointIndex index);

 private:
  struct DimensionData {
    std::array<int32_t, 64 / 4> small_attribute_quotas;

    int32_t default_quota;

    absl::flat_hash_map<int64_t, int32_t> large_attribute_quotas;

    DimensionData();

    void SetQuota(int32_t value);
    int32_t& MutableQuota(int64_t crowding_attribute);
  };

  ConstSpan<int64_t> datapoint_index_to_crowding_attribute_;
  ConstSpan<std::string> crowding_dimension_names_;
  static constexpr int kExpectedMaxNumDimensions = 8;
  std::optional<int> weight_dimension_index_;
  std::bitset<32> used_dimensions_mask_ = 0;
  absl::InlinedVector<DimensionData, kExpectedMaxNumDimensions> dimension_;
};

class CrowdingAttributesGuard {
 public:
  absl::Status Append(const GenericFeatureVector::Crowding& crowding,
                      vector<int64_t>& destination,
                      vector<std::string>* dimension_names);

 private:
  absl::Status AppendSingleDimensional(
      const GenericFeatureVector::Crowding& crowding,
      vector<int64_t>& destination);

  absl::Status AppendMultiDimensional(
      const GenericFeatureVector::Crowding& crowding,
      vector<int64_t>& destination, vector<std::string>* dimension_names);

  enum class State { kEmpty, kSingleDimensional, kMultiDimensional };
  State state_ = State::kEmpty;
};

}  // namespace research_scann

#endif
