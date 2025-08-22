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

#include "scann/utils/crowding_multi_dimensional.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

CrowdingMultiDimensional::CrowdingMultiDimensional(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute,
    ConstSpan<std::string> crowding_dimension_names)
    : datapoint_index_to_crowding_attribute_(
          datapoint_index_to_crowding_attribute),
      crowding_dimension_names_(crowding_dimension_names),
      dimension_(crowding_dimension_names.size()) {
  auto it = absl::c_find(crowding_dimension_names_, kQuotaWeightDimensionName);
  if (it != crowding_dimension_names_.end()) {
    weight_dimension_index_ =
        static_cast<int>(it - crowding_dimension_names_.begin());
  }
}

void CrowdingMultiDimensional::SetQuota(absl::string_view dimension,
                                        int quota) {
  auto it = absl::c_find(crowding_dimension_names_, dimension);
  if (it != crowding_dimension_names_.end()) {
    const size_t dimension_index = it - crowding_dimension_names_.begin();
    if (dimension_index < used_dimensions_mask_.size()) {
      used_dimensions_mask_[dimension_index] = true;
    }
    dimension_[dimension_index].SetQuota(quota);
  }
}

void CrowdingMultiDimensional::SetQuota(absl::string_view dimension,
                                        int64_t crowding_attribute, int quota) {
  auto it = absl::c_find(crowding_dimension_names_, dimension);
  if (it != crowding_dimension_names_.end()) {
    const size_t dimension_index = it - crowding_dimension_names_.begin();
    if (dimension_index < used_dimensions_mask_.size()) {
      used_dimensions_mask_[dimension_index] = true;
    }
    dimension_[dimension_index].MutableQuota(crowding_attribute) = quota;
  }
}

bool CrowdingMultiDimensional::Add(DatapointIndex index) {
  const size_t num_dimensions = crowding_dimension_names_.size();
  ConstSpan<int64_t> crowding_attributes =
      datapoint_index_to_crowding_attribute_.subspan(index * num_dimensions,
                                                     num_dimensions);
  const int weight = weight_dimension_index_.has_value()
                         ? crowding_attributes[*weight_dimension_index_]
                         : 1;
  for (int i = 0; i < num_dimensions; ++i) {
    if (i < used_dimensions_mask_.size() && !used_dimensions_mask_[i]) continue;
    int& quota = dimension_[i].MutableQuota(crowding_attributes[i]);
    if (quota < 1) {
      for (int j = 0; j < i; ++j) {
        dimension_[j].MutableQuota(crowding_attributes[j]) += weight;
      }
      return false;
    }
    quota -= weight;
  }
  return true;
}

CrowdingMultiDimensional::DimensionData::DimensionData()
    : default_quota(std::numeric_limits<int32_t>::max()) {
  SetQuota(std::numeric_limits<int32_t>::max());
}

void CrowdingMultiDimensional::DimensionData::SetQuota(int32_t value) {
  default_quota = value;

  for (auto& value : small_attribute_quotas) {
    value = default_quota;
  }
}

int& CrowdingMultiDimensional::DimensionData::MutableQuota(
    int64_t crowding_attribute) {
  if (0 <= crowding_attribute &&
      crowding_attribute < small_attribute_quotas.size()) {
    return small_attribute_quotas[crowding_attribute];
  }

  auto it =
      large_attribute_quotas.try_emplace(crowding_attribute, default_quota)
          .first;
  return it->second;
}

absl::Status CrowdingAttributesGuard::Append(
    const GenericFeatureVector::Crowding& crowding,
    vector<int64_t>& destination, vector<std::string>* dimension_names) {
  switch (state_) {
    case State::kEmpty: {
      if (!crowding.multidimensional_crowding_attributes().empty() ||
          !crowding.multidimensional_crowding_dimensions().empty()) {
        state_ = State::kMultiDimensional;
        return AppendMultiDimensional(crowding, destination, dimension_names);
      } else {
        state_ = State::kSingleDimensional;
        return AppendSingleDimensional(crowding, destination);
      }
    }
    case State::kSingleDimensional:
      return AppendSingleDimensional(crowding, destination);
    case State::kMultiDimensional:
      return AppendMultiDimensional(crowding, destination, dimension_names);
  }
  return OkStatus();
}

absl::Status CrowdingAttributesGuard::AppendSingleDimensional(
    const GenericFeatureVector::Crowding& crowding,
    vector<int64_t>& destination) {
  if (!crowding.multidimensional_crowding_attributes().empty() ||
      !crowding.multidimensional_crowding_dimensions().empty()) {
    return FailedPreconditionError(
        "Cannot mix multidimensional and single dimensional crowding "
        "attributes.");
  }
  destination.push_back(crowding.crowding_attribute());
  return OkStatus();
}

absl::Status CrowdingAttributesGuard::AppendMultiDimensional(
    const GenericFeatureVector::Crowding& crowding,
    vector<int64_t>& destination, vector<std::string>* dimension_names) {
  if (dimension_names == nullptr) {
    return FailedPreconditionError(
        "Dimension names must be provided for multidimensional crowding.");
  }
  if (crowding.has_crowding_attribute()) {
    return FailedPreconditionError(
        "Cannot mix multidimensional and single dimensional crowding "
        "attributes.");
  }
  if (crowding.multidimensional_crowding_dimensions_size() !=
      crowding.multidimensional_crowding_attributes_size()) {
    return FailedPreconditionError(
        "The number of crowding dimensions must match the number of crowding "
        "attributes.");
  }
  if (!dimension_names->empty()) {
    if (!absl::c_equal(*dimension_names,
                       crowding.multidimensional_crowding_dimensions())) {
      return FailedPreconditionError(
          "All GFVs in a dataset need to have the same crowding dimensions.");
    }
  } else {
    absl::flat_hash_set<std::string> unique_dimensions(
        crowding.multidimensional_crowding_dimensions().begin(),
        crowding.multidimensional_crowding_dimensions().end());
    if (unique_dimensions.size() !=
        crowding.multidimensional_crowding_dimensions_size()) {
      return FailedPreconditionError("Crowding dimensions must be unique.");
    }
    if (absl::c_any_of(unique_dimensions, [](const auto& dimension) {
          return dimension.empty();
        })) {
      return FailedPreconditionError(
          "Crowding dimension name must not be empty.");
    }
    *dimension_names = std::vector<std::string>(
        crowding.multidimensional_crowding_dimensions().begin(),
        crowding.multidimensional_crowding_dimensions().end());
  }
  destination.insert(destination.end(),
                     crowding.multidimensional_crowding_attributes().begin(),
                     crowding.multidimensional_crowding_attributes().end());
  return OkStatus();
}

}  // namespace research_scann
