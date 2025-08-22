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



#include "scann/tree_x_hybrid/tree_x_params.h"

#include <cstdint>
#include <utility>

#include "scann/base/search_parameters.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

namespace research_scann {

TreeXOptionalParameters::TreeXOptionalParameters() {}
TreeXOptionalParameters::~TreeXOptionalParameters() {}

Status TreeXOptionalParameters::EnablePreTokenization(
    vector<int32_t> leaf_tokens_to_search) {
  if (leaf_tokens_to_search.empty()) {
    return InvalidArgumentError(
        "leaf_tokens_to_search cannot be empty on calls to "
        "EnablePreTokenization.");
  }

  if (pre_tokenization_enabled()) {
    return FailedPreconditionError(
        "Pre-tokenization cannot be enabled if it is already enabled.");
  }

  leaf_tokens_to_search_ = std::move(leaf_tokens_to_search);
  return OkStatus();
}

Status TreeXOptionalParameters::EnablePreTokenizationWithDistances(
    vector<pair<DatapointIndex, float>> centers_to_search) {
  if (centers_to_search.empty()) {
    return InvalidArgumentError(
        "centers_to_search cannot be empty on calls to "
        "EnablePreTokenizationWithDistances.");
  }

  if (pre_tokenization_with_distances_enabled()) {
    return FailedPreconditionError(
        "Pre-tokenization with distances cannot be enabled if it is already "
        "enabled.");
  }

  if (pre_tokenization_enabled()) {
    return FailedPreconditionError(
        "Pre-tokenization cannot be enabled if it is already enabled.");
  }

  centers_to_search_ = std::move(centers_to_search);

  leaf_tokens_to_search_.reserve(centers_to_search_.size());
  for (const auto& center : centers_to_search_) {
    leaf_tokens_to_search_.push_back(center.first);
  }

  return OkStatus();
}

}  // namespace research_scann
