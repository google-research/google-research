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

#include "scann/base/search_parameters.h"

#include "scann/base/restrict_allowlist.h"

namespace research_scann {

Status SearchParameters::Validate(bool reordering_enabled) const {
  if (pre_reordering_num_neighbors() <= 0) {
    return InvalidArgumentError("pre_reordering_num_neighbors must be > 0.");
  }

  if (per_crowding_attribute_pre_reordering_num_neighbors() <= 0) {
    return InvalidArgumentError(
        "per_crowding_attribute_pre_reordering_num_neighbors must be > 0.");
  }

  if (per_crowding_attribute_post_reordering_num_neighbors() <= 0) {
    return InvalidArgumentError(
        "per_crowding_attribute_post_reordering_num_neighbors must be > 0.");
  }

  if (std::isnan(pre_reordering_epsilon())) {
    return InvalidArgumentError(
        "pre_reordering_epsilon must be set to a non-NaN value.");
  }

  if (reordering_enabled) {
    if (post_reordering_num_neighbors() <= 0) {
      return InvalidArgumentError(
          "post_reordering_num_neighbors must be > 0 if reordering is "
          "enabled.");
    }

    if (std::isnan(post_reordering_epsilon())) {
      return InvalidArgumentError(
          "post_reordering_epsilon must be set to a "
          "non-NaN value if reordering is enabled.");
    }
  }

  return OkStatus();
}

void SearchParameters::SetUnspecifiedParametersFrom(
    const SearchParameters& defaults) {
  DCHECK(this);
  DCHECK(&defaults);

  if (pre_reordering_num_neighbors() == -1) {
    set_pre_reordering_num_neighbors(defaults.pre_reordering_num_neighbors());
  }

  if (post_reordering_num_neighbors() == -1) {
    set_post_reordering_num_neighbors(defaults.post_reordering_num_neighbors());
  }

  if (std::isnan(pre_reordering_epsilon())) {
    set_pre_reordering_epsilon(defaults.pre_reordering_epsilon());
  }

  if (std::isnan(post_reordering_epsilon())) {
    set_post_reordering_epsilon(defaults.post_reordering_epsilon());
  }
}

}  // namespace research_scann
