// Copyright 2023 The Google Research Authors.
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

#include "scann/tree_x_hybrid/internal/utils.h"

namespace research_scann {

StatusOr<bool> ValidateDatapointsByToken(
    const vector<std::vector<DatapointIndex>>& datapoints_by_token,
    DatapointIndex num_datapoints) {
  bool is_disjoint = true;

  vector<bool> global_bitmap(num_datapoints, false);
  vector<bool> seen_twice(num_datapoints, false);

  for (const std::vector<DatapointIndex>& dp_list : datapoints_by_token) {
    flat_hash_set<DatapointIndex> duplicates;
    for (DatapointIndex dp_index : dp_list) {
      if (!duplicates.insert(dp_index).second) {
        return InvalidArgumentError(
            absl::StrCat("Duplicate datapoint index within a partition of "
                         "datapoints_by_token:  ",
                         dp_index, "."));
      }
      if (dp_index >= num_datapoints) {
        return OutOfRangeError(
            "Datapoint index in datapoints_by_token is >= number of "
            "datapoints in database (%d vs. %d).",
            dp_index, num_datapoints);
      }
      if (global_bitmap[dp_index]) {
        is_disjoint = false;
        if (seen_twice[dp_index]) {
          return InvalidArgumentError(
              StrCat("Datapoint ", dp_index,
                     " represented more than twice in datapoints_by_token.  "
                     "Only TWO_CENTER_ORTHOGONALITY_AMPLIFIED database "
                     "spilling is supported in tree-X hybrid."));
        } else {
          seen_twice[dp_index] = true;
        }
      } else {
        global_bitmap[dp_index] = true;
      }
    }
  }

  const DatapointIndex num_missing =
      std::count(global_bitmap.begin(), global_bitmap.end(), false);
  if (num_missing > 0) {
    auto false_it =
        std::find(global_bitmap.begin(), global_bitmap.end(), false);
    const size_t first_missing = false_it - global_bitmap.begin();
    return InvalidArgumentError(absl::StrCat(
        "Found ", num_missing,
        " datapoint(s) "
        "that are not represented in any partition.  First missing "
        "datapoint index = ",
        first_missing, "."));
  }

  return is_disjoint;
}

}  // namespace research_scann
