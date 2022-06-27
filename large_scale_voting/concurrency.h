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

#ifndef EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_CONCURRENCY_H_
#define EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_CONCURRENCY_H_

#include <deque>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "status_macros.h"

namespace large_scale_voting {

// Since threaded recursive itertion, i.e. `num_workers` is unused. Depth first
// search.
template <typename WorkItemType>
absl::Status ProcessRecursivelyDFS(
    int num_workers, const WorkItemType& initial_work_item,
    absl::FunctionRef<
        absl::StatusOr<std::vector<WorkItemType>>(const WorkItemType&)>
        processing_function) {
  std::deque<WorkItemType> work_item_queue;
  work_item_queue.push_back(initial_work_item);

  while (!work_item_queue.empty()) {
    WorkItemType work_item = std::move(work_item_queue.front());
    work_item_queue.pop_front();

    while (true) {
      ASSIGN_OR_RETURN(std::vector<WorkItemType> new_work_items,
                       processing_function(work_item));
      if (new_work_items.empty()) {
        break;
      }

      // Do the first item right away.
      work_item = std::move(new_work_items[0]);

      // Add the rest into the queue.
      // Insert into front so we get a depth first search.
      for (size_t i = 1; i < new_work_items.size(); ++i) {
        work_item_queue.push_front(std::move(new_work_items[i]));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace large_scale_voting

#endif  // EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_CONCURRENCY_H_
