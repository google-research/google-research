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

#ifndef SCANN_UTILS_MULTI_STAGE_BATCH_PIPELINE_H_
#define SCANN_UTILS_MULTI_STAGE_BATCH_PIPELINE_H_

#include <algorithm>
#include <cstddef>
#include <tuple>

#include "scann/utils/common.h"

namespace research_scann {

template <int kBatchSize, typename... StageCallbacks>
void RunMultiStageBatchPipeline(size_t task_count,
                                std::tuple<StageCallbacks...> stage_callbacks) {
  constexpr size_t kStageCount =
      std::tuple_size_v<std::tuple<StageCallbacks...>>;
  static_assert(kStageCount >= 2, "At least two stages are required");
  static_assert(kStageCount <= 4, "At most four stages are supported");
  static_assert(kBatchSize % 8 == 0 && kBatchSize >= 8 && kBatchSize <= 128,
                "Not supported batch size");

  for (size_t batch_start = 0; batch_start < task_count;
       batch_start += kBatchSize) {
    const size_t batch_end =
        std::min<size_t>(batch_start + kBatchSize, task_count);

    const auto& stage1_cb = std::get<0>(stage_callbacks);
    for (size_t task_idx = batch_start; task_idx < batch_end; ++task_idx) {
      stage1_cb(task_idx, task_idx - batch_start);
    }

    const auto& stage2_cb = std::get<1>(stage_callbacks);
    for (size_t task_idx = batch_start; task_idx < batch_end; ++task_idx) {
      stage2_cb(task_idx, task_idx - batch_start);
    }
    if constexpr (kStageCount >= 3) {
      const auto& stage3_cb = std::get<2>(stage_callbacks);
      for (size_t task_idx = batch_start; task_idx < batch_end; ++task_idx) {
        stage3_cb(task_idx, task_idx - batch_start);
      }
    }
    if constexpr (kStageCount == 4) {
      const auto& stage4_cb = std::get<3>(stage_callbacks);
      for (size_t task_idx = batch_start; task_idx < batch_end; ++task_idx) {
        stage4_cb(task_idx, task_idx - batch_start);
      }
    }
  }
}

}  // namespace research_scann

#endif
