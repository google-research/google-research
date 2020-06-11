// Copyright 2020 The Google Research Authors.
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

/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SCANN__HASHES_INTERNAL_LUT16_SSE4_H_
#define SCANN__HASHES_INTERNAL_LUT16_SSE4_H_

#ifdef __x86_64__

#include "scann/hashes/internal/lut16_args.h"
#include "scann/utils/intrinsics/attributes.h"
#include "scann/utils/types.h"

namespace tensorflow {
namespace scann_ops {
namespace asymmetric_hashing_internal {

template <size_t kBatchSize, bool kPrefetch>
class LUT16Sse4 {
 public:
  SCANN_SSE4_OUTLINE static void GetInt16Distances(LUT16Args<int16_t> args);
  SCANN_SSE4_OUTLINE static void GetInt32Distances(LUT16Args<int32_t> args);

  SCANN_SSE4_OUTLINE static void GetTopInt16Distances(
      LUT16ArgsTopN<int16_t> args);
  SCANN_SSE4_OUTLINE static void GetTopFloatDistances(
      LUT16ArgsTopN<float> args);

  SCANN_SSE4_OUTLINE static void GetFloatDistances(
      LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers);
};

SCANN_INSTANTIATE_CLASS_FOR_LUT16_BATCH_SIZES(extern, LUT16Sse4);

}  // namespace asymmetric_hashing_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
#endif
