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

#ifndef SCANN_HASHES_INTERNAL_LUT16_AVX512_H_
#define SCANN_HASHES_INTERNAL_LUT16_AVX512_H_
#include <cstdint>

#ifdef __x86_64__

#include "scann/hashes/internal/lut16_args.h"
#include "scann/utils/types.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

void Avx512PlatformSpecificSwizzle(uint8_t* packed_dataset, int num_datapoints,
                                   int num_codes);

template <size_t kBatchSize, PrefetchStrategy kPrefetch>
class LUT16Avx512 {
 public:
  static SCANN_AVX512_OUTLINE void GetInt16Distances(LUT16Args<int16_t> args);
  static SCANN_AVX512_OUTLINE void GetInt32Distances(LUT16Args<int32_t> args);

  static SCANN_AVX512_OUTLINE void GetFloatDistances(
      LUT16Args<float> args, ConstSpan<float> inv_fp_multipliers);

  static void GetTopInt16Distances(LUT16ArgsTopN<int16_t> args);
  static void GetTopFloatDistances(LUT16ArgsTopN<float> args);
};

SCANN_INSTANTIATE_CLASS_FOR_LUT16_BATCH_SIZES(extern, LUT16Avx512);

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
#endif
