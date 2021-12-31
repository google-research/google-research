// Copyright 2021 The Google Research Authors.
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

#ifndef SCANN_HASHES_INTERNAL_LUT16_AVX512_SWIZZLE_H_
#define SCANN_HASHES_INTERNAL_LUT16_AVX512_SWIZZLE_H_
#include <cstdint>
#ifdef __x86_64__

#include "scann/utils/intrinsics/attributes.h"
#include "tensorflow/core/platform/types.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

void Avx512Swizzle128(const uint8_t* src, uint8_t* dst);

void Avx512Swizzle32(const uint8_t* src, uint8_t* dst);

void Avx512PlatformSpecificSwizzle(uint8_t* packed_dataset, int num_datapoints,
                                   int num_codes_per_dp);

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
#endif
