// Copyright 2026 The Google Research Authors.
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

#ifndef SCANN_HASHES_INTERNAL_LUT16_NEON_H_
#define SCANN_HASHES_INTERNAL_LUT16_NEON_H_

#include <cstdint>

#include "scann/utils/intrinsics/attributes.h"

#if defined(__aarch64__)
namespace research_scann {
namespace asymmetric_hashing_internal {
namespace neon {

SCANN_INLINE uint32_t ComputePushMask(const int16_t* dist16, int16_t threshold);

}  // namespace neon
}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#include "scann/hashes/internal/lut16_neon.inc"
#endif  // defined(__aarch64__)

#endif  // SCANN_HASHES_INTERNAL_LUT16_NEON_H_
