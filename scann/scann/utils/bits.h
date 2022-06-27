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

#ifndef SCANN_UTILS_BITS_H_
#define SCANN_UTILS_BITS_H_

#include <cstddef>
#include <cstdint>

#include "absl/numeric/bits.h"
#include "scann/utils/types.h"

namespace research_scann {

inline uint32_t NextPowerOfTwo32(uint32_t x) {
  return (x & (x - 1)) ? 1UL << (32 - absl::countl_zero(x)) : x;
}

inline uint64_t NextPowerOfTwo64(uint64_t x) {
  return (x & (x - 1)) ? 1ULL << (64 - absl::countl_zero(x)) : x;
}

inline size_t NextPowerOfTwo(size_t x) {
  if (sizeof(size_t) == sizeof(uint64_t)) {
    return NextPowerOfTwo64(x);
  } else {
    return NextPowerOfTwo32(x);
  }
}

inline bool IsPowerOfTwo(uint64_t x) { return x && ((x & (x - 1)) == 0); }

SCANN_INLINE uint32_t GetFinalMask32(size_t num_datapoints) {
  const size_t remainder_bits = num_datapoints % 32;
  return remainder_bits ? (1u << remainder_bits) - 1 : 0xFFFFFFFF;
}

}  // namespace research_scann

#endif
