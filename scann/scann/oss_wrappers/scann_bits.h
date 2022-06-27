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

#ifndef SCANN_OSS_WRAPPERS_SCANN_BITS_H_
#define SCANN_OSS_WRAPPERS_SCANN_BITS_H_

#include <cstddef>
#include <cstdint>

#include "absl/numeric/bits.h"

namespace research_scann {
namespace bits {

#if (defined(__i386__) || defined(__x86_64__)) && defined(__GNUC__)

inline int FindLSBSetNonZero(uint32_t n) { return __builtin_ctz(n); }

inline int FindLSBSetNonZero64(uint64_t n) { return __builtin_ctzll(n); }

#else

int FindLSBSetNonZero(uint32_t n);

inline int FindLSBSetNonZero64(uint64_t n) {
  const uint32_t bottombits = static_cast<uint32_t>(n);
  if (bottombits == 0) {
    return 32 + FindLSBSetNonZero(static_cast<uint32_t>(n >> 32));
  } else {
    return FindLSBSetNonZero(bottombits);
  }
}

#endif

extern const char num_bits[];

int Count(const void *m, int num_bytes);

inline int Log2Floor(uint32_t n) { return absl::bit_width(n) - 1; }
inline int Log2FloorNonZero(uint32_t n) { return Log2Floor(n); }
inline int Log2Floor64(uint64_t n) { return absl::bit_width(n) - 1; }
inline int Log2FloorNonZero64(uint64_t n) { return Log2Floor64(n); }

inline int FindMSBSetNonZero(uint32_t n) { return Log2FloorNonZero(n); }
inline int FindMSBSetNonZero64(uint64_t n) { return Log2FloorNonZero64(n); }

int Log2Ceiling(uint32_t n);
int Log2Ceiling64(uint64_t n);

}  // namespace bits
}  // namespace research_scann

#endif
