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

#ifndef SCANN__OSS_WRAPPERS_SCANN_BITS_H_
#define SCANN__OSS_WRAPPERS_SCANN_BITS_H_

#include <cstddef>
#include <cstdint>

namespace tensorflow {
namespace scann_ops {
namespace bits {

#if (defined(__i386__) || defined(__x86_64__)) && defined(__GNUC__)

inline int CountLeadingZeros32(uint32_t n) {
  if (n == 0) {
    return sizeof(n) * 8;
  }
  return __builtin_clz(n);
}

inline int CountOnes(uint32_t n) { return __builtin_popcount(n); }

inline int Log2Floor(uint32_t n) { return n == 0 ? -1 : 31 ^ __builtin_clz(n); }

inline int Log2FloorNonZero(uint32_t n) { return 31 ^ __builtin_clz(n); }

inline int FindLSBSetNonZero(uint32_t n) { return __builtin_ctz(n); }

inline int CountLeadingZeros64(uint64_t n) {
  if (n == 0) {
    return sizeof(n) * 8;
  }
  return __builtin_clzll(n);
}

inline int CountOnes64(uint64_t n) { return __builtin_popcountll(n); }

inline int Log2Floor64(uint64_t n) {
  return n == 0 ? -1 : 63 ^ __builtin_clzll(n);
}

inline int Log2FloorNonZero64(uint64_t n) { return 63 ^ __builtin_clzll(n); }

inline int FindLSBSetNonZero64(uint64_t n) { return __builtin_ctzll(n); }

#else

int CountLeadingZeros32(uint32_t n);

inline int CountOnes(uint32_t n) {
  n -= ((n >> 1) & 0x55555555);
  n = ((n >> 2) & 0x33333333) + (n & 0x33333333);
  return static_cast<int>((((n + (n >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24);
}

int Log2Floor(uint32_t n);

inline int Log2FloorNonZero(uint32_t n) { return Log2Floor(n); }

int FindLSBSetNonZero(uint32_t n);

inline int CountOnes64(uint64_t n) {
  return CountOnes(n >> 32) + CountOnes(n & 0xffffffff);
}

inline int CountLeadingZeros64(uint64_t n) {
  return ((n >> 32) ? CountLeadingZeros32(n >> 32)
                    : 32 + CountLeadingZeros32(n));
}

inline int Log2Floor64(uint64_t n) {
  const uint32_t topbits = static_cast<uint32_t>(n >> 32);
  if (topbits == 0) {
    return Log2Floor(static_cast<uint32_t>(n));
  } else {
    return 32 + Log2FloorNonZero(topbits);
  }
}

inline int Log2FloorNonZero64(uint64_t n) { return Log2FloorNonZero64(n); }

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

inline int CountOnesInByte(unsigned char n) { return num_bits[n]; }

inline int FindMSBSetNonZero(uint32_t n) { return Log2FloorNonZero(n); }
inline int FindMSBSetNonZero64(uint64_t n) { return Log2FloorNonZero64(n); }

int Log2Ceiling(uint32_t n);
int Log2Ceiling64(uint64_t n);

}  // namespace bits
}  // namespace scann_ops
}  // namespace tensorflow

#endif
