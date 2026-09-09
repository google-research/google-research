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

#if defined(__x86_64__)

#define HWY_COMPILE_ONLY_STATIC

#define HWY_BASELINE_TARGETS HWY_AVX2

#include "scann/utils/hwy-compact.h"

#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "hwy/highway.h"
#include "scann/utils/common.h"

namespace research_scann {

constexpr size_t kNumLanes = 8;

namespace hn = hwy::N_AVX2;

template <typename T>
SCANN_INLINE HWY_ATTR void DCompressStore(hn::Vec256<T> val_v,
                                          uint8_t* mask_bits,
                                          hn::Simd<T, kNumLanes, 0> tag,
                                          T* HWY_RESTRICT dst_unaligned) {
  StoreU(hn::CompressBits(val_v, mask_bits), tag, dst_unaligned);
}

SCANN_OUTLINE HWY_ATTR size_t HwyCompact(uint32_t* indices, float* values,
                                         const uint32_t* masks,
                                         size_t n_masks) {
  HWY_FULL(uint32_t) indices_v;
  HWY_FULL(float) values_v;
  DCHECK_EQ(Lanes(indices_v), kNumLanes);
  DCHECK_EQ(Lanes(values_v), kNumLanes);

  uint8_t mask_bits[8] = {};

  size_t write_idx = 0;
  for (size_t mask_idx : Seq(n_masks)) {
    const uint8_t* mask = reinterpret_cast<const uint8_t*>(&masks[mask_idx]);
    for (size_t i = 0; i < 32; i += kNumLanes) {
      const size_t read_idx = mask_idx * 32 + i;
      static_assert(kNumLanes == 8);
      mask_bits[0] = mask[i / 8];

      DCompressStore(LoadU(indices_v, indices + read_idx), mask_bits, indices_v,
                     indices + write_idx);

      DCompressStore(LoadU(values_v, values + read_idx), mask_bits, values_v,
                     values + write_idx);

      write_idx += hwy::PopCount(mask_bits);
    }
  }

  return write_idx;
}

}  // namespace research_scann
#elif defined(__aarch64__)

#define HWY_COMPILE_ONLY_STATIC

#define HWY_BASELINE_TARGETS HWY_NEON

#include <cstddef>
#include <cstdint>

#include "absl/log/check.h"
#include "hwy/highway.h"
#include "scann/utils/common.h"
#include "scann/utils/hwy-compact.h"

namespace research_scann {

namespace hn = hwy::HWY_NAMESPACE;
using DU8 = hn::FixedTag<uint8_t, 16>;
using DU32 = hn::FixedTag<uint32_t, 4>;
using DF32 = hn::FixedTag<float, 4>;

HWY_INLINE static hn::Vec<DU8> PermutationForCompress(uint32_t mask) {
  DCHECK_LT(mask, 16);

  alignas(16) static constexpr uint8_t indices[16 * 16] = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 0,  1,
      2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 4,  5,  6,  7,
      0,  1,  2,  3,  8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,
      6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 8,  9,  10, 11, 0,  1,  2,  3,
      4,  5,  6,  7,  12, 13, 14, 15, 0,  1,  2,  3,  8,  9,  10, 11, 4,  5,
      6,  7,  12, 13, 14, 15, 4,  5,  6,  7,  8,  9,  10, 11, 0,  1,  2,  3,
      12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
      0,  1,  2,  3,  12, 13, 14, 15, 4,  5,  6,  7,  8,  9,  10, 11, 4,  5,
      6,  7,  12, 13, 14, 15, 0,  1,  2,  3,  8,  9,  10, 11, 0,  1,  2,  3,
      4,  5,  6,  7,  12, 13, 14, 15, 8,  9,  10, 11, 8,  9,  10, 11, 12, 13,
      14, 15, 0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  8,  9,  10, 11,
      12, 13, 14, 15, 4,  5,  6,  7,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
      14, 15, 0,  1,  2,  3,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
      12, 13, 14, 15};
  return hn::Load(DU8{}, indices + 16 * mask);
}

SCANN_OUTLINE HWY_ATTR size_t HwyCompact(uint32_t* indices, float* values,
                                         const uint32_t* masks,
                                         size_t n_masks) {
  size_t write_idx = 0;
  for (size_t mask_idx : Seq(n_masks)) {
    auto mask = masks[mask_idx];
    for (size_t i = 0; i < 32; i += hn::Lanes(DU32{}), mask >>= 4) {
      const size_t read_idx = mask_idx * 32 + i;
      auto vindices = hn::LoadU(DU32{}, indices + read_idx);
      auto vvalues = hn::LoadU(DF32{}, values + read_idx);

      auto mask_u4 = mask & 0xF;
      auto p = PermutationForCompress(mask_u4);
      vindices = hn::BitCast(
          DU32{}, hn::TableLookupBytes(hn::BitCast(DU8{}, vindices), p));
      vvalues = hn::BitCast(
          DF32{}, hn::TableLookupBytes(hn::BitCast(DU8{}, vvalues), p));

      hn::StoreU(vindices, DU32{}, indices + write_idx);
      hn::StoreU(vvalues, DF32{}, values + write_idx);

      write_idx += __builtin_popcount(mask_u4);
    }
  }

  return write_idx;
}
}  // namespace research_scann

#endif
