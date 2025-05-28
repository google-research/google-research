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

#ifdef __x86_64__

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

template <typename V>
SCANN_INLINE HWY_ATTR void DCompressStore(
    hwy::N_AVX2::Vec256<V> val_v, uint64_t mask_bits,
    hwy::N_AVX2::Simd<V, kNumLanes, 0> tag, V* HWY_RESTRICT dst_unaligned) {
  StoreU(hwy::N_AVX2::detail::Compress(val_v, mask_bits), tag, dst_unaligned);
}

SCANN_OUTLINE HWY_ATTR size_t HwyCompact(uint32_t* indices, float* values,
                                         const uint32_t* masks,
                                         size_t n_masks) {
  HWY_FULL(uint32_t) indices_v;
  HWY_FULL(float) values_v;
  DCHECK_EQ(Lanes(indices_v), kNumLanes);
  DCHECK_EQ(Lanes(values_v), kNumLanes);

  size_t write_idx = 0;
  for (size_t mask_idx : Seq(n_masks)) {
    const uint8_t* mask = reinterpret_cast<const uint8_t*>(&masks[mask_idx]);
    for (size_t i = 0; i < 32; i += kNumLanes) {
      const size_t read_idx = mask_idx * 32 + i;

      uint64_t mask_bits = 0;
      hwy::CopyBytes<1>(mask + i / 8, &mask_bits);

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
#endif
