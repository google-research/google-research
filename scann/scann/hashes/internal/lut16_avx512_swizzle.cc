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

#include <cstdint>
#ifdef __x86_64__
#include "scann/hashes/internal/lut16_avx512_swizzle.h"
#include "scann/utils/common.h"
#include "scann/utils/intrinsics/avx512.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

SCANN_AVX512_OUTLINE void Avx512Swizzle128(const uint8_t* src, uint8_t* dst) {
  Avx512<uint8_t> orig = Avx512<uint8_t>::Load(src);

  Avx512<uint8_t> low_nib_mask = Avx512<uint8_t>::Broadcast(0x0F);

  Avx512<uint8_t> even_nibs = orig & low_nib_mask;
  Avx512<uint8_t> odd_nibs =
      Avx512<uint8_t>(Avx512<uint16_t>(orig) >> 4) & low_nib_mask;

  constexpr uint64_t kAA = 0;
  constexpr uint64_t kBB = 8;

  Avx512<uint64_t> hi256_idxs = _mm512_set_epi64(
      kBB + 7, kBB + 6, kBB + 5, kBB + 4, kAA + 7, kAA + 6, kAA + 5, kAA + 4);
  Avx512<uint8_t> hi256_nibs =
      _mm512_permutex2var_epi64(*even_nibs, *hi256_idxs, *odd_nibs);
  Avx512<uint64_t> lo256_idxs =

      _mm512_set_epi64(kBB + 3, kBB + 2, kBB + 1, kBB + 0, kAA + 3, kAA + 2,
                       kAA + 1, kAA + 0);
  Avx512<uint8_t> lo256_nibs =
      _mm512_permutex2var_epi64(*even_nibs, *lo256_idxs, *odd_nibs);

  hi256_nibs = Avx512<uint8_t>(Avx512<uint16_t>(hi256_nibs) << 4);

  Avx512<uint8_t> new_bytes = hi256_nibs + lo256_nibs;

  constexpr int kDDBB = 0b11'11'10'10;
  constexpr int kCCAA = 0b01'01'00'00;
  Avx512<uint8_t> new_bytes_ddbb = _mm512_permutex_epi64(*new_bytes, kDDBB);
  Avx512<uint8_t> new_bytes_ccaa = _mm512_permutex_epi64(*new_bytes, kCCAA);

  Avx512<uint8_t> interleaved =
      _mm512_unpackhi_epi8(*new_bytes_ccaa, *new_bytes_ddbb);

  interleaved.Store(dst);
}

SCANN_AVX512_OUTLINE void Avx512Swizzle32(const uint8_t* src, uint8_t* dst) {
  array<uint8_t, 32> nibbles;
  for (size_t j : Seq(16)) {
    nibbles[2 * j + 0] = (src[j] >> 0) & 0x0F;
    nibbles[2 * j + 1] = (src[j] >> 4) & 0x0F;
  }
  for (size_t j : Seq(16)) {
    dst[j] = (nibbles[j + 0] << 0) + (nibbles[j + 16] << 4);
  }
}

SCANN_AVX512_OUTLINE void Avx512PlatformSpecificSwizzle(uint8_t* packed_dataset,
                                                        int num_datapoints,
                                                        int num_codes_per_dp) {
  size_t num_32dp_simd_iters = DivRoundUp(num_datapoints, 32);

  const size_t num_256dp_simd_iters = num_32dp_simd_iters / 8;
  num_32dp_simd_iters %= 8;

  const size_t num_128dp_simd_iters = num_32dp_simd_iters / 4;
  num_32dp_simd_iters %= 4;

  using Block = array<uint8_t, 16>;
  Block* blocks = reinterpret_cast<Block*>(packed_dataset);
  vector<Block> transposed(8 * num_codes_per_dp);

  for (auto _ : Seq(num_256dp_simd_iters)) {
    for (size_t jj : Seq(num_codes_per_dp)) {
      transposed[8 * jj + 0] = blocks[0 * num_codes_per_dp + jj];
      transposed[8 * jj + 1] = blocks[1 * num_codes_per_dp + jj];
      transposed[8 * jj + 2] = blocks[2 * num_codes_per_dp + jj];
      transposed[8 * jj + 3] = blocks[3 * num_codes_per_dp + jj];
      transposed[8 * jj + 4] = blocks[4 * num_codes_per_dp + jj];
      transposed[8 * jj + 5] = blocks[5 * num_codes_per_dp + jj];
      transposed[8 * jj + 6] = blocks[6 * num_codes_per_dp + jj];
      transposed[8 * jj + 7] = blocks[7 * num_codes_per_dp + jj];
    }

    for (size_t jj : Seq(2 * num_codes_per_dp)) {
      const uint8_t* src =
          reinterpret_cast<const uint8_t*>(&transposed[4 * jj]);
      uint8_t* dst = reinterpret_cast<uint8_t*>(&blocks[4 * jj]);
      Avx512Swizzle128(src, dst);
    }
    blocks += 8 * num_codes_per_dp;
  }

  for (auto _ : Seq(num_128dp_simd_iters)) {
    for (size_t jj : Seq(num_codes_per_dp)) {
      transposed[4 * jj + 0] = blocks[0 * num_codes_per_dp + jj];
      transposed[4 * jj + 1] = blocks[1 * num_codes_per_dp + jj];
      transposed[4 * jj + 2] = blocks[2 * num_codes_per_dp + jj];
      transposed[4 * jj + 3] = blocks[3 * num_codes_per_dp + jj];
    }

    for (size_t jj : Seq(num_codes_per_dp)) {
      const uint8_t* src =
          reinterpret_cast<const uint8_t*>(&transposed[4 * jj]);
      uint8_t* dst = reinterpret_cast<uint8_t*>(&blocks[4 * jj]);
      Avx512Swizzle128(src, dst);
    }
    blocks += 4 * num_codes_per_dp;
  }

  for (auto _ : Seq(num_32dp_simd_iters)) {
    for (size_t jj : Seq(num_codes_per_dp)) {
      uint8_t* ptr = reinterpret_cast<uint8_t*>(&blocks[1 * jj]);
      Avx512Swizzle32(ptr, ptr);
    }
    blocks += 1 * num_codes_per_dp;
  }
}

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif
