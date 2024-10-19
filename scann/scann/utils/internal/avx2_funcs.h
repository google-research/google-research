// Copyright 2024 The Google Research Authors.
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

#ifndef SCANN_UTILS_INTERNAL_AVX2_FUNCS_H_
#define SCANN_UTILS_INTERNAL_AVX2_FUNCS_H_
#ifdef __x86_64__

#include "scann/utils/intrinsics/avx2.h"
#include "scann/utils/types.h"

namespace research_scann {

class AvxFunctionsAvx2Fma {
 public:
  SCANN_AVX2_INLINE static __m256 Int8ToFloatLower(__m128i x) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x));
  }

  SCANN_AVX2_INLINE static __m256 Int8ToFloatUpper(__m128i x) {
    return Int8ToFloatLower(_mm_srli_si128(x, 8));
  }

  SCANN_AVX2_INLINE static __m256 MultiplyAdd(__m256 mul1, __m256 mul2,
                                              __m256 add) {
    return _mm256_fmadd_ps(mul1, mul2, add);
  }
  SCANN_AVX2_INLINE static __m256d MultiplyAdd(__m256d mul1, __m256d mul2,
                                               __m256d add) {
    return _mm256_fmadd_pd(mul1, mul2, add);
  }
  SCANN_AVX2_INLINE static __m128d MultiplyAdd(__m128d mul1, __m128d mul2,
                                               __m128d add) {
    return _mm_fmadd_pd(mul1, mul2, add);
  }

  SCANN_AVX2_INLINE static __m256 MultiplySub(__m256 mul1, __m256 mul2,
                                              __m256 base) {
    return _mm256_fnmadd_ps(mul1, mul2, base);
  }

  SCANN_AVX2_INLINE static __m256 SseToAvx(__m128 x) {
    return _mm256_insertf128_ps(_mm256_setzero_ps(), x, 0);
  }

  SCANN_AVX2_INLINE static float Sum8(__m256 x) {
    const __m128 upper = _mm256_extractf128_ps(x, 1);
    const __m128 lower = _mm256_castps256_ps128(x);
    __m128 sum = _mm_add_ps(upper, lower);
    sum = _mm_add_ps(
        sum, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(sum), 8)));
    return sum[0] + sum[1];
  }
};

}  // namespace research_scann

#endif
#endif
