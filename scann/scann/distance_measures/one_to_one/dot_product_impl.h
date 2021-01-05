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

#ifndef SCANN__DISTANCE_MEASURES_ONE_TO_ONE_DOT_PRODUCT_IMPL_H_
#define SCANN__DISTANCE_MEASURES_ONE_TO_ONE_DOT_PRODUCT_IMPL_H_
#ifdef __x86_64__

#include "scann/data_format/datapoint.h"

#ifndef SCANN_SIMD_INLINE
#define SCANN_SIMD_INLINE SCANN_INLINE
#endif

namespace tensorflow {
namespace scann_ops {
namespace dp_internal {

template <typename AvxFuncs>
SCANN_SIMD_INLINE double DenseDotProductInt8FloatAvxImpl(const int8_t* aptr,
                                                         const float* bptr,
                                                         size_t length) {
  const int8_t* aend = aptr + length;

  auto as_m128i = [](const int8_t* x) -> __m128i* {
    return reinterpret_cast<__m128i*>(const_cast<int8_t*>(x));
  };

  __m256 accumulator0 = _mm256_setzero_ps();
  __m256 accumulator1 = _mm256_setzero_ps();
  for (; aptr + 16 <= aend; aptr += 16, bptr += 16) {
    __m128i avals = _mm_loadu_si128(as_m128i(aptr));
    __m256 avals0 = AvxFuncs::Int8ToFloatLower(avals);
    __m256 bvals0 = _mm256_loadu_ps(bptr);
    accumulator0 = AvxFuncs::MultiplyAdd(avals0, bvals0, accumulator0);

    __m256 avals1 = AvxFuncs::Int8ToFloatUpper(avals);
    __m256 bvals1 = _mm256_loadu_ps(bptr + 8);
    accumulator1 = AvxFuncs::MultiplyAdd(avals1, bvals1, accumulator1);
  }

  if (aptr + 8 <= aend) {
    __m128i avals = _mm_loadl_epi64(as_m128i(aptr));
    __m256 avals0 = AvxFuncs::Int8ToFloatLower(avals);
    __m256 bvals0 = _mm256_loadu_ps(bptr);
    accumulator0 = AvxFuncs::MultiplyAdd(avals0, bvals0, accumulator0);
    aptr += 8;
    bptr += 8;
  }

  if (aptr + 4 <= aend) {
    __m128i avals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(aptr));
    __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
    __m128 bvals0 = _mm_loadu_ps(bptr);
    __m128 prod = _mm_mul_ps(avals0, bvals0);
    accumulator0 = _mm256_add_ps(
        accumulator0, _mm256_insertf128_ps(_mm256_setzero_ps(), prod, 0));
    aptr += 4;
    bptr += 4;
  }

  float scalar_accumulator =
      AvxFuncs::Sum8(_mm256_add_ps(accumulator0, accumulator1));

  DCHECK_LT(aend - aptr, 4);
  for (; aptr < aend; ++aptr, ++bptr) {
    scalar_accumulator += static_cast<float>(*aptr) * *bptr;
  }

  return static_cast<double>(scalar_accumulator);
}

template <typename AvxFuncs>
SCANN_SIMD_INLINE double DenseDotProductInt8FloatFloatAvxImpl(
    const int8_t* aptr, const float* bptr, const float* cptr, size_t length) {
  const int8_t* aend = aptr + length;

  auto as_m128i = [](const int8_t* x) -> __m128i* {
    return reinterpret_cast<__m128i*>(const_cast<int8_t*>(x));
  };

  __m256 accumulator0 = _mm256_setzero_ps();
  __m256 accumulator1 = _mm256_setzero_ps();
  for (; aptr + 16 <= aend; aptr += 16, bptr += 16, cptr += 16) {
    __m128i avals = _mm_loadu_si128(as_m128i(aptr));
    __m256 avals0 = AvxFuncs::Int8ToFloatLower(avals);
    __m256 bvals0 = _mm256_loadu_ps(bptr);
    __m256 cvals0 = _mm256_loadu_ps(cptr);
    __m256 bcvals0 = _mm256_mul_ps(bvals0, cvals0);
    accumulator0 = AvxFuncs::MultiplyAdd(avals0, bcvals0, accumulator0);

    __m256 avals1 = AvxFuncs::Int8ToFloatUpper(avals);
    __m256 bvals1 = _mm256_loadu_ps(bptr + 8);
    __m256 cvals1 = _mm256_loadu_ps(cptr + 8);
    __m256 bcvals1 = _mm256_mul_ps(bvals1, cvals1);
    accumulator1 = AvxFuncs::MultiplyAdd(avals1, bcvals1, accumulator1);
  }

  if (aptr + 8 <= aend) {
    __m128i avals = _mm_loadl_epi64(as_m128i(aptr));
    __m256 avals0 = AvxFuncs::Int8ToFloatLower(avals);
    __m256 bvals0 = _mm256_loadu_ps(bptr);
    __m256 cvals0 = _mm256_loadu_ps(cptr);
    __m256 bcvals0 = _mm256_mul_ps(bvals0, cvals0);
    accumulator0 = AvxFuncs::MultiplyAdd(avals0, bcvals0, accumulator0);
    aptr += 8;
    bptr += 8;
    cptr += 8;
  }

  if (aptr + 4 <= aend) {
    __m128i avals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(aptr));
    __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
    __m128 bvals0 = _mm_loadu_ps(bptr);
    __m128 cvals0 = _mm_loadu_ps(cptr);
    __m128 prod = _mm_mul_ps(_mm_mul_ps(avals0, bvals0), cvals0);
    accumulator0 = _mm256_add_ps(
        accumulator0, _mm256_insertf128_ps(_mm256_setzero_ps(), prod, 0));
    aptr += 4;
    bptr += 4;
    cptr += 4;
  }

  float scalar_accumulator =
      AvxFuncs::Sum8(_mm256_add_ps(accumulator0, accumulator1));

  DCHECK_LT(aend - aptr, 4);
  for (; aptr < aend; ++aptr, ++bptr, ++cptr) {
    scalar_accumulator += static_cast<float>(*aptr) * *bptr * *cptr;
  }

  return static_cast<double>(scalar_accumulator);
}

template <typename AvxFuncs>
SCANN_SIMD_INLINE double DenseDotProductInt8Int8FloatAvxImpl(const int8_t* aptr,
                                                             const int8_t* bptr,
                                                             const float* cptr,
                                                             size_t length) {
  const int8_t* aend = aptr + length;

  auto as_m128i = [](const int8_t* x) -> __m128i* {
    return reinterpret_cast<__m128i*>(const_cast<int8_t*>(x));
  };

  __m256 accumulator0 = _mm256_setzero_ps();
  __m256 accumulator1 = _mm256_setzero_ps();
  for (; aptr + 16 <= aend; aptr += 16, bptr += 16, cptr += 16) {
    __m128i avals = _mm_loadu_si128(as_m128i(aptr));
    __m128i bvals = _mm_loadu_si128(as_m128i(bptr));
    __m256 avals0 = AvxFuncs::Int8ToFloatLower(avals);
    __m256 bvals0 = AvxFuncs::Int8ToFloatLower(bvals);
    __m256 cvals0 = _mm256_loadu_ps(cptr);
    __m256 bcvals0 = _mm256_mul_ps(bvals0, cvals0);
    accumulator0 = AvxFuncs::MultiplyAdd(avals0, bcvals0, accumulator0);

    __m256 avals1 = AvxFuncs::Int8ToFloatUpper(avals);
    __m256 bvals1 = AvxFuncs::Int8ToFloatUpper(bvals);
    __m256 cvals1 = _mm256_loadu_ps(cptr + 8);
    __m256 bcvals1 = _mm256_mul_ps(bvals1, cvals1);
    accumulator1 = AvxFuncs::MultiplyAdd(avals1, bcvals1, accumulator1);
  }

  if (aptr + 8 <= aend) {
    __m128i avals = _mm_loadl_epi64(as_m128i(aptr));
    __m128i bvals = _mm_loadl_epi64(as_m128i(bptr));
    __m256 avals0 = AvxFuncs::Int8ToFloatLower(avals);
    __m256 bvals0 = AvxFuncs::Int8ToFloatLower(bvals);
    __m256 cvals0 = _mm256_loadu_ps(cptr);
    __m256 bcvals0 = _mm256_mul_ps(bvals0, cvals0);
    accumulator0 = AvxFuncs::MultiplyAdd(avals0, bcvals0, accumulator0);
    aptr += 8;
    bptr += 8;
    cptr += 8;
  }

  if (aptr + 4 <= aend) {
    __m128i avals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(aptr));
    __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
    __m128i bvals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(bptr));
    __m128 bvals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(bvals));
    __m128 cvals0 = _mm_loadu_ps(cptr);
    __m128 prod = _mm_mul_ps(_mm_mul_ps(avals0, bvals0), cvals0);
    accumulator0 = _mm256_add_ps(
        accumulator0, _mm256_insertf128_ps(_mm256_setzero_ps(), prod, 0));
    aptr += 4;
    bptr += 4;
    cptr += 4;
  }

  float scalar_accumulator =
      AvxFuncs::Sum8(_mm256_add_ps(accumulator0, accumulator1));

  DCHECK_LT(aend - aptr, 4);
  for (; aptr < aend; ++aptr, ++bptr, ++cptr) {
    scalar_accumulator +=
        static_cast<float>(*aptr) * static_cast<float>(*bptr) * *cptr;
  }

  return static_cast<double>(scalar_accumulator);
}

}  // namespace dp_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
#endif
