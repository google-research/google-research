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

#ifndef SCANN_UTILS_INTRINSICS_HORIZONTAL_SUM_H_
#define SCANN_UTILS_INTRINSICS_HORIZONTAL_SUM_H_

#include "scann/utils/intrinsics/simd.h"

namespace research_scann {
namespace fallback {

SCANN_INLINE float HorizontalSum(Simd<float> a) { return a.Unwrap(); }
SCANN_INLINE float HorizontalSum(Simd<double> a) { return a.Unwrap(); }

template <typename FloatT>
SCANN_INLINE void HorizontalSum2X(Simd<FloatT> a, Simd<FloatT> b,
                                  FloatT* resulta, FloatT* resultb) {
  *resulta = a.Unwrap();
  *resultb = b.Unwrap();
}

template <typename FloatT>
SCANN_INLINE void HorizontalSum4X(Simd<FloatT> a, Simd<FloatT> b,
                                  Simd<FloatT> c, Simd<FloatT> d,
                                  FloatT* resulta, FloatT* resultb,
                                  FloatT* resultc, FloatT* resultd) {
  *resulta = a.Unwrap();
  *resultb = b.Unwrap();
  *resultc = c.Unwrap();
  *resultd = d.Unwrap();
}

}  // namespace fallback

#ifdef __x86_64__

namespace sse4 {

SCANN_INLINE float HorizontalSum(Sse4<float> x) {
  x += Sse4<float>(_mm_shuffle_ps(*x, *x, 0b11'10'11'10));

  x += Sse4<float>(_mm_shuffle_ps(*x, *x, 0b11'10'01'01));

  return x.GetLowElement();
}

SCANN_INLINE double HorizontalSum(Sse4<double> x) {
  x += Sse4<double>(_mm_shuffle_pd(*x, *x, 0b1'1));

  return x.GetLowElement();
}

template <typename FloatT>
SCANN_INLINE void HorizontalSum2X(Sse4<FloatT> a, Sse4<FloatT> b,
                                  FloatT* resulta, FloatT* resultb) {
  *resulta = HorizontalSum(a);
  *resultb = HorizontalSum(b);
}

template <typename FloatT>
SCANN_INLINE void HorizontalSum4X(Sse4<FloatT> a, Sse4<FloatT> b,
                                  Sse4<FloatT> c, Sse4<FloatT> d,
                                  FloatT* resulta, FloatT* resultb,
                                  FloatT* resultc, FloatT* resultd) {
  HorizontalSum2X(a, b, resulta, resultb);
  HorizontalSum2X(c, d, resultc, resultd);
}

}  // namespace sse4

namespace avx1 {

SCANN_AVX1_INLINE float HorizontalSum(Avx1<float> a) {
  Sse4<float> upper = _mm256_extractf128_ps(*a, 1);
  Sse4<float> lower = _mm256_castps256_ps128(*a);
  return sse4::HorizontalSum(upper + lower);
}

SCANN_AVX1_INLINE double HorizontalSum(Avx1<double> a) {
  Sse4<double> upper = _mm256_extractf128_pd(*a, 1);
  Sse4<double> lower = _mm256_castpd256_pd128(*a);
  return sse4::HorizontalSum(upper + lower);
}

template <typename FloatT>
SCANN_AVX1_INLINE Avx1<FloatT> Sum128BitLanes(Avx1<FloatT> a, Avx1<FloatT> b) {
  static_assert(IsSameAny<FloatT, float, double>());
  constexpr int kDestLoEqALo = 0x00;
  constexpr int kDestLoEqAHi = 0x01;
  constexpr int kDestHiEqBLo = 0x20;
  constexpr int kDestHiEqBHi = 0x30;
  Avx1<FloatT> term0, term1;
  if constexpr (IsSame<FloatT, float>()) {
    term0 = _mm256_permute2f128_ps(*a, *b, kDestLoEqALo + kDestHiEqBLo);
    term1 = _mm256_permute2f128_ps(*a, *b, kDestLoEqAHi + kDestHiEqBHi);
  }
  if constexpr (IsSame<FloatT, double>()) {
    term0 = _mm256_permute2f128_pd(*a, *b, kDestLoEqALo + kDestHiEqBLo);
    term1 = _mm256_permute2f128_pd(*a, *b, kDestLoEqAHi + kDestHiEqBHi);
  }
  return term0 + term1;
}

SCANN_AVX1_INLINE Avx1<float> Sum64BitLanes(Avx1<float> a, Avx1<float> b) {
  auto term0 = _mm256_shuffle_ps(*a, *b, 0b11'10'01'00);
  auto term1 = _mm256_shuffle_ps(*a, *b, 0b01'00'11'10);
  return term0 + term1;
}

SCANN_AVX1_INLINE void HorizontalSum2X(Avx1<float> a, Avx1<float> b,
                                       float* resulta, float* resultb) {
  auto sum = *Sum128BitLanes(a, b);

  sum += _mm256_shuffle_ps(sum, sum, 0b11'10'11'10);

  sum += _mm256_shuffle_ps(sum, sum, 0b11'10'01'01);

  *resulta = sum[0];
  *resultb = sum[4];
}

SCANN_AVX1_INLINE void HorizontalSum3X(Avx1<float> a, Avx1<float> b,
                                       Avx1<float> c, float* resulta,
                                       float* resultb, float* resultc) {
  Avx1<float> ac = Sum128BitLanes(a, c);
  Avx1<float> bg = b + Avx1<float>(_mm256_permute2f128_ps(*b, *b, 1));
  auto abcg = *Sum64BitLanes(ac, bg);

  abcg += _mm256_shuffle_ps(abcg, abcg, 0b11'11'01'01);

  *resulta = abcg[0];
  *resultb = abcg[2];
  *resultc = abcg[4];
}

SCANN_AVX1_INLINE void HorizontalSum4X(Avx1<float> a, Avx1<float> b,
                                       Avx1<float> c, Avx1<float> d,
                                       float* resulta, float* resultb,
                                       float* resultc, float* resultd) {
  Avx1<float> ac = Sum128BitLanes(a, c);
  Avx1<float> bd = Sum128BitLanes(b, d);
  auto abcd = *Sum64BitLanes(ac, bd);

  abcd += _mm256_shuffle_ps(abcd, abcd, 0b11'11'01'01);

  *resulta = abcd[0];
  *resultb = abcd[2];
  *resultc = abcd[4];
  *resultd = abcd[6];
}

SCANN_AVX1_INLINE void HorizontalSum2X(Avx1<double> a, Avx1<double> b,
                                       double* resulta, double* resultb) {
  auto sum = *Sum128BitLanes(a, b);

  sum += _mm256_shuffle_pd(sum, sum, 0b11'11);

  *resulta = sum[0];
  *resultb = sum[2];
}

SCANN_AVX1_INLINE void HorizontalSum4X(Avx1<double> a, Avx1<double> b,
                                       Avx1<double> c, Avx1<double> d,
                                       double* resulta, double* resultb,
                                       double* resultc, double* resultd) {
  HorizontalSum2X(a, b, resulta, resultb);
  HorizontalSum2X(c, d, resultc, resultd);
}

}  // namespace avx1

namespace avx2 {

using ::research_scann::avx1::HorizontalSum;
using ::research_scann::avx1::HorizontalSum2X;
using ::research_scann::avx1::HorizontalSum3X;
using ::research_scann::avx1::HorizontalSum4X;

}  // namespace avx2

namespace avx512 {

SCANN_AVX512_INLINE float HorizontalSum(Avx512<float> a) {
  return _mm512_reduce_add_ps(*a);
}

SCANN_AVX512_INLINE double HorizontalSum(Avx512<double> a) {
  return _mm512_reduce_add_pd(*a);
}

SCANN_AVX512_INLINE void HorizontalSum2X(Avx512<float> a, Avx512<float> b,
                                         float* resulta, float* resultb) {
  *resulta = _mm512_reduce_add_ps(*a);
  *resultb = _mm512_reduce_add_ps(*b);
}

SCANN_AVX512_INLINE void HorizontalSum2X(Avx512<double> a, Avx512<double> b,
                                         double* resulta, double* resultb) {
  *resulta = _mm512_reduce_add_pd(*a);
  *resultb = _mm512_reduce_add_pd(*b);
}

SCANN_AVX512_INLINE void HorizontalSum4X(Avx512<float> a, Avx512<float> b,
                                         Avx512<float> c, Avx512<float> d,
                                         float* resulta, float* resultb,
                                         float* resultc, float* resultd) {
  *resulta = _mm512_reduce_add_ps(*a);
  *resultb = _mm512_reduce_add_ps(*b);
  *resultc = _mm512_reduce_add_ps(*c);
  *resultd = _mm512_reduce_add_ps(*d);
}

SCANN_AVX512_INLINE void HorizontalSum4X(Avx512<double> a, Avx512<double> b,
                                         Avx512<double> c, Avx512<double> d,
                                         double* resulta, double* resultb,
                                         double* resultc, double* resultd) {
  *resulta = _mm512_reduce_add_pd(*a);
  *resultb = _mm512_reduce_add_pd(*b);
  *resultc = _mm512_reduce_add_pd(*c);
  *resultd = _mm512_reduce_add_pd(*d);
}

}  // namespace avx512

#endif

}  // namespace research_scann

#endif
