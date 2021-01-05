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

#include "scann/distance_measures/one_to_one/l1_distance_sse4.h"
#ifdef __x86_64__

#include "scann/utils/intrinsics/sse4.h"

namespace tensorflow {
namespace scann_ops {
namespace l1_internal {

SCANN_SSE4_OUTLINE double DenseL1NormSse4(const DatapointPtr<float>& a,
                                          const DatapointPtr<float>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());

  constexpr int32_t abs_mask_scalar = 0x7FFFFFFF;
  static const __m128 abs_mask_vector = _mm_castsi128_ps(_mm_set_epi32(
      abs_mask_scalar, abs_mask_scalar, abs_mask_scalar, abs_mask_scalar));
  auto abs_ps = [](__m128 x) SCANN_SSE4_INLINE_LAMBDA {
    return _mm_and_ps(x, abs_mask_vector);
  };

  const float* aptr = a.values();
  const float* bptr = b.values();
  const float* aend = aptr + a.nonzero_entries();

  __m128 accumulator0 = _mm_setzero_ps();
  __m128 accumulator1 = _mm_setzero_ps();
  while (aptr + 8 <= aend) {
    __m128 avals0 = _mm_loadu_ps(aptr);
    __m128 bvals0 = _mm_loadu_ps(bptr);
    __m128 avals1 = _mm_loadu_ps(aptr + 4);
    __m128 bvals1 = _mm_loadu_ps(bptr + 4);

    __m128 diff0 = abs_ps(_mm_sub_ps(avals0, bvals0));
    __m128 diff1 = abs_ps(_mm_sub_ps(avals1, bvals1));

    accumulator0 = _mm_add_ps(accumulator0, diff0);
    accumulator1 = _mm_add_ps(accumulator1, diff1);

    aptr += 8;
    bptr += 8;
  }

  if (aptr + 4 <= aend) {
    __m128 avals = _mm_loadu_ps(aptr);
    __m128 bvals = _mm_loadu_ps(bptr);
    __m128 diff = abs_ps(_mm_sub_ps(avals, bvals));
    accumulator0 = _mm_add_ps(accumulator0, diff);
    aptr += 4;
    bptr += 4;
  }

  if (aptr + 2 <= aend) {
    __m128 avals = _mm_setzero_ps();
    __m128 bvals = _mm_setzero_ps();
    avals = _mm_loadh_pi(avals, reinterpret_cast<const __m64*>(aptr));
    bvals = _mm_loadh_pi(bvals, reinterpret_cast<const __m64*>(bptr));
    __m128 diff = abs_ps(_mm_sub_ps(avals, bvals));
    accumulator1 = _mm_add_ps(accumulator1, diff);
    aptr += 2;
    bptr += 2;
  }

  if (aptr < aend) {
    accumulator0[0] += std::abs(aptr[0] - bptr[0]);
  }

  __m128 accumulator = _mm_add_ps(accumulator0, accumulator1);
  accumulator = _mm_hadd_ps(accumulator, accumulator);
  accumulator = _mm_hadd_ps(accumulator, accumulator);
  return accumulator[0];
}

SCANN_SSE4_OUTLINE double DenseL1NormSse4(const DatapointPtr<double>& a,
                                          const DatapointPtr<double>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());

  constexpr int32_t abs_mask_scalar = 0x7FFFFFFF;
  constexpr int32_t all_set = 0xFFFFFFFF;
  static const __m128d abs_mask_vector = _mm_castsi128_pd(
      _mm_set_epi32(abs_mask_scalar, all_set, abs_mask_scalar, all_set));
  auto abs_pd = [](__m128d x) SCANN_SSE4_INLINE_LAMBDA {
    return _mm_and_pd(x, abs_mask_vector);
  };

  __m128d accumulator0 = _mm_setzero_pd();
  __m128d accumulator1 = _mm_setzero_pd();
  const double* aptr = a.values();
  const double* bptr = b.values();
  const double* aend = aptr + a.nonzero_entries();
  while (aptr + 4 <= aend) {
    __m128d avals0 = _mm_loadu_pd(aptr);
    __m128d bvals0 = _mm_loadu_pd(bptr);
    __m128d avals1 = _mm_loadu_pd(aptr + 2);
    __m128d bvals1 = _mm_loadu_pd(bptr + 2);
    __m128d diff0 = abs_pd(_mm_sub_pd(avals0, bvals0));
    __m128d diff1 = abs_pd(_mm_sub_pd(avals1, bvals1));
    accumulator0 = _mm_add_pd(accumulator0, diff0);
    accumulator1 = _mm_add_pd(accumulator1, diff1);
    aptr += 4;
    bptr += 4;
  }

  if (aptr + 2 <= aend) {
    __m128d avals0 = _mm_loadu_pd(aptr);
    __m128d bvals0 = _mm_loadu_pd(bptr);
    __m128d diff0 = abs_pd(_mm_sub_pd(avals0, bvals0));
    accumulator0 = _mm_add_pd(accumulator0, diff0);
    aptr += 2;
    bptr += 2;
  }

  __m128d accumulator = _mm_add_pd(accumulator0, accumulator1);
  accumulator = _mm_hadd_pd(accumulator, accumulator);
  double result = accumulator[0];

  if (aptr < aend) {
    result += std::abs(*aptr - *bptr);
  }

  return result;
}

}  // namespace l1_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
