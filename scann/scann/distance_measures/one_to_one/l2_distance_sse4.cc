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

#include "scann/distance_measures/one_to_one/l2_distance_sse4.h"
#ifdef __x86_64__

#include "scann/utils/intrinsics/sse4.h"

namespace tensorflow {
namespace scann_ops {
namespace l2_internal {

template <typename Byte, typename SseFuncs>
SCANN_SSE4_INLINE double DenseSquaredL2DistanceByteImpl(const Byte* aptr,
                                                        const Byte* bptr,
                                                        size_t length) {
  const Byte* aend = aptr + length;

  auto as_m128i = [](const Byte* x) SCANN_SSE4_INLINE_LAMBDA -> __m128i* {
    return reinterpret_cast<__m128i*>(const_cast<Byte*>(x));
  };

  auto get_terms = [&as_m128i](const Byte* aptr, const Byte* bptr)
                       SCANN_SSE4_INLINE_LAMBDA {
                         __m128i avals = _mm_loadu_si128(as_m128i(aptr));
                         __m128i bvals = _mm_loadu_si128(as_m128i(bptr));
                         __m128i diff = SseFuncs::AbsDiff(avals, bvals);

                         __m128i lower = SseFuncs::ZeroExtendLower8To16(diff);
                         __m128i upper = SseFuncs::ZeroExtendUpper8To16(diff);
                         lower = _mm_mullo_epi16(lower, lower);
                         upper = _mm_mullo_epi16(upper, upper);
                         return std::make_pair(lower, upper);
                       };

  uint32_t scalar_accumulator = 0;
  if (aptr + 4 <= aend) {
    __m128i accumulator0 = _mm_setzero_si128();
    __m128i accumulator1 = _mm_setzero_si128();

    auto do_accumulations = [&accumulator0, &accumulator1](
                                __m128i term) SCANN_SSE4_INLINE_LAMBDA {
      accumulator0 =
          _mm_add_epi32(accumulator0, SseFuncs::ZeroExtendLower16To32(term));
      accumulator1 =
          _mm_add_epi32(accumulator1, SseFuncs::ZeroExtendUpper16To32(term));
    };

    for (; aptr + 16 <= aend; aptr += 16, bptr += 16) {
      const pair<__m128i, __m128i> terms = get_terms(aptr, bptr);
      do_accumulations(terms.first);
      do_accumulations(terms.second);
    }

    if (aptr + 8 <= aend) {
      __m128i avals = _mm_loadl_epi64(as_m128i(aptr));
      __m128i bvals = _mm_loadl_epi64(as_m128i(bptr));
      __m128i diff = SseFuncs::AbsDiff(avals, bvals);
      __m128i lower = SseFuncs::ZeroExtendLower8To16(diff);
      lower = _mm_mullo_epi16(lower, lower);
      do_accumulations(lower);
      aptr += 8;
      bptr += 8;
    }

    if (aptr + 4 <= aend) {
      __m128i avals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(aptr));
      __m128i bvals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(bptr));
      __m128i diff = SseFuncs::AbsDiff(avals, bvals);
      __m128i lower = _mm_unpacklo_epi8(diff, _mm_setzero_si128());
      lower = _mm_mullo_epi16(lower, lower);
      do_accumulations(lower);
      aptr += 4;
      bptr += 4;
    }

    scalar_accumulator =
        SseFuncs::HorizontalSum32(_mm_add_epi32(accumulator0, accumulator1));
  }

  DCHECK_LT(aend - aptr, 4);
  for (; aptr < aend; ++aptr, ++bptr) {
    int32_t diff = static_cast<int32_t>(*aptr) - static_cast<int32_t>(*bptr);
    scalar_accumulator += diff * diff;
  }

  return static_cast<double>(scalar_accumulator);
}

class SseFunctionsSse4 {
 public:
  SCANN_SSE4_INLINE static __m128i ZeroExtendLower8To16(__m128i v) {
    return _mm_cvtepu8_epi16(v);
  }

  SCANN_SSE4_INLINE static __m128i ZeroExtendLower16To32(__m128i v) {
    return _mm_cvtepu16_epi32(v);
  }

  SCANN_SSE4_INLINE static __m128i ZeroExtendUpper8To16(__m128i v) {
    return _mm_unpackhi_epi8(v, _mm_setzero_si128());
  }

  SCANN_SSE4_INLINE static __m128i ZeroExtendUpper16To32(__m128i v) {
    return _mm_unpackhi_epi16(v, _mm_setzero_si128());
  }

  SCANN_SSE4_INLINE static uint32_t HorizontalSum32(__m128i v) {
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    return _mm_cvtsi128_si32(v);
  }
};

class SignedSquaredL2SseFunctionsSse4 : public SseFunctionsSse4 {
 public:
  SCANN_SSE4_INLINE static __m128i AbsDiff(__m128i a, __m128i b) {
    return _mm_sub_epi8(_mm_max_epi8(a, b), _mm_min_epi8(a, b));
  }
};

class UnsignedSquaredL2SseFunctionsSse4 : public SseFunctionsSse4 {
 public:
  SCANN_SSE4_INLINE static __m128i AbsDiff(__m128i a, __m128i b) {
    return _mm_sub_epi8(_mm_max_epu8(a, b), _mm_min_epu8(a, b));
  }
};

SCANN_SSE4_OUTLINE double DenseSquaredL2DistanceSse4(
    const DatapointPtr<uint8_t>& a, const DatapointPtr<uint8_t>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  return DenseSquaredL2DistanceByteImpl<uint8_t,
                                        UnsignedSquaredL2SseFunctionsSse4>(
      a.values(), b.values(), a.nonzero_entries());
}

SCANN_SSE4_OUTLINE double DenseSquaredL2DistanceSse4(
    const DatapointPtr<int8_t>& a, const DatapointPtr<int8_t>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  return DenseSquaredL2DistanceByteImpl<int8_t,
                                        SignedSquaredL2SseFunctionsSse4>(
      a.values(), b.values(), a.nonzero_entries());
}

SCANN_SSE4_OUTLINE double DenseSquaredL2DistanceSse4(
    const DatapointPtr<float>& a, const DatapointPtr<float>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());

  auto get_terms = [](const float* aptr, const float* bptr)
                       SCANN_SSE4_INLINE_LAMBDA {
                         __m128 avals = _mm_loadu_ps(aptr);
                         __m128 bvals = _mm_loadu_ps(bptr);
                         __m128 diff = _mm_sub_ps(avals, bvals);
                         return _mm_mul_ps(diff, diff);
                       };

  const float* aptr = a.values();
  const float* bptr = b.values();
  const float* aend = aptr + a.nonzero_entries();
  __m128 accumulator = _mm_setzero_ps();
  if (aptr + 8 <= aend) {
    __m128 accumulator0 = get_terms(aptr, bptr);
    __m128 accumulator1 = get_terms(aptr + 4, bptr + 4);
    aptr += 8;
    bptr += 8;
    for (; aptr + 8 <= aend; aptr += 8, bptr += 8) {
      accumulator0 = _mm_add_ps(accumulator0, get_terms(aptr, bptr));
      accumulator1 = _mm_add_ps(accumulator1, get_terms(aptr + 4, bptr + 4));
    }

    accumulator = _mm_add_ps(accumulator0, accumulator1);
  }

  if (aptr + 4 <= aend) {
    accumulator = _mm_add_ps(accumulator, get_terms(aptr, bptr));
    aptr += 4;
    bptr += 4;
  }

  if (aptr + 2 <= aend) {
    __m128 avals = _mm_setzero_ps();
    __m128 bvals = _mm_setzero_ps();
    avals = _mm_loadh_pi(avals, reinterpret_cast<const __m64*>(aptr));
    bvals = _mm_loadh_pi(bvals, reinterpret_cast<const __m64*>(bptr));
    __m128 diff = _mm_sub_ps(avals, bvals);
    __m128 squared = _mm_mul_ps(diff, diff);
    accumulator = _mm_add_ps(accumulator, squared);
    aptr += 2;
    bptr += 2;
  }

  if (aptr < aend) {
    accumulator[0] += (aptr[0] - bptr[0]) * (aptr[0] - bptr[0]);
  }

  accumulator = _mm_hadd_ps(accumulator, accumulator);
  accumulator = _mm_hadd_ps(accumulator, accumulator);
  return accumulator[0];
}

SCANN_SSE4_OUTLINE double DenseSquaredL2DistanceSse4(
    const DatapointPtr<double>& a, const DatapointPtr<double>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());

  auto get_terms = [](const double* aptr, const double* bptr)
                       SCANN_SSE4_INLINE_LAMBDA {
                         __m128d avals = _mm_loadu_pd(aptr);
                         __m128d bvals = _mm_loadu_pd(bptr);
                         __m128d diff = _mm_sub_pd(avals, bvals);
                         return _mm_mul_pd(diff, diff);
                       };

  const double* aptr = a.values();
  const double* bptr = b.values();
  const double* aend = aptr + a.nonzero_entries();
  __m128d accumulator = _mm_setzero_pd();
  if (aptr + 4 <= aend) {
    __m128d accumulator0 = get_terms(aptr, bptr);
    __m128d accumulator1 = get_terms(aptr + 2, bptr + 2);
    aptr += 4;
    bptr += 4;
    for (; aptr + 4 <= aend; aptr += 4, bptr += 4) {
      accumulator0 = _mm_add_pd(accumulator0, get_terms(aptr, bptr));
      accumulator1 = _mm_add_pd(accumulator1, get_terms(aptr + 2, bptr + 2));
    }

    accumulator = _mm_add_pd(accumulator0, accumulator1);
  }

  if (aptr + 2 <= aend) {
    accumulator = _mm_add_pd(accumulator, get_terms(aptr, bptr));
    aptr += 2;
    bptr += 2;
  }

  accumulator = _mm_hadd_pd(accumulator, accumulator);
  double result = accumulator[0];

  if (aptr < aend) {
    const double diff = *aptr - *bptr;
    result += diff * diff;
  }

  return result;
}

}  // namespace l2_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
