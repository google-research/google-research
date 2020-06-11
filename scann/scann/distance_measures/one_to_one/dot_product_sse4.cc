// Copyright 2020 The Google Research Authors.
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

// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/distance_measures/one_to_one/dot_product_sse4.h"
#ifdef __x86_64__

#include "scann/utils/intrinsics/sse4.h"

namespace tensorflow {
namespace scann_ops {
namespace dp_internal {

template <typename Byte, typename SseFuncs>
SCANN_SSE4_INLINE double DenseDotProductByteImpl(const Byte* aptr,
                                                 const Byte* bptr,
                                                 size_t length) {
  const Byte* aend = aptr + length;

  auto as_m128i = [](const Byte* x) SCANN_SSE4_INLINE_LAMBDA -> __m128i* {
    return reinterpret_cast<__m128i*>(const_cast<Byte*>(x));
  };

  conditional_t<IsSignedType<Byte>(), int32_t, uint32_t> scalar_accumulator = 0;
  if (aptr + 4 <= aend) {
    __m128i accumulator0 = _mm_setzero_si128();
    __m128i accumulator1 = _mm_setzero_si128();

    auto do_accumulations =
        [&accumulator0, &accumulator1](__m128i term) SCANN_SSE4_INLINE_LAMBDA {
          accumulator0 =
              _mm_add_epi32(accumulator0, SseFuncs::ExtendLower16To32(term));
          accumulator1 =
              _mm_add_epi32(accumulator1, SseFuncs::ExtendUpper16To32(term));
        };

    for (; aptr + 16 <= aend; aptr += 16, bptr += 16) {
      __m128i avals = _mm_loadu_si128(as_m128i(aptr));
      __m128i bvals = _mm_loadu_si128(as_m128i(bptr));

      __m128i avals_low = SseFuncs::ExtendLower8To16(avals);
      __m128i avals_high = SseFuncs::ExtendUpper8To16(avals);
      __m128i bvals_low = SseFuncs::ExtendLower8To16(bvals);
      __m128i bvals_high = SseFuncs::ExtendUpper8To16(bvals);
      do_accumulations(_mm_mullo_epi16(avals_low, bvals_low));
      do_accumulations(_mm_mullo_epi16(avals_high, bvals_high));
    }

    if (aptr + 8 <= aend) {
      __m128i avals = _mm_loadl_epi64(as_m128i(aptr));
      __m128i bvals = _mm_loadl_epi64(as_m128i(bptr));
      __m128i avals_low = SseFuncs::ExtendLower8To16(avals);
      __m128i bvals_low = SseFuncs::ExtendLower8To16(bvals);
      do_accumulations(_mm_mullo_epi16(avals_low, bvals_low));
      aptr += 8;
      bptr += 8;
    }

    if (aptr + 4 <= aend) {
      __m128i avals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(aptr));
      __m128i bvals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(bptr));
      __m128i avals_low = SseFuncs::ExtendLower8To16(avals);
      __m128i bvals_low = SseFuncs::ExtendLower8To16(bvals);
      do_accumulations(_mm_mullo_epi16(avals_low, bvals_low));
      aptr += 4;
      bptr += 4;
    }

    scalar_accumulator =
        SseFuncs::HorizontalSum32(_mm_add_epi32(accumulator0, accumulator1));
  }

  DCHECK_LT(aend - aptr, 4);
  for (; aptr < aend; ++aptr, ++bptr) {
    scalar_accumulator +=
        static_cast<int32_t>(*aptr) * static_cast<int32_t>(*bptr);
  }

  return static_cast<double>(scalar_accumulator);
}

class UnsignedDotProductSseFunctionsSse4 {
 public:
  SCANN_SSE4_INLINE static __m128i ExtendLower8To16(__m128i v) {
    return _mm_cvtepu8_epi16(v);
  }

  SCANN_SSE4_INLINE static __m128i ExtendUpper8To16(__m128i v) {
    return _mm_unpackhi_epi8(v, _mm_setzero_si128());
  }

  SCANN_SSE4_INLINE static __m128i ExtendLower16To32(__m128i v) {
    return _mm_cvtepu16_epi32(v);
  }

  SCANN_SSE4_INLINE static __m128i ExtendUpper16To32(__m128i v) {
    return _mm_unpackhi_epi16(v, _mm_setzero_si128());
  }

  SCANN_SSE4_INLINE static uint32_t HorizontalSum32(__m128i v) {
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    return _mm_cvtsi128_si32(v);
  }
};

class SignedDotProductSseFunctionsSse4 {
 public:
  SCANN_SSE4_INLINE static __m128i ExtendLower8To16(__m128i v) {
    return _mm_cvtepi8_epi16(v);
  }

  SCANN_SSE4_INLINE static __m128i ExtendUpper8To16(__m128i v) {
    return _mm_cvtepi8_epi16(_mm_srli_si128(v, 8));
  }

  SCANN_SSE4_INLINE static __m128i ExtendLower16To32(__m128i v) {
    return _mm_cvtepi16_epi32(v);
  }

  SCANN_SSE4_INLINE static __m128i ExtendUpper16To32(__m128i v) {
    return _mm_cvtepi16_epi32(_mm_srli_si128(v, 8));
  }

  SCANN_SSE4_INLINE static uint32_t HorizontalSum32(__m128i v) {
    v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
    v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
    return _mm_cvtsi128_si32(v);
  }
};

SCANN_SSE4_OUTLINE double DenseDotProductSse4(const DatapointPtr<uint8_t>& a,
                                              const DatapointPtr<uint8_t>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  return DenseDotProductByteImpl<uint8_t, UnsignedDotProductSseFunctionsSse4>(
      a.values(), b.values(), a.nonzero_entries());
}

SCANN_SSE4_OUTLINE double DenseDotProductSse4(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<int8_t>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  return DenseDotProductByteImpl<int8_t, SignedDotProductSseFunctionsSse4>(
      a.values(), b.values(), a.nonzero_entries());
}

SCANN_SSE4_OUTLINE double DenseDotProductSse4(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<float>& b) {
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  const int8_t* aptr = a.values();
  const float* bptr = b.values();
  const int8_t* aend = aptr + a.nonzero_entries();
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());

  auto as_m128i = [](const int8_t* x) SCANN_SSE4_INLINE_LAMBDA -> __m128i* {
    return reinterpret_cast<__m128i*>(const_cast<int8_t*>(x));
  };

  float scalar_accumulator = 0.0f;
  if (aptr + 4 <= aend) {
    __m128 accumulator0 = _mm_setzero_ps();
    __m128 accumulator1 = _mm_setzero_ps();

    for (; aptr + 16 <= aend; aptr += 16, bptr += 16) {
      __m128i avals = _mm_loadu_si128(as_m128i(aptr));
      __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
      __m128 bvals0 = _mm_loadu_ps(bptr);
      accumulator0 = _mm_add_ps(accumulator0, _mm_mul_ps(avals0, bvals0));

      __m128 avals1 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32((_mm_srli_si128(avals, 4))));
      __m128 bvals1 = _mm_loadu_ps(bptr + 4);
      accumulator1 = _mm_add_ps(accumulator1, _mm_mul_ps(avals1, bvals1));

      __m128 avals2 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(avals, 8)));
      __m128 bvals2 = _mm_loadu_ps(bptr + 8);
      accumulator0 = _mm_add_ps(accumulator0, _mm_mul_ps(avals2, bvals2));

      __m128 avals3 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(avals, 12)));
      __m128 bvals3 = _mm_loadu_ps(bptr + 12);
      accumulator1 = _mm_add_ps(accumulator1, _mm_mul_ps(avals3, bvals3));
    }

    if (aptr + 8 <= aend) {
      __m128i avals = _mm_loadl_epi64(as_m128i(aptr));
      __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
      __m128 bvals0 = _mm_loadu_ps(bptr);
      accumulator0 = _mm_add_ps(accumulator0, _mm_mul_ps(avals0, bvals0));

      __m128 avals1 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(avals, 4)));
      __m128 bvals1 = _mm_loadu_ps(bptr + 4);
      accumulator1 = _mm_add_ps(accumulator1, _mm_mul_ps(avals1, bvals1));
      aptr += 8;
      bptr += 8;
    }

    if (aptr + 4 <= aend) {
      __m128i avals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(aptr));
      __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
      __m128 bvals0 = _mm_loadu_ps(bptr);
      accumulator0 = _mm_add_ps(accumulator0, _mm_mul_ps(avals0, bvals0));
      aptr += 4;
      bptr += 4;
    }

    __m128 accumulator = _mm_add_ps(accumulator0, accumulator1);
    accumulator = _mm_hadd_ps(accumulator, accumulator);
    accumulator = _mm_hadd_ps(accumulator, accumulator);
    scalar_accumulator = accumulator[0];
  }

  DCHECK_LT(aend - aptr, 4);
  for (; aptr < aend; ++aptr, ++bptr) {
    scalar_accumulator += static_cast<float>(*aptr) * *bptr;
  }

  return static_cast<double>(scalar_accumulator);
}

SCANN_SSE4_OUTLINE double DenseDotProductSse4(const DatapointPtr<float>& a,
                                              const DatapointPtr<float>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());

  auto get_terms = [](const float* aptr, const float* bptr)
                       SCANN_SSE4_INLINE_LAMBDA {
                         __m128 avals = _mm_loadu_ps(aptr);
                         __m128 bvals = _mm_loadu_ps(bptr);
                         return _mm_mul_ps(avals, bvals);
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
    __m128 prod = _mm_mul_ps(avals, bvals);
    accumulator = _mm_add_ps(accumulator, prod);
    aptr += 2;
    bptr += 2;
  }

  if (aptr < aend) {
    accumulator[0] += aptr[0] * bptr[0];
  }

  accumulator = _mm_hadd_ps(accumulator, accumulator);
  accumulator = _mm_hadd_ps(accumulator, accumulator);
  return accumulator[0];
}

SCANN_SSE4_OUTLINE double DenseDotProductSse4(const DatapointPtr<double>& a,
                                              const DatapointPtr<double>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());

  auto get_terms = [](const double* aptr, const double* bptr)
                       SCANN_SSE4_INLINE_LAMBDA {
                         __m128d avals = _mm_loadu_pd(aptr);
                         __m128d bvals = _mm_loadu_pd(bptr);
                         return _mm_mul_pd(avals, bvals);
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
    result += *aptr * *bptr;
  }

  return result;
}

SCANN_SSE4_OUTLINE double DenseDotProductSse4(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<float>& b,
                                              const DatapointPtr<float>& c) {
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  DCHECK(c.IsDense());
  const int8_t* aptr = a.values();
  const float* bptr = b.values();
  const float* cptr = c.values();
  const int8_t* aend = aptr + a.nonzero_entries();
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK_EQ(a.nonzero_entries(), c.nonzero_entries());

  auto as_m128i = [](const int8_t* x) SCANN_SSE4_INLINE_LAMBDA -> __m128i* {
    return reinterpret_cast<__m128i*>(const_cast<int8_t*>(x));
  };

  float scalar_accumulator = 0.0f;
  if (aptr + 4 <= aend) {
    __m128 accumulator0 = _mm_setzero_ps();
    __m128 accumulator1 = _mm_setzero_ps();

    for (; aptr + 16 <= aend; aptr += 16, bptr += 16, cptr += 16) {
      __m128i avals = _mm_loadu_si128(as_m128i(aptr));
      __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
      __m128 bvals0 = _mm_loadu_ps(bptr);
      __m128 cvals0 = _mm_loadu_ps(cptr);
      __m128 prod0 = _mm_mul_ps(_mm_mul_ps(avals0, bvals0), cvals0);
      accumulator0 = _mm_add_ps(accumulator0, prod0);

      __m128 avals1 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32((_mm_srli_si128(avals, 4))));
      __m128 bvals1 = _mm_loadu_ps(bptr + 4);
      __m128 cvals1 = _mm_loadu_ps(cptr + 4);
      __m128 prod1 = _mm_mul_ps(_mm_mul_ps(avals1, bvals1), cvals1);
      accumulator1 = _mm_add_ps(accumulator1, prod1);

      __m128 avals2 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(avals, 8)));
      __m128 bvals2 = _mm_loadu_ps(bptr + 8);
      __m128 cvals2 = _mm_loadu_ps(cptr + 8);
      __m128 prod2 = _mm_mul_ps(_mm_mul_ps(avals2, bvals2), cvals2);
      accumulator0 = _mm_add_ps(accumulator0, prod2);

      __m128 avals3 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(avals, 12)));
      __m128 bvals3 = _mm_loadu_ps(bptr + 12);
      __m128 cvals3 = _mm_loadu_ps(cptr + 12);
      __m128 prod3 = _mm_mul_ps(_mm_mul_ps(avals3, bvals3), cvals3);
      accumulator1 = _mm_add_ps(accumulator1, prod3);
    }

    if (aptr + 8 <= aend) {
      __m128i avals = _mm_loadl_epi64(as_m128i(aptr));
      __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
      __m128 bvals0 = _mm_loadu_ps(bptr);
      __m128 cvals0 = _mm_loadu_ps(cptr);
      __m128 prod0 = _mm_mul_ps(_mm_mul_ps(avals0, bvals0), cvals0);
      accumulator0 = _mm_add_ps(accumulator0, prod0);

      __m128 avals1 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(avals, 4)));
      __m128 bvals1 = _mm_loadu_ps(bptr + 4);
      __m128 cvals1 = _mm_loadu_ps(cptr + 4);
      __m128 prod1 = _mm_mul_ps(_mm_mul_ps(avals1, bvals1), cvals1);
      accumulator1 = _mm_add_ps(accumulator1, prod1);
      aptr += 8;
      bptr += 8;
      cptr += 8;
    }

    if (aptr + 4 <= aend) {
      __m128i avals = _mm_cvtsi32_si128(*reinterpret_cast<const int*>(aptr));
      __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
      __m128 bvals0 = _mm_loadu_ps(bptr);
      __m128 cvals0 = _mm_loadu_ps(cptr);
      __m128 prod0 = _mm_mul_ps(_mm_mul_ps(avals0, bvals0), cvals0);
      accumulator0 = _mm_add_ps(accumulator0, prod0);
      aptr += 4;
      bptr += 4;
      cptr += 4;
    }

    __m128 accumulator = _mm_add_ps(accumulator0, accumulator1);
    accumulator = _mm_hadd_ps(accumulator, accumulator);
    accumulator = _mm_hadd_ps(accumulator, accumulator);
    scalar_accumulator = accumulator[0];
  }

  DCHECK_LT(aend - aptr, 4);
  for (; aptr < aend; ++aptr, ++bptr, ++cptr) {
    scalar_accumulator += static_cast<float>(*aptr) * *bptr * *cptr;
  }

  return static_cast<double>(scalar_accumulator);
}

SCANN_SSE4_OUTLINE double DenseDotProductSse4(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<int8_t>& b,
                                              const DatapointPtr<float>& c) {
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  DCHECK(c.IsDense());
  const int8_t* aptr = a.values();
  const int8_t* bptr = b.values();
  const float* cptr = c.values();
  const int8_t* aend = aptr + a.nonzero_entries();
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK_EQ(a.nonzero_entries(), c.nonzero_entries());

  auto as_m128i = [](const int8_t* x) SCANN_SSE4_INLINE_LAMBDA -> __m128i* {
    return reinterpret_cast<__m128i*>(const_cast<int8_t*>(x));
  };

  float scalar_accumulator = 0.0f;
  if (aptr + 4 <= aend) {
    __m128 accumulator0 = _mm_setzero_ps();
    __m128 accumulator1 = _mm_setzero_ps();

    for (; aptr + 16 <= aend; aptr += 16, bptr += 16, cptr += 16) {
      __m128i avals = _mm_loadu_si128(as_m128i(aptr));
      __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
      __m128i bvals = _mm_loadu_si128(as_m128i(bptr));
      __m128 bvals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(bvals));

      __m128 cvals0 = _mm_loadu_ps(cptr);
      __m128 prod0 = _mm_mul_ps(_mm_mul_ps(avals0, bvals0), cvals0);
      accumulator0 = _mm_add_ps(accumulator0, prod0);

      __m128 avals1 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32((_mm_srli_si128(avals, 4))));
      __m128 bvals1 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32((_mm_srli_si128(bvals, 4))));
      __m128 cvals1 = _mm_loadu_ps(cptr + 4);
      __m128 prod1 = _mm_mul_ps(_mm_mul_ps(avals1, bvals1), cvals1);
      accumulator1 = _mm_add_ps(accumulator1, prod1);

      __m128 avals2 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(avals, 8)));
      __m128 bvals2 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32((_mm_srli_si128(bvals, 8))));
      __m128 cvals2 = _mm_loadu_ps(cptr + 8);
      __m128 prod2 = _mm_mul_ps(_mm_mul_ps(avals2, bvals2), cvals2);
      accumulator0 = _mm_add_ps(accumulator0, prod2);

      __m128 avals3 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(avals, 12)));
      __m128 bvals3 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(bvals, 12)));
      __m128 cvals3 = _mm_loadu_ps(cptr + 12);
      __m128 prod3 = _mm_mul_ps(_mm_mul_ps(avals3, bvals3), cvals3);
      accumulator1 = _mm_add_ps(accumulator1, prod3);
    }

    if (aptr + 8 <= aend) {
      __m128i avals = _mm_loadl_epi64(as_m128i(aptr));
      __m128 avals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(avals));
      __m128i bvals = _mm_loadl_epi64(as_m128i(bptr));
      __m128 bvals0 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(bvals));
      __m128 cvals0 = _mm_loadu_ps(cptr);
      __m128 prod0 = _mm_mul_ps(_mm_mul_ps(avals0, bvals0), cvals0);
      accumulator0 = _mm_add_ps(accumulator0, prod0);

      __m128 avals1 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(avals, 4)));
      __m128 bvals1 =
          _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(bvals, 4)));
      __m128 cvals1 = _mm_loadu_ps(cptr + 4);
      __m128 prod1 = _mm_mul_ps(_mm_mul_ps(avals1, bvals1), cvals1);
      accumulator1 = _mm_add_ps(accumulator1, prod1);
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
      __m128 prod0 = _mm_mul_ps(_mm_mul_ps(avals0, bvals0), cvals0);
      accumulator0 = _mm_add_ps(accumulator0, prod0);
      aptr += 4;
      bptr += 4;
      cptr += 4;
    }

    __m128 accumulator = _mm_add_ps(accumulator0, accumulator1);
    accumulator = _mm_hadd_ps(accumulator, accumulator);
    accumulator = _mm_hadd_ps(accumulator, accumulator);
    scalar_accumulator = accumulator[0];
  }

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
