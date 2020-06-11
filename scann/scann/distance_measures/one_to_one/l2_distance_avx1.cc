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

#include "scann/distance_measures/one_to_one/l2_distance_avx1.h"
#ifdef __x86_64__

#include "scann/utils/intrinsics/avx1.h"

namespace tensorflow {
namespace scann_ops {
namespace l2_internal {

SCANN_AVX1_INLINE __m256d AddTerms256(__m256d accumulator, const double* aptr,
                                      const double* bptr) {
  __m256d avals = _mm256_loadu_pd(aptr);
  __m256d bvals = _mm256_loadu_pd(bptr);
  __m256d diff = _mm256_sub_pd(avals, bvals);
  return _mm256_add_pd(_mm256_mul_pd(diff, diff), accumulator);
}

SCANN_AVX1_OUTLINE double DenseSquaredL2DistanceAvx1(
    const DatapointPtr<double>& a, const DatapointPtr<double>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());

  const double* aptr = a.values();
  const double* bptr = b.values();
  const double* aend = aptr + a.nonzero_entries();
  __m256d accumulator = _mm256_setzero_pd();
  if (aptr + 8 <= aend) {
    __m256d accumulator0 = AddTerms256(_mm256_setzero_pd(), aptr, bptr);
    __m256d accumulator1 = AddTerms256(_mm256_setzero_pd(), aptr + 4, bptr + 4);
    aptr += 8;
    bptr += 8;
    for (; aptr + 8 <= aend; aptr += 8, bptr += 8) {
      accumulator0 = AddTerms256(accumulator0, aptr, bptr);
      accumulator1 = AddTerms256(accumulator1, aptr + 4, bptr + 4);
    }
    accumulator = _mm256_add_pd(accumulator0, accumulator1);
  }

  if (aptr + 4 <= aend) {
    accumulator = AddTerms256(accumulator, aptr, bptr);
    aptr += 4;
    bptr += 4;
  }

  __m128d upper = _mm256_extractf128_pd(accumulator, 1);
  const __m128d lower = _mm256_castpd256_pd128(accumulator);
  if (aptr + 2 <= aend) {
    __m128d avals = _mm_loadu_pd(aptr);
    __m128d bvals = _mm_loadu_pd(bptr);
    __m128d diff = _mm_sub_pd(avals, bvals);
    upper = _mm_add_pd(_mm_mul_pd(diff, diff), upper);
    aptr += 2;
    bptr += 2;
  }
  __m128d sum = _mm_add_pd(upper, lower);
  double result = sum[0] + sum[1];

  if (aptr < aend) {
    const double to_square = *aptr - *bptr;
    result += to_square * to_square;
  }
  return result;
}

}  // namespace l2_internal
}  // namespace scann_ops
}  // namespace tensorflow

#endif
