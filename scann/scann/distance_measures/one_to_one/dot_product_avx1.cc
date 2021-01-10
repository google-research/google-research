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

#include "scann/distance_measures/one_to_one/dot_product_avx1.h"

#ifdef __x86_64__

#include "scann/data_format/datapoint.h"
#include "scann/utils/internal/avx_funcs.h"
#include "scann/utils/intrinsics/avx1.h"

namespace research_scann {
namespace dp_internal {

#define SCANN_SIMD_ATTRIBUTE SCANN_AVX1
#include "scann/distance_measures/one_to_one/dot_product_impl.inc"
#undef SCANN_SIMD_ATTRIBUTE

SCANN_AVX1_OUTLINE double DenseDotProductAvx1(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<float>& b) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  return DenseDotProductInt8FloatAvxImpl<AvxFunctionsAvx>(
      a.values(), b.values(), a.nonzero_entries());
}

SCANN_AVX1_OUTLINE double DenseDotProductAvx1(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<float>& b,
                                              const DatapointPtr<float>& c) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK_EQ(a.nonzero_entries(), c.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  DCHECK(c.IsDense());
  return DenseDotProductInt8FloatFloatAvxImpl<AvxFunctionsAvx>(
      a.values(), b.values(), c.values(), a.nonzero_entries());
}

SCANN_AVX1_OUTLINE double DenseDotProductAvx1(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<int8_t>& b,
                                              const DatapointPtr<float>& c) {
  DCHECK_EQ(a.nonzero_entries(), b.nonzero_entries());
  DCHECK_EQ(a.nonzero_entries(), c.nonzero_entries());
  DCHECK(a.IsDense());
  DCHECK(b.IsDense());
  DCHECK(c.IsDense());
  return DenseDotProductInt8Int8FloatAvxImpl<AvxFunctionsAvx>(
      a.values(), b.values(), c.values(), a.nonzero_entries());
}

}  // namespace dp_internal
}  // namespace research_scann

#endif
