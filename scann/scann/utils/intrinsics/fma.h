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

#ifndef SCANN_UTILS_INTRINSICS_FMA_H_
#define SCANN_UTILS_INTRINSICS_FMA_H_

#include "scann/utils/index_sequence.h"
#include "scann/utils/intrinsics/highway.h"
#include "scann/utils/intrinsics/simd.h"

namespace research_scann {

#ifdef __x86_64__

namespace avx512 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX512

#include "scann/utils/intrinsics/fma.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx512

namespace avx2 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX2
#include "scann/utils/intrinsics/fma.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx2

namespace avx1 {
#define SCANN_SIMD_ATTRIBUTE SCANN_AVX1
#include "scann/utils/intrinsics/fma.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace avx1

namespace sse4 {
#define SCANN_SIMD_ATTRIBUTE SCANN_SSE4
#include "scann/utils/intrinsics/fma.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace sse4

#endif

namespace fallback {
#define SCANN_SIMD_ATTRIBUTE
#include "scann/utils/intrinsics/fma.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace fallback

#if HWY_HAVE_CONSTEXPR_LANES
HWY_BEFORE_NAMESPACE();
namespace highway {
#define SCANN_SIMD_ATTRIBUTE
#include "scann/utils/intrinsics/fma.inc"
#undef SCANN_SIMD_ATTRIBUTE
}  // namespace highway
HWY_AFTER_NAMESPACE();
#endif

}  // namespace research_scann

#endif
