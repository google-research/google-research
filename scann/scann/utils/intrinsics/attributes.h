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

#ifndef SCANN_UTILS_INTRINSICS_ATTRIBUTES_H_
#define SCANN_UTILS_INTRINSICS_ATTRIBUTES_H_

#ifdef __x86_64__

#define SCANN_SSE4
#define SCANN_AVX1 __attribute((target("avx")))
#define SCANN_AVX2 __attribute((target("avx,avx2,fma")))
#define SCANN_AVX512 \
  __attribute((target("avx,avx2,fma,avx512f,avx512dq,avx512bw")))

#else

#define SCANN_SSE4
#define SCANN_AVX1
#define SCANN_AVX2
#define SCANN_AVX512

#endif

#define SCANN_SIMD_INLINE SCANN_SIMD_ATTRIBUTE SCANN_INLINE
#define SCANN_SIMD_INLINE_LAMBDA SCANN_SIMD_ATTRIBUTE SCANN_INLINE_LAMBDA
#define SCANN_SIMD_OUTLINE SCANN_SIMD_ATTRIBUTE SCANN_OUTLINE

#define SCANN_SSE4_INLINE SCANN_SSE4 SCANN_INLINE
#define SCANN_SSE4_INLINE_LAMBDA SCANN_SSE4 SCANN_INLINE_LAMBDA
#define SCANN_SSE4_OUTLINE SCANN_SSE4 SCANN_OUTLINE

#define SCANN_AVX1_INLINE SCANN_AVX1 SCANN_INLINE
#define SCANN_AVX1_INLINE_LAMBDA SCANN_AVX1 SCANN_INLINE_LAMBDA
#define SCANN_AVX1_OUTLINE SCANN_AVX1 SCANN_OUTLINE

#define SCANN_AVX2_INLINE SCANN_AVX2 SCANN_INLINE
#define SCANN_AVX2_INLINE_LAMBDA SCANN_AVX2 SCANN_INLINE_LAMBDA
#define SCANN_AVX2_OUTLINE SCANN_AVX2 SCANN_OUTLINE

#define SCANN_AVX512_INLINE SCANN_AVX512 SCANN_INLINE
#define SCANN_AVX512_INLINE_LAMBDA SCANN_AVX512 SCANN_INLINE_LAMBDA
#define SCANN_AVX512_OUTLINE SCANN_AVX512 SCANN_OUTLINE

#endif
