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

#ifndef SCANN__UTILS_INTRINSICS_ATTRIBUTES_H_
#define SCANN__UTILS_INTRINSICS_ATTRIBUTES_H_

#ifdef __x86_64__

#define SCANN_AVX512_ATTRIBUTE \
  __attribute((target("avx,avx2,fma,avx512f,avx512dq,avx512bw")))
#define SCANN_AVX2_ATTRIBUTE __attribute((target("avx,avx2,fma")))
#define SCANN_AVX1_ATTRIBUTE __attribute((target("avx")))
#define SCANN_SSE4_ATTRIBUTE

#else

#define SCANN_AVX512_ATTRIBUTE
#define SCANN_AVX2_ATTRIBUTE
#define SCANN_AVX1_ATTRIBUTE
#define SCANN_SSE4_ATTRIBUTE

#endif

#define SCANN_AVX512_INLINE SCANN_AVX512_ATTRIBUTE SCANN_INLINE
#define SCANN_AVX512_INLINE_LAMBDA SCANN_AVX512_ATTRIBUTE SCANN_INLINE_LAMBDA
#define SCANN_AVX512_OUTLINE SCANN_AVX512_ATTRIBUTE SCANN_OUTLINE

#define SCANN_AVX2_INLINE SCANN_AVX2_ATTRIBUTE SCANN_INLINE
#define SCANN_AVX2_INLINE_LAMBDA SCANN_AVX2_ATTRIBUTE SCANN_INLINE_LAMBDA
#define SCANN_AVX2_OUTLINE SCANN_AVX2_ATTRIBUTE SCANN_OUTLINE

#define SCANN_AVX1_INLINE SCANN_AVX1_ATTRIBUTE SCANN_INLINE
#define SCANN_AVX1_INLINE_LAMBDA SCANN_AVX1_ATTRIBUTE SCANN_INLINE_LAMBDA
#define SCANN_AVX1_OUTLINE SCANN_AVX1_ATTRIBUTE SCANN_OUTLINE

#define SCANN_SSE4_INLINE SCANN_SSE4_ATTRIBUTE SCANN_INLINE
#define SCANN_SSE4_INLINE_LAMBDA SCANN_SSE4_ATTRIBUTE SCANN_INLINE_LAMBDA
#define SCANN_SSE4_OUTLINE SCANN_SSE4_ATTRIBUTE SCANN_OUTLINE

#endif
