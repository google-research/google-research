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

#ifndef SCANN_UTILS_INTRINSICS_AVX512_VNNI_H_
#define SCANN_UTILS_INTRINSICS_AVX512_VNNI_H_

#include "scann/utils/intrinsics/avx512.h"
#include "scann/utils/intrinsics/flags.h"

#ifdef __x86_64__

namespace research_scann {
namespace avx512_vnni {

static constexpr PlatformGeneration kPlatformGeneration =
    kCascadelakeAvx512Vnni;

SCANN_INLINE string_view SimdName() { return "AVX-512-VNNI"; }
SCANN_INLINE bool RuntimeSupportsSimd() { return RuntimeSupportsAvx512Vnni(); }

template <typename T, size_t... kTensorNumRegisters>
using Simd = Avx512<T, kTensorNumRegisters...>;

template <typename T, size_t kTensorNumElements0, size_t... kTensorNumElements>
using SimdFor = Avx512For<T, kTensorNumElements0, kTensorNumElements...>;

using Zeros = Avx512Zeros;
using Uninitialized = Avx512Uninitialized;

}  // namespace avx512_vnni
}  // namespace research_scann

#endif
#endif
