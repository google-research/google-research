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

#ifndef SCANN_UTILS_INTRINSICS_FLAGS_H_
#define SCANN_UTILS_INTRINSICS_FLAGS_H_

#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "scann/utils/common.h"
#include "scann/utils/types.h"

ABSL_DECLARE_FLAG(bool, ignore_avx512);

ABSL_DECLARE_FLAG(bool, ignore_avx2);

namespace research_scann {
namespace flags_internal {

extern bool should_use_avx2;
extern bool should_use_avx512;

}  // namespace flags_internal

#ifdef __x86_64__

#ifdef SCANN_FORCE_SSE4
inline bool RuntimeSupportsAvx1() { return false; }
inline bool RuntimeSupportsAvx2() { return false; }
inline bool RuntimeSupportsAvx512() { return false; }
#else

inline bool RuntimeSupportsAvx1() { return true; }
inline bool RuntimeSupportsAvx2() { return flags_internal::should_use_avx2; }
inline bool RuntimeSupportsAvx512() {
  return flags_internal::should_use_avx512;
}
#endif

inline bool RuntimeSupportsSse4() { return true; }

#else

inline bool RuntimeSupportsAvx2() { return false; }
inline bool RuntimeSupportsAvx512() { return false; }
inline bool RuntimeSupportsSse4() { return false; }
inline bool RuntimeSupportsAvx1() { return false; }

#endif

enum PlatformGeneration {

  kFallbackForNonX86 = 99,

  kHighway = 98,

  kBaselineSse4 = 0,

  kSandyBridgeAvx1 = 1,

  kHaswellAvx2 = 2,

  kSkylakeAvx512 = 3,
};

inline string_view PlatformName(PlatformGeneration x86_arch) {
  switch (x86_arch) {
    case kFallbackForNonX86:
      return "FallbackForNonX86";
    case kBaselineSse4:
      return "SSE4";
    case kSandyBridgeAvx1:
      return "AVX1";
    case kHaswellAvx2:
      return "AVX2";
    case kSkylakeAvx512:
      return "AVX512";
    default:
      return "INVALID_X86_ARCH";
  }
}

class ScopedPlatformOverride {
 public:
  SCANN_DECLARE_IMMOBILE_CLASS(ScopedPlatformOverride);

  explicit ScopedPlatformOverride(PlatformGeneration generation);

  ~ScopedPlatformOverride();

  bool IsSupported();

 private:
  bool original_avx2_;
  bool original_avx512_;
};

ScopedPlatformOverride TestHookOverridePlatform(PlatformGeneration generation);

}  // namespace research_scann

#endif
