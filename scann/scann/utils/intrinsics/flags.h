// Copyright 2026 The Google Research Authors.
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

ABSL_DECLARE_FLAG(bool, ignore_neon);

ABSL_DECLARE_FLAG(bool, ignore_neon_dotprod);

ABSL_DECLARE_FLAG(bool, ignore_neon_i8mm);

ABSL_DECLARE_FLAG(bool, ignore_sve);

ABSL_DECLARE_FLAG(bool, ignore_sve2);

namespace research_scann {
namespace flags_internal {

extern bool should_use_avx2;
extern bool should_use_avx512;
extern bool should_use_avx512_vnni;
extern bool should_use_amx;
bool TryEnableAmx();

extern bool should_use_neon;
extern bool should_use_neon_dotprod;
extern bool should_use_neon_i8mm;
extern bool should_use_sve;
extern bool should_use_sve2;
}  // namespace flags_internal

#ifdef __x86_64__

#ifdef SCANN_FORCE_SSE4
inline bool RuntimeSupportsAvx1() { return false; }
inline bool RuntimeSupportsAvx2() { return false; }
inline bool RuntimeSupportsAvx512() { return false; }
inline bool RuntimeSupportsAvx512Vnni() { return false; }
inline bool RuntimeSupportsAmx() { return false; }
#else

inline bool RuntimeSupportsAvx1() { return true; }
inline bool RuntimeSupportsAvx2() { return flags_internal::should_use_avx2; }
inline bool RuntimeSupportsAvx512() {
  return flags_internal::should_use_avx512;
}
inline bool RuntimeSupportsAvx512Vnni() {
  return flags_internal::should_use_avx512_vnni;
}

#if (defined(__clang__) && __clang_major__ < 20) || defined(MEMORY_SANITIZER)

inline bool RuntimeSupportsAmx() { return false; }

#else

#define SCANN_HAVE_AMX
inline bool RuntimeSupportsAmx() {
  if (!flags_internal::should_use_amx) return false;

  static bool amx_enabled = flags_internal::TryEnableAmx();
  return amx_enabled;
}

#endif

#endif

inline bool RuntimeSupportsSse4() { return true; }

#else

inline bool RuntimeSupportsAvx2() { return false; }
inline bool RuntimeSupportsAvx512() { return false; }
inline bool RuntimeSupportsAvx512Vnni() { return false; }
inline bool RuntimeSupportsAmx() { return false; }
inline bool RuntimeSupportsSse4() { return false; }
inline bool RuntimeSupportsAvx1() { return false; }

#endif

#ifdef __aarch64__
inline bool RuntimeSupportsNeon() { return flags_internal::should_use_neon; }
inline bool RuntimeSupportsNeonDotprod() {
  return flags_internal::should_use_neon_dotprod;
}
inline bool RuntimeSupportsNeonI8MM() {
  return flags_internal::should_use_neon_i8mm;
}
inline bool RuntimeSupportsSVE() { return flags_internal::should_use_sve; }
inline bool RuntimeSupportsSVE2() { return flags_internal::should_use_sve2; }
#else
inline bool RuntimeSupportsNeon() { return false; }
inline bool RuntimeSupportsNeonDotprod() { return false; }
inline bool RuntimeSupportsNeonI8MM() { return false; }
inline bool RuntimeSupportsSVE() { return false; }
inline bool RuntimeSupportsSVE2() { return false; }
#endif

enum PlatformGeneration {

  kFallbackForNonX86 = 99,

  kHighway = 98,

  kBaselineSse4 = 0,

  kSandyBridgeAvx1 = 1,

  kHaswellAvx2 = 2,

  kSkylakeAvx512 = 3,

  kCascadelakeAvx512Vnni = 4,

  kSapphireRapidsAmx = 5,
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
    case kCascadelakeAvx512Vnni:
      return "AVX512_VNNI";
    case kSapphireRapidsAmx:
      return "AMX";
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
  bool original_avx512_vnni_;
  bool original_amx_;
};

ScopedPlatformOverride TestHookOverridePlatform(PlatformGeneration generation);

}  // namespace research_scann

#endif
