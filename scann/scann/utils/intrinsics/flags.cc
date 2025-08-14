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

#include "scann/utils/intrinsics/flags.h"

#include <sys/syscall.h>

#include "hwy/highway.h"
#include "hwy/targets.h"
#include "scann/oss_wrappers/scann_cpu_info.h"

ABSL_FLAG(bool, ignore_avx512, false,
          "Ignore the presence of AVX512 instructions when assigning "
          "function pointers at ScaNN startup.  Useful for testing and "
          "debugging.");

ABSL_FLAG(bool, ignore_avx2, false,
          "Ignore the presence of AVX2/FMA instructions when assigning "
          "function pointers at ScaNN startup.  Useful for testing and "
          "debugging.");

ABSL_FLAG(bool, ignore_avx512_vnni, false,
          "Ignore AVX512_VNNI.  NOTE:  AVX512_VNNI support is currently "
          "experimental and therefore disabled by default.");

ABSL_FLAG(bool, ignore_amx, false,
          "Ignore AMX.  NOTE:  AMX support is currently experimental and "
          "therefore disabled by default.");

ABSL_RETIRED_FLAG(bool, ignore_avx, false, "Ignore AVX1.");

ABSL_RETIRED_FLAG(bool, ignore_sse4, false, "Ignore SSE4");

namespace research_scann {
namespace flags_internal {

bool should_use_avx2 = port::TestCPUFeature(port::AVX2);
bool should_use_avx512 = port::TestCPUFeature(port::AVX512F) &&
                         port::TestCPUFeature(port::AVX512DQ) &&
                         port::TestCPUFeature(port::AVX512BW);
bool should_use_avx512_vnni =
    should_use_avx512 && port::TestCPUFeature(port::AVX512_VNNI);
bool should_use_amx = should_use_avx512_vnni &&
                      port::TestCPUFeature(port::AVX512_BF16) &&
                      port::TestCPUFeature(port::AVX512IFMA) &&
                      port::TestCPUFeature(port::AVX512VBMI) &&
                      port::TestCPUFeature(port::AMX_INT8) &&
                      port::TestCPUFeature(port::AMX_BF16) &&
                      port::TestCPUFeature(port::AMX_TILE);

bool TryEnableAmx() {
#ifdef __x86_64__

  constexpr int ARCH_REQ_XCOMP_PERM = 0x1023;
  constexpr int XFEATURE_XTILEDATA = 18;
  int rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  bool success = (rc == 0);
  if (!success) {
    LOG(ERROR) << "Failed to enable AMX: " << rc;
  }
  return success;
#else
  return false;
#endif
}

}  // namespace flags_internal

ScopedPlatformOverride::ScopedPlatformOverride(PlatformGeneration generation) {
  original_avx2_ = flags_internal::should_use_avx2;
  original_avx512_ = flags_internal::should_use_avx512;
  original_avx512_vnni_ = flags_internal::should_use_avx512_vnni;
  original_amx_ = flags_internal::should_use_amx;
  flags_internal::should_use_avx2 = false;
  flags_internal::should_use_avx512 = false;
  flags_internal::should_use_avx512_vnni = false;
  flags_internal::should_use_amx = false;

  hwy::DisableTargets(0);
  switch (generation) {
    case kSapphireRapidsAmx:
      flags_internal::should_use_amx = true;
      ABSL_FALLTHROUGH_INTENDED;

    case kCascadelakeAvx512Vnni:
      flags_internal::should_use_avx512_vnni = true;
      ABSL_FALLTHROUGH_INTENDED;

    case kSkylakeAvx512:
      flags_internal::should_use_avx512 = true;
      ABSL_FALLTHROUGH_INTENDED;

    case kHaswellAvx2:
      flags_internal::should_use_avx2 = true;

      hwy::DisableTargets(HWY_AVX3 | (HWY_AVX3 - 1));
      ABSL_FALLTHROUGH_INTENDED;

    case kSandyBridgeAvx1:
    case kBaselineSse4:
    case kFallbackForNonX86:
      hwy::DisableTargets(HWY_AVX2 | (HWY_AVX2 - 1));
      break;

    default:
      LOG(FATAL) << "Unexpected Case: " << generation;
  }
}

ScopedPlatformOverride::~ScopedPlatformOverride() {
  flags_internal::should_use_avx2 = original_avx2_;
  flags_internal::should_use_avx512 = original_avx512_;
  flags_internal::should_use_avx512_vnni = original_avx512_vnni_;
  flags_internal::should_use_amx = original_amx_;
}

bool ScopedPlatformOverride::IsSupported() {
  if (flags_internal::should_use_avx512 &&
      !(port::TestCPUFeature(port::AVX512F) &&
        port::TestCPUFeature(port::AVX512DQ))) {
    LOG(WARNING) << "The CPU lacks AVX512 support! (skipping some tests)";
    return false;
  }
  if (flags_internal::should_use_avx2 && !port::TestCPUFeature(port::AVX2)) {
    LOG(WARNING) << "The CPU lacks AVX2 support! (skipping some tests)";
    return false;
  }
  if (flags_internal::should_use_avx512_vnni &&
      !port::TestCPUFeature(port::AVX512_VNNI)) {
    LOG(WARNING) << "The CPU lacks AVX512_VNNI support! (skipping some tests)";
    return false;
  }
  if (flags_internal::should_use_amx &&
      !(port::TestCPUFeature(port::AMX_INT8) &&
        port::TestCPUFeature(port::AMX_BF16) &&
        port::TestCPUFeature(port::AMX_TILE))) {
    LOG(WARNING) << "The CPU lacks AMX support! (skipping some tests)";
    return false;
  }
  return true;
}

ScopedPlatformOverride TestHookOverridePlatform(PlatformGeneration generation) {
  return ScopedPlatformOverride(generation);
}

}  // namespace research_scann
