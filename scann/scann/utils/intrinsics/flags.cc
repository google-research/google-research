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

#include "hwy/highway.h"
#include "hwy/targets.h"
#include "scann/oss_wrappers/scann_cpu_info.h"

ABSL_FLAG(bool, ignore_avx512, false,
          "Ignore the presence of AVX512 instructions when assigning "
          "function pointers at ScaNN startup.  Useful for testing and "
          "debugging.  NOTE:  AVX512 support is currently experimental and "
          "therefore disabled by default.");

ABSL_FLAG(bool, ignore_avx2, false,
          "Ignore the presence of AVX2/FMA instructions when assigning "
          "function pointers at ScaNN startup.  Useful for testing and "
          "debugging.");

ABSL_RETIRED_FLAG(bool, ignore_avx, false, "Ignore AVX1.");

ABSL_RETIRED_FLAG(bool, ignore_sse4, false, "Ignore SSE4");

namespace research_scann {
namespace flags_internal {

bool should_use_avx2 = port::TestCPUFeature(port::AVX2);
bool should_use_avx512 = port::TestCPUFeature(port::AVX512F) &&
                         port::TestCPUFeature(port::AVX512DQ) &&
                         port::TestCPUFeature(port::AVX512BW);

}  // namespace flags_internal

ScopedPlatformOverride::ScopedPlatformOverride(PlatformGeneration generation) {
  original_avx2_ = flags_internal::should_use_avx2;
  original_avx512_ = flags_internal::should_use_avx512;
  flags_internal::should_use_avx2 = false;
  flags_internal::should_use_avx512 = false;

  hwy::DisableTargets(0);
  switch (generation) {
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
  return true;
}

ScopedPlatformOverride TestHookOverridePlatform(PlatformGeneration generation) {
  return ScopedPlatformOverride(generation);
}

}  // namespace research_scann
