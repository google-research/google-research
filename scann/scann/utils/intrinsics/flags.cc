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

#include "scann/utils/intrinsics/flags.h"
#include "tensorflow/core/platform/cpu_info.h"

ABSL_FLAG(bool, ignore_avx512, false,
          "Ignore the presence of AVX512 instructions when assigning "
          "function pointers at ScaNN startup.  Useful for testing and "
          "debugging.  NOTE:  AVX512 support is currently experimental and "
          "therefore disabled by default.");

ABSL_FLAG(bool, ignore_avx2, false,
          "Ignore the presence of AVX2/FMA instructions when assigning "
          "function pointers at ScaNN startup.  Useful for testing and "
          "debugging.");

ABSL_FLAG(bool, ignore_avx, false,
          "Ignore the presence of AVX and higher instructions when assigning "
          "function pointers at ScaNN startup.  Useful for testing and "
          "debugging.");

ABSL_RETIRED_FLAG(bool, ignore_sse4, false, "Ignore SSE4");

namespace tensorflow {
namespace scann_ops {
namespace flags_internal {

bool should_use_sse4 = port::TestCPUFeature(port::SSE4_2);
bool should_use_avx1 = port::TestCPUFeature(port::AVX);
bool should_use_avx2 = port::TestCPUFeature(port::AVX2);
bool should_use_avx512 =
    port::TestCPUFeature(port::AVX512F) && port::TestCPUFeature(port::AVX512DQ);

}  // namespace flags_internal

ScopedPlatformOverride::ScopedPlatformOverride(PlatformGeneration generation) {
  original_avx1_ = flags_internal::should_use_avx1;
  original_avx2_ = flags_internal::should_use_avx2;
  original_avx512_ = flags_internal::should_use_avx512;
  original_sse4_ = flags_internal::should_use_sse4;
  flags_internal::should_use_sse4 = false;
  flags_internal::should_use_avx1 = false;
  flags_internal::should_use_avx2 = false;
  flags_internal::should_use_avx512 = false;
  switch (generation) {
    case kSkylakeAvx512:
      flags_internal::should_use_avx512 = true;
      ABSL_FALLTHROUGH_INTENDED;

    case kHaswellAvx2:
      flags_internal::should_use_avx2 = true;
      ABSL_FALLTHROUGH_INTENDED;

    case kSandyBridgeAvx1:
      flags_internal::should_use_avx1 = true;
      ABSL_FALLTHROUGH_INTENDED;

    case kBaselineSse4:
      flags_internal::should_use_sse4 = true;
      break;

    case kFallbackForNonX86:
      break;

    default:
      LOG(FATAL) << "Unexpected Case: " << generation;
  }
}

ScopedPlatformOverride::~ScopedPlatformOverride() {
  flags_internal::should_use_avx1 = original_avx1_;
  flags_internal::should_use_avx2 = original_avx2_;
  flags_internal::should_use_avx512 = original_avx512_;
  flags_internal::should_use_sse4 = original_sse4_;
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
  if (flags_internal::should_use_avx1 && !port::TestCPUFeature(port::AVX)) {
    LOG(WARNING) << "The CPU lacks AVX1 support! (skipping some tests)";
    return false;
  }
  if (flags_internal::should_use_sse4 && !port::TestCPUFeature(port::SSE4_2)) {
    LOG(WARNING) << "This CPU lacks SSE4.2 support! (skipping some tests)";
  }
  return true;
}

ScopedPlatformOverride TestHookOverridePlatform(PlatformGeneration generation) {
  return ScopedPlatformOverride(generation);
}

}  // namespace scann_ops
}  // namespace tensorflow
