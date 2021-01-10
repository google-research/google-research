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

#ifndef SCANN_OSS_WRAPPERS_SCANN_MALLOC_EXTENSION_H_
#define SCANN_OSS_WRAPPERS_SCANN_MALLOC_EXTENSION_H_

#include <stddef.h>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"

namespace research_scann {

class MallocExtension {
 public:
  static bool GetNumericProperty(absl::string_view property, size_t* value) {
    return false;
  }
  static absl::optional<size_t> GetNumericProperty(absl::string_view property) {
    return absl::nullopt;
  }
  static void ReleaseMemoryToSystem(size_t num_bytes) {}
  static size_t GetAllocatedSize(const void* p) { return 0; }
};

}  // namespace research_scann

#endif
