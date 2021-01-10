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

#include "scann/utils/memory_logging.h"

#include "absl/strings/str_cat.h"
#include "scann/oss_wrappers/scann_malloc_extension.h"

namespace research_scann {

size_t GfvStorage(const GenericFeatureVector& gfv) {
  size_t result = sizeof(gfv);
  result += gfv.feature_index().Capacity() * sizeof(gfv.feature_index(0));
  result +=
      gfv.feature_value_int64().Capacity() * sizeof(gfv.feature_value_int64(0));
  result +=
      gfv.feature_value_float().Capacity() * sizeof(gfv.feature_value_float(0));
  result += gfv.feature_value_double().Capacity() *
            sizeof(gfv.feature_value_double(0));
  if (gfv.has_data_id_str()) {
    result += gfv.data_id_str().size();
  }

  return result;
}

size_t GfvStorage(const vector<GenericFeatureVector>& gfvs) {
  size_t result = sizeof(gfvs);
  for (const auto& gfv : gfvs) {
    result += GfvStorage(gfv);
  }

  return result;
}

std::string GetTcMallocLogString() {
  size_t allocated_bytes =
      *MallocExtension::GetNumericProperty("generic.current_allocated_bytes");
  size_t free_bytes =
      *MallocExtension::GetNumericProperty("tcmalloc.pageheap_free_bytes");
  size_t unmapped_bytes =
      *MallocExtension::GetNumericProperty("tcmalloc.pageheap_unmapped_bytes");
  size_t heap_size = *MallocExtension::GetNumericProperty("generic.heap_size");

  return absl::StrCat("From TCMalloc:  ", allocated_bytes / bytes_in_mb,
                      "MB allocated, ", free_bytes / bytes_in_mb,
                      "MB free on page heap, ", unmapped_bytes / bytes_in_mb,
                      "MB unmapped.  Heap size = ", heap_size / bytes_in_mb,
                      "MB.");
}

}  // namespace research_scann
