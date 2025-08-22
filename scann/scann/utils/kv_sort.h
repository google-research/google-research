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

#ifndef SCANN_UTILS_KV_SORT_H_
#define SCANN_UTILS_KV_SORT_H_

#include <bit>
#include <cstdint>

#include "hwy/contrib/sort/vqsort-inl.h"
#include "scann/utils/common.h"

namespace research_scann {

static constexpr uint32_t FloatToUint32Sortkey(float val) {
  uint32_t temp = absl::bit_cast<uint32_t>(val);
  return temp & 0x80000000 ? ~temp : temp ^ 0x80000000;
}

static constexpr float Uint32SortkeyToFloat(uint32_t val) {
  return absl::bit_cast<float>(val & 0x80000000 ? val ^ 0x80000000 : ~val);
}

enum class KVSortOrder { kAscending, kDescending };

inline void KVSort(MutableSpan<float> keys, MutableSpan<uint32_t> values,
                   KVSortOrder sort_order) {
  std::vector<hwy::K32V32> kvs;
  kvs.reserve(keys.size());
  for (int i = 0; i < keys.size(); ++i) {
    kvs.push_back({.value = values[i], .key = FloatToUint32Sortkey(keys[i])});
  }
  if (sort_order == KVSortOrder::kAscending) {
    hwy::HWY_NAMESPACE::VQSortStatic(kvs.data(), kvs.size(),
                                     hwy::SortAscending());
  } else {
    hwy::HWY_NAMESPACE::VQSortStatic(kvs.data(), kvs.size(),
                                     hwy::SortDescending());
  }
  for (int i = 0; i < keys.size(); ++i) {
    values[i] = kvs[i].value;
    keys[i] = Uint32SortkeyToFloat(kvs[i].key);
  }
}

}  // namespace research_scann

#endif
