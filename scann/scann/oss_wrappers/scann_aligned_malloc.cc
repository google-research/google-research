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

#include "scann/oss_wrappers/scann_aligned_malloc.h"

namespace tensorflow {
namespace scann_ops {

#if (defined(__STD_C_VERSION__) && (__STD_C_VERSION__ >= 201112L)) || \
    (__cplusplus >= 201703L) || defined(_ISOC11_SOURCE)
void *aligned_malloc(size_t size, size_t minimum_alignment) {
  size = (size + minimum_alignment - 1) / minimum_alignment * minimum_alignment;
  return aligned_alloc(minimum_alignment, size);
}

void aligned_free(void *aligned_memory) { free(aligned_memory); }

#elif defined(_MSC_VER)
#include <malloc.h>

void *aligned_malloc(size_t size, size_t minimum_alignment) {
  return _aligned_alloc(size, minimum_alignment);
}

void aligned_free(void *aligned_memory) { _aligned_free(aligned_memory); }

#else
#endif

}  // namespace scann_ops
}  // namespace tensorflow
