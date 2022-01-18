// Copyright 2022 The Google Research Authors.
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

#if GOOGLE_CUDA
#include <cusparse.h>

#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "sparse/ops/cc/common.h"

namespace sgk {

namespace {

using HandleMap = absl::flat_hash_map<cudaStream_t, cusparseHandle_t>;

// Get singleton map of cuda streams to cusparse handles.
HandleMap* GetHandleMap() {
  static HandleMap handle_map;
  return &handle_map;
}

}  // namespace

cusparseHandle_t GetHandleForStream(cudaStream_t stream) {
  HandleMap* handle_map = GetHandleMap();
  auto it = handle_map->find(stream);
  if (it == handle_map->end()) {
    // Allocate new cusparse handle and set the stream.
    cusparseHandle_t new_handle;
    CUSPARSE_CALL(cusparseCreate(&new_handle));
    CUSPARSE_CALL(cusparseSetStream(new_handle, stream));
    it = handle_map->insert(std::make_pair(stream, new_handle)).first;
  }
  return it->second;
}

}  // namespace sgk
#endif  // GOOGLE_CUDA
