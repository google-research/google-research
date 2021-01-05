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

#ifndef SGK_SPARSE_OPS_CC_COMMON_H_
#define SGK_SPARSE_OPS_CC_COMMON_H_

#include <cusparse.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace sgk {

// Fixed-width integer types from TensorFlow.
using ::tensorflow::int32;
using ::tensorflow::int64;
using ::tensorflow::uint32;

// HACK: TensorFlow has no support for int32 data on GPU (b/25387198).
// To get around this, we use uint32 tensors to store our int32 data
// even though we pass them to the CUDA kernels as int32. This helper
// makes the accessing of uint32 data as int32 data cleaner.
template <size_t kDims>
int32* AsInt32(tensorflow::Tensor* t) {
  return reinterpret_cast<int32*>(t->tensor<uint32, kDims>().data());
}

template <size_t kDims>
const int32* AsInt32(const tensorflow::Tensor& t) {
  return reinterpret_cast<const int32*>(t.tensor<uint32, kDims>().data());
}

#define CUDA_CALL(code)                                     \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    CHECK_EQ(status, cudaSuccess) << "CUDA Error: " << err; \
  } while (0)

#define CUSPARSE_CALL(code)                                        \
  do {                                                             \
    cusparseStatus_t status = code;                                \
    CHECK_EQ(status, CUSPARSE_STATUS_SUCCESS) << "CuSparse Error"; \
  } while (0)

}  // namespace sgk

#endif  // SGK_SPARSE_OPS_CC_COMMON_H_
