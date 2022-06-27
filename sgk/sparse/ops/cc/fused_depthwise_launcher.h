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

#ifndef SGK_SPARSE_OPS_CC_FUSED_DEPTHWISE_LAUNCHER_H_
#define SGK_SPARSE_OPS_CC_FUSED_DEPTHWISE_LAUNCHER_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

#if GOOGLE_CUDA
/**
 * @brief Helper to launch fused depthwise conv kernel on the device.
 */
void LaunchFusedDepthwiseConv(const Eigen::GpuDevice& d, int n, int c, int h,
                              int w, const float* input, int kernel_size,
                              int padding, int stride, const float* filter,
                              const float* bias, float* output);
#endif  // GOOGLE_CUDA

}  // namespace sgk

#endif  // SGK_SPARSE_OPS_CC_FUSED_DEPTHWISE_LAUNCHER_H_
