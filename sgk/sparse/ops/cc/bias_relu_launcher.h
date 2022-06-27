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

#ifndef SGK_SPARSE_OPS_CC_BIAS_RELU_LAUNCHER_H_
#define SGK_SPARSE_OPS_CC_BIAS_RELU_LAUNCHER_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

#if GOOGLE_CUDA
void LaunchBiasRelu(const Eigen::GpuDevice &device, int n, int c, int d,
                    const float *in, const float *bias, float *out);
#endif  // GOOGLE_CUDA

}  // namespace sgk
#endif  // SGK_SPARSE_OPS_CC_BIAS_RELU_LAUNCHER_H_
