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

#include "sparse/ops/cc/fused_softmax_launcher.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

using ::tensorflow::Tensor;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::errors::InvalidArgument;

template <typename Device, typename T>
class FusedSoftmaxOp : public tensorflow::OpKernel {
 public:
  explicit FusedSoftmaxOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Collect the input tensor.
    const Tensor& input = context->input(0);

    // Validate the input shapes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input.shape()),
                InvalidArgument("Expected 2-dimension input."));

    // Allocate the output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    // Launch the kernel.
    LaunchFusedSoftmax(context->eigen_device<Device>(), input.dim_size(0),
                       input.dim_size(1), input.flat<float>().data(),
                       output->flat<float>().data());
  }
};

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("FusedSoftmax").Device(tensorflow::DEVICE_GPU),
                        FusedSoftmaxOp<Eigen::GpuDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace sgk
