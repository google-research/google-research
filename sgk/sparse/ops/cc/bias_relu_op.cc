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

#include "sparse/ops/cc/bias_relu_launcher.h"
#include "sparse/ops/cc/common.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

using ::tensorflow::Tensor;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::errors::InvalidArgument;

template <typename Device, typename T>
class BiasReluOp : public tensorflow::OpKernel {
 public:
  explicit BiasReluOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& in = context->input(0);
    const Tensor& bias = context->input(1);

    // Validate the input shapes.
    OP_REQUIRES(context, in.dims() >= 2 && in.dims() <= 4,
                InvalidArgument("Expected 2-4 dimensional input"));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                InvalidArgument("Expected 1-dimension bias"));
    OP_REQUIRES(context, bias.dim_size(0) == in.dim_size(1),
                InvalidArgument("Expected one bias value for each channel."));

    // Get the problem shape.
    int n = in.dim_size(0);
    int c = in.dim_size(1);
    int d = 1;
    if (in.dims() == 3) {
      d = in.dim_size(2);
    } else if (in.dims() == 4) {
      d = in.dim_size(2) * in.dim_size(3);
    }

    // Allocate the output tensor.
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, in.shape(), &out));

    // Launch the kernel.
    LaunchBiasRelu(context->eigen_device<Device>(), n, c, d,
                   in.flat<float>().data(), bias.flat<float>().data(),
                   out->flat<float>().data());
  }
};

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("BiasRelu").Device(tensorflow::DEVICE_GPU),
                        BiasReluOp<Eigen::GpuDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace sgk
