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

#include "sparse/ops/cc/common.h"
#include "sparse/ops/cc/transpose_launcher.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::errors::InvalidArgument;

template <typename Device, typename T>
class CsrTransposeOp : public tensorflow::OpKernel {
 public:
  explicit CsrTransposeOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Collect the input & output tensors.
    const Tensor& m_tensor = context->input(0);
    const Tensor& n_tensor = context->input(1);
    const Tensor& values = context->input(2);
    const Tensor& row_offsets = context->input(3);
    const Tensor& column_indices = context->input(4);

    // Validate the input shapes.
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(m_tensor.shape()),
                InvalidArgument("Expected scalar for argument 'm'."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(n_tensor.shape()),
                InvalidArgument("Expected scalar for argument 'n'."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(values.shape()),
                InvalidArgument("Expected 1-dimension values tensor."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(row_offsets.shape()),
                InvalidArgument("Expected 1-dimension row_offsets tensor."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(column_indices.shape()),
                InvalidArgument("Expected 1-dimension column_indices tensor."));
    OP_REQUIRES(context, values.dim_size(0) == column_indices.dim_size(0),
                InvalidArgument("Expected same number of values and indices"));

    // Get the problem shape.
    int m = m_tensor.tensor<int32, 0>().data()[0];
    int n = n_tensor.tensor<int32, 0>().data()[0];
    int nonzeros = values.dim_size(0);

    // Validate row offsets size.
    OP_REQUIRES(context, row_offsets.dim_size(0) == m + 1,
                InvalidArgument("Expected m+1 row offsets."));

    // Allocate the output tensor.
    Tensor* output_values = nullptr;
    Tensor* output_row_offsets = nullptr;
    Tensor* output_column_indices = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{nonzeros},
                                                     &output_values));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{n + 1},
                                                     &output_row_offsets));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{nonzeros},
                                                     &output_column_indices));

    // (Possibly) get a temporary buffer to work in.
    Tensor workspace;
    AllocateTransposeWorkspace(
        context, context->eigen_device<Device>(), m, n, nonzeros,
        values.tensor<float, 1>().data(), AsInt32<1>(row_offsets),
        AsInt32<1>(column_indices), output_values->tensor<float, 1>().data(),
        AsInt32<1>(output_row_offsets), AsInt32<1>(output_column_indices),
        &workspace);

    // Launch the kernel.
    LaunchTranspose(
        context->eigen_device<Device>(), m, n, nonzeros,
        values.tensor<float, 1>().data(), AsInt32<1>(row_offsets),
        AsInt32<1>(column_indices), output_values->tensor<float, 1>().data(),
        AsInt32<1>(output_row_offsets), AsInt32<1>(output_column_indices),
        workspace.tensor<float, 1>().data());
  }
};

REGISTER_KERNEL_BUILDER(Name("CsrTranspose").Device(tensorflow::DEVICE_CPU),
                        CsrTransposeOp<Eigen::ThreadPoolDevice, float>);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("CsrTranspose")
                            .Device(tensorflow::DEVICE_GPU)
                            .HostMemory("m")
                            .HostMemory("n"),
                        CsrTransposeOp<Eigen::GpuDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace sgk
