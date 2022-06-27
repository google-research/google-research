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
#include "sparse/ops/cc/sparse_softmax_launcher.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

using ::tensorflow::Tensor;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::errors::InvalidArgument;

template <typename Device, typename T>
class SparseSoftmaxOp : public tensorflow::OpKernel {
 public:
  explicit SparseSoftmaxOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // Collect the input & output tensors.
    const Tensor& values = context->input(0);
    const Tensor& row_indices = context->input(1);
    const Tensor& row_offsets = context->input(2);
    const Tensor& column_indices = context->input(3);

    // Validate the input shapes.
    OP_REQUIRES(context, TensorShapeUtils::IsVector(row_indices.shape()),
                InvalidArgument("Expected 1-dimension row_indices tensor."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(row_offsets.shape()),
                InvalidArgument("Expected 1-dimension row_offsets tensor."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(column_indices.shape()),
                InvalidArgument("Expected 1-dimension column_indices tensor."));
    OP_REQUIRES(context, row_indices.dim_size(0) + 1 == row_offsets.dim_size(0),
                InvalidArgument("Expected 1 more row index than offset."));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsVector(values.shape()) || values.dims() == 2,
        InvalidArgument("Expected 1-dim or 2-dim values tensor."));

    // Get the problem shape.
    //
    // NOTE: The kernel doesn't actually need the n argument. Pass garbage,
    // since we can't pull it off the sparse matrix representation.
    int m = row_indices.dim_size(0);
    int n = -1;
    int nonzeros = column_indices.dim_size(0);
    int dim_offset = values.dims() - 1;
    int replication = dim_offset == 1 ? values.dim_size(0) : 1;

    // Validate the sparse matrix shape.
    OP_REQUIRES(context, values.dim_size(dim_offset) == nonzeros,
                InvalidArgument("Num values must equal num col indices."));

    // Allocate the output tensor.
    Tensor* output_values = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, values.shape(), &output_values));

    // Launch the kernel for each step of computation.
    //
    // TODO(tgale): This could be accelerated by supported replicated/batched
    // execution in the kernel. Running the kernel is a loop like this could
    // incur significant overhead from kernel launch latency if the computation
    // is cheap.
    for (int idx = 0; idx < replication; ++idx) {
      LaunchSparseSoftmax(context->eigen_device<Device>(), m, n, nonzeros,
                          values.flat<float>().data() + nonzeros * idx,
                          AsInt32<1>(row_indices), AsInt32<1>(row_offsets),
                          AsInt32<1>(column_indices),
                          output_values->flat<float>().data() + nonzeros * idx);
    }
  }
};

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("CsrSoftmax").Device(tensorflow::DEVICE_GPU),
                        SparseSoftmaxOp<Eigen::GpuDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace sgk
