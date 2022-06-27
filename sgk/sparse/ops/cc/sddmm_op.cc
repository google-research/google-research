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
#include "sparse/ops/cc/sddmm_launcher.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

using ::tensorflow::Tensor;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::errors::InvalidArgument;

template <typename Device, typename T>
class SddmmOp : public tensorflow::OpKernel {
 public:
  explicit SddmmOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_lhs", &transpose_lhs_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_rhs", &transpose_rhs_));

    // NOTE: We currently do not support transposition for either argument.
    OP_REQUIRES(context, !transpose_lhs_,
                InvalidArgument("transpose_lhs=True not yet supported."));
    OP_REQUIRES(context, transpose_rhs_,
                InvalidArgument("transpose_rhs=False not yet supported."));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    // Collect the input & output tensors.
    const Tensor& m_tensor = context->input(0);
    const Tensor& n_tensor = context->input(1);
    const Tensor& row_indices = context->input(2);
    const Tensor& row_offsets = context->input(3);
    const Tensor& column_indices = context->input(4);
    const Tensor& lhs_matrix = context->input(5);
    const Tensor& rhs_matrix = context->input(6);

    // Validate the input shapes.
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(m_tensor.shape()),
                InvalidArgument("Expected scalar for argument 'm'."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(n_tensor.shape()),
                InvalidArgument("Expected scalar for argument 'n'."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(row_indices.shape()),
                InvalidArgument("Expected 1-dimension row_indices tensor."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(row_offsets.shape()),
                InvalidArgument("Expected 1-dimension row_offsets tensor."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(column_indices.shape()),
                InvalidArgument("Expected 1-dimension column_indices tensor."));
    OP_REQUIRES(context, row_indices.dim_size(0) + 1 == row_offsets.dim_size(0),
                InvalidArgument("Expected 1 more row index than offset."));
    OP_REQUIRES(context, lhs_matrix.dim_size(1) == rhs_matrix.dim_size(1),
                InvalidArgument("Last dim of input matrices must match."));
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(lhs_matrix.shape()) ||
                    lhs_matrix.dims() == 3,
                InvalidArgument("Expected 2-dim or 3-dim lhs matrix tensor."));
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(rhs_matrix.shape()) ||
                    rhs_matrix.dims() == 3,
                InvalidArgument("Expected 2-dim or 3-dim rhs matrix tensor."));

    // TODO(tgale): We can lift this constraint to support arbitrary replication
    // of rhs/lhs matrix. For example, if lhs is a 3-tensor and rhs is a matrix
    // we can compute `lhs.shape[0]` sddmms with each kernel using the same rhs
    // matrix.
    OP_REQUIRES(context, rhs_matrix.dims() == lhs_matrix.dims(),
                InvalidArgument("rhs and lhs must match number of dims."));

    // Get the problem shape.
    int m = m_tensor.tensor<int32, 0>().data()[0];
    int n = n_tensor.tensor<int32, 0>().data()[0];
    int nonzeros = column_indices.dim_size(0);

    int dim_offset = lhs_matrix.dims() - 2;
    int k = lhs_matrix.dim_size(dim_offset + 1);
    int replication = dim_offset == 1 ? lhs_matrix.dim_size(0) : 1;

    // Validate the sparse matrix shape.
    OP_REQUIRES(context, row_indices.dim_size(0) == m,
                InvalidArgument("Num row indices and 'm' must match."));
    OP_REQUIRES(context, lhs_matrix.dim_size(dim_offset) == m,
                InvalidArgument("First dim of lhs must match output rows."));
    OP_REQUIRES(context, rhs_matrix.dim_size(dim_offset) == n,
                InvalidArgument("First dim of lhs must match output cols."));

    // If we're going to run multiple sddmms, the first dimension of the
    // matrices must match.
    OP_REQUIRES(context,
                replication == 1 || replication == rhs_matrix.dim_size(0),
                InvalidArgument("First dim of lhs & rhs must match"));

    // Allocate the output tensor.
    Tensor* output_values = nullptr;
    tensorflow::TensorShape output_shape = {nonzeros};
    if (replication > 1) {
      output_shape = {replication, nonzeros};
    }
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_values));

    // Launch the kernel.
    //
    // TODO(tgale): This could be accelerated by supported replicated/batched
    // execution in the kernel. Running the kernel is a loop like this could
    // incur significant overhead from kernel launch latency if the computation
    // is cheap.
    for (int idx = 0; idx < replication; ++idx) {
      LaunchSddmm(context->eigen_device<Device>(), m, k, n, nonzeros,
                  AsInt32<1>(row_indices), AsInt32<1>(row_offsets),
                  AsInt32<1>(column_indices),
                  lhs_matrix.flat<float>().data() + m * k * idx,
                  rhs_matrix.flat<float>().data() + k * n * idx,
                  output_values->flat<float>().data() + nonzeros * idx);
    }
  }

 private:
  bool transpose_lhs_, transpose_rhs_;
};

REGISTER_KERNEL_BUILDER(Name("Sddmm").Device(tensorflow::DEVICE_CPU),
                        SddmmOp<Eigen::ThreadPoolDevice, float>);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Sddmm")
                            .Device(tensorflow::DEVICE_GPU)
                            .HostMemory("m")
                            .HostMemory("n"),
                        SddmmOp<Eigen::GpuDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace sgk
