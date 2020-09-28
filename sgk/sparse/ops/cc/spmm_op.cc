// Copyright 2020 The Google Research Authors.
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
#include "sparse/ops/cc/spmm_launcher.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace sgk {

using ::tensorflow::Tensor;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::errors::InvalidArgument;

template <typename Device, typename T>
class SpmmOp : public tensorflow::OpKernel {
 public:
  explicit SpmmOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_lhs", &transpose_lhs_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_rhs", &transpose_rhs_));

    // NOTE: We currently do not support transposition for either argument.
    OP_REQUIRES(context, !transpose_lhs_,
                InvalidArgument("transpose_lhs=True not yet supported."));
    OP_REQUIRES(context, !transpose_rhs_,
                InvalidArgument("transpose_rhs=True not yet supported."));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    ComputeHelper(context, /*bias_ptr=*/nullptr);
  }

  void ComputeHelper(tensorflow::OpKernelContext* context,
                     const float* bias_ptr) {
    // Collect the input & output tensors.
    const Tensor& m_tensor = context->input(0);
    const Tensor& k_tensor = context->input(1);
    const Tensor& values = context->input(2);
    const Tensor& row_indices = context->input(3);
    const Tensor& row_offsets = context->input(4);
    const Tensor& column_indices = context->input(5);
    const Tensor& dense_matrix = context->input(6);

    // Validate the input shapes.
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(m_tensor.shape()),
                InvalidArgument("Expected scalar for argument 'm'."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(k_tensor.shape()),
                InvalidArgument("Expected scalar for argument 'k'."));
    OP_REQUIRES(
        context,
        TensorShapeUtils::IsVector(values.shape()) || values.dims() == 2,
        InvalidArgument("Expected 1-dim or 2-dim values tensor."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(row_indices.shape()),
                InvalidArgument("Expected 1-dimension row_indices tensor."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(row_offsets.shape()),
                InvalidArgument("Expected 1-dimension row_offsets tensor."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(column_indices.shape()),
                InvalidArgument("Expected 1-dimension column_indices tensor."));
    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(dense_matrix.shape()) ||
                    dense_matrix.dims() == 3,
                InvalidArgument("Expected 2 or 3-dim dense matrix tensor."));
    OP_REQUIRES(context, row_indices.dim_size(0) + 1 == row_offsets.dim_size(0),
                InvalidArgument("Expected one more row index than offset."));

    // TODO(tgale): We can lift this constraint to support arbitrary replication
    // of rhs/lhs matrix. For example, if lhs is a 3-tensor and rhs is a matrix
    // we can compute `lhs.shape[0]` spmms with each kernel using the same rhs
    // matrix.
    OP_REQUIRES(context, values.dims() == dense_matrix.dims() - 1,
                InvalidArgument("Values and rhs must be replicated the same."));

    // Get the problem shape.
    int m = m_tensor.tensor<int32, 0>().data()[0];
    int k = k_tensor.tensor<int32, 0>().data()[0];
    int nonzeros = column_indices.dim_size(0);

    int dim_offset = dense_matrix.dims() - 2;
    int n = dense_matrix.dim_size(dim_offset + 1);
    int replication = dim_offset == 1 ? dense_matrix.dim_size(0) : 1;

    // Validate the sparse matrix and dense matrix shapes match.
    OP_REQUIRES(context, values.dim_size(dim_offset) == nonzeros,
                InvalidArgument("Num values must equal num col indices."));
    OP_REQUIRES(context, row_indices.dim_size(0) == m,
                InvalidArgument("Num row indices and 'm' must match."));
    OP_REQUIRES(context, dense_matrix.dim_size(dim_offset) == k,
                InvalidArgument("Inner matrix dimensions must match."));

    // If we're going to run multiple spmms, the first dimension of the
    // matrices must match.
    OP_REQUIRES(context, replication == 1 || replication == values.dim_size(0),
                InvalidArgument("First dim of values and rhs must match"));

    // Allocate the output tensor.
    Tensor* output_matrix = nullptr;
    tensorflow::TensorShape output_shape = {m, n};
    if (replication > 1) {
      output_shape = {replication, m, n};
    }
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_matrix));

    // TODO(tgale): Add type checks on meta-data tensors to make sure our
    // casting is safe.
    //
    // TODO(tgale): This could be accelerated by supported replicated/batched
    // execution in the kernel. Running the kernel is a loop like this could
    // incur significant overhead from kernel launch latency if the computation
    // is cheap.
    for (int idx = 0; idx < replication; ++idx) {
      LaunchSpmm(context->eigen_device<Device>(), m, k, n, nonzeros,
                 values.flat<float>().data() + nonzeros * idx,
                 AsInt32<1>(row_indices), AsInt32<1>(row_offsets),
                 AsInt32<1>(column_indices),
                 dense_matrix.flat<float>().data() + k * n * idx, bias_ptr,
                 output_matrix->flat<float>().data() + m * n * idx);
    }
  }

 private:
  bool transpose_lhs_, transpose_rhs_;
};

template <typename Device, typename T>
class FusedSpmmOp : public SpmmOp<Device, T> {
 public:
  explicit FusedSpmmOp(tensorflow::OpKernelConstruction* context)
      : SpmmOp<Device, T>(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& m_tensor = context->input(0);
    const Tensor& bias = context->input(7);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(m_tensor.shape()),
                InvalidArgument("Expected scalar for argument 'm'."));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                InvalidArgument("Expected vector for argument 'bias'."));
    int m = m_tensor.tensor<int32, 0>().data()[0];
    OP_REQUIRES(context, bias.dim_size(0) == m,
                InvalidArgument("Num biases size and 'm' must match."));
    this->ComputeHelper(context, bias.tensor<float, 1>().data());
  }
};

REGISTER_KERNEL_BUILDER(Name("Spmm").Device(tensorflow::DEVICE_CPU),
                        SpmmOp<Eigen::ThreadPoolDevice, float>);
REGISTER_KERNEL_BUILDER(Name("FusedSpmm").Device(tensorflow::DEVICE_CPU),
                        FusedSpmmOp<Eigen::ThreadPoolDevice, float>);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("Spmm").Device(tensorflow::DEVICE_GPU).HostMemory("m").HostMemory("k"),
    SpmmOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("FusedSpmm")
                            .Device(tensorflow::DEVICE_GPU)
                            .HostMemory("m")
                            .HostMemory("k"),
                        FusedSpmmOp<Eigen::GpuDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace sgk
