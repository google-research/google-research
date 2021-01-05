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

#include "sparse/ops/cc/common.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/padding.h"

namespace sgk {

tensorflow::Status SpmmShapeFn(
    tensorflow::shape_inference::InferenceContext* c) {
  using tensorflow::shape_inference::ShapeHandle;
  ShapeHandle lhs_rows;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &lhs_rows));

  ShapeHandle rhs_shape, output_shape;
  if (c->Rank(c->input(6)) == 3) {
    rhs_shape = c->input(6);
    output_shape = c->MakeShape(
        {c->Dim(rhs_shape, 0), c->Dim(lhs_rows, 0), c->Dim(rhs_shape, 2)});
  } else {
    TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 2, &rhs_shape));
    output_shape = c->MakeShape({c->Dim(lhs_rows, 0), c->Dim(rhs_shape, 1)});
  }

  c->set_output(0, output_shape);
  return tensorflow::Status::OK();
}

REGISTER_OP("Spmm")
    .Attr("transpose_lhs: bool = false")
    .Attr("transpose_rhs: bool = false")
    .Input("m: int32")
    .Input("k: int32")
    .Input("values: float")
    .Input("row_indices: uint32")
    .Input("row_offsets: uint32")
    .Input("column_indices: uint32")
    .Input("dense_matrix: float")
    .Output("output_matrix: float")
    .SetShapeFn(SpmmShapeFn)
    .Doc(R"doc(
Compute the product of a sparse matrix and a dense matrix to produce a
dense output matrix. The sparse matrix is stored in compressed sparse
row format.

m: [1], the number of rows in the input sparse matrix.
k: [1], the number of columns in the input sparse matrix.
values: [nonzeros], the nonzero values of the sparse matrix.
row_indices: [m], row indices from 0-{m-1} optionally reordered for load
    balancing.
row_offsets: [m+1], offsets for the rows of the sparse matrix.
column_indices: [nonzeros], column indices for each nonzero in the sparse
    matrix.
dense_matrix: [k, n], dense matrix to multiply the sparse matrix by.
output_matrix: [m, n], output dense matrix to store the result.
)doc");

REGISTER_OP("FusedSpmm")
    .Attr("transpose_lhs: bool = false")
    .Attr("transpose_rhs: bool = false")
    .Input("m: int32")
    .Input("k: int32")
    .Input("values: float")
    .Input("row_indices: uint32")
    .Input("row_offsets: uint32")
    .Input("column_indices: uint32")
    .Input("dense_matrix: float")
    .Input("bias: float")
    .Output("output_matrix: float")
    .SetShapeFn(SpmmShapeFn);

REGISTER_OP("Sddmm")
    .Attr("transpose_lhs: bool = false")
    .Attr("transpose_rhs: bool = false")
    .Input("m: int32")
    .Input("n: int32")
    .Input("row_indices: uint32")
    .Input("row_offsets: uint32")
    .Input("column_indices: uint32")
    .Input("lhs_matrix: float")
    .Input("rhs_matrix: float")
    .Output("output_values: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      using tensorflow::shape_inference::ShapeHandle;
      ShapeHandle nonzeros;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &nonzeros));

      ShapeHandle output_shape = nonzeros;
      if (c->Rank(c->input(5)) == 3) {
        ShapeHandle lhs_shape = c->input(5);
        output_shape =
            c->MakeShape({c->Dim(lhs_shape, 0), c->Dim(nonzeros, 0)});
      }
      c->set_output(0, output_shape);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Computes the product of two dense matrices where a subset of the outputs
are requested. Which outputs are to be computed are specified by a sparse
matrix stored in compressed sparse row format.

Currently only supports having the right-hand matrix transposed.

m: [1], the number of rows in the input sparse matrix.
n: [1], the number of columns in the input sparse matrix.
row_indices: [m], row indices from 0-{m-1} optionally reordered for load
    balancing.
row_offsets: [m+1], offsets for the rows of the sparse matrix.
column_indices: [nonzeros], column indices for each nonzero in the sparse
    matrix.
lhs_matrix: [m, k], left-hand, dense matrix operand to the matrix product.
rhs_matrix: [n, k], right-hand, dense matrix operand to the matrix product.
output_values: [nonzeros], the nonzero values of the sparse matrix.
)doc");

// NOTE: We can't tell how many columns are in a compressed sparse row matrix
// from the data structures alone. The necessary information is in the host
// tensors `m` and `n`, but we can't access this during shape inference.
REGISTER_OP("CsrTranspose")
    .Input("m: int32")
    .Input("n: int32")
    .Input("values: float")
    .Input("row_offsets: uint32")
    .Input("column_indices: uint32")
    .Output("output_values : float")
    .Output("output_row_offsets : uint32")
    .Output("output_column_indices : uint32")
    .Doc(R"doc(
Transposes a compressed sparse row matrix.

m: [1], the number of rows in the input sparse matrix.
n: [1], the number of columns in the input sparse matrix.
values: [nonzeros], the nonzero values of the input sparse matrix.
row_offsets: [m+1], offsets for the rows of the input sparse matrix.
column_indices: [nonzeros], column indices for each nonzero in the
    input sparse matrix.
output_values: [nonzeros], the nonzero values of the output sparse matrix.
output_row_offsets: [m+1], offsets for the rows of the output sparse matrix.
output_column_indices: [nonzeros], column indices for each nonzero in the
    output sparse matrix.
)doc");

REGISTER_OP("Csr2idx")
    .Input("m: int32")
    .Input("n: int32")
    .Input("row_offsets: uint32")
    .Input("column_indices: uint32")
    .Output("linear_indices : uint32")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      using tensorflow::shape_inference::ShapeHandle;
      ShapeHandle nonzeros;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &nonzeros));
      c->set_output(0, nonzeros);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Converts a compressed sparse row matrix to linear format.

Converts `column_index[i]` to `column_index[i] * row_index * n`, where
`row_index` is the row that this column index belongs to. We call this
"index format" or "1-dimensional coordinate format".

m: [1], the number of rows in the input sparse matrix.
n: [1], the number of columns in the input sparse matrix.
row_offsets: [m+1], offsets for the rows of the sparse matrix.
column_indices: [nonzeros], column indices for each nonzero in the
    sparse matrix.
linear_indices: [nonzeros], the linear indices for the sparse matrix.
)doc");

tensorflow::Status DepthwiseShapeFn(
    tensorflow::shape_inference::InferenceContext* c) {
  using tensorflow::shape_inference::DimensionHandle;
  using tensorflow::shape_inference::ShapeHandle;
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

  if (strides.size() != 4) {
    return tensorflow::errors::InvalidArgument(
        "DepthwiseConv2D requires the stride attribute to contain 4 "
        "values, "
        "but got: ",
        strides.size());
  }

  // Only supports NCHW.
  std::string data_format = "NCHW";
  int32 stride_rows = strides[2];
  int32 stride_cols = strides[3];

  DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
  DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
  DimensionHandle in_cols_dim = c->Dim(input_shape, 3);

  DimensionHandle filter_rows_dim = c->Dim(filter_shape, 1);
  DimensionHandle filter_cols_dim = c->Dim(filter_shape, 2);
  DimensionHandle input_depth = c->Dim(filter_shape, 0);
  DimensionHandle depth_multiplier = c->Dim(filter_shape, 3);

  // Check that the input depths are compatible.
  TF_RETURN_IF_ERROR(
      c->Merge(c->Dim(input_shape, 1), input_depth, &input_depth));

  DimensionHandle output_depth;
  TF_RETURN_IF_ERROR(c->Multiply(input_depth, depth_multiplier, &output_depth));

  tensorflow::Padding padding_type = tensorflow::EXPLICIT;
  std::vector<int64> padding;
  TF_RETURN_IF_ERROR(c->GetAttr("explicit_paddings", &padding));

  if (padding.size() != 4) {
    return tensorflow::errors::InvalidArgument(
        "DepthwiseConv2D requires the padding attribute to contain 4 "
        "values, "
        "but got: ",
        padding.size());
  }
  int64 pad_rows = padding[2];
  int64 pad_cols = padding[3];

  DimensionHandle output_rows, output_cols;

  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, in_rows_dim, filter_rows_dim, /* dilation_rate = */ 1, stride_rows,
      padding_type, pad_rows, pad_rows, &output_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeFromDimsV2(
      c, in_cols_dim, filter_cols_dim, /* dilation_rate = */ 1, stride_cols,
      padding_type, pad_cols, pad_cols, &output_cols));

  ShapeHandle output_shape =
      c->MakeShape({batch_size_dim, output_depth, output_rows, output_cols});
  c->set_output(0, output_shape);
  return tensorflow::Status::OK();
}

REGISTER_OP("DepthwiseConv")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("strides: list(int)")
    .Attr(tensorflow::GetExplicitPaddingsAttrString())
    .SetShapeFn(DepthwiseShapeFn);

REGISTER_OP("FusedDepthwiseConv")
    .Input("input: T")
    .Input("filter: T")
    .Input("bias: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("strides: list(int)")
    .Attr(tensorflow::GetExplicitPaddingsAttrString())
    .SetShapeFn(DepthwiseShapeFn);

REGISTER_OP("BiasRelu")
    .Input("in: float")
    .Input("bias: float")
    .Output("out: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      using tensorflow::shape_inference::ShapeHandle;
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &input_shape));
      c->set_output(0, input_shape);
      return tensorflow::Status::OK();
    });

REGISTER_OP("CsrSoftmax")
    .Input("input_values : float")
    .Input("row_indices: uint32")
    .Input("row_offsets: uint32")
    .Input("column_indices: uint32")
    .Output("output_values : float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      using tensorflow::shape_inference::ShapeHandle;
      ShapeHandle values_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &values_shape));
      c->set_output(0, values_shape);
      return tensorflow::Status::OK();
    });

REGISTER_OP("FusedSoftmax")
    .Input("input : float")
    .Output("output : float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      using tensorflow::shape_inference::ShapeHandle;
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
      c->set_output(0, input_shape);
      return tensorflow::Status::OK();
    });

}  // namespace sgk
