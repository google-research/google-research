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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

typedef tensorflow::int32 int32;

namespace {

REGISTER_OP("SubmanifoldSparseConv2D")
    .Input("coordinates: int32")
    .Input("num_valid_coordinates: int32")
    .Input("input_features: float")
    .Input("filter: float")
    .Output("output_features: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      using tensorflow::shape_inference::ShapeHandle;
      ShapeHandle input_feature_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &input_feature_shape));
      ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &filter_shape));
      ShapeHandle output_feature_shape = c->MakeShape(
          {c->Dim(input_feature_shape, 0), c->Dim(input_feature_shape, 1),
           c->Dim(filter_shape, 3)});
      c->set_output(0, output_feature_shape);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Run convolutions on 2D points whose locations are specified by `coordinates`.
For each point, it runs a regular conv2d using `input_features` and `filter`.
But different from a traditional convolution, it doesn't run convolution on
points that are not present in `coordinates`.

This is also called "Submanifold convolutions", for more details please refer
to

Benjamin Graham and Laurens van der Maaten. Submanifold sparse convolutional
networks. arXiv preprint arXiv:1706.01307, 2017.

coordinates: [batch_size, max_num_coords_per_batch, 2], the padded 2D
    coordinates. max_num_coords_per_batch is the max number of coordinates in
    each batch item.
num_valid_coordinates: [batch_size], the number of valid coordinates per batch
    item. Only the top num_valid_coordinates[i] entries in coordinates[i],
    input_features[i], and output_features[i] are valid. The rest of the entries
    are paddings.
input_features: [batch_size, max_num_coords_per_batch, in_channels] where
    in_channels is the channel size of the input feature.
filter: [filter_height, filter_width, in_channels, out_channels], the
    convolution filter (kernel) values.
output_features: [batch_size, max_num_coords_per_batch, out_channels] where
    out_channels is the channel size of the output feature.
)doc");

REGISTER_OP("SubmanifoldSparseConv3D")
    .Input("coordinates: int32")
    .Input("num_valid_coordinates: int32")
    .Input("input_features: float")
    .Input("filter: float")
    .Output("output_features: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      using tensorflow::shape_inference::ShapeHandle;
      ShapeHandle input_feature_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &input_feature_shape));
      ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 5, &filter_shape));
      ShapeHandle output_feature_shape = c->MakeShape(
          {c->Dim(input_feature_shape, 0), c->Dim(input_feature_shape, 1),
           c->Dim(filter_shape, 4)});
      c->set_output(0, output_feature_shape);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Run convolutions on 3D points whose locations are specified by `coordinates`.
For each point, it runs a regular conv3d using `input_features` and `filter`.
But different from a traditional convolution, it doesn't run convolution on
points that are not present in `coordinates`.

This is also called "Submanifold convolutions", for more details please refer
to

Benjamin Graham and Laurens van der Maaten. Submanifold sparse convolutional
networks. arXiv preprint arXiv:1706.01307, 2017.

coordinates: [batch_size, max_num_coords_per_batch, 3], the padded 3D
    coordinates. max_num_coords_per_batch is the max number of coordinates in
    each batch item.
num_valid_coordinates: [batch_size], the number of valid coordinates per batch
    item. Only the top num_valid_coordinates[i] entries in coordinates[i],
    input_features[i], and output_features[i] are valid. The rest of the entries
    are paddings.
input_features: [batch_size, max_num_coords_per_batch, in_channels] where
    in_channels is the channel size of the input feature.
filter: [filter_depth, filter_height, filter_width, in_channels, out_channels],
    the convolution filter (kernel) values.
output_features: [batch_size, max_num_coords_per_batch, out_channels] where
    out_channels is the channel size of the output feature.
)doc");

REGISTER_OP("SubmanifoldSparseConv2DBackpropInput")
    .Input("coordinates: int32")
    .Input("num_valid_coordinates: int32")
    .Input("input_features: float")
    .Input("filter: float")
    .Input("out_backprop: float")
    .Output("output: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle input_feature_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &input_feature_shape));
      c->set_output(0, input_feature_shape);
      return tensorflow::Status::OK();
    });

REGISTER_OP("SubmanifoldSparseConv3DBackpropInput")
    .Input("coordinates: int32")
    .Input("num_valid_coordinates: int32")
    .Input("input_features: float")
    .Input("filter: float")
    .Input("out_backprop: float")
    .Output("output: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle input_feature_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &input_feature_shape));
      c->set_output(0, input_feature_shape);
      return tensorflow::Status::OK();
    });

REGISTER_OP("SubmanifoldSparseConv2DBackpropFilter")
    .Input("coordinates: int32")
    .Input("num_valid_coordinates: int32")
    .Input("input_features: float")
    .Input("filter: float")
    .Input("out_backprop: float")
    .Output("output: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &filter_shape));
      c->set_output(0, filter_shape);
      return tensorflow::Status::OK();
    });

REGISTER_OP("SubmanifoldSparseConv3DBackpropFilter")
    .Input("coordinates: int32")
    .Input("num_valid_coordinates: int32")
    .Input("input_features: float")
    .Input("filter: float")
    .Input("out_backprop: float")
    .Output("output: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::shape_inference::ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 5, &filter_shape));
      c->set_output(0, filter_shape);
      return tensorflow::Status::OK();
    });

}  // namespace
