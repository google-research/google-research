# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Sparse Conv. Operations."""

import tensorflow as tf
from tf3d.ops import gen_car_ops

submanifold_sparse_conv2d = gen_car_ops.submanifold_sparse_conv2d
submanifold_sparse_conv3d = gen_car_ops.submanifold_sparse_conv3d
submanifold_sparse_conv2d_backprop_input = (
    gen_car_ops.submanifold_sparse_conv2d_backprop_input)
submanifold_sparse_conv3d_backprop_input = (
    gen_car_ops.submanifold_sparse_conv3d_backprop_input)
submanifold_sparse_conv2d_backprop_filter = (
    gen_car_ops.submanifold_sparse_conv2d_backprop_filter)
submanifold_sparse_conv3d_backprop_filter = (
    gen_car_ops.submanifold_sparse_conv3d_backprop_filter)


@tf.RegisterGradient('SubmanifoldSparseConv2D')
def _submanifold_sparse_conv2d_grad(op, grad):
  """The gradients for `submanifold_sparse_conv2d`.

  Args:
    op: The `submanifold_sparse_conv2d` Operation that we are differentiating,
      which we can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `submanifold_sparse_conv2d`
      op.

  Returns:
    Gradients with respect to the input of `submanifold_sparse_conv2d`.
  """
  (coordinates, num_valid_coordinates, input_features, filters) = op.inputs
  coordinates_grad = None
  num_valid_coordinates_grad = None
  input_features_grad = submanifold_sparse_conv2d_backprop_input(
      coordinates, num_valid_coordinates, input_features, filters, grad)
  filters_grad = submanifold_sparse_conv2d_backprop_filter(
      coordinates, num_valid_coordinates, input_features, filters, grad)
  return [
      coordinates_grad, num_valid_coordinates_grad, input_features_grad,
      filters_grad
  ]


@tf.RegisterGradient('SubmanifoldSparseConv3D')
def _submanifold_sparse_conv3d_grad(op, grad):
  """The gradients for `submanifold_sparse_conv3d`.

  Args:
    op: The `submanifold_sparse_conv3d` Operation that we are differentiating,
      which we can use to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `submanifold_sparse_conv3d`
      op.

  Returns:
    Gradients with respect to the input of `submanifold_sparse_conv3d`.
  """
  (coordinates, num_valid_coordinates, input_features, filters) = op.inputs
  coordinates_grad = None
  num_valid_coordinates_grad = None
  input_features_grad = submanifold_sparse_conv3d_backprop_input(
      coordinates, num_valid_coordinates, input_features, filters, grad)
  filters_grad = submanifold_sparse_conv3d_backprop_filter(
      coordinates, num_valid_coordinates, input_features, filters, grad)
  return [
      coordinates_grad, num_valid_coordinates_grad, input_features_grad,
      filters_grad
  ]
