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

"""Defines helpful utilities for custom recurrent cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def check_rnn_weight_shapes(kernel_weights, bias_weights, num_units):
  """Helper to verify input weight shapes.

  Args:
    kernel_weights: 2-tuple of Variables for the RNN kernel weights.
    bias_weights: Variable for the RNN bias weights.
    num_units: The number of units in the RNN layer.

  Raises:
    RuntimeError: If the kernel_weights is not a 2-tupel of 2D variables
      with equal shape and second dimension of size `num_units` or if the
      bias_weights is not a 1D tensor of size `num_units`.
  """
  if len(kernel_weights) != 2:
    raise RuntimeError("Incorrect number of variational parameters.")
  if kernel_weights[0].shape != kernel_weights[1].shape:
    raise RuntimeError("Variational parameters must be the same shape.")

  means_shape = kernel_weights[0].get_shape().as_list()
  if len(means_shape) != 2:
    raise RuntimeError("Kernel weights must be two dimensional.")
  if means_shape[1] != num_units:
    raise RuntimeError(
        "Kernel weights second dimension must be size `num_units`")

  bias_shape = bias_weights.shape
  if len(bias_shape) != 1:
    raise RuntimeError("Bias weights must be one dimensional.")
  if bias_shape[0] != num_units:
    raise RuntimeError("Bias weights must be size `num_units`")


def check_lstm_weight_shapes(kernel_weights, bias_weights, num_units):
  """Helper to verify input weight shapes.

  Args:
    kernel_weights: Variable for the LSTM kernel weights.
    bias_weights: Variable for the LSTM bias weights.
    num_units: The number of units in the LSTM layer.

  Raises:
    RuntimeError: If the kernel_weights is not a 2-tupel of 2D variables
      with equal shape and second dimension of size `4 * num_units` or if the
      bias_weights is not a 1D tensor of size `4 * num_units`.
  """
  if len(kernel_weights) != 2:
    raise RuntimeError("Incorrect number of variational parameters.")
  if kernel_weights[0].shape != kernel_weights[1].shape:
    raise RuntimeError("Variational parameters must be the same shape.")

  means_shape = kernel_weights[0].get_shape().as_list()
  if len(means_shape) != 2:
    raise RuntimeError("Kernel weights must be two dimensional.")
  if means_shape[1] != 4 * num_units:
    raise RuntimeError(
        "Kernel weights second dimension must be size `4 * num_units`")

  bias_shape = bias_weights.shape
  if len(bias_shape) != 1:
    raise RuntimeError("Bias weights must be one dimensional.")
  if bias_shape[0] != 4 * num_units:
    raise RuntimeError("Bias weights must be size `4 * num_units`")
