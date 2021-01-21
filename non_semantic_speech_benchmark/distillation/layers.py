# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Compression layers for Non-Semantic Speech project."""

from absl import logging
import tensorflow as tf


class CompressedDense(tf.keras.layers.Dense):
  """A compressed Dense keras layer with the compression op.

  The compression_obj.get_spec().rank must be divisibe by
  compression_obj.get_spec().input_block_size. The input size to the layer
  must be divisible by compression_obj.get_spec().input_block_size which
  in turn must be divisible by compression_obj.get_spec().rank.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               compression_obj=None,
               **kwargs):
    """Initializer.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      bias_constraint: Constraint function applied to the bias vector.
      compression_obj: Compression object contaning compression parameters  The
        compression_obj.get_spec().rank must be divisibe by
        compression_obj.get_spec().input_block_size. The input size to the layer
        must be divisible by compression_obj.get_spec().input_block_size which
        in turn must be divisible by compression_obj.get_spec().rank.
      **kwargs: additional keyword arguments.
    """

    super().__init__(units, activation, use_bias, kernel_initializer,
                     bias_initializer, kernel_regularizer, bias_regularizer,
                     activity_regularizer, kernel_constraint, bias_constraint,
                     **kwargs)
    self.compression_obj = compression_obj
    self.compression_op = None
    self.alpha = -1

  def build(self, input_shape):
    super().build(input_shape)
    self.compression_op = self.compression_obj.apply_compression_keras(
        self.kernel, layer=self)

    logging.info(
        'in build kernel a_matrix b_matrix and c_matrix shape is %s %s %s %s',
        self.kernel.shape, self.compression_op.a_matrix_tfvar.shape,
        self.compression_op.b_matrix_tfvar.shape,
        self.compression_op.c_matrix_tfvar.shape)

  def call(self, inputs, training=True):
    self.compression_op.maybe_run_update_step()
    return self.activation(
        self.compression_op.compressed_matmul_keras(inputs) + self.bias)
