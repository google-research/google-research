# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# Copyright 2024 Google LLC
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

"""Modules used for implementing adaptive low rank tuning."""

import tensorflow as tf


class AdaptiveEinsumDense(tf.keras.layers.Layer):
  """Adaptive low rank dense layer.

  This will be used to replace the key, query, and values components
  in the TF MultiHeadAttention module.

  Attributes:
    full_rank_layer: the full rank dense layer used in MultiHeadAttention.
    scaling: scaling during adaptation.
    equation: equation used for tensor multiplication.
    activation: activation function.
    full_rank_kernel: kernel in full_rank_layer.
    bias: bias in full_rank_layer.
    adaptive_a: first low rank tensor factor.
    adaptive_b: second low rank tensor factor.
  """

  def __init__(
      self, dense_layer, rank = 4, scaling = 8
  ):
    """Adaptive low rank module initialization.

    Args:
      dense_layer: the full rank dense layer used in MultiHeadAttention.
      rank: desired rank of the low rank module.
      scaling: scaling factor used during adaptation.
    """
    super().__init__()
    # dense layer is instance of EinsumDense
    self.full_rank_layer_equation = dense_layer.equation
    self.scaling = scaling / rank
    # multiplication of two adaptor tensors.
    self.equation = "...ab,...bc->...ac"
    self.activation = dense_layer.activation
    self.full_rank_kernel = self.add_weight(
        shape=dense_layer.kernel.shape,
        initializer=tf.initializers.Constant(dense_layer.kernel),
        trainable=False,
    )
    self.bias = dense_layer.bias
    kernel_shape = list(self.full_rank_kernel.shape)
    adaptive_a_shape = kernel_shape[:-1] + [rank]
    adaptive_b_shape = kernel_shape
    adaptive_b_shape[-2] = rank
    self.adaptive_a = self.add_weight(
        "adaptive_a",
        shape=adaptive_a_shape,
        initializer="glorot_uniform",
        trainable=True,
    )
    self.adaptive_b = self.add_weight(
        "adaptive_b",
        shape=adaptive_b_shape,
        initializer="glorot_uniform",
        trainable=True,
    )

  def call(self, inputs):
    """forward pass for adaptive low rank tuning.

    Args:
      inputs: training inputs.

    Returns:
      Computed output using low rank adaptation.
    """
    full_weight = tf.einsum(self.equation, self.adaptive_a, self.adaptive_b)
    ret = tf.einsum(
        self.full_rank_layer_equation,
        inputs,
        self.full_rank_kernel + full_weight,
    )
    # TODO(yihed): successively multiply A and B with inputs to save memory.E.g.
    # inputs_low_rank = tf.einsum("...nd,dr->...nr", inputs, self.adaptive_a)
    # ret = tf.matmul(inputs_low_rank, self.adaptive_b)
    # ret = tf.matmul(inputs, self.full_rank_kernel) + ret

    if self.bias is not None:
      ret += self.bias
    if self.activation is not None:
      ret = self.activation(ret)
    return ret
