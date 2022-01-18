# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Wrapper layers for ETC.

These layers wrap other layers to add skip connections (e.g. residual blocks)
or gating mechanisms.
"""

from typing import Optional, Sequence, Text, Union

import tensorflow as tf

from etcmodel.layers import helpers
from etcmodel.layers import recomputing_dropout


class ResidualBlock(tf.keras.layers.Layer):
  """Residual network block.

  This is a flexible residual block wrapper around a user-provided
  `inner_layer`, which is just a fully-connected 2-layer network by default.
  Normalization and dropout are applied in the following order
  by default (as used by the original Transformer layers in
  https://arxiv.org/abs/1706.03762):
    output = normalization(input + dropout(inner_layer(input)))

  Alternatively, there's an option to use the "pre-activation" order from
  https://arxiv.org/abs/1603.05027 instead:
    output = input + dropout(inner_layer(normalization(input)))
  """

  def __init__(self,
               inner_layer: Optional[tf.keras.layers.Layer] = None,
               normalization_layer: Optional[
                   Union[tf.keras.layers.Layer,
                         Sequence[tf.keras.layers.Layer]]] = None,
               dropout_probability: float = 0.0,
               use_pre_activation_order: bool = False,
               inner_intermediate_size: Optional[int] = None,
               inner_activation='relu',
               inner_kernel_initializer=None,
               name: Text = 'residual_block',
               **kwargs):
    """Init.

    Args:
      inner_layer: Keras layer to apply as the inner layer in the residual
        block. The output of the layer must have the same shape as the input. By
        default, a 2-layer fully-connected network (via `DenseLayers`) is
        created based on the `inner_...` arguments below.
      normalization_layer: Normalization layer to apply. If `inner_layer`
        expects multiple inputs/outputs, then this should be a sequence of
        layers, one for each input. By default this is initialized to a single
        `tf.keras.layers.LayerNormalization` layer, so it must be given when
        expecting multiple `inner_layer` inputs.
      dropout_probability: The probability of dropping out a value when applying
        dropout for the block.
      use_pre_activation_order: If True, use "pre-activation" order (see class
        docstring for details).
      inner_intermediate_size: Size of intermediate fully-connected layer.
        Defaults to the input layer size. Ignored if `inner_layer` is not None.
      inner_activation: Activation function for the intermediate layer. Ignored
        if `inner_layer` is not None.
      inner_kernel_initializer: Initializer to use for fully-connected kernel
        weights. Bias weights are always initialized to 0. Ignored if
        `inner_layer` is not None.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(ResidualBlock, self).__init__(name=name, **kwargs)

    if normalization_layer is None:
      normalization_layer = tf.keras.layers.LayerNormalization(
          axis=-1, epsilon=1e-12, name='layer_norm')
    if isinstance(normalization_layer, Sequence):
      normalization_layers = normalization_layer
    else:
      normalization_layers = [normalization_layer]
    # Inner layer may be created later. Assign `normalization_layers` attribute
    # first, so that the variable order remains the same regardless.
    self.normalization_layers = normalization_layers
    self.inner_layer = inner_layer
    self.dropout_probability = dropout_probability
    self.use_pre_activation_order = use_pre_activation_order
    self.inner_intermediate_size = inner_intermediate_size
    self.inner_activation = inner_activation
    self.inner_kernel_initializer = inner_kernel_initializer
    self.dropout_layers = [
        recomputing_dropout.RecomputingDropout(rate=dropout_probability)
        for _ in self.normalization_layers
    ]

  def build(self, input_shape: tf.TensorShape) -> None:
    """Keras build function.

    Args:
      input_shape: TensorShape of the input.
    """
    if self.inner_layer is None:
      input_size = input_shape.as_list()[-1]
      if input_size is None:
        raise ValueError('Static input layer size must be known.')
      if self.inner_intermediate_size is None:
        self.inner_intermediate_size = input_size
      self.inner_layer = helpers.DenseLayers(
          hidden_sizes=[self.inner_intermediate_size, input_size],
          activation=self.inner_activation,
          use_bias=True,
          kernel_initializer=self.inner_kernel_initializer)

    super(ResidualBlock, self).build(input_shape)

  def call(self,
           inputs: Union[tf.Tensor, Sequence[tf.Tensor]],
           training=None,
           **kwargs) -> Union[tf.Tensor, Sequence[tf.Tensor]]:
    """Calls the layer.

    Args:
      inputs: <float32>[batch_size, ..., input_size] Tensor or sequence of
        tensors. In the sequence case, all the tensors will be passed to
        `inner_layer` as positional arguments, and the output of `inner_layer`
        must be a same-length sequence of tensors with exactly the same shapes.
      training: For Keras, optional boolean scalar tensor or Python boolean
        indicating whether the call is meant for training or inference.
      **kwargs: Additional keyword arguments to pass to `inner_layer`.

    Returns:
      Float Tensor of same shape as `inputs`.
    """
    if isinstance(inputs, Sequence):
      input_is_singleton = False
    else:
      input_is_singleton = True
      inputs = [inputs]

    if len(inputs) != len(self.normalization_layers):
      raise ValueError(
          'Number of inputs ({}) does not match number of normalization layers '
          '({}).'.format(len(inputs), len(self.normalization_layers)))

    if self.use_pre_activation_order:
      tensors = _zip_layer_sequence(
          self.normalization_layers, inputs, training=training)
      tensors = self.inner_layer(*tensors, training=training, **kwargs)
      if not isinstance(tensors, Sequence):
        tensors = [tensors]
      tensors = _zip_layer_sequence(
          self.dropout_layers, tensors, training=training)
      outputs = [x + y for x, y in zip(inputs, tensors)]
    else:
      tensors = self.inner_layer(*inputs, training=training, **kwargs)
      if not isinstance(tensors, Sequence):
        tensors = [tensors]
      tensors = _zip_layer_sequence(
          self.dropout_layers, tensors, training=training)
      tensors = [x + y for x, y in zip(inputs, tensors)]
      outputs = _zip_layer_sequence(
          self.normalization_layers, tensors, training=training)

    if input_is_singleton:
      return outputs[0]
    else:
      return outputs


def _zip_layer_sequence(layers: Sequence[tf.keras.layers.Layer],
                        tensors: Sequence[tf.Tensor], **kwargs):
  """Applies a sequence of layers to a sequence of tensors of the same size."""
  return [layer(tensor, **kwargs) for layer, tensor in zip(layers, tensors)]
