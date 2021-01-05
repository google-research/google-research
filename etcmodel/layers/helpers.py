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

"""Miscellaneous helper layers for ETC."""

from typing import Any, Sequence, Text

import tensorflow as tf


class DenseLayers(tf.keras.layers.Layer):
  """Convenience class for stacking Dense layers.

  The result is a simple fully-connected network with `len(hidden_sizes)`
  layers, the last of which is the output. The given activation function
  will be applied to every layer except for the last one, which will not
  have any activation.
  """

  def __init__(self,
               hidden_sizes: Sequence[int],
               activation=None,
               use_bias: bool = True,
               kernel_initializer=None,
               bias_initializer='zeros',
               name: Text = 'dense_layers',
               **kwargs):
    """Init.

    Args:
      hidden_sizes: List of hidden layer sizes, one for each layer in order. The
        last integer will be the layer size of the output.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied (i.e. "linear" activation: a(x) = x). Note that
          the last layer will not have any activation applied.
      use_bias: Boolean, whether the layers use a bias vector.
      kernel_initializer: Initializer for the kernel weights matrices.
      bias_initializer: Initializer for the bias vectors.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    super(DenseLayers, self).__init__(name=name, **kwargs)

    if any(size < 1 for size in hidden_sizes):
      raise ValueError('All sizes in `hidden_sizes` must be positive.')

    self.hidden_sizes = hidden_sizes
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

    self.layers = []
    for i, layer_size in enumerate(hidden_sizes):
      linear_layer = tf.keras.layers.Dense(
          units=layer_size,
          activation=None,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          name='layer_%d' % i)
      self.layers.append(linear_layer)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Calls the layer.

    Args:
      inputs: <float32>[batch_size, ..., input_size].

    Returns:
      <float32>[batch_size, ..., output_size].
    """
    output = inputs
    for i, layer in enumerate(self.layers):
      is_last_layer = (i == len(self.layers) - 1)
      output = layer(output)
      if not is_last_layer:
        output = self.activation(output)
    return output


class TrackedLambda(tf.keras.layers.Layer):
  """Custom layer defined by a function and its dependencies.

  This is similar to `tf.keras.layers.Lambda`, except it takes a list of all
  dependencies (layers and/or variables) that the function depends on in order
  to avoid the issue mentioned here:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda#variables
  """

  def __init__(self,
               function,
               dependencies: Sequence[Any] = (),
               name: Text = 'tracked_lambda',
               **kwargs):
    """Init.

    Args:
      function: The function to be evaluated. All arguments to `call` will be
        forwarded to this function.
      dependencies: Sequence of all variables or layers used by `function`.
      name: Name of the layer.
      **kwargs: Forwarded to super.
    """
    # Set the function as the `call` method before calling
    # the keras layer super constructor, so that Keras can
    # inspect the arguments of the function and determine
    # whether it needs to try passing a `training` argument or not.
    self.call = function
    super(TrackedLambda, self).__init__(name=name, **kwargs)

    self.function = function
    self.dependencies = list(dependencies)
