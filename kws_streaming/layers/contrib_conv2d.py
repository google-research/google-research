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

"""A layer which applies normalization before the activation function."""
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes


class ContribConv2D(tf.keras.layers.Conv2D):
  """A Keras implementation of contrib.layers.conv2d.

  This implementation applies a normalizer to the output before the activation
  function.
  """

  def __init__(self,
               normalizer_fn=None,
               normalizer_params=None,
               activation=None,
               use_bias=True,
               mode=Modes.TRAINING,
               **kwargs):
    """Initialization of the layer.

    Args:
      normalizer_fn: The normalization function to use (in any).
      normalizer_params: The parameters passed to the normalization function.
      activation: The activation function for the layer. In case a normalization
        function is applied, the activation function gets manually applied
        *after* the normalization function.
      use_bias: Whether to use bias during training. If a normalizer function is
        applied, is set to False regardless of the value passed.
      mode: The type of mode the layer is in.
      **kwargs: Parameters that would normally be passed to the Conv2D layer
    """
    self.normalizer_fn = normalizer_fn
    self.normalizer_params = None
    self.activation_fn = None

    self.training = (mode == Modes.TRAINING)

    if normalizer_fn is not None:
      self.normalizer_params = normalizer_params or {}
      self.activation_fn = activation
      activation = None
      use_bias = False

    super(ContribConv2D, self).__init__(
        activation=activation, use_bias=use_bias, **kwargs)

  def build(self, input_shape):
    super(ContribConv2D, self).build(input_shape)
    if self.normalizer_fn is not None:
      self.norm_layer = self.normalizer_fn(**self.normalizer_params)
    else:
      self.norm_layer = tf.keras.layers.Lambda(lambda x: x)

    if self.activation_fn is not None:
      self.activation_layer = tf.keras.layers.Activation(self.activation_fn)

  def call(self, inputs):
    outputs = super(ContribConv2D, self).call(inputs)
    outputs = self.norm_layer(outputs, training=self.training)
    if self.activation_fn is not None:
      outputs = self.activation_layer(outputs)
    return outputs

  def get_config(self):
    config = super(ContribConv2D, self).get_config()
    config.update({
        'normalizer_fn': self.normalizer_fn,
        'normalizer_params': self.normalizer_params,
        'activation_fn': self.activation_fn,
        'training': self.training
    })
    return config
