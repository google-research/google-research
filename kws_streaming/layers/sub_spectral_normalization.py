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

"""Sub spectral normalization layer."""
from typing import Any, Dict

import tensorflow_model_optimization as tfmot

from kws_streaming.layers.compat import tf


class SubSpectralNormalization(tf.keras.layers.Layer):
  """Sub spectral normalization layer.

  It is based on paper:
  "SUBSPECTRAL NORMALIZATION FOR NEURAL AUDIO DATA PROCESSING"
  https://arxiv.org/pdf/2103.13620.pdf
  """

  def __init__(self, sub_groups, **kwargs):
    super(SubSpectralNormalization, self).__init__(**kwargs)
    self.sub_groups = sub_groups

    self.batch_norm = tf.keras.layers.BatchNormalization()

  def call(self, inputs):
    # expected input: [N, Time, Frequency, Channels]
    if inputs.shape.rank != 4:
      raise ValueError('input_shape.rank:%d must be 4' % inputs.shape.rank)

    input_shape = inputs.shape.as_list()
    if input_shape[2] % self.sub_groups:
      raise ValueError('input_shape[2]: %d must be divisible by '
                       'self.sub_groups %d ' %
                       (input_shape[2], self.sub_groups))

    net = inputs
    if self.sub_groups == 1:
      net = self.batch_norm(net)
    else:
      target_shape = [
          input_shape[1], input_shape[2] // self.sub_groups,
          input_shape[3] * self.sub_groups
      ]
      net = tf.keras.layers.Reshape(target_shape)(net)
      net = self.batch_norm(net)
      net = tf.keras.layers.Reshape(input_shape[1:])(net)
    return net

  def get_config(self):
    config = {'sub_groups': self.sub_groups}
    base_config = super(SubSpectralNormalization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# Quantization aware training support for custom operations.


def quantizable_sub_spectral_normalization(is_quantize = False,
                                           **kwargs):
  """Functional API with quantization annotations.

  Args:
    is_quantize: bool, if True quantize using layer annotation with custom
      quantization config.
    **kwargs: Any

  Returns:
    keras layer
  """
  if is_quantize:
    return tfmot.quantization.keras.quantize_annotate_layer(
        SubSpectralNormalization(**kwargs), DoNotQuantizeConfig())
  else:
    return SubSpectralNormalization(**kwargs)


class DoNotQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
  """QuantizeConfig which does not quantize any part of the layer."""

  def get_weights_and_quantizers(
      self, layer
  ):
    return []

  def get_activations_and_quantizers(
      self, layer
  ):
    return []

  def set_quantize_weights(self, layer,
                           quantize_weights):
    return []

  def set_quantize_activations(
      self, layer, quantize_activations
  ):
    return []

  def get_output_quantizers(
      self, layer
  ):
    return []

  def get_config(self):
    return {}


def quantization_scopes():
  """Quantization scope for all known custom ops."""
  return {
      'SubSpectralNormalization': SubSpectralNormalization,
      'DoNotQuantizeConfig': DoNotQuantizeConfig,
  }
