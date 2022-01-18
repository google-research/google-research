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

"""Quantization functions."""

from absl import logging
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry


class NoOpActivationConfig(
    default_8bit_quantize_registry.Default8BitConvQuantizeConfig):
  """8BitConvQuantizeConfig without activation quantization.

    It is useful for conv + batch_norm quantization aware training, so that
    TFlite can fold these layers later.
  """

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_activations(self, layer, quantize_activations):
    pass


def quantize_layer(layer, apply_quantization=True, quantize_config=None):
  """Quantizes a layer.

  It is useful for quantization aware training
  Args:
    layer: input layer to quantize
    apply_quantization: if True layer is quantized, otherwise not
    quantize_config: quantization config for special cases such as
      sequence of convolution and batch normalization

  Returns:
    quantized layer
  """
  if apply_quantization:
    scheme = tfmot.quantization.keras.default_8bit.Default8BitQuantizeScheme()

    quantize_registry = scheme.get_quantize_registry()

    if not quantize_registry.supports(layer):
      logging.info('layer is not supported: %s', str(layer))
      return layer

    if quantize_config is None:
      quantize_config = quantize_registry.get_quantize_config(layer)
    return quantize_wrapper.QuantizeWrapper(layer, quantize_config)
  else:
    return layer
