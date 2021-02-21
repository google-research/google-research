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

"""Quantization functions."""

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_wrapper


def quantize_layer(layer, apply_quantization=True):
  """Quantizes a layer.

  It is useful for quantization aware training
  Args:
    layer: input layer to quantize
    apply_quantization: if True layer is quantized, otherwise not
      returned

  Returns:
    quantized layer
  """
  if apply_quantization:
    scheme = tfmot.quantization.keras.default_8bit.Default8BitQuantizeScheme()

    quantize_registry = scheme.get_quantize_registry()

    if not quantize_registry.supports(layer):
      return layer
    quantize_config = quantize_registry.get_quantize_config(layer)
    return quantize_wrapper.QuantizeWrapper(layer, quantize_config)
  else:
    return layer
