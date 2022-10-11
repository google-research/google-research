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
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_registry
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import default_n_bit_quantize_configs
from tensorflow_model_optimization.python.core.quantization.keras.experimental.default_n_bit import default_n_bit_quantize_registry


class NBitNoOpActivationConfig(
    default_n_bit_quantize_registry.DefaultNBitConvQuantizeConfig):
  """DefaultNBitConvQuantizeConfig without activation quantization.

    It is useful for conv + batch_norm quantization aware training, so that
    TFlite can fold these layers later.
  """

  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_activations(self, layer, quantize_activations):
    pass


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


def get_conv_bn_quantize_config(flags, nbit_weight_8bit=False):
  """Returns quantize_config for wrapped Conv2D layer followed by batch norm.

  For quantized conv2d layer followed by batch normalization, it specifies
  activations are not quantized using either regular or n-bit TFMOT scheme. It
  enables batch normalization folding during conversion to TFLite.

  Args:
    flags: model/data parameters.
    nbit_weight_8bit: if True use 8-bit weights in n-bit quantization,
      otherwise use flags.nbit_weight_bits.


  Returns:
    quantize_config or None for float model.
  """
  if flags.quantize:
    if flags.use_quantize_nbit:
      return NBitNoOpActivationConfig(
          ['kernel'], ['activation'],
          False,
          num_bits_weight=flags.nbit_weight_bits
          if not nbit_weight_8bit
          else 8,
          num_bits_activation=flags.nbit_activation_bits)
    else:
      return NoOpActivationConfig(['kernel'], ['activation'], False)
  else:
    return None


def get_no_op_quantize_config(flags):
  """Returns config without quantization according to TFMOT scheme.

  For batch normalization layers during training.  It enables batch
  normalization folding during conversion to TFLite.

  Args:
    flags: data/model parameters.

  Returns:
    quantize_config
  """
  if flags is None or not flags.use_quantize_nbit:
    return default_8bit_quantize_configs.NoOpQuantizeConfig()
  else:
    return default_n_bit_quantize_configs.NoOpQuantizeConfig()


def quantize_layer(layer, apply_quantization=None, quantize_config=None,
                   flags=None, nbit_weight_8bit=False):
  """Quantizes a layer.

  It is useful for quantization aware training

  Args:
    layer: input layer to quantize
    apply_quantization: if True layer is quantized, otherwise not
    quantize_config: quantization config for special cases such as
      sequence of convolution and batch normalization (e.g.:NoOpQuantizeConfig).
    flags: data/model parameters.
    nbit_weight_8bit: if True use 8-bit weights in n-bit quantization, otherwise
      use flags.nbit_weight_bits.

  Returns:
    quantized layer or layer without changes.

  Raise:
    ValueError if BatchNormalization quantize_config is not NoOpQuantizeConfig,
               or (flags.quantize and apply_quantization are not equal).
  """
  if flags is None:
    if apply_quantization is None:
      apply_quantization = True  # Legacy support.
  else:
    if apply_quantization is None:
      apply_quantization = flags.quantize
    elif apply_quantization != flags.quantize:
      raise ValueError('flags.quantize and apply_quantization are not equal.')

  if apply_quantization:
    # Quantize the layer using one of the two TFMOT quantization schemes.

    if flags is not None and flags.use_quantize_nbit:
      # Use TF MOT default_n_bit scheme: model layer quantize_config can define
      # other values for num_bits_weight and num_bits_activation.
      scheme = tfmot.quantization.keras.experimental.default_n_bit.DefaultNBitQuantizeScheme(
          num_bits_weight=flags.nbit_weight_bits if not nbit_weight_8bit else 8,
          num_bits_activation=flags.nbit_activation_bits,
          )

    else:
      # Use TF MOT default_8bit scheme.
      scheme = tfmot.quantization.keras.default_8bit.Default8BitQuantizeScheme()

    quantize_registry = scheme.get_quantize_registry()

    if layer.__class__.__name__ == 'BatchNormalization':
      if not isinstance(
          quantize_config,
          (default_8bit_quantize_configs.NoOpQuantizeConfig
           if not (flags is not None and flags.use_quantize_nbit)
           else default_n_bit_quantize_configs.NoOpQuantizeConfig),
      ):
        raise ValueError('Unexpected quantize_config for batchnorm: %s' %
                         quantize_config.__class__.__name__)
    elif not quantize_registry.supports(layer):
      logging.info('layer is not supported: %s', str(layer))
      return layer

    if quantize_config is None:
      quantize_config = quantize_registry.get_quantize_config(layer)

    quantize_config_str = (f'{layer.name:>30} '
                           f'{quantize_config.__class__.__name__}')
    logging.info(quantize_config_str)

    return quantize_wrapper.QuantizeWrapperV2(layer, quantize_config)
  else:
    return layer


def quantize_scope():
  """Returns quantize scope with custom objects."""
  return tfmot.quantization.keras.quantize_scope({
      'NBitNoOpActivationConfig': NBitNoOpActivationConfig,
      'NoOpActivationConfig': NoOpActivationConfig,
  })
