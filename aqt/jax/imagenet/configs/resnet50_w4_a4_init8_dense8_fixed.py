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

"""Resnet50 quantized model."""

from aqt.jax.imagenet.configs import base_config


def get_config(quant_target=base_config.QuantTarget.weights_and_fixed_acts):
  """Gets Resnet50 config for 4 bits weights and fixed activation quantization.

    conv_init and last dense layer quantized to 8bit.

  Args:
   quant_target: quantization target, of type QuantTarget.

  Returns:
   ConfigDict instance.
  """
  config = base_config.get_config(
      imagenet_type=base_config.ImagenetType.resnet50,
      quant_target=quant_target)
  config.weight_prec = 4
  config.quant_act.prec = 4
  config.model_hparams.conv_init.weight_prec = 8
  config.model_hparams.conv_init.quant_act.prec = 8
  config.model_hparams.dense_layer.weight_prec = 8
  config.model_hparams.dense_layer.quant_act.prec = 8
  return config
