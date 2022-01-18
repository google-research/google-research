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

"""Resnet50 quantized model."""

from aqt.jax.imagenet.configs import base_config


def get_config(quant_target=base_config.QuantTarget.weights_and_fixed_acts):
  """Gets Resnet50 config for 8 bits weights and fixed activation quantization.

  conv_init and last dense layer not quantized as these are the most
  sensitive layers in the model.

  Args:
   quant_target: quantization target, of type QuantTarget.

  Returns:
   ConfigDict instance.
  """
  config = base_config.get_config(
      imagenet_type=base_config.ImagenetType.resnet50,
      quant_target=quant_target)
  config.weight_prec = 8
  config.quant_act.prec = 8
  return config
