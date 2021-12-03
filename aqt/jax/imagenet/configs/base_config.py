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

"""Defines a reasonable baseline configuration for Resnet.

Specific experimental configurations should override these values as needed.
"""

import enum
from typing import List

from aqt.jax.imagenet.configs_script import config_schema


class ImagenetType(enum.Enum):
  """Enum to distinguish between different resnet architectures.

  Details of the architecture is at ../imagenet.png from
  He *et al.*, 2015 https://arxiv.org/pdf/1512.03385.pdf.
  """
  resnet29 = enum.auto()
  resnet50 = enum.auto()
  resnet101 = enum.auto()
  resnet152 = enum.auto()
  resnet200 = enum.auto()

  def get_residual_layers(self):
    """Returns number of resudual blocks corresponding to the architecture."""
    if self == self.resnet29:
      return [2, 2, 3, 2]
    elif self == self.resnet50:
      return [3, 4, 6, 3]
    elif self == self.resnet101:
      return [3, 4, 23, 3]
    elif self == self.resnet152:
      return [3, 8, 36, 3]
    elif self == self.resnet200:
      return [3, 24, 36, 3]
    else:
      raise ValueError(f"ImagenetType {self.value} is unknown.")


# TODO(shivaniagrawal): share the QuantTargets
class QuantTarget(enum.Enum):
  """Specifies which part of the model to quantize."""
  # No quantization
  none = enum.auto()

  # Quantize only the model weights. Activations are left unquantized.
  weights_only = enum.auto()

  # Quantize all the weights and activations. Activations use fixed
  # pre-specified clipping bounds.
  weights_and_fixed_acts = enum.auto()

  # Quantize all the weights and activations. Activations use clipping bounds
  # derived from activation statistics collected during training.
  weights_and_auto_acts = enum.auto()


def get_base_config(imagenet_type, quant_target):
  """Returns config that sets hyperparameters common to all quant targets.

  Fields in that config can then be overridden to customize a configuration.

  Note that two hyperparameters, model architecture kind and whether to
  automatically find clipping bounds for activations, have to be specified in
  advance as keyword arguments to this function instead of being overridden in
  the returned configdict. That is because these parameters affect the name and
  number of fields in the configdict instance, which can't be changed after
  creation: there will be one set of overridable parameters per layer in the
  configdict, and the field names in the 'quant_act' fields change depending on
  'quant_target'.

  Args:
    imagenet_type: Resnet model architecture.
    quant_target: Given quantization target, helpful for making decision whether
      to get config for automatic bounds alculation for activations or fixed
      bounds.

  Returns:
    A ConfigDict instance suitable for WMT training.
  """
  resnet_layers = imagenet_type.get_residual_layers()
  num_blocks = sum(resnet_layers)

  use_auto_acts = True if quant_target == QuantTarget.weights_and_auto_acts else False
  config = config_schema.get_config(
      num_blocks=num_blocks, use_auto_acts=use_auto_acts)
  config.update({
      "base_learning_rate": 0.1,
      "momentum": 0.9,
      "weight_decay": 0.0001,
      "activation_bound_update_freq": -1,
      "activation_bound_start_step": -1,
      "prec": None,
      "quant_type": "fake_quant",
      "weight_quant_granularity": "per_channel",
      "act_function": "relu",
      "shortcut_ch_shrink_method": "none",
      "shortcut_ch_expand_method": "none",
      "shortcut_spatial_method": "none",
      "lr_scheduler": {
          "warmup_epochs": 5,
          "cooldown_epochs": 50,
          "scheduler": "cosine",
          "num_epochs": 250,
          "endlr": 0.0,
          "knee_lr": 1e-5,
          "knee_epochs": 125,
      },
      "optimizer": "sgd",
      "adam": {
          "beta1": 0.9,
          "beta2": 0.999
      },
      "early_stop_steps": -1,  # -1 means no early stop
      "weight_quant_start_step": 0,  # 0 means turned on by default
      "teacher_model": "labels",
      "is_teacher": True,  # by default train the vanilla resnet
      "seed": 0,
  })

  proj_layers = [sum(resnet_layers[:x]) for x in range(len(resnet_layers))]
  for idx in range(num_blocks):
    if idx not in proj_layers:
      config.model_hparams.residual_blocks[idx].conv_proj = None
      config.model_hparams.residual_blocks[
          idx].conv_1.quant_act.input_distribution = "positive"

  config.model_hparams.filter_multiplier = 1.
  config.model_hparams.se_ratio = 0.5
  config.model_hparams.init_group = 32
  config.half_shift = False

  return config


def get_auto_acts_config(imagenet_type):
  """Returns config appropriate for automatic activation quantization."""
  config = get_base_config(
      imagenet_type=imagenet_type,
      quant_target=QuantTarget.weights_and_auto_acts)
  config.quant_act.bounds.update({  # update default values for auto acts
      "initial_bound": -1.0,
      "stddev_coeff": 3.0,
      "absdev_coeff": 3.0,
      "mix_coeff": 0.0,
      "reset_stats": False,
      "granularity": "per_channel",
      "ema_coeff": 0.1,
      "use_cams": False,
      "exclude_zeros": True,
      "use_mean_of_max": True,
      "use_old_code": True,
  })
  config.activation_bound_start_step = 7500
  config.activation_bound_update_freq = -1
  return config


def get_weights_only_config(imagenet_type):
  """Returns config for weights-only quantization."""
  config = get_base_config(
      imagenet_type=imagenet_type, quant_target=QuantTarget.weights_only)
  config.quant_act.bounds = None
  return config


def get_fixed_acts_config(imagenet_type):
  """Returns config for activation quantization with fixed bounds."""
  config = get_base_config(
      imagenet_type=imagenet_type,
      quant_target=QuantTarget.weights_and_fixed_acts)
  config.quant_act.bounds = 6.0
  return config


def get_config(quant_target, imagenet_type):
  """Returns config for a given quantization target and layer count."""
  if quant_target == QuantTarget.weights_only:
    return get_weights_only_config(imagenet_type=imagenet_type)
  elif quant_target == QuantTarget.weights_and_fixed_acts:
    return get_fixed_acts_config(imagenet_type=imagenet_type)
  elif quant_target == QuantTarget.weights_and_auto_acts:
    return get_auto_acts_config(imagenet_type=imagenet_type)
  elif quant_target == QuantTarget.none:
    return get_base_config(
        imagenet_type=imagenet_type, quant_target=quant_target)
  else:
    raise ValueError(f"quant_target {quant_target} not understood.")
