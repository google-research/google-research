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

"""Defines a reasonable baseline configuration for the WMT transformer.

Specific experimental configurations should override these values as needed.
"""

import enum

from aqt.jax.wmt_mlperf.hparams_config_scripts import config_schema


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


def get_base_config(n_layers, use_auto_acts, fp_quant):
  """Returns config that sets hyperparameters common to all quantization targets.

  Fields in that config can then be overridden to customize a configuration.

  Note that two hyperparameters, the number of layers and whether to
  automatically find clipping bounds for activations, have to be specified in
  advance as keyword arguments to this function instead of being overridden in
  the returned configdict. That is because these parameters affect the name and
  number of fields in the configdict instance, which can't be changed after
  creation: there will be one set of overridable parameters per layer in the
  configdict, and the field names in the 'quant_act' fields change depending on
  'use_auto_acts'.

  Args:
    n_layers: Number of layers in the encoder and decoder (eg, n_layers=3 mean
      three encoder layers and three decoder layers).
    use_auto_acts: Whether to use automatic bounds calculation for activations
      (True) or fixed bounds.
    fp_quant: Whether to use floating point quantization. Defaults to False for
      integer quantization.

  Returns:
    A ConfigDict instance suitable for WMT training.
  """
  config = config_schema.get_config(
      use_auto_acts=use_auto_acts, n_layers=n_layers, fp_quant=fp_quant)
  config.half_shift = False
  config.update({
      "learning_rate_schedule": {
          "factors": "constant * linear_warmup * rsqrt_decay",
          "base_learning_rate": 0.0625,
          "warmup_steps": 1000,
          "decay_factor": 0.5,
          "steps_per_decay": 20000,
          "steps_per_cycle": 100000,
      },
      "per_host_batch_size": 256,
      "num_train_steps": 200000,
      "weight_decay": 0.25,
      "beta1": 0.9,
      "beta2": 0.98,
      "eps": 1e-9,
      "random_seed": 0,
      "hardware_rng": True,
      "activation_bound_update_freq": -1,
      "activation_bound_start_step": -1,
      "weight_outlier_regularization": 0.0,
      "prefer_int8_to_int32_dot": True,
      "model_hparams": {
          "emb_dim": 1024,
          "num_heads": 16,
          "qkv_dim": 1024,
          "mlp_dim": 4096,
          "share_embeddings": True,
          "logits_via_embedding": True,
      },
      "weight_outlier_regularization_regex": "^.*kernel$",
      "weight_quant_granularity": "per_channel"
  })
  if not fp_quant:
    config.prec = None
    config.quant_type = "aqt"
  else:
    config.prec.is_scaled = False
    config.quant_type = "fake_quant"
  layernorm_config = config_schema.get_layer_norm_config(
      quantized=False, quantized_reductions=False)
  config = config_schema.set_global_layer_norm_config(config, layernorm_config)
  return config


def get_auto_acts_config(n_layers, fp_quant):
  """Returns config appropriate for automatic activation quantization."""
  config = get_base_config(
      n_layers=n_layers, use_auto_acts=True, fp_quant=fp_quant)
  config.quant_act.bounds.update({
      "initial_bound": -1.0,
      "stddev_coeff": 3.0,
      "absdev_coeff": 0.0,
      "mix_coeff": 1.0,
      "reset_stats": False,
      "ema_coeff": 0.1,
      "use_cams": False,
      "exclude_zeros": True,
      "use_mean_of_max": False,
      "granularity": "per_channel"
  })
  config.attention.quant_act.bounds.granularity = "per_tensor"
  config.weight_outlier_regularization = 1.0
  config.activation_bound_start_step = 20000
  return config


def get_weights_only_config(n_layers, fp_quant):
  """Returns config for weights-only quantization."""
  config = get_base_config(
      n_layers=n_layers, use_auto_acts=False, fp_quant=fp_quant)
  config.quant_act.bounds = None
  return config


def get_fixed_acts_config(n_layers, fp_quant):
  """Returns config for activation quantization with fixed bounds."""
  config = get_base_config(
      n_layers=n_layers, use_auto_acts=False, fp_quant=fp_quant)
  config.quant_act.bounds = 1.0
  config.mlp_block.dense_2.quant_act.bounds = 6.0
  config.attention.attn_acts.attn_act_v.bounds = 6.0
  config.attention.dense_out.quant_act.bounds = 2.0
  return config


def get_config(quant_target,
               n_layers,
               fp_quant = False):
  """Returns config for a given quantization target and layer count."""
  if quant_target == QuantTarget.weights_and_auto_acts:
    return get_auto_acts_config(n_layers=n_layers, fp_quant=fp_quant)
  elif quant_target == QuantTarget.weights_only:
    return get_weights_only_config(n_layers=n_layers, fp_quant=fp_quant)
  elif quant_target == QuantTarget.weights_and_fixed_acts:
    return get_fixed_acts_config(n_layers=n_layers, fp_quant=fp_quant)
  elif quant_target == QuantTarget.none:
    return get_base_config(
        n_layers=n_layers, use_auto_acts=False, fp_quant=fp_quant)
  else:
    raise ValueError(f"quant_target {quant_target} not understood.")
