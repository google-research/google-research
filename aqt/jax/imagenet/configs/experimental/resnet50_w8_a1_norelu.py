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

import copy
import ml_collections
from aqt.jax.imagenet.configs import base_config
from aqt.jax.imagenet.configs.paper.resnet50_bfloat16 import get_config as bfloat16_paper_config
from aqt.jax.imagenet.configs.paper.resnet50_w4_a4_init8_dense8_auto import get_config as w4a4auto_paper_config
from aqt.jax.imagenet.configs.paper.resnet50_w8_a8_auto import get_config as w8a8auto_paper_config


def get_config(quant_target=base_config.QuantTarget.weights_and_auto_acts):
  """Gets Resnet50 config for 8 bits weights and 1 bit auto activation quantization.

  conv_init and last dense layer not quantized as these are the most
  sensitive layers in the model.

  Args:
   quant_target: quantization target, of type QuantTarget.

  Returns:
   ConfigDict instance.
  """

  def set_init_bound_coeff(field):
    # input should be a class field so that the changes in this function
    # will be global to the class even without a return value
    field.cams_coeff = 0.0
    field.cams_stddev_coeff = 0.0
    field.mean_of_max_coeff = 0.0
    field.stddev_coeff = 0.0
    field.absdev_coeff = 0.0
    field.fixed_bound = 0.0
    field.granularity = "per_channel"
    field.use_old_code = False

  def reset_bound_for_convinit_dense(config):
    # reset bound haparams for conv_init and dense layers
    # use mean_of_max to automatically calculate the bound values
    set_init_bound_coeff(config.model_hparams.dense_layer.quant_act.bounds)
    config.model_hparams.dense_layer.quant_act.bounds.initial_bound = -1
    config.model_hparams.dense_layer.quant_act.bounds.mean_of_max_coeff = 1.0
    set_init_bound_coeff(config.model_hparams.conv_init.quant_act.bounds)
    config.model_hparams.conv_init.quant_act.bounds.initial_bound = -1
    config.model_hparams.conv_init.quant_act.bounds.mean_of_max_coeff = 1.0
    return config

  # create an init config which the sweep configs will be based on
  config_init = base_config.get_config(
      imagenet_type=base_config.ImagenetType.resnet50,
      quant_target=quant_target)
  config_init.weight_prec = 8
  config_init.quant_act.prec = 1
  config_init.half_shift = True
  config_init.base_learning_rate = 2e-5
  config_init.activation_bound_start_step = 7500
  # set act function and shortcut method to each conv layer
  config_init.act_function = "none"
  config_init.shortcut_ch_shrink_method = "none"
  config_init.shortcut_ch_expand_method = "none"
  config_init.shortcut_spatial_method = "none"
  # set learning rate scheduler
  config_init.lr_scheduler.num_epochs = 250
  config_init.lr_scheduler.warmup_epochs = 5
  config_init.lr_scheduler.cooldown_epochs = 0
  config_init.lr_scheduler.scheduler = "linear"
  # -1 means no early stopping by default
  config_init.early_stop_steps = -1
  # optimizer params
  config_init.optimizer = "adam"
  config_init.adam.beta1 = 0.9
  config_init.adam.beta2 = 0.999
  # Conv_init and dense layers will have floating-point weights and acts
  config_init.model_hparams.conv_init.weight_prec = None
  config_init.model_hparams.conv_init.quant_act.prec = None
  config_init.model_hparams.dense_layer.weight_prec = None
  config_init.model_hparams.dense_layer.quant_act.prec = None
  # set all of the input distributions to "symmetric"
  config_init.model_hparams.dense_layer.quant_act.input_distribution = "symmetric"
  config_init.model_hparams.conv_init.quant_act.input_distribution = "symmetric"
  for residual_block in config_init.model_hparams.residual_blocks:
    residual_block.conv_1.quant_act.input_distribution = "symmetric"
    residual_block.conv_2.quant_act.input_distribution = "symmetric"
    residual_block.conv_3.quant_act.input_distribution = "symmetric"
    if residual_block.conv_proj is not None:
      residual_block.conv_proj.quant_act.input_distribution = "symmetric"
  # set bound hparams to all zero for activations
  # will update one of the bound hparams at a time in sweep configs
  set_init_bound_coeff(config_init.quant_act.bounds)
  # set initial bound value
  config_init.quant_act.bounds.initial_bound = 2.0
  # name of the experiment on TB
  config_init.metadata.hyper_str = "w8a1"

  # create a collection of config files for sweeping
  sweep_config = ml_collections.ConfigDict()
  configs = []

  # leaderboard configs for testing purpose
  configs.append(bfloat16_paper_config())
  configs.append(w8a8auto_paper_config())
  configs.append(w4a4auto_paper_config())

  # baseline: act_function [none, bprelu], no additional shortcuts
  for act_function in ["none", "bprelu"]:
    for fix_bound in [3.0]:
      config = copy.deepcopy(config_init)
      config.act_function = act_function
      config.quant_act.bounds.fixed_bound = fix_bound
      # reset bound haparams for conv_init and dense layers
      config = reset_bound_for_convinit_dense(config)
      config.metadata.hyper_str += f"_{act_function}_baseline"
      configs.append(config)

  # Turn on sc1, sc2, sc3
  # Sweep sc1, sc3 both with different methods
  for act_function in ["bprelu"]:
    for fix_bound in [3.0]:
      for shortcut_shrink_method in ["consecutive"]:
        for shortcut_expand_method in ["zeropad"]:
          config = copy.deepcopy(config_init)
          config.act_function = act_function
          config.quant_act.bounds.fixed_bound = fix_bound
          config.shortcut_ch_shrink_method = shortcut_shrink_method
          config.shortcut_ch_expand_method = shortcut_expand_method
          config.shortcut_spatial_method = "max_pool"
          # reset bound haparams for conv_init and dense layers
          config = reset_bound_for_convinit_dense(config)
          config.metadata.hyper_str += f"_{act_function}_sc123_shrink_{shortcut_shrink_method}_expand_{shortcut_expand_method}"
          configs.append(config)

  sweep_config.configs = configs
  return sweep_config
