# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Default config for training implicit models."""

import ml_collections


def get_config():
  """Default configs for the experiments."""
  config = ml_collections.ConfigDict()

  # Dataset Configs
  config.dataset = get_dataset_config()
  # Model Configs
  config.model = get_model_config()
  # Training Configs
  config.train = get_train_config()

  config.loss = get_loss_config()
  # Evaluation Configs
  config.eval = get_eval_config()
  # Render Configs
  config.render = get_render_config()

  config.seed = 3
  config.dev_run = False

  config.trial = 0  # Dummy for repeated runs.
  config.lock()
  return config


def get_dataset_config():
  """Configs for the dataset."""
  dataset_config = ml_collections.ConfigDict()

  # We set the base directories for different datasets here.

  dataset_config.name = "davis"
  dataset_config.basedir = ""
  dataset_config.scene = "longboard"
  dataset_config.num_objects = -1

  dataset_config.batch_size = 4

  # Depths.
  dataset_config.min_depth = -1.0
  dataset_config.max_depth = -1.0

  # Image resolution.
  dataset_config.image_height = -1
  dataset_config.image_width = -1
  dataset_config.white_bkgd = False

  return dataset_config


def get_model_config():
  """Configurations for the model."""
  model_config = ml_collections.ConfigDict()

  model_config.name = "ldi"

  # -------------------------------------------
  # Unet Parameters.
  model_config.unet_feat_scales = (1, 2, 4, 8, 8)
  model_config.unet_out_dim = 32
  model_config.unet_num_res_blocks = 3

  model_config.num_ldi_layers = 1
  model_config.use_rts = True

  return model_config


def get_train_config():
  """Configurations relating to training."""
  train_config = ml_collections.ConfigDict()

  train_config.scheduler = "cosine"
  train_config.lr_init = 2.0e-3
  train_config.warmup_epochs = 2
  train_config.weight_decay = 0.0
  train_config.warmup_steps = 2500
  train_config.lr_final = 2.0e-5
  # A multiplier on the learning rate when the step
  # is < lr_delay_steps
  train_config.lr_delay_mult = 0.1

  # The gradient clipping magnitude (disabled if == 0).
  train_config.grad_max_norm = 0
  train_config.grad_max_val = 0
  train_config.max_steps = 100000
  train_config.switch_steps = 15000
  train_config.num_epochs = 2000
  train_config.checkpoint_every_steps = 1000
  train_config.log_loss_every_steps = 500
  train_config.render_every_steps = 10000
  train_config.gc_every_steps = 10000

  train_config.overfit = False
  train_config.use_gt_rgb = False
  train_config.use_gt_disp = False
  train_config.use_gt_mask = True
  train_config.use_gt_scene_flow = True
  train_config.crop_projection = False

  train_config.flow_warmup_steps = 15000

  return train_config


def get_loss_config():
  """Configurations for the loss."""
  loss_config = ml_collections.ConfigDict()
  loss_config.rgb_layer_alpha = 1.0
  loss_config.disp_layer_alpha = 1.0
  loss_config.mask_layer_alpha = 1.0
  loss_config.mask_l0_alpha = 1.0

  loss_config.op_l2_alpha = 1.0
  loss_config.op_cosine_alpha = 1.0

  loss_config.proj_rgb_alpha = 1.0
  loss_config.src_rgb_recon_alpha = 1.0
  loss_config.src_static_rgb_alpha = 1.0

  loss_config.disp_smooth_alpha = 0.05

  loss_config.layer_rgb_cons_alpha = 1.0

  loss_config.bg_rgb_cons_alpha = 1.0

  loss_config.bg_rgb_proj_alpha = 1.0

  # Futher away timestep.
  loss_config.proj_far_rgb_alpha = 1.0
  loss_config.proj_far_rgb_recon_alpha = 1.0

  loss_config.shadow_smooth_alpha = 0.05
  loss_config.fg_rgb_smooth_alpha = 0.05
  loss_config.fg_alpha_reg_l0_alpha = 0.0005
  loss_config.fg_alpha_reg_l1_alpha = 0.01
  loss_config.fg_overlap_alpha = 1.0
  loss_config.fg_mask_alpha = 0.05

  # mask alpha for 3d
  loss_config.fg_alpha_smooth_alpha = 0.05
  loss_config.fg_alpha_reg_l0_alpha = 0.0005
  loss_config.fg_alpha_reg_l1_alpha = 0.01
  loss_config.mask_alpha = 0.05

  loss_config.src_rgb_grad_alpha = 0.3
  loss_config.recover_bg_alpha = 1.0

  return loss_config


def get_render_config():
  """Configurations for the model."""
  render_config = ml_collections.ConfigDict()
  render_config.soft_zbuff = True

  return render_config


def get_eval_config():
  """Configuration relation to model evaluation."""
  eval_config = ml_collections.ConfigDict()

  eval_config.eval_once = False
  eval_config.save_output = True

  return eval_config


