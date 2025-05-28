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
  # LF configs
  config.lightfield = get_lf_config()
  # Training Configs
  config.train = get_train_config()
  # Evaluation Configs
  config.eval = get_eval_config()

  config.seed = 33
  config.dev_run = False

  config.trial = 0  # Dummy for repeated runs.
  config.lock()
  return config


def get_dataset_config():
  """Configs for the dataset."""
  dataset_config = ml_collections.ConfigDict()

  # We set the base directories for different datasets here.

  dataset_config.name = "ff_epipolar"
  dataset_config.scene = ""
  #--------------------------------------------------------
  # llff + ibrnet collected scenes.
  #--------------------------------------------------------
  dataset_config.ff_base_dir = ""
  # The downsampling factor of images, 0 for no downsample
  dataset_config.ff_image_height = 756
  dataset_config.ff_image_width = 1008

  #--------------------------------------------------------
  # dtu.
  #--------------------------------------------------------
  dataset_config.dtu_base_dir = ""
  dataset_config.dtu_factor = 1
  dataset_config.dtu_light_idx = 3

  #--------------------------------------------------------
  # Eval FF.
  #--------------------------------------------------------
  dataset_config.eval_ff_dir = ""
  dataset_config.eval_ff_image_height = 756
  dataset_config.eval_ff_image_width = 1008

  #--------------------------------------------------------
  # Eval llff.
  #--------------------------------------------------------
  dataset_config.eval_llff_dir = ""
  dataset_config.eval_llff_image_height = 756
  dataset_config.eval_llff_image_width = 1008

  dataset_config.batch_size = 4096
  dataset_config.batching = "single_image"
  dataset_config.cam_transform = False

  # Render generated images if set to True
  dataset_config.render_path = False
  dataset_config.spherify = False
  # will take every 1/N images as LLFF test set.
  dataset_config.llffhold = 8
  # If True, generate rays through the center of each pixel.
  # Note: While this is the correct way to handle rays, it
  # is not the way rays are handled in the original NeRF paper.
  dataset_config.use_pixel_centers = False
  # to store height and width

  dataset_config.normalize = True
  dataset_config.num_interpolation_views = 10

  # For generalization evaluation
  dataset_config.eval_dataset = "llff"
  dataset_config.eval_scene = ""
  dataset_config.blender_factor = 0

  # For llff eval
  dataset_config.eval_llff_image_height = 756
  dataset_config.eval_llff_image_width = 1008

  return dataset_config


def get_model_config():
  """Configurations for the model."""
  model_config = ml_collections.ConfigDict()

  model_config.name = "gpnr"
  model_config.near = 0.
  model_config.far = 1.

  # add a skip connection to the output vector of every
  # skip_layer layers.
  model_config.skip_layer = 4
  model_config.num_rgb_channels = 3
  model_config.num_sigma_channels = 1
  model_config.randomized = False

  # Position encoding config
  model_config.mapping_type = "positional_encoding"
  # Min and max degree for positional encoding for points
  model_config.min_deg_point = 0
  model_config.max_deg_point = 10
  # Degree of positional encoding for view directions
  model_config.deg_view = 4
  model_config.net_activation = "relu"
  model_config.rgb_activation = "sigmoid"
  model_config.sigma_activation = "relu"

  model_config.white_bkgd = False

  #------------------------------------
  # For Transformer
  model_config.epi_transformer_layers = 8
  model_config.view_transformer_layers = 8
  model_config.transformer_heads = 1
  model_config.qkv_dim = 256
  model_config.transformer_mlp_dim = 256
  #------------------------------------
  # Epipolar conv features
  model_config.conv_feature_dim = (32,)
  model_config.patch_size = 11

  #--------------------------------------
  # For epipolar projection
  model_config.num_projections = 127
  model_config.interpolation_type = "rounding"
  model_config.mask_invalid_projection = False

  model_config.return_attn = False

  model_config.init_final_precision = "DEFAULT"
  model_config.normalize_ref_image = False

  # Predict RGB from interpolated features.
  model_config.predict_feature_rgb = True

  return model_config


def get_lf_config():
  """Configurations relationg to lf representation."""
  lf_config = ml_collections.ConfigDict()
  lf_config.name = "plucker"
  lf_config.encoding_name = "positional_encoding"
  # Min and max degree for positional encoding for points
  lf_config.min_deg_point = 0
  lf_config.max_deg_point = 10
  return lf_config


def get_train_config():
  """Configurations relating to training."""
  train_config = ml_collections.ConfigDict()

  train_config.pretrain_dir = ""
  train_config.switch_scene_iter = 500
  train_config.scheduler = "linear"
  train_config.lr_init = 2.0e-3
  train_config.warmup_epochs = 2
  train_config.weight_decay = 0.
  train_config.warmup_steps = 2500
  train_config.lr_final = 2.0e-5
  # train_config.lr_delay_steps = 2500
  # A multiplier on the learning rate when the step
  # is < lr_delay_steps
  train_config.lr_delay_mult = 0.1

  # The gradient clipping magnitude (disabled if == 0).
  train_config.grad_max_norm = 0
  train_config.grad_max_val = 0
  train_config.max_steps = 250000
  train_config.num_epochs = 180
  train_config.checkpoint_every_steps = 1000
  train_config.log_loss_every_steps = 500
  train_config.render_every_steps = 5000
  train_config.gc_every_steps = 10000

  return train_config


def get_eval_config():
  """Configuration relation to model evaluation."""
  eval_config = ml_collections.ConfigDict()

  eval_config.eval_once = False
  eval_config.save_output = True
  # the size of chunks for evaluation inferences,
  # set to the value that fits your GPU/TPU memory.
  eval_config.chunk = 4096
  eval_config.inference = False

  eval_config.mvsn_style = False
  eval_config.return_coarse = False
  eval_config.checkpoint_step = -1

  return eval_config


def get_hyper(h):
  return h.product([
      h.sweep(
          "ml_config.dataset.eval_scene",
          h.categorical([
              "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"
          ]))
  ])
