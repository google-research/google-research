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

"""Config for developer run"""

from light_field_neural_rendering.configs import defaults


def get_config():
  """Default configs for the experiments"""
  config = defaults.get_config()

  config.dev_run = True
  # For Dataset
  # Undistort
  # Change to the directory conatining the scenes
  config.dataset.base_dir = "path/to/dataset"
  config.dataset.scene = "scene_name"  # Specify the name of the scene
  config.dataset.name = "ff_epipolar"
  config.dataset.factor = 8

  config.dataset.batching = "single_image"
  config.dataset.batch_size = 4
  config.dataset.num_interpolation_views = 2
  # For models
  config.model.name = "lfnr"
  config.model.net_depth = 1
  config.model.net_width = 4
  config.model.net_depth_condition = 1
  config.model.net_width_condition = 1
  config.model.max_deg_point = 1
  config.model.deg_view = 2
  config.model.num_coarse_samples = 2
  config.model.num_fine_samples = 4
  config.model.transformer_layers = 1
  config.model.transformer_heads = 1
  config.model.qkv_dim = 3
  config.model.transformer_mlp_dim = 3

  config.model.num_projections = 6
  config.model.use_learned_embedding = True
  config.model.learned_embedding_mode = "concat"
  config.model.mask_invalid_projection = True
  config.model.use_conv_features = True
  config.model.conv_feature_dim = (2,)

  # For training
  config.train.max_steps = 5
  config.train.warmup_steps = 1
  config.train.num_epochs = 1
  config.train.warmup_epochs = 0
  # For logging and eval
  config.train.checkpoint_every_steps = 1
  config.train.log_loss_every_steps = 1
  config.train.render_every_steps = 2
  # For eval
  config.eval.eval_once = True

  return config
