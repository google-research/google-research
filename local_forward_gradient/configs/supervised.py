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

# pylint: skip-file
"""Supervised config."""
import ml_collections


def get_config():
  # For ImageNet L/32/64
  config = ml_collections.ConfigDict()
  config.dataset = 'imagenet2012'
  config.exp = 'forward_grad_activations'
  config.mom = 0.9
  config.lr = 0.05
  config.num_epochs = 120
  config.schedule = 'warmup_cosine'
  config.warmup_epochs = 10
  config.init_scheme = 'kaiming'
  config.num_groups = 64
  config.wd = 1e-4
  config.num_blocks = 4
  config.num_patches = 32
  config.num_channel_mlp_units = 2048
  config.num_passes = 1
  config.optimizer = 'sgd'
  config.batch_size = 256
  config.aug = True
  config.head_lr = 1.0
  config.begin_ln = True
  config.middle_ln = True
  config.last_layer_ln = True
  config.inter_ln = True
  config.augcolor = True
  config.area_lb = 0.3
  config.use_gcs = False
  config.train_eval = True
  config.spatial_loss = True
  config.modular_loss = True
  return config
