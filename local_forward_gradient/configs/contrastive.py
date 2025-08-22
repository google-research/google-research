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
"""Contrastive config."""
import ml_collections


def get_config():
  # For ImageNet L/32/64
  config = ml_collections.ConfigDict()
  config.dataset = 'imagenet2012'
  config.exp = 'forward_grad_activations'
  config.mom = 0.9
  config.lr = 0.1
  config.num_epochs = 800
  config.schedule = 'warmup_cosine'
  config.warmup_epochs = 10
  config.init_scheme = 'kaiming'
  config.num_groups = 64
  config.wd = 1e-4
  config.num_blocks = 4
  config.num_patches = 32
  config.num_channel_mlp_units = 2048
  config.num_passes = 1
  config.optimizer = 'lars'
  config.batch_size = 128  # 128x16
  config.aug = True
  config.head_lr = 1.0
  config.begin_ln = True
  config.middle_ln = True
  config.last_layer_ln = True
  config.area_lb = 0.08
  config.use_gcs = False
  config.linear_scale = True
  return config
