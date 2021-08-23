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

"""TCN config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """TCN config."""

  config = _get_config()

  config.algorithm = "tcn"
  config.optim.train_max_iters = 4_000
  config.frame_sampler.strategy = "window"
  config.frame_sampler.num_frames_per_sequence = 40
  config.model.model_type = "resnet18_linear"
  config.model.normalize_embeddigs = False
  config.model.learnable_temp = False
  config.loss.tcn.pos_radius = 1
  config.loss.tcn.neg_radius = 4
  config.loss.tcn.num_pairs = 2
  config.loss.tcn.margin = 1.0
  config.loss.tcn.temperature = 0.1

  return config
