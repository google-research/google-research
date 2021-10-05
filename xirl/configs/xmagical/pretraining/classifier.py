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

"""Goal classifier config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """Goal classifier config."""

  config = _get_config()

  config.algorithm = "goal_classifier"
  config.optim.train_max_iters = 6_000
  config.frame_sampler.strategy = "last_and_randoms"
  config.frame_sampler.num_frames_per_sequence = 15
  config.model.model_type = "resnet18_classifier"
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False

  return config
