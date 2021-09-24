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

"""LIFS config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """LIFS config."""

  config = _get_config()

  config.algorithm = "lifs"
  config.optim.train_max_iters = 8_000
  config.frame_sampler.strategy = "variable_strided"
  config.model.model_type = "resnet18_linear_ae"
  config.model.embedding_size = 32
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False
  config.loss.lifs.temperature = 0.1
  config.eval.downstream_task_evaluators = [
      "reward_visualizer",
      "kendalls_tau",
      "reconstruction_visualizer",
  ]

  return config
