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

from configs.pretrain_default import get_config as _get_config


def get_config():
  """LIFS config."""

  config = _get_config()

  config.ALGORITHM = "lifs"
  config.OPTIM.TRAIN_MAX_ITERS = 8_000
  config.FRAME_SAMPLER.STRATEGY = "variable_strided"
  config.MODEL.MODEL_TYPE = "resnet18_linear_ae"
  config.MODEL.EMBEDDING_SIZE = 32
  config.MODEL.NORMALIZE_EMBEDDINGS = False
  config.MODEL.LEARNABLE_TEMP = False
  config.LOSS.LIFS.TEMPERATURE = 0.1
  config.EVAL.DOWNSTREAM_TASK_EVALUATORS = [
      "reward_visualizer",
      "kendalls_tau",
      "reconstruction_visualizer",
  ]

  return config
