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

"""TCC config."""

from configs.pretraining.default import get_config as _get_config


def get_config():
  """TCC config."""

  config = _get_config()

  config.ALGORITHM = "tcc"
  config.OPTIM.TRAIN_MAX_ITERS = 4_000
  config.FRAME_SAMPLER.STRATEGY = "uniform"
  config.FRAME_SAMPLER.UNIFORM_SAMPLER.OFFSET = 0
  config.FRAME_SAMPLER.NUM_FRAMES_PER_SEQUENCE = 40
  config.MODEL.MODEL_TYPE = "resnet18_linear"
  config.MODEL.EMBEDDING_SIZE = 32
  config.MODEL.NORMALIZE_EMBEDDINGS = False
  config.MODEL.LEARNABLE_TEMP = False
  config.LOSS.TCC.STOCHASTIC_MATCHING = False
  config.LOSS.TCC.LOSS_TYPE = "regression_mse"
  config.LOSS.TCC.SIMILARITY_TYPE = "l2"
  config.LOSS.TCC.SOFTMAX_TEMPERATURE = 1.0

  return config
