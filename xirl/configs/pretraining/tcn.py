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

from configs.pretraining.default import get_config as _get_config


def get_config():
  """TCN config."""

  config = _get_config()

  config.ALGORITHM = "tcn"
  config.OPTIM.TRAIN_MAX_ITERS = 4_000
  config.FRAME_SAMPLER.STRATEGY = "window"
  config.FRAME_SAMPLER.NUM_FRAMES_PER_SEQUENCE = 40
  config.MODEL.MODEL_TYPE = "resnet18_linear"
  config.MODEL.NORMALIZE_EMBEDDINGS = False
  config.MODEL.LEARNABLE_TEMP = False
  config.LOSS.TCN.POS_RADIUS = 1
  config.LOSS.TCN.NEG_RADIUS = 4
  config.LOSS.TCN.NUM_PAIRS = 2
  config.LOSS.TCN.MARGIN = 1.0
  config.LOSS.TCN.TEMPERATURE = 0.1

  return config
