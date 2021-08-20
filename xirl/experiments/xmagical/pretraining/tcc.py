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

from configs.pretrain_default import get_config as _get_config


def get_config():
  """TCC config."""

  config = _get_config()

  config.algorithm = "tcc"
  config.optim.train_max_iters = 4_000
  config.frame_sampler.strategy = "uniform"
  config.frame_sampler.uniform_sampler.offset = 0
  config.frame_sampler.num_frames_per_sequence = 40
  config.model.model_type = "resnet18_linear"
  config.model.embedding_size = 32
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False
  config.loss.tcc.stochastic_matching = False
  config.loss.tcc.loss_type = "regression_mse"
  config.loss.tcc.similarity_type = "l2"
  config.loss.tcc.softmax_temperature = 1.0

  return config
