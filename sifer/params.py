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

"""Config for launching experiment."""

import ml_collections
import numpy as np
from sifer.utils import misc


def get_config():
  """Get the default configuration."""

  config = ml_collections.ConfigDict()
  config.debug = False
  config.dataset = 'Waterbirds'
  config.algorithm = 'ERM'
  config.output_folder_name = '/tmp/'
  config.experiment_dir = '/tmp/'
  config.data_dir = '/tmp/data'
  config.output_dir = '/tmp/output'
  config.hparams_seed = 0
  config.seed = 0
  config.steps = None

  config.es_metric = 'overall:adjusted_accuracy'
  config.pretrained = True
  config.checkpoint_freq = None

  # architectures and pre-training sources
  config.image_arch = 'resnet_sup_in1k'

  config.resnet18 = True
  config.nonlinear_classifier = False
  config.default_hparams = False
  config.last_layer_dropout = 0.

  config.lr = 1e-3
  config.weight_decay = 1e-4
  config.aux_depth = 2
  config.aux_width = 256
  config.aux_pos = 1
  config.forget_after_iters = 5
  config.aux_lr = 1e-2
  config.forget_lr = 1e-2
  config.aux_weight_decay = 1e-4
  config.batch_size = 32
  return config


def update_config(config):
  """Update config."""
  algorithm = config.algorithm
  # dataset = config.dataset

  def _hparam(name, default_value, random_val_fn):
    if config.hparams_seed == 0:
      return default_value
    random_state = np.random.RandomState(
        misc.seed_hash(config.hparams_seed, name)
    )
    return random_val_fn(random_state)

  config.lr = _hparam('lr', 1e-3, lambda r: 10 ** r.uniform(-4, -1))
  config.weight_decay = _hparam(
      'weight_decay', 1e-4, lambda r: 10 ** r.uniform(-6, -3)
  )
  if 'SiFeR' in algorithm:
    config.aux_pos = _hparam('aux_pos', 1, lambda r: r.choice([1, 2, 3, 4]))
    config.aux_depth = _hparam('aux_depth', 2, lambda r: r.uniform(1, 6))
    config.aux_width = _hparam('aux_width', 256, lambda r: 256)
    config.forget_after_iters = _hparam('forget_after_iters', 5, lambda r: 15)
    config.aux_lr = _hparam('aux_lr', 1e-2, lambda r: 10 ** r.uniform(-4, -1))
    config.forget_lr = _hparam(
        'forget_lr', 1e-2, lambda r: 10 ** r.uniform(-4, -1)
    )
    config.aux_weight_decay = _hparam(
        'aux_weight_decay', 1e-4, lambda r: 10 ** r.uniform(-6, -3)
    )
  return config
