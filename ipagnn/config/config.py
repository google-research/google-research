# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Interpreter model config."""

import ml_collections

from ipagnn.config import constants
from ipagnn.config import overrides_lib


Config = ml_collections.ConfigDict


def default_config():
  """Gets the default config for the interpreter model."""
  config = Config()
  config.overrides = ''
  config.debug = False

  config.setup = Config()
  config.setup.setup_dataset = True
  config.setup.setup_model = True

  config.logging = Config()
  config.logging.summary_freq = 500
  config.logging.save_freq = 500

  config.runner = Config()
  config.runner.mode = 'train'
  config.runner.method = 'supervised'
  config.runner.model_config = 'assert'  # keep, load, or assert
  config.runner.dataset_config = 'keep'  # keep, load, or assert
  config.runner.restart_behavior = 'restore'  # abort or restore

  config.checkpoint = Config()
  config.checkpoint.run_dir = ''
  config.checkpoint.path = ''
  config.checkpoint.id = 0

  config.dataset = Config()
  config.dataset.name = 'control_flow_programs/decimal-large-state-L10'
  config.dataset.version = 'default'  # Set to use an explicit dataset version.
  config.dataset.split = 'default'
  config.dataset.representation = 'code'  # code, trace
  config.dataset.max_length = 10000
  config.dataset.batch_size = 128
  config.dataset.batch = True
  config.dataset.in_memory = False
  config.dataset.max_examples = 0

  config.train = Config()
  config.train.total_steps = 0  # 0 means no limit.

  config.opt = Config()
  config.opt.learning_rate = 0.0003
  config.opt.learning_rate_factors = 'constant'
  config.opt.clip_by = 'global_norm'
  config.opt.clip_value = 5.0

  # Model configs.
  config.model = Config()
  config.model.name = 'IPAGNN'
  config.model.hidden_size = 200

  config.model.rnn_cell = Config()
  config.model.rnn_cell.layers = 2

  config.model.ipagnn = Config()
  config.model.ipagnn.checkpoint = True

  config.model.interpolant = Config()
  config.model.interpolant.init_with_code_embeddings = False
  config.model.interpolant.apply_code_rnn = False
  config.model.interpolant.apply_dense = False
  config.model.interpolant.apply_gru = True
  config.model.interpolant.use_b = True
  config.model.interpolant.use_p = False
  config.model.interpolant.use_ipa = True  # Alt. to use_parent_embeddings
  config.model.interpolant.use_parent_embeddings = False  # Alt. to use_ipa
  config.model.interpolant.use_child_embeddings = True
  config.model.interpolant.normalize = False
  config.model.interpolant.name = 'interpolant'  # Unused.

  config.initialization = Config()
  config.initialization.maxval = 1.0

  # Analysis configs.
  config.analysis = Config()
  config.analysis.xid = 0
  config.analysis.experiment_dir = ''

  # Other configs.
  config.eval_name = ''
  config.eval_steps = 1000
  # Number of seconds to wait without receiving checkpoint before timing out.
  config.eval_timeout = 30 * 60  # 30 minutes.

  # Launcher configs are meant to be used only by the xm launcher.
  config.launcher = Config()
  config.launcher.sweep = 'empty_sweep'  # See sweeps.py.
  config.launcher.experiment_kind = constants.TRAIN_AND_EVAL
  config.launcher.extra_tags = ''
  config.launcher.eval_dataset_names = ''
  config.launcher.max_work_units = 100

  config.index = 0
  return config


def adhoc_overrides(config):
  """Use this space for adhoc overrides. Nothing should be checked in here."""
  del config  # Unused.


def get_config():
  """Gets the config for the interpreter model."""
  config = default_config()
  override_names = config.overrides.split(',')
  overrides_lib.apply_overrides(config, override_names=override_names)
  adhoc_overrides(config)
  # default_overrides are the overrides that are already applied when you call
  # get_config. Additional overrides may be specified at the command line or by
  # the launcher by setting config.overrides.
  config.default_overrides = config.overrides
  config.lock()
  return config
