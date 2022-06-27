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

"""Config overrides."""

from ipagnn.config import constants


def overrides_debug(config):
  config.debug = True


def overrides_nodebug(config):
  config.debug = False


def overrides_local(config):
  """Configs for iterating quickly on a workstation."""
  config.debug = True
  config.dataset.batch_size = min(3, config.dataset.batch_size)
  config.logging.summary_freq = 1
  config.logging.save_freq = 10
  config.eval_steps = 15
  config.model.hidden_size = 60
  config.model.rnn_cell.layers = 2


def overrides_2step(config):
  config.train.total_steps = 2


def overrides_20step(config):
  config.train.total_steps = 20


def overrides_eval(config, xid=None):
  config.launcher.experiment_kind = constants.MULTI_DATASET_EVAL_JOB
  config.runner.model_config = 'load'
  config.runner.dataset_config = 'keep'
  if xid is not None:
    config.analysis.xid = int(xid)
