# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Config overrides."""

from ipagnn.config import constants


def overrides_paper_partial(config):
  """IPAGNN paper experiment, partial programs."""
  config.launcher.sweep = 'sweep_paper'
  config.runner.restart_behavior = 'abort'  # abort or restore
  config.launcher.experiment_kind = constants.JAX_DATASET_COMPARISON

  config.setup.setup_dataset = True
  config.setup.setup_model = True

  config.dataset.name = 'control_flow_programs/decimal-large-state-L10-partial'
  config.launcher.eval_dataset_names = ','.join([
      'control_flow_programs/decimal-large-state-L2-partial',
      'control_flow_programs/decimal-large-state-L5-partial',
      'control_flow_programs/decimal-large-state-L10-partial',
      'control_flow_programs/decimal-large-state-L20-partial',
      'control_flow_programs/decimal-large-state-L30-partial',
      'control_flow_programs/decimal-large-state-L40-partial',
      'control_flow_programs/decimal-large-state-L50-partial',
      'control_flow_programs/decimal-large-state-L60-partial',
      'control_flow_programs/decimal-large-state-L70-partial',
      'control_flow_programs/decimal-large-state-L80-partial',
      'control_flow_programs/decimal-large-state-L90-partial',
      'control_flow_programs/decimal-large-state-L100-partial',
  ])
  config.dataset.in_memory = False
  config.dataset.max_examples = 3 * 3500000

  config.model.name = 'IPAGNN'
  config.model.hidden_size = 200

  config.dataset.batch_size = 32
  config.train.total_steps = 2000000
  config.eval_steps = 200
  config.opt.learning_rate = 3e-2
  config.opt.learning_rate_factors = 'constant'

  config.logging.summary_freq = 5000
  config.logging.save_freq = 5000


def overrides_paper_full(config):
  """IPAGNN paper experiment, full programs."""
  config.launcher.sweep = 'sweep_paper'
  config.runner.restart_behavior = 'abort'  # abort, restore, replace, new
  config.launcher.experiment_kind = constants.JAX_DATASET_COMPARISON

  config.setup.setup_dataset = True
  config.setup.setup_model = True

  config.dataset.name = 'control_flow_programs/decimal-large-state-L10'
  config.launcher.eval_dataset_names = ','.join([
      'control_flow_programs/decimal-large-state-L2',
      'control_flow_programs/decimal-large-state-L5',
      'control_flow_programs/decimal-large-state-L10',
      'control_flow_programs/decimal-large-state-L20',
      'control_flow_programs/decimal-large-state-L30',
      'control_flow_programs/decimal-large-state-L40',
      'control_flow_programs/decimal-large-state-L50',
      'control_flow_programs/decimal-large-state-L60',
      'control_flow_programs/decimal-large-state-L70',
      'control_flow_programs/decimal-large-state-L80',
      'control_flow_programs/decimal-large-state-L90',
      'control_flow_programs/decimal-large-state-L100',
  ])
  config.dataset.in_memory = False
  config.dataset.max_examples = 3 * 3500000

  config.model.name = 'IPAGNN'
  config.model.hidden_size = 200

  config.dataset.batch_size = 32
  config.train.total_steps = 2000000
  config.eval_steps = 200
  config.opt.learning_rate = 3e-2
  config.opt.learning_rate_factors = 'constant'

  config.logging.summary_freq = 5000
  config.logging.save_freq = 5000
