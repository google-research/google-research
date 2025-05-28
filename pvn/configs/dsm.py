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

"""Default configuration.

Note: These are not necessarily the parameters used for all experiments,
but are a sane default starting point.
"""

from ml_collections import config_dict


def _get_offline_config(
    config,
):  # pyformat: disable
  """Get offline/train config."""
  offline = config_dict.ConfigDict()
  offline.num_auxiliary_tasks = 100
  offline.batch_size = 256

  # Atari agent performs 200M iterations of 1M frames per iteration.
  # 1M frames / frame skip of 4 = 250k agent steps
  # 250k agent steps / 4 agent steps per SGD update
  #     = 62,500 SGD updates per 1M frames
  # Therefore, we would see 62,500 * 200 * 32 = 400,000,000 transitions
  offline.num_grad_updates = 25 * 62_500

  offline.discount = config.get_ref('discount')
  offline.target_params_update_every = None
  offline.target_params_soft_update_tau = 0.99

  # Log 10 times every iteration
  offline.log_metrics_every = 62_500 // 10
  # Checkpoint every five iteration
  offline.checkpoint_every = 62_500 * 5

  # === Dataset ===
  offline.dataset = config_dict.ConfigDict()
  offline.dataset.name = 'atari'
  offline.dataset.game = config.get_ref('game')
  offline.dataset.run = config.get_ref('run')
  offline.dataset.batch_size = offline.get_ref('batch_size')
  offline.dataset.split = 'full'

  # === DSM Model ===
  offline.model = config_dict.ConfigDict()
  offline.model.name = 'DsmNetwork'
  offline.model.num_auxiliary_tasks = offline.get_ref('num_auxiliary_tasks')

  # === DSM Indicator Functions ===
  indicator_type = config_dict.FieldReference('rnd', str)
  target_reward_proportion = config_dict.FieldReference(0.01, float)

  offline.indicator = config_dict.ConfigDict()
  offline.indicator.target_reward_proportion = target_reward_proportion
  offline.indicator.type = indicator_type
  offline.indicator.module = config_dict.ConfigDict()

  if indicator_type.get() == 'hash':
    offline.indicator.module.name = 'StackedMultiplyShiftHashIndicator'
    offline.indicator.module.target_reward_proportion = target_reward_proportion
    offline.indicator.module.num_auxiliary_tasks = offline.get_ref(
        'num_auxiliary_tasks'
    )
  elif indicator_type.get() == 'rnd':
    offline.indicator.module.name = 'StackedNatureDqnIndicator'
    offline.indicator.module.num_auxiliary_tasks = offline.get_ref(
        'num_auxiliary_tasks'
    )
    offline.indicator.module.tasks_per_module = 10
    offline.indicator.module.width_multiplier = 1.0
    # 1 iteration warmup period
    offline.indicator.num_qr_warmup_steps = 1 * 62_500
    # Allow num_grad_updates QR steps
    offline.indicator.num_qr_steps = offline.get_oneway_ref('num_grad_updates')
    offline.indicator.train_on_unthresholded_rewards = False
  else:
    raise ValueError(f'Invalid indicator type {indicator_type}.')

  # === DSM Optimizer ===
  offline.optim = config_dict.ConfigDict()
  offline.optim.name = 'adam'
  offline.optim.learning_rate = 6.25e-5 * (
      (offline.get_ref('batch_size') // 32) ** (1 / 4)
  )
  offline.optim.b1 = 0.9
  offline.optim.b2 = 0.999
  offline.optim.eps = 1.5e-4

  return offline


def _get_online_config(
    config,
):
  """Get online/eval config."""
  online = config_dict.ConfigDict()
  online.game = config.get_ref('game')
  online.agent = 'dqn'
  online.use_distributional_rl = False
  online.num_hidden_layers = 0
  online.hidden_layer_width = 512
  # PER exponent settings are in `online.py`
  online.use_prioritized_replay = False
  online.num_steps = 1_000_000

  online.dqn = config_dict.ConfigDict()
  online.dqn.epsilon = 0.01
  online.dqn.eval_epsilon = 1e-3
  online.dqn.learning_rate = 6.25e-5
  online.dqn.adam_eps = 1.5e-4
  online.dqn.discount = config.get_ref('discount')
  online.dqn.n_step = 5
  online.dqn.target_update_period = 2_000
  online.dqn.max_gradient_norm = 10.0
  online.dqn.batch_size = 32
  online.dqn.min_replay_size = 2_000
  online.dqn.max_replay_size = 1_000_000
  online.dqn.prefetch_size = 4
  online.dqn.samples_per_insert = 8  # 1 batches every 4 env steps
  online.dqn.samples_per_insert_tolerance_rate = 0.1
  online.dqn.num_sgd_steps_per_step = 1

  return online


def _get_encoder_config(_):
  encoder = config_dict.ConfigDict()
  encoder.name = 'ImpalaEncoder'
  encoder.num_features = 4096
  encoder.width_multiplier = 8.0

  return encoder


def get_config():
  """Get config."""

  config = config_dict.ConfigDict()
  config.run = 1
  config.seed = config.get_oneway_ref('run')
  config.game = 'Pong'
  config.discount = 0.99

  config.encoder = _get_encoder_config(config)
  config.offline = _get_offline_config(config)
  config.online = _get_online_config(config)

  return config
