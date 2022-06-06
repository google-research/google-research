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

"""Config for distributed puddle world experiments."""
from ml_collections import config_dict
from aux_tasks.grid import utils


def get_config(method):
  """Default config."""
  batch_size = config_dict.FieldReference(64)
  alpha = config_dict.FieldReference(0.001)
  num_tasks = config_dict.FieldReference(50)

  env_config = config_dict.ConfigDict({
      'name':
          'pw',
      'task':
          'random_policy',
      'pw':
          config_dict.ConfigDict({
              'arena': 'sutton',
              'num_bins': 20,
              'rollout_length': 50,
              'gamma': 0.9,
              'samples_for_ground_truth': 100_000
          }),
      'gym':
          config_dict.ConfigDict({
              'id': None,
          }),
      'random':
          config_dict.ConfigDict({
              'num_states': 2048,
              'num_actions': 5
          })
  })

  eval_config = config_dict.ConfigDict({
      'num_tasks': 1000,
      'seed': 32,
      'learning_rate': 1e-3
  })

  methods = {
      'naive':
          config_dict.ConfigDict({
              'method': 'naive',
              'stop_grad': False,
              'rcond': 1e-5,
              'module': config_dict.ConfigDict({'name': 'ImplicitModule'}),
          }),
      'naive++':
          config_dict.ConfigDict({
              'method': 'naive',
              'stop_grad': True,
              'rcond': 1e-5,
              'module': config_dict.ConfigDict({'name': 'ImplicitModule'}),
          }),
      'explicit':
          config_dict.ConfigDict({
              'method':
                  'explicit',
              'module':
                  config_dict.ConfigDict({
                      'name': 'ExplicitModule',
                      'num_tasks': num_tasks
                  })
          }),
      'naive_implicit':
          config_dict.ConfigDict({
              'method': 'naive_implicit',
              'module': config_dict.ConfigDict({'name': 'ImplicitModule'}),
              'alpha': alpha,
          }),
      'implicit':
          config_dict.ConfigDict({
              'method':
                  'implicit',
              'module':
                  config_dict.ConfigDict({'name': 'ImplicitModule'}),
              'alpha':
                  alpha,
              'batch_sizes':
                  utils.ImplicitBatchSizes(
                      main=batch_size.get(),
                      cov=batch_size.get(),
                      weight=batch_size.get())
          }),
  }

  # TODO(joshgreaves): Use smaller embedding dim.
  encoder_config = config_dict.ConfigDict({
      'num_layers': 3,
      'num_units': 1024,
      'embedding_dim': 256
  })

  config = config_dict.ConfigDict()

  config.train = methods.get(method)
  config.env = env_config
  config.eval = eval_config
  config.encoder = encoder_config
  config.alpha = alpha

  config.num_tasks = num_tasks

  config.seed = 0
  config.learning_rate = 0.00005

  config.num_train_steps = 100_000
  config.num_eval_steps = 50_000

  config.num_eval_points = 1_200  # 3 points for each square in a 20x20.
  config.num_eval_rollouts = 100  # The number of rollouts to average over.

  config.log_metrics_every = 50

  if config.train.method == 'implicit' and (batch_sizes := config.train.get(
      'batch_sizes', None)):
    config.batch_size = batch_sizes.samples_required
  else:
    config.batch_size = batch_size

  return config


def get_hyper(h, method):
  """Get hyper sweep."""

  sweeps = {'implicit': [h.sweep('alpha', h.discrete([0.01, 0.1, 0.25, 0.5]))]}

  return h.product([
      h.sweep('num_tasks', h.discrete([100, 1_000, 10_000, 100_000,
                                       1_000_000])),
      # h.sweep('batch_size', h.discrete([32, 64, 128])),
      h.sweep('learning_rate', h.discrete([1e-4, 5e-4, 1e-3, 5e-3])),
      *sweeps.get(method, [])
  ])
