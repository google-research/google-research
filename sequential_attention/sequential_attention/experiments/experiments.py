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

"""Run feature selection with Sequential Attention."""

import os

from absl import app

from sequential_attention.experiments import hyperparams_sa


def get_cmd(params):
  """Get command for running experiments with input parameters."""
  experiments_dir = os.path.dirname(os.path.realpath(__file__))
  experiment_file = os.path.join(experiments_dir, 'run.py')
  cmd = ['python', experiment_file]
  for param in params:
    cmd.append(f'--{param}={params[param]}')
  return ' '.join(cmd)


def main(_):
  base_name = 'experiment'
  base_dir = '/tmp/model_dir'
  parameters = []
  def get_params(seed, name):
    return {
        'data_name': name,
        'algo': 'sa',  # sa, lly, seql, gl, omp
        'deep_layers': '67',
        'alpha': 0,
        'batch_size': hyperparams_sa.BATCH[name],
        'num_epochs_select': hyperparams_sa.EPOCHS[name],
        'num_epochs_fit': hyperparams_sa.EPOCHS_FIT[name],
        'learning_rate': hyperparams_sa.LEARNING_RATE[name],
        'decay_steps': hyperparams_sa.DECAY_STEPS[name],
        'decay_rate': hyperparams_sa.DECAY_RATE[name],
        'num_selected_features': 50,
        'seed': seed,
        'enable_batch_norm': True,
        'num_inputs_to_select_per_step': 1,
        'model_dir': f'{base_dir}/{name}/{base_name}_seed_{seed}/',
    }

  for seed in [1, 2, 3, 4, 5]:
    parameters += [
        get_params(seed, name)
        for name in ['mice', 'mnist', 'fashion', 'isolet', 'coil', 'activity']
    ]

  for params in parameters:
    cmd = get_cmd(params)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
  app.run(main)
