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

r"""Compute SR on toy  environments and excess risk bounds.
"""

import collections
import json
import os
import pickle

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from generalization_representations_rl_aistats22.minigrid import approximation_error
from generalization_representations_rl_aistats22.minigrid import estimation_error
from generalization_representations_rl_aistats22.minigrid import rl_basics
from generalization_representations_rl_aistats22.minigrid import successor_representation
from generalization_representations_rl_aistats22.minigrid import toy_mdps


flags.DEFINE_string('base_dir', None,
                    'Base directory to store stats.')
flags.DEFINE_string(
    'env_name', 'Torus1dMDP',
    'Name of environment(s) to load/create. If None, will '
    'create a set of random MDPs.')
flags.DEFINE_integer('num_runs', 5, 'Number of runs for each MDP.')
flags.DEFINE_integer('num_states', 400, 'Number of states in MDP.')
flags.DEFINE_integer('num_samples', 300, 'Fixed number of samples used.')
flags.DEFINE_bool('verbose', False, 'Whether to print verbose messages.')
flags.DEFINE_multi_string('reward_type',
                          ['all_ones', 'one_hot', 'gaussian'],
                          'list of rewards used')
flags.DEFINE_float('gamma', 0.9, 'Discount factor.')
flags.DEFINE_string(
    'custom_base_dir_from_hparams', None,
    'If not None, will set the base_directory prefixed with '
    'the value of this flag, and a subdirectory using game '
    'name and hparam settings.')


FLAGS = flags.FLAGS

# Dictionary mapping env name to constructor.
MDPS = {
    'StarMDP': toy_mdps.StarMDP,
    'DisconnectedMDP': toy_mdps.DisconnectedMDP,
    'FullyConnectedMDP': toy_mdps.FullyConnectedMDP,
    'Torus1dMDP': toy_mdps.Torus1dMDP,
    'Torus2dMDP': toy_mdps.Torus2dMDP,
    'ChainMDP': toy_mdps.ChainMDP,
}


def get_rewardvector(num_states, reward_type):  # pylint: disable=invalid-name
  """Get a particular type of reward vector."""
  if reward_type == 'gaussian':
    reward_function = np.random.normal(size=(num_states))
    reward_function /= np.linalg.norm(
        reward_function, ord=np.inf)  # normalize to R_max=1
  if reward_type == 'one_hot':
    reward_function = np.zeros(num_states)
    reward_function[num_states-1] = 1.
  if reward_type == 'all_ones':
    reward_function = np.ones(num_states)
  return reward_function


def main(_):
  flags.mark_flags_as_required(['base_dir'])
  if FLAGS.custom_base_dir_from_hparams is not None:
    FLAGS.base_dir = os.path.join(FLAGS.base_dir,
                                  FLAGS.custom_base_dir_from_hparams)
  base_dir = os.path.join(FLAGS.base_dir, FLAGS.env_name, f'{FLAGS.num_states}',
                          f'{FLAGS.num_runs}')
  base_dir = os.path.join(base_dir, 'successor_representation')
  if not tf.io.gfile.exists(base_dir):
    tf.io.gfile.makedirs(base_dir)

  env = MDPS[FLAGS.env_name](FLAGS.num_states)
  # We add the discount factor to the environment.
  env.gamma = FLAGS.gamma

  k_range = np.arange(1, FLAGS.num_samples // 2+1)
  dimension = np.arange(1, env.num_states + 1)
  data = collections.defaultdict(dict)
  policy = rl_basics.policy_random(env)
  sr, _ = successor_representation.sr_closed_form(
      env, policy, gamma=env.gamma)
  F, sigma, _ = np.linalg.svd(sr)  # pylint: disable=invalid-name

  y = estimation_error.coherence_vec(dimension, F)
  y = y.reshape(1, -1)
  data['coherence'] = y
  estimation_error.plot_coherence(base_dir, data['coherence'])

  data['singular_values'] = sigma.reshape(1, env.num_states)
  approximation_error.plot_singular_values(base_dir, data['singular_values'])

  for reward_type in FLAGS.reward_type:
    reward_function = get_rewardvector(env.num_states, reward_type)
    values = sr @ reward_function

    estim_errors = np.array([
        estimation_error.generate_and_estime(FLAGS.num_samples, d, F, values,
                                             FLAGS.num_runs) for d in k_range
    ])
    data['estim_error'][reward_type] = estim_errors

    sty_errors = [
        estimation_error.stylized_theoretical_bound(F, d, values,
                                                    FLAGS.num_samples)
        for d in k_range
    ]
    data['sty_error'][reward_type] = sty_errors

    y = estimation_error.orth_proj_vec(dimension, F, values) / env.num_states
    data['approx_error'][reward_type] = y

    data['excess_risk'][reward_type] = estim_errors + y[:FLAGS.num_samples //
                                                        2].reshape(-1, 1)

  approximation_error.plot_approx_error_rewards(base_dir, dimension,
                                                data['approx_error'],
                                                FLAGS.reward_type,
                                                FLAGS.num_states, FLAGS.gamma,
                                                FLAGS.env_name)

  estimation_error.plot_excess_error_rewards(
      base_dir, k_range, data['excess_risk'], FLAGS.reward_type,
      FLAGS.num_samples, FLAGS.num_states, FLAGS.gamma, FLAGS.env_name)

  estimation_error.plot_sty_bound_rewards(base_dir, k_range,
                                          data['sty_error'],
                                          FLAGS.reward_type, FLAGS.num_samples,
                                          FLAGS.num_states, FLAGS.gamma,
                                          FLAGS.env_name)

  path = os.path.join(base_dir, 'data_' + '.pkl')
  with tf.gfile.GFile(path, 'wb') as f:
    pickle.dump(data, f)

if __name__ == '__main__':
  app.run(main)
