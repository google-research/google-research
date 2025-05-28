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

"""Run Dichotomy of Control on Bandit."""

import getpass
from absl import app
from absl import flags

import numpy as np
import os
import tensorflow.compat.v2 as tf

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment

from dice_rl.environments.bernoulli_bandit import BernoulliBandit
from dice_rl.environments import suites
from dice_rl.estimators import estimator as estimator_lib
import dice_rl.utils.common as common_utils
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_agents_onpolicy_dataset import TFAgentsOnpolicyDataset
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
from dichotomy_of_control import utils
from dichotomy_of_control.models.tabular_bc import TabularBC
from dichotomy_of_control.models.tabular_dt import TabularDT
from dichotomy_of_control.models.tabular_sdt import TabularSDT

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'bernoulli_bandit', 'Environment name.')
flags.DEFINE_integer('env_seed', 0, 'Env random seed.')
flags.DEFINE_integer('num_trajectory', 1000,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 1,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.1, 'How close to target policy.')
flags.DEFINE_bool('tabular_obs', True, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '/tmp/dichotomy_of_control',
                    'Directory to load dataset from.')

flags.DEFINE_string('algo_name', 'doc', 'One of [doc, bc, bc_pos, dt].')

flags.DEFINE_float('prior_weight', 1., 'Weight of prior.')
flags.DEFINE_float('energy_weight', 1., 'Weight of energy loss.')

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_integer('num_steps', 10_000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 1000,
                     'Number of pretraining training steps.')
flags.DEFINE_integer('seed', 1, 'Training random seed.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')


def get_onpolicy_dataset(env_name, tabular_obs, policy_fn, policy_info_spec):
  """Gets target policy."""
  if env_name == 'bernoulli_bandit':
    env = BernoulliBandit(bernoulli_prob=FLAGS.alpha)
    env = gym_wrapper.GymWrapper(env)
  elif 'FrozenLake-v1' in env_name:
    if len(env_name.split('-')) == 3:
      no_slip_prob = float(env_name.split('-')[-1])
      env_name = 'FrozenLake-v1'
    else:
      no_slip_prob = 1. / 3
    env = suites.load_gym(env_name, gym_kwargs={'no_slip_prob': no_slip_prob})
    env.seed(FLAGS.env_seed)
  else:
    raise ValueError('Unknown environment: %s.' % env_name)

  tf_env = tf_py_environment.TFPyEnvironment(env)
  tf_policy = common_utils.TFAgentsWrappedPolicy(
      tf_env.time_step_spec(),
      tf_env.action_spec(),
      policy_fn,
      policy_info_spec,
      emit_log_probability=True)

  return TFAgentsOnpolicyDataset(tf_env, tf_policy)


def main(argv):
  env_name = FLAGS.env_name
  env_seed = FLAGS.env_seed
  tabular_obs = FLAGS.tabular_obs
  num_trajectory = FLAGS.num_trajectory
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  load_dir = FLAGS.load_dir
  algo_name = FLAGS.algo_name
  learning_rate = FLAGS.learning_rate
  num_steps = FLAGS.num_steps
  batch_size = FLAGS.batch_size
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                    ENV_NAME=env_name,
                    TAB=tabular_obs,
                    ALPHA=alpha,
                    SEED=env_seed,
                    NUM_TRAJ=num_trajectory,
                    MAX_TRAJ=max_trajectory_length)
  directory = os.path.join(load_dir, hparam_str)
  print('Loading dataset.')
  dataset = Dataset.load(directory)
  print('num loaded steps', dataset.num_steps)
  print('num loaded total steps', dataset.num_total_steps)
  print('num loaded episodes', dataset.num_episodes)
  print('num loaded total episodes', dataset.num_total_episodes)
  estimate = estimator_lib.get_fullbatch_average(dataset, by_steps=False)
  print('data per step avg', estimate)

  hparam_dict = {
      'env_name': env_name,
      'algo_name': algo_name,
      'alpha': alpha,
      'env_seed': env_seed,
      'seed': FLAGS.seed,
      'num_trajectory': num_trajectory,
      'max_trajectory_length': max_trajectory_length,
      'learning_rate': learning_rate,
      'prior_weight': FLAGS.prior_weight,
      'energy_weight': FLAGS.energy_weight,
  }

  pos_only = False
  if 'bc' in algo_name:
    algo = TabularBC(dataset.spec, learning_rate=learning_rate)
    if algo_name == 'bc_pos':
      pos_only = True
  elif algo_name == 'dt':
    algo = TabularDT(dataset.spec, learning_rate=learning_rate)
  elif algo_name == 'doc':
    algo = TabularSDT(
        dataset.spec,
        learning_rate=learning_rate,
        energy_weight=FLAGS.energy_weight)
    algo.prepare_dataset(dataset)
  else:
    raise ValueError('algo %s not supported' % algo_name)
  dataset = utils.convert_to_tf_dataset(
      dataset, max_trajectory_length, batch_size, pos_only=pos_only)
  data_iter = iter(dataset)

  for step in range(num_steps):
    batch = next(data_iter)
    info_dict = algo.train_step(batch)
    if step % FLAGS.eval_interval == 0:
      for k, v in info_dict.items():
        print(k, v)

      policy_fn, policy_info_spec = algo.get_policy()
      onpolicy_data = get_onpolicy_dataset(env_name, tabular_obs, policy_fn,
                                           policy_info_spec)
      onpolicy_episodes, valid_steps = onpolicy_data.get_episode(
          num_trajectory,
          truncate_episode_at=max_trajectory_length)

      mask = ((1 - tf.cast(onpolicy_episodes.is_last(), tf.float32)) *
              tf.cast(valid_steps, tf.float32))
      episode_reward = np.mean(
          np.sum(onpolicy_episodes.reward * mask, axis=-1))
      print('eval/reward', episode_reward)


if __name__ == '__main__':
  app.run(main)
