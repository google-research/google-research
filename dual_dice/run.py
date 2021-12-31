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

"""Run off-policy policy evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

import dual_dice.algos.dual_dice as dual_dice
import dual_dice.algos.neural_dual_dice as neural_dual_dice
import dual_dice.gridworld.environments as gridworld_envs
import dual_dice.gridworld.policies as gridworld_policies
import dual_dice.transition_data as transition_data

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 1, 'Initial NumPy random seed.')
flags.DEFINE_integer('num_seeds', 1, 'How many seeds to run.')
flags.DEFINE_integer('num_trajectories', 200,
                     'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 400,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 0.0,
                   'Higher alpha corresponds to using '
                   'behavior policy that is closer to target policy.')
flags.DEFINE_float('gamma', 0.995, 'Discount factor.')
flags.DEFINE_bool('tabular_obs', True, 'Use tabular observations?')
flags.DEFINE_bool('tabular_solver', True, 'Use tabular solver?')
flags.DEFINE_string('env_name', 'grid', 'Environment to evaluate on.')
flags.DEFINE_string('solver_name', 'dice', 'Type of solver to use.')
flags.DEFINE_string('save_dir', None, 'Directory to save results to.')
flags.DEFINE_float('function_exponent', 1.5,
                   'Exponent for f function in DualDICE.')
flags.DEFINE_bool('deterministic_env', False, 'assume deterministic env.')
flags.DEFINE_integer('batch_size', 512,
                     'batch_size for training models.')
flags.DEFINE_integer('num_steps', 200000,
                     'num_steps for training models.')
flags.DEFINE_integer('log_every', 500, 'log after certain number of steps.')

flags.DEFINE_float('nu_learning_rate', 0.0001, 'nu lr')
flags.DEFINE_float('zeta_learning_rate', 0.001, 'z lr')

flags.register_validator(
    'solver_name',
    lambda value: value in ['dice'],
    message='Unknown solver.')
flags.register_validator(
    'env_name',
    lambda value: value in ['grid'],
    message='Unknown environment.')
flags.register_validator(
    'alpha', lambda value: 0 <= value <= 1, message='Invalid value.')


def get_env_and_policies(env_name, tabular_obs, alpha):
  """Get environment and policies."""
  if env_name == 'grid':
    length = 10
    env = gridworld_envs.GridWalk(length, tabular_obs)
    policy0 = gridworld_policies.get_behavior_gridwalk_policy(
        env, tabular_obs, alpha)
    policy1 = gridworld_policies.get_target_gridwalk_policy(env, tabular_obs)
    env.discrete_actions = True
  else:
    ValueError('Environment is not supported.')
  return env, policy0, policy1


def get_solver(solver_name, env, gamma, tabular_solver,
               summary_writer, summary_prefix):
  """Create solver object."""
  if tabular_solver:
    if solver_name == 'dice':
      return dual_dice.TabularDualDice(env.num_states, env.num_actions, gamma)
    else:
      raise ValueError('Solver is not supported.')
  else:
    neural_solver_params = neural_dual_dice.NeuralSolverParameters(
        env.state_dim,
        env.action_dim,
        gamma,
        discrete_actions=env.discrete_actions,
        deterministic_env=FLAGS.deterministic_env,
        nu_learning_rate=FLAGS.nu_learning_rate,
        zeta_learning_rate=FLAGS.zeta_learning_rate,
        batch_size=FLAGS.batch_size,
        num_steps=FLAGS.num_steps,
        log_every=FLAGS.log_every,
        summary_writer=summary_writer,
        summary_prefix=summary_prefix)

    if solver_name == 'dice':
      return neural_dual_dice.NeuralDualDice(
          parameters=neural_solver_params,
          function_exponent=FLAGS.function_exponent)
    else:
      raise ValueError('Solver is not supported.')


def count_state_frequency(data, num_states, gamma):
  state_counts = np.zeros([num_states])
  for transition_tuple in data.iterate_once():
    state_counts[transition_tuple.state] += gamma ** transition_tuple.time_step
  return state_counts / np.sum(state_counts)


def main(argv):
  del argv
  start_seed = FLAGS.seed
  num_seeds = FLAGS.num_seeds
  num_trajectories = FLAGS.num_trajectories
  max_trajectory_length = FLAGS.max_trajectory_length
  alpha = FLAGS.alpha
  gamma = FLAGS.gamma
  nu_learning_rate = FLAGS.nu_learning_rate
  zeta_learning_rate = FLAGS.zeta_learning_rate
  tabular_obs = FLAGS.tabular_obs
  tabular_solver = FLAGS.tabular_solver
  if tabular_solver and not tabular_obs:
    raise ValueError('Tabular solver can only be used with tabular obs.')
  env_name = FLAGS.env_name
  solver_name = FLAGS.solver_name
  save_dir = FLAGS.save_dir

  hparam_format = ('{ENV}_{ALPHA}_{NUM_TRAJ}_{TRAJ_LEN}_'
                   '{N_LR}_{Z_LR}_{GAM}_{SOLVER}')
  solver_str = (solver_name + tabular_solver * '-tab' +
                '-%.1f' % FLAGS.function_exponent)
  hparam_str = hparam_format.format(
      ENV=env_name + tabular_obs * '-tab',
      ALPHA=alpha,
      NUM_TRAJ=num_trajectories,
      TRAJ_LEN=max_trajectory_length,
      GAM=gamma,
      N_LR=nu_learning_rate,
      Z_LR=zeta_learning_rate,
      SOLVER=solver_str)

  if save_dir:
    summary_dir = os.path.join(save_dir, hparam_str)
    if num_seeds == 1:
      summary_dir = os.path.join(summary_dir, 'seed%d' % start_seed)
    summary_writer = tf.summary.FileWriter(summary_dir)
  else:
    summary_writer = None

  env, policy0, policy1 = get_env_and_policies(env_name, tabular_obs, alpha)

  results = []
  for seed in range(start_seed, start_seed + num_seeds):
    print('Seed', seed)
    if num_seeds == 1:
      summary_prefix = ''
    else:
      summary_prefix = 'seed%d/' % seed
    np.random.seed(seed)
    # Off-policy data.
    (behavior_data, behavior_avg_episode_rewards,
     behavior_avg_step_rewards) = transition_data.collect_data(
         env,
         policy0,
         num_trajectories,
         max_trajectory_length,
         gamma=gamma)
    print('Behavior average episode rewards', behavior_avg_episode_rewards)
    print('Behavior average step rewards', behavior_avg_step_rewards)
    # Oracle on-policy data.
    (target_data, target_avg_episode_rewards,
     target_avg_step_rewards) = transition_data.collect_data(
         env,
         policy1,
         num_trajectories,
         max_trajectory_length,
         gamma=gamma)
    print('Target (oracle) average episode rewards', target_avg_episode_rewards)
    print('Target (oracle) average step rewards', target_avg_step_rewards)

    if tabular_obs:
      behavior_state_frequency = count_state_frequency(behavior_data,
                                                       env.num_states, gamma)
      target_state_frequency = count_state_frequency(target_data,
                                                     env.num_states, gamma)
      empirical_density_ratio = (
          target_state_frequency / (1e-8 + behavior_state_frequency))
      print('Empirical state density ratio', empirical_density_ratio[:4], '...')
    del target_data  # Don't use oracle in later code.

    # Get solver.
    density_estimator = get_solver(solver_name, env, gamma, tabular_solver,
                                   summary_writer, summary_prefix)
    # Solve for estimated density ratios.
    est_avg_rewards = density_estimator.solve(behavior_data, policy1)
    # Close estimator properly.
    density_estimator.close()
    print('Estimated (solver: %s) average step reward' % solver_name,
          est_avg_rewards)
    results.append(
        [behavior_avg_step_rewards, target_avg_step_rewards, est_avg_rewards])

  if save_dir is not None:
    filename = os.path.join(save_dir, '%s.npy' % hparam_str)
    print('Saving results to %s' % filename)
    if not tf.gfile.IsDirectory(save_dir):
      tf.gfile.MkDir(save_dir)
    with tf.gfile.GFile(filename, 'w') as f:
      np.save(f, np.array(results))
  print('Done!')


if __name__ == '__main__':
  app.run(main)
