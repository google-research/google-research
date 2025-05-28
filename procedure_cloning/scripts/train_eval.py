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

"""Run procedure cloning on maze BFS."""

from absl import app
from absl import flags

import numpy as np
import os
import io

import tensorflow.compat.v2 as tf

import dice_rl.environments.gridworld.maze as maze
import procedure_cloning.data.dataset as dataset
from procedure_cloning.models.behavioral_cloning import BehavioralCloning
from procedure_cloning.models.behavioral_cloning_bfs import BehavioralCloningBFS


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'maze:16-tunnel', 'Environment name.')
flags.DEFINE_integer('train_seeds', 5, 'Number of training tasks.')
flags.DEFINE_integer('test_seeds', 1, 'Number of test tasks.')
flags.DEFINE_integer('num_trajectory', 4, 'Number of trajectories to collect.')
flags.DEFINE_integer('max_trajectory_length', 100,
                     'Cutoff trajectory at this step.')
flags.DEFINE_float('alpha', 1., 'How close to target policy.')
flags.DEFINE_bool('tabular_obs', False, 'Whether to use tabular observations.')
flags.DEFINE_string('load_dir', '/tmp/procedure_cloning/',
                    'Directory to load dataset from.')
flags.DEFINE_string('save_dir', '/tmp/procedure_cloning/',
                    'Directory to save result to.')

flags.DEFINE_enum('algo_name', 'pc',
                  ['bc', 'aug_bc', 'pc', 'aux_bc'],
                  'Algorithm name.')

flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_integer('num_steps', 500_000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 10_000,
                     'Number of steps between each evaluation.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('num_eval_episodes', 5, 'Number of eval episodes.')
flags.DEFINE_integer('max_eval_episode_length', 100, 'Number of eval episodes.')

flags.DEFINE_integer('seed', 1, 'Random seed.')


def get_env(env_name, env_seed):
  if 'maze:' in env_name:
    # Format is in maze:<size>-<type>
    name, wall_type = env_name.split('-')
    size = int(name.split(':')[-1])
    env = maze.Maze(size, wall_type, maze_seed=env_seed)
  else:
    raise ValueError('Unknown environment: %s.' % env_name)
  return env


def evaluate(env, policy):
  maze_map = env.get_maze_map(stacked=True)

  total_returns = 0.0
  for i in range(FLAGS.num_eval_episodes):
    obs = env.reset()
    for j in range(FLAGS.max_eval_episode_length):
      action = policy.act(obs, maze_map)
      obs, reward, done, _ = env.step(action)
      total_returns += reward
      if done:
        break
  return total_returns / FLAGS.num_eval_episodes

def main(argv):
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)

  hparam_dict = {
      'env_name': FLAGS.env_name,
      'train_seeds': FLAGS.train_seeds,
      'test_seeds': FLAGS.test_seeds,
      'num_trajectory': FLAGS.num_trajectory,
      'algo_name': FLAGS.algo_name,
  }
  hparam_str = ','.join(
      ['%s=%s' % (k, str(hparam_dict[k])) for k in sorted(hparam_dict.keys())])
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, hparam_str))

  all_envs = [get_env(FLAGS.env_name, seed) for seed in range(
      FLAGS.train_seeds + FLAGS.test_seeds)]

  if FLAGS.algo_name == 'bc':
    train_dataset, test_dataset, max_len, _ = dataset.load_datasets(
        FLAGS.load_dir,
        FLAGS.train_seeds,
        FLAGS.test_seeds,
        FLAGS.batch_size,
        FLAGS.env_name,
        FLAGS.num_trajectory,
        FLAGS.max_trajectory_length,
        build_value_map=True,
        build_bfs_sequence=False)
    algo = BehavioralCloning(
        all_envs[0].size, learning_rate=FLAGS.learning_rate, augment=False)
  elif FLAGS.algo_name == 'aug_bc':
    train_dataset, test_dataset, max_len, _ = dataset.load_datasets(
        FLAGS.load_dir,
        FLAGS.train_seeds,
        FLAGS.test_seeds,
        FLAGS.batch_size,
        FLAGS.env_name,
        FLAGS.num_trajectory,
        FLAGS.max_trajectory_length,
        build_value_map=True,
        build_bfs_sequence=False)
    algo = BehavioralCloning(
        all_envs[0].size, learning_rate=FLAGS.learning_rate, augment=True)
  elif FLAGS.algo_name == 'pc':
    train_dataset, test_dataset = dataset.load_2d_datasets(
        FLAGS.load_dir,
        FLAGS.train_seeds,
        FLAGS.test_seeds,
        FLAGS.batch_size,
        FLAGS.env_name,
        FLAGS.num_trajectory,
        FLAGS.max_trajectory_length,
        full_sequence=True)
    algo = BehavioralCloningBFS(
        all_envs[0].size,
        all_envs[0].n_action,
        learning_rate=FLAGS.learning_rate)
  elif FLAGS.algo_name == 'aux_bc':
    train_dataset, test_dataset = dataset.load_2d_datasets(
        FLAGS.load_dir,
        FLAGS.train_seeds,
        FLAGS.test_seeds,
        FLAGS.batch_size,
        FLAGS.env_name,
        FLAGS.num_trajectory,
        FLAGS.max_trajectory_length,
        full_sequence=False)
    algo = BehavioralCloningBFS(
        all_envs[0].size,
        all_envs[0].n_action,
        aux_weight=1.,
        learning_rate=FLAGS.learning_rate)
  else:
    raise NotImplementedError

  train_iter = iter(train_dataset)
  test_iter = iter(test_dataset)

  for step in range(FLAGS.num_steps):
    info_dict = algo(train_iter, training=True)

    if step % FLAGS.eval_interval == 0:
      with summary_writer.as_default():
        info_dict = algo(train_iter, training=True, generate=True)
        for k, v in info_dict.items():
          tf.summary.scalar(f'train/{k}', v, step=step)
          print('train', k, v)

        info_dict = algo(test_iter, training=False, generate=True)
        for k, v in info_dict.items():
          tf.summary.scalar(f'eval/{k}', v, step=step)
          print('eval', k, v)

      train_success = evaluate(all_envs[0], algo)
      test_successes = []
      for seed in range(FLAGS.train_seeds, FLAGS.train_seeds + FLAGS.test_seeds):
        ret = evaluate(all_envs[seed], algo)
        test_successes.append(ret)

      with summary_writer.as_default():
        tf.summary.scalar('train/success_mean', train_success, step=step)
        tf.summary.scalar('eval/success_mean', np.mean(test_successes), step=step)
        tf.summary.scalar('eval/success_std', np.std(test_successes), step=step)
        tf.summary.scalar('eval/success_max', np.max(test_successes), step=step)
        tf.summary.scalar('eval/success_min', np.min(test_successes), step=step)
        print('train/success', train_success)
        print('eval/success', np.mean(test_successes), np.std(test_successes))


if __name__ == '__main__':
  app.run(main)
