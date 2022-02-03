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

"""Collect demonstrations for Adroit using an expert policy."""

import os
import pickle

from absl import app
from absl import flags
import gym
import numpy as np

from rrlfd import adroit_ext  # pylint: disable=unused-import
from rrlfd.bc import pickle_dataset
from tensorflow.io import gfile


flags.DEFINE_enum('task', None, ['door', 'hammer', 'pen', 'relocate'],
                  'Adroit task for which to collect demonstrations.')
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to record.')
flags.DEFINE_integer('seed', 0, 'Experiment seed.')
flags.DEFINE_boolean('increment_seed', False,
                     'If True, increment seed at every episode.')
flags.DEFINE_integer('image_size', None, 'Size of rendered images.')

flags.DEFINE_string('expert_policy_dir', None,
                    'Path to pickle file with expert policy.')
flags.DEFINE_boolean('record_failed', False,
                     'If True, save failed demonstrations.')
flags.DEFINE_string('logdir', None, 'Location to save demonstrations to.')
flags.DEFINE_string('run_id', None,
                    'If set, a custom string to append to saved demonstrations '
                    'file name.')

FLAGS = flags.FLAGS


def env_loop(env, agent, num_episodes, log_path, record_failed, seed,
             increment_seed, compress_images=True):
  """Loop for collecting demonstrations with an agent in a Gym environment."""
  if log_path is None:
    log_f = None
    success_f = None
    demo_writer = None
  else:
    log_f = gfile.GFile(log_path + '_log.txt', 'w')
    success_f = gfile.GFile(log_path + '_success.txt', 'w')
    demo_writer = pickle_dataset.DemoWriter(log_path + '.pkl', compress_images)
    print('Writing demos to', log_path + '.pkl')
  e = 0
  # Counter to keep track of seed offset, if not recording failed episodes.
  skipped_seeds = 0
  num_successes = 0
  num_attempts = 0
  min_reward, max_reward = np.inf, -np.inf
  while e < num_episodes:
    if e % 10 == 0 and e > 0:
      print(f'Episode {e} / {num_episodes}; '
            f'Success rate {num_successes} / {num_attempts}')
    if increment_seed:
      env.seed(seed + skipped_seeds + e)
    obs = env.reset()

    done = False
    _, agent_info = agent.get_action(obs['original_obs'])
    action = agent_info['evaluation']
    observations = []
    actions = []
    rewards = []
    # For envs with non-Markovian success criteria, track required fields.
    goals_achieved = []

    while not done:
      observations.append(obs)
      actions.append(action)
      obs, reward, done, info = env.step(action)
      rewards.append(reward)
      min_reward = min(min_reward, reward)
      max_reward = max(max_reward, reward)
      _, agent_info = agent.get_action(obs['original_obs'])
      action = agent_info['evaluation']
      if 'goal_achieved' in info:
        goals_achieved.append(info['goal_achieved'])

    # Environment defines success criteria based on full episode.
    success_percentage = env.evaluate_success(
        [{'env_infos': {'goal_achieved': goals_achieved}}])
    success = bool(success_percentage)

    num_successes += int(success)
    num_attempts += 1
    if success:
      print(f'{e}: success')
      if log_f is not None:
        log_f.write(f'{e}: success\n')
        log_f.flush()
      if success_f is not None:
        success_f.write('success\n')
        success_f.flush()
    else:
      if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
        print(f'{e}: failure: time limit')
      else:
        print(f'{e}: failure')
      if log_f is not None:
        if 'TimeLimit.truncated' in info  and info['TimeLimit.truncated']:
          log_f.write(f'{e}: failure: time limit \n')
        else:
          log_f.write(f'{e}: failure\n')
        log_f.flush()
      if success_f is not None:
        success_f.write('failure\n')
        success_f.flush()

    if success or record_failed:
      e += 1
      if demo_writer is not None:
        demo_writer.write_episode(observations, actions, rewards)
    elif not record_failed:
      skipped_seeds += 1

  print(f'Done; Success rate {num_successes} / {num_attempts}')
  print('min reward', min_reward)
  print('max reward', max_reward)
  if log_f is not None:
    log_f.write(f'Done; Success rate {num_successes} / {num_attempts}\n')
    log_f.write(f'min reward {min_reward}\n')
    log_f.write(f'max reward {max_reward}\n')
    log_f.close()


def main(_):
  with gfile.GFile(
      os.path.join(FLAGS.expert_policy_dir, f'{FLAGS.task}.pickle'), 'rb') as f:
    agent = pickle.load(f)
  env = gym.make(f'visual-{FLAGS.task}-v0')
  env.seed(FLAGS.seed)
  im_size = FLAGS.image_size
  if im_size is not None:
    env.env.im_size = im_size

  if FLAGS.logdir is None:
    log_path = None
  else:
    logdir = os.path.join(FLAGS.logdir, f'{FLAGS.task}')
    run_id = '' if FLAGS.run_id is None else '_' + FLAGS.run_id
    if FLAGS.record_failed:
      run_id += '_all'
    if im_size is not None and im_size != adroit_ext.camera_kwargs['im_size']:
      run_id += f'_{im_size}px'
    increment_str = 'i' if FLAGS.increment_seed else ''
    log_path = os.path.join(
        logdir, f's{FLAGS.seed}{increment_str}_e{FLAGS.num_episodes}{run_id}')
    gfile.makedirs(os.path.dirname(log_path))
    print('Writing to', log_path)
  env_loop(env, agent, FLAGS.num_episodes, log_path, FLAGS.record_failed,
           FLAGS.seed, FLAGS.increment_seed)


if __name__ == '__main__':
  app.run(main)
