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

"""Evaluation for behavioural cloning policies."""

import pickle

from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile
from rrlfd.bc import pickle_dataset

flags.DEFINE_string('hand_vil_episodes_path', None, 'Pickled eval.')

FLAGS = flags.FLAGS


def log_success_rate(num_episodes, success_rate, summary_writer, summary_key):
  if summary_writer is not None:
    with summary_writer.as_default():
      tf.summary.scalar(
          summary_key + '_success_rate' if summary_key else 'success_rate',
          success_rate, step=num_episodes)
      summary_writer.flush()


def separate_robot_info(robot_info, agent):
  """Separate concatenated proprioceptive data into dictionary items."""
  obs = {}
  acc_size = 0
  for feat in agent.visible_state_features:
    size = len(agent.signals_obs_space.mean[feat])
    obs[feat] = robot_info[acc_size:acc_size + size]
    acc_size += size
  assert np.array_equal(
      robot_info,
      np.concatenate([obs[feat] for feat in agent.visible_state_features]))
  return obs


def separate_episode_robot_info(episode_robot_info, agent):
  obs = []
  for t in range(len(episode_robot_info)):
    robot_info = episode_robot_info[t]['robot_info']
    t_obs = separate_robot_info(robot_info, agent)
    obs.append(t_obs)
  return obs


def equal_actions(a1, a2):
  """Test whether the dicts a1 and a2 are equal in each action component."""
  # Assumes a1 and a2 are drawn from the same action space.
  return np.all([np.all(np.equal(a1[k], a2[k])) for k in a1.keys()])


def eval_policy(env, seed, increment_seed, agent, num_episodes, eval_path=None,
                num_videos_to_save=0, summary_writer=None, summary_key='',
                stop_if_stuck=False, verbose=False):
  """Evaluate policy on env for num_episodes episodes."""
  num_successes = 0
  success_rates = {}
  if eval_path is None:
    log_f = None
    success_f = None
    episode_length_f = None
    eval_writer = None
  else:
    log_f = gfile.GFile(eval_path + '_log.txt', 'w')
    success_f = gfile.GFile(eval_path + '_success.txt', 'w')
    episode_length_f = gfile.GFile(eval_path + '_lengths.txt', 'w')
    eval_writer = pickle_dataset.DemoWriter(eval_path + '.pkl')
  if not increment_seed:
    env.seed(seed)

  hand_vil_episodes = None
  if FLAGS.hand_vil_episodes_path is not None:
    with gfile.GFile(FLAGS.hand_vil_episodes_path, 'rb') as f:
      hand_vil_episodes = pickle.load(f)
    hand_vil_actions = hand_vil_episodes['actions']
    hand_vil_images = hand_vil_episodes['rgb']
    hand_vil_robot_info = [
        separate_episode_robot_info(e_infos, agent)
        for e_infos in hand_vil_episodes['env_infos']]

  for e in range(num_episodes):
    if e % 10 == 0 and e > 0:
      success_rate = num_successes / e
      if verbose:
        print(f'Episode {e} / {num_episodes}; Success rate {num_successes} / '
              f'{e} ({success_rate * 100:.4f}%)')
      if (e % 100 == 0 and e > 0) or e == num_episodes - 1:
        success_rates[e] = success_rate
        log_success_rate(e, success_rate, summary_writer, summary_key)

    if increment_seed:
      env.seed(seed)
      seed += 1
    obs = env.reset()

    done = False
    observations = []
    actions = []
    step_count = 0
    prev_stacked_obs = None
    # For envs with non-Markovian success criteria, track required fields.
    goals_achieved = []

    while not done:
      if hand_vil_episodes is not None:
        obs = hand_vil_robot_info[e][step_count]
        obs['rgb'] = hand_vil_images[e][step_count]
      action, stacked_obs = agent.get_action(
          obs, observations, env, return_stacked_obs=True)
      if hand_vil_episodes is not None:
        if not np.allclose(action,
                           hand_vil_actions[e][step_count], atol=5e-6):
          raise ValueError('Actions from agent and from trajectory diverge: '
                           f'{action} vs {hand_vil_actions[e][step_count]}')
      if prev_stacked_obs is not None and stop_if_stuck:
        prev_img, prev_signals = prev_stacked_obs  # pylint: disable=unpacking-non-sequence
        img, signals = stacked_obs
        obs_stuck = np.all(np.equal(img, prev_img))
        # Note: target position has even higher noise.
        signals_stuck = np.all(np.isclose(signals, prev_signals))
        act_stuck = equal_actions(action, actions[-1])
        if obs_stuck and signals_stuck and act_stuck:
          info['failure_message'] = 'Stuck' or info['failure_message']
          break
      prev_stacked_obs = stacked_obs
      observations.append(obs)
      actions.append(action)
      obs, unused_reward, done, info = env.step(action)
      step_count += 1
      if (hand_vil_episodes is not None
          and step_count >= len(hand_vil_robot_info[e])):
        print('episode ends at', step_count, 'done =', done)
      if 'goal_achieved' in info:
        # Environment defines success criteria based on full episode.
        goals_achieved.append(info['goal_achieved'])
        success_percentage = env.evaluate_success(
            [{'env_infos': {'goal_achieved': goals_achieved}}])
        success = bool(success_percentage)
        done = done or success
      else:
        success = False

    if verbose:
      print(step_count, info)
    # Success is directly exposed in environment info.
    success = success or ('success' in info and info['success'])

    num_successes += int(success)
    if success:
      if verbose:
        print(f'{e}: success')
      if log_f is not None:
        log_f.write(f'{e}: success\n')
        log_f.flush()
      if success_f is not None:
        success_f.write('success\n')
        success_f.flush()
    else:
      if verbose:
        if 'failure_message' in info:
          print(f'{e}: failure:', info['failure_message'])
        elif 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
          print(f'{e}: failure: time limit')
        else:
          print(f'{e}: failure')
      if log_f is not None:
        if 'failure_message' in info:
          log_f.write(f'{e}: failure: ' + info['failure_message'] + '\n')
        elif 'TimeLimit.truncated' in info  and info['TimeLimit.truncated']:
          log_f.write(f'{e}: failure: time limit \n')
        else:
          log_f.write(f'{e}: failure\n')
        log_f.flush()
      if success_f is not None:
        success_f.write('failure\n')
        success_f.flush()
    if episode_length_f is not None:
      # TODO(minttu): Save env infos for later.
      episode_length_f.write(str(step_count) + '\n')
      episode_length_f.flush()
    if e < num_videos_to_save and eval_writer is not None:
      eval_writer.write_episode(observations, actions)

  success_rate = num_successes / num_episodes
  success_rates[num_episodes] = success_rate
  log_success_rate(num_episodes, success_rate, summary_writer, summary_key)
  print(
      f'Done; Success rate {num_successes} / {num_episodes} '
      f'({success_rate * 100:.4f}%)')
  if log_f is not None:
    log_f.write(
        f'Done; Success rate {num_successes} / {num_episodes} '
        f'({success_rate * 100:.4f}%)\n')
    log_f.close()
  return success_rates
