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

"""Loads D4RL dataset from pickle files."""
import typing

import d4rl
import gym
import numpy as np
import tensorflow as tf


def qlearning_and_window_dataset(env, sliding_window=1,
                                 dataset=None, terminate_on_end=False, **kwargs):
  if dataset is None:
    dataset = env.get_dataset(**kwargs)

  N = dataset['rewards'].shape[0]
  obs_ = []
  next_obs_ = []
  action_ = []
  reward_ = []
  done_ = []

  window_obs_ = []
  window_next_obs_ = []
  window_action_ = []
  window_reward_ = []
  window_done_ = []

  # The newer version of the dataset adds an explicit
  # timeouts field. Keep old method for backwards compatability.
  if 'timeouts' in dataset:
    use_timeouts = True

  episode_step = 0
  episode_return = 0
  sliding_obs = []
  sliding_act = []
  sliding_reward = []
  sliding_done = []
  episode_start = True
  for i in range(N-1):
    obs = dataset['observations'][i]
    new_obs = dataset['observations'][i+1]
    action = dataset['actions'][i]
    reward = dataset['rewards'][i]
    done_bool = bool(dataset['terminals'][i])

    if episode_start:
      sliding_obs = [obs] * sliding_window
      sliding_act = [0 * action] * sliding_window
      sliding_reward = [0 * reward] * sliding_window
      sliding_done = [-1.] * sliding_window  # -1 for 'before start'.

    sliding_obs.append(obs)
    sliding_act.append(action)
    sliding_reward.append(reward)
    sliding_done.append(done_bool)

    sliding_obs.pop(0)
    sliding_act.pop(0)
    sliding_done.pop(0)
    sliding_reward.pop(0)

    if use_timeouts:
      final_timestep = dataset['timeouts'][i]
    else:
      final_timestep = (episode_step == env._max_episode_steps - 1)
    if (not terminate_on_end) and final_timestep:
      # Skip this transition and don't apply terminals on the last step of an episode
      print('Finished episode with length %d, return %.2f.' % (
          episode_step, episode_return))
      episode_step = 0
      episode_return = 0
      episode_start = True
      continue
    if done_bool or final_timestep:
      print('Finished episode with length %d, return %.2f.' % (
          episode_step, episode_return))
      episode_step = 0
      episode_return = 0
      episode_start = True
    else:
      episode_start = False

    obs_.append(obs)
    next_obs_.append(new_obs)
    action_.append(action)
    reward_.append(reward)
    done_.append(done_bool)

    window_obs_.append(sliding_obs[:])
    window_next_obs_.append(sliding_obs[1:] + [new_obs])
    window_action_.append(sliding_act[:])
    window_reward_.append(sliding_reward[:])
    window_done_.append(sliding_done[:])

    episode_step += 1
    episode_return += reward

  return {
      'observations': np.array(obs_),
      'actions': np.array(action_),
      'next_observations': np.array(next_obs_),
      'rewards': np.array(reward_),
      'terminals': np.array(done_),
  }, {
      'observations': np.array(window_obs_),
      'actions': np.array(window_action_),
      'next_observations': np.array(window_next_obs_),
      'rewards': np.array(window_reward_),
      'terminals': np.array(window_done_),
  }


def create_d4rl_env_and_dataset(
    task_name,
    batch_size,
    sliding_window = None,
    data_size = None,
    state_mask_fn = None,
):
  """Create gym environment and dataset for d4rl.

  Args:
    task_name: Name of d4rl task.
    batch_size: Mini batch size.
    sliding_window: If specified, creates a second dataset with sliding window.
    data_size: If specified, data is truncated.
  Returns:
    Gym env and dataset.
  """
  env = gym.make(task_name)
  dataset, window_dataset = qlearning_and_window_dataset(
      env, sliding_window=sliding_window or 1)

  states = np.array(dataset['observations'], dtype=np.float32)
  actions = np.array(dataset['actions'], dtype=np.float32)
  rewards = np.array(dataset['rewards'], dtype=np.float32)
  discounts = np.array(np.logical_not(dataset['terminals']), dtype=np.float32)
  next_states = np.array(dataset['next_observations'], dtype=np.float32)
  if state_mask_fn:
    states = state_mask_fn(states)
    next_states = state_mask_fn(next_states)

  if data_size:
    states = states[:data_size]
    actions = actions[:data_size]
    rewards = rewards[:data_size]
    discounts = discounts[:data_size]
    next_states = next_states[:data_size]

  dataset = tf.data.Dataset.from_tensor_slices(
      (states, actions, rewards, discounts, next_states)).cache().shuffle(
          states.shape[0], reshuffle_each_iteration=True).repeat().batch(
              batch_size,
              drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

  if sliding_window:
    window_states = np.array(window_dataset['observations'], dtype=np.float32)
    window_actions = np.array(window_dataset['actions'], dtype=np.float32)
    window_rewards = np.array(window_dataset['rewards'], dtype=np.float32)
    window_discounts = np.array(np.logical_not(window_dataset['terminals']), dtype=np.float32)
    window_next_states = np.array(window_dataset['next_observations'], dtype=np.float32)

    if data_size:
      window_states = window_states[:data_size]
      window_actions = window_actions[:data_size]
      window_rewards = window_rewards[:data_size]
      window_discounts = window_discounts[:data_size]
      window_next_states = window_next_states[:data_size]

    window_dataset = tf.data.Dataset.from_tensor_slices(
        (window_states, window_actions, window_rewards, window_discounts, window_next_states))
    window_dataset = window_dataset.cache().shuffle(
        window_states.shape[0], reshuffle_each_iteration=True).repeat().batch(
            batch_size,
            drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  else:
    window_dataset = dataset
  return env, dataset, window_dataset
