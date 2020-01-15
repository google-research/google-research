# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utils for data loading and preprocessing."""

import random
import numpy as np


def load_expert_data(filename):
  """Loads expert trajectoris from a file.

  Args:
    filename: a filename to load the data.

  Returns:
    Numpy arrays that contain states, actions, next_states and dones
  """
  with open(filename, 'rb') as fin:
    expert_data = np.load(fin)
    expert_data = {key: expert_data[key] for key in expert_data.files}

    expert_states = expert_data['states']
    expert_actions = expert_data['actions']
    expert_next_states = expert_data['next_states']
    expert_dones = expert_data['dones']
  return expert_states, expert_actions, expert_next_states, expert_dones


def subsample_trajectories(expert_states, expert_actions, expert_next_states,
                           expert_dones, num_trajectories):
  """Extracts a random subset of trajectories.

  Args:
    expert_states: A numpy array with expert states.
    expert_actions: A numpy array with expert states.
    expert_next_states: A numpy array with expert states.
    expert_dones: A numpy array with expert states.
    num_trajectories: A number of trajectories to extract.

  Returns:
      Numpy arrays that contain states, actions, next_states and dones.
  """
  expert_states_traj = [[]]
  expert_actions_traj = [[]]
  expert_next_states_traj = [[]]
  expert_dones_traj = [[]]

  for i in range(expert_states.shape[0]):
    expert_states_traj[-1].append(expert_states[i])
    expert_actions_traj[-1].append(expert_actions[i])
    expert_next_states_traj[-1].append(expert_next_states[i])
    expert_dones_traj[-1].append(expert_dones[i])

    if expert_dones[i] and i < expert_states.shape[0] - 1:
      expert_states_traj.append([])
      expert_actions_traj.append([])
      expert_next_states_traj.append([])
      expert_dones_traj.append([])

  shuffle_inds = list(range(len(expert_states_traj)))
  random.shuffle(shuffle_inds)
  shuffle_inds = shuffle_inds[:num_trajectories]
  expert_states_traj = [expert_states_traj[i] for i in shuffle_inds]
  expert_actions_traj = [expert_actions_traj[i] for i in shuffle_inds]
  expert_next_states_traj = [expert_next_states_traj[i] for i in shuffle_inds]
  expert_dones_traj = [expert_dones_traj[i] for i in shuffle_inds]

  def concat_trajectories(trajectories):
    return np.concatenate(trajectories, 0)

  expert_states = concat_trajectories(expert_states_traj)
  expert_actions = concat_trajectories(expert_actions_traj)
  expert_next_states = concat_trajectories(expert_next_states_traj)
  expert_dones = concat_trajectories(expert_dones_traj)

  return expert_states, expert_actions, expert_next_states, expert_dones


def add_absorbing_states(expert_states, expert_actions, expert_next_states,
                         expert_dones, env):
  """Adds absorbing states to trajectories.

  Args:
    expert_states: A numpy array with expert states.
    expert_actions: A numpy array with expert states.
    expert_next_states: A numpy array with expert states.
    expert_dones: A numpy array with expert states.
    env: A gym environment.

  Returns:
      Numpy arrays that contain states, actions, next_states and dones.
  """

  # First add 0 indicator to all non-absorbing states.
  expert_states = np.pad(expert_states, ((0, 0), (0, 1)), mode='constant')
  expert_next_states = np.pad(
      expert_next_states, ((0, 0), (0, 1)), mode='constant')

  expert_states = [x for x in expert_states]
  expert_next_states = [x for x in expert_next_states]
  expert_actions = [x for x in expert_actions]
  expert_dones = [x for x in expert_dones]

  # Add absorbing states.
  i = 0
  current_len = 0
  while i < len(expert_states):
    current_len += 1
    if expert_dones[i] and current_len < env._max_episode_steps:  # pylint: disable=protected-access
      current_len = 0
      expert_states.insert(i + 1, env.get_absorbing_state())
      expert_next_states[i] = env.get_absorbing_state()
      expert_next_states.insert(i + 1, env.get_absorbing_state())
      expert_actions.insert(i + 1, np.zeros((env.action_space.shape[0],)))
      expert_dones[i] = 0.0
      expert_dones.insert(i + 1, 1.0)
      i += 1
    i += 1

  expert_states = np.stack(expert_states)
  expert_next_states = np.stack(expert_next_states)
  expert_actions = np.stack(expert_actions)
  expert_dones = np.stack(expert_dones)

  return expert_states, expert_actions, expert_next_states, expert_dones
