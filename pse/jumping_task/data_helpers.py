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

# Lint as: python3
"""Helpers for creating imitation data."""

from absl import logging
import gym_jumping_task
import numpy as np

from pse.jumping_task import gym_helpers
OBSTACLE_COLORS = gym_jumping_task.COLORS
JUMP_DISTANCE = 13


def stack_obs(observation_list):
  """Returns a list with stacked consecutive observations."""
  zero_observation = [np.zeros_like(observation_list[0])]
  return np.stack(
      (zero_observation + observation_list[:-2], observation_list[:-1]),
      axis=-1)


def generate_optimal_trajectory(obstacle_position,
                                floor_height,
                                obstacle_color='WHITE'):
  """Returns an optimal trajectory for JumpyWorld with specified parameters."""

  if obstacle_color == 'WHITE':
    env = gym_helpers.create_gym_environment('jumping-task')
  elif obstacle_color in OBSTACLE_COLORS:
    env = gym_helpers.create_gym_environment(
        'jumping-colors-task', obstacle_color=obstacle_color)
  else:
    raise ValueError('Please pass a valid `obstacle_color`')
  initial_obs = env.environment._reset(  # pylint: disable=protected-access
      obstacle_position=obstacle_position,
      floor_height=floor_height)
  terminal = False
  obs = initial_obs
  observations, rewards, actions = [], [], []
  counter = 0
  while not terminal:
    counter += 1
    if (counter == obstacle_position -
        JUMP_DISTANCE) and (obstacle_color != OBSTACLE_COLORS.GREEN):
      action = 1
    else:
      action = 0
    next_obs, reward, terminal, _ = env.step(action)
    rewards.append(reward)
    observations.append(obs)
    actions.append(action)
    obs = next_obs
  assert sum(rewards) >= counter + 1, 'Trajectory not optimal!'
  observations.append(obs)
  return (observations, actions, rewards)


def _generate_imitation_data(min_obstacle_position=20,
                             max_obstacle_position=45,
                             min_floor_height=10,
                             max_floor_height=20,
                             obstacle_color='WHITE'):
  """Generates the imitation learning data for Jumpy World."""
  all_data = {}
  for position in range(min_obstacle_position, max_obstacle_position + 1):
    all_data[position] = {}
    for height in range(min_floor_height, max_floor_height + 1):
      observations, actions, rewards = generate_optimal_trajectory(
          position, height, obstacle_color=obstacle_color)
      observations = stack_obs(observations)
      all_data[position][height] = (observations, actions, rewards)
    logging.info('Obstacle position %d done.', position)
  return all_data


def generate_imitation_data(min_obstacle_position=20,
                            max_obstacle_position=45,
                            min_floor_height=10,
                            max_floor_height=20,
                            use_colors=False):
  """Generate imitation data with uncolored or colored obstacles."""
  imitation_data = {}
  colors = OBSTACLE_COLORS if use_colors else ['WHITE']
  for obstacle_color in colors:
    if isinstance(obstacle_color, str):
      color_key = obstacle_color
    else:
      color_key = obstacle_color.name
    imitation_data[color_key] = _generate_imitation_data(
        min_obstacle_position=min_obstacle_position,
        max_obstacle_position=max_obstacle_position,
        min_floor_height=min_floor_height,
        max_floor_height=max_floor_height,
        obstacle_color=obstacle_color)
  return imitation_data


def generate_validation_tight_grid(training_positions,
                                   pos_diff,
                                   height_diff,
                                   min_obstacle_position=20,
                                   max_obstacle_position=45,
                                   min_floor_height=10,
                                   max_floor_height=20):
  """Calculates validation positions for tight grid."""
  all_pos = []

  def is_valid_position(x, y):
    return (min_obstacle_position <= x < max_obstacle_position) and \
    (min_floor_height <= y <= max_floor_height)

  for x, y in training_positions:
    x_0 = x + pos_diff // 2
    y_0 = y + height_diff // 2
    for i in range(-5, 5):
      y_new = y_0 + i * height_diff
      if is_valid_position(x_0, y_new):
        all_pos.append((x_0, y_new))

    for i in range(-5, 3):
      x_new = x_0 + i * pos_diff
      if is_valid_position(x_new, y_0):
        all_pos.append((x_new, y_0))
  return set(all_pos)


def generate_training_positions(min_obstacle_position=20,
                                max_obstacle_position=45,
                                min_floor_height=10,
                                max_floor_height=20,
                                positions_train_diff=5,
                                heights_train_diff=5,
                                random_tasks=False,
                                seed=0):
  """Generate positions for training."""
  if random_tasks:
    obstacle_positions = list(
        range(min_obstacle_position, max_obstacle_position + 1))
    floor_heights = list(range(min_floor_height, max_floor_height + 1))
    num_positions = (len(obstacle_positions) // positions_train_diff) + 1
    num_heights = (len(floor_heights) // heights_train_diff) + 1
    num_train_positions = num_positions * num_heights
    np.random.seed(seed)
    obstacle_positions_train = np.random.choice(
        obstacle_positions, size=num_train_positions)
    floor_heights_train = np.random.choice(
        floor_heights, size=num_train_positions)
    training_positions = list(
        zip(obstacle_positions_train, floor_heights_train))
  else:
    obstacle_positions_train = list(
        range(min_obstacle_position, max_obstacle_position + 1,
              positions_train_diff))
    floor_heights_train = list(
        range(min_floor_height, max_floor_height + 1, heights_train_diff))
    training_positions = []
    for pos in obstacle_positions_train:
      for height in floor_heights_train:
        training_positions.append((pos, height))
  return training_positions


def _training_data(imitation_data, training_positions):
  """Picks the JumpyWorld environments used for training."""
  x_train, y_train = [], []
  for (position, height) in training_positions:
    obs, ac, _ = imitation_data[position][height]
    x_train.extend(obs)
    y_train.extend(ac)
  return x_train, y_train


def training_data(imitation_data, training_positions):
  """Picks the JumpyWorld environments used for training."""
  x_train, y_train = [], []
  for imitation_color_data in imitation_data.values():
    x_train_color, y_train_color = _training_data(
        imitation_color_data, training_positions)
    if x_train is None:
      x_train, y_train = x_train_color, y_train_color
    else:
      x_train.extend(x_train_color)
      y_train.extend(y_train_color)
  return np.array(
      x_train, dtype=np.float32), np.array(
          y_train, dtype=np.float32)


def sample_train_pair_index(training_positions, print_log=False):
  """Returns the index for a random pair of training environments."""
  index1, index2 = 0, 0
  while index1 == index2:
    index1, index2 = np.random.randint(0, high=len(training_positions), size=2)
  pos1, pos2 = training_positions[index1], training_positions[index2]
  if print_log:
    logging.info('Obs1, Height1 -> %d %d; Obs2, Height1 -> %d %d', pos1[0],
                 pos1[1], pos2[0], pos2[1])
  return pos1, pos2


def generate_optimal_data_tuple(imitation_data,
                                training_positions,
                                print_log=False):
  """Returns a tuple of {(obs_1, actions_1}, (obs_2, actions_2)}."""
  (obs1, height1), (obs2, height2) = sample_train_pair_index(
      training_positions, print_log=print_log)
  return (imitation_data[obs1][height1],
          imitation_data[obs2][height2])
