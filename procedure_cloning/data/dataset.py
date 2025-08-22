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

import os
import numpy as np
import tensorflow.compat.v2 as tf
from multiprocessing import dummy as multiprocessing

from dice_rl.data.dataset import Dataset
import dice_rl.environments.gridworld.maze as maze

from procedure_cloning import utils


def load_datasets(load_dir,
                  train_seeds,
                  test_seeds,
                  batch_size,
                  env_name,
                  num_trajectory,
                  max_trajectory_length,
                  stacked=True,
                  build_value_map=False,
                  build_bfs_sequence=False):
  pool = multiprocessing.Pool(100)

  def load_dataset_env(seed):
    name, wall_type = env_name.split('-')
    size = int(name.split(':')[-1])
    env = maze.Maze(size, wall_type, maze_seed=seed)
    hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                  'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                      ENV_NAME=env_name,
                      TAB=False,
                      ALPHA=1.0,
                      SEED=seed,
                      NUM_TRAJ=num_trajectory,
                      MAX_TRAJ=max_trajectory_length)
    directory = os.path.join(load_dir, hparam_str)
    dataset = Dataset.load(directory)
    return dataset, env

  datasets_envs = pool.map(load_dataset_env, range(train_seeds + test_seeds))

  observations = []
  actions = []
  maze_maps = []
  value_maps = []
  bfs_sequences = []
  max_len = 0
  max_bfs_len = 0
  for (dataset, env) in datasets_envs:
    episodes, valid_steps = dataset.get_all_episodes()
    max_len = max(max_len, valid_steps.shape[1])

    env_steps = dataset.get_all_steps(num_steps=1)
    observation = tf.squeeze(env_steps.observation, axis=1)
    action = tf.squeeze(tf.cast(env_steps.action, tf.int32), axis=1)

    observations.append(observation)
    actions.append(action)
    maze_map = env.get_maze_map(stacked=stacked)
    maze_maps.append(
        tf.repeat(maze_map[None, Ellipsis], env_steps.observation.shape[0], axis=0))

    value_map = tf.cast(maze.get_value_map(env), tf.float32)
    value_maps.append(
        tf.repeat(value_map[None, Ellipsis], env_steps.observation.shape[0], axis=0))

    bfs_sequence = []
    for i in range(observation.shape[0]):
      bfs_sequence_single = maze.get_bfs_sequence(
          env, observation[i].numpy().astype(int), include_maze_layout=True)
      max_bfs_len = max(max_bfs_len, len(bfs_sequence_single))
      bfs_sequence.append(bfs_sequence_single)
    bfs_sequences.append(bfs_sequence)

  train_data = (tf.concat(observations[:train_seeds],
                          axis=0), tf.concat(actions[:train_seeds], axis=0),
                tf.concat(maze_maps[:train_seeds], axis=0))

  test_data = (tf.concat(observations[train_seeds:],
                         axis=0), tf.concat(actions[train_seeds:], axis=0),
               tf.concat(maze_maps[train_seeds:], axis=0))

  if build_value_map:
    train_data += (tf.concat(value_maps[:train_seeds], axis=0),)
    test_data += (tf.concat(value_maps[train_seeds:], axis=0),)
  if build_bfs_sequence:
    train_sequences = [
        seq for bfs_sequence in bfs_sequences[:train_seeds]
        for seq in bfs_sequence
    ]
    test_sequences = [
        seq for bfs_sequence in bfs_sequences[train_seeds:]
        for seq in bfs_sequence
    ]
    vocab_size = datasets_envs[0][1].n_action + datasets_envs[0][1].num_maze_keys
    train_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        train_sequences, maxlen=max_bfs_len, padding='post', value=vocab_size)
    test_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        test_sequences, maxlen=max_bfs_len, padding='post', value=vocab_size)
    train_data += (tf.concat(train_sequences, axis=0),)
    test_data += (tf.concat(test_sequences, axis=0),)

  train_dataset = tf.data.Dataset.from_tensor_slices(
      train_data).cache().shuffle(
          train_data[0].shape[0], reshuffle_each_iteration=True).repeat().batch(
              batch_size,
              drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_data).cache().shuffle(
      test_data[0].shape[0], reshuffle_each_iteration=True).repeat().batch(
          batch_size,
          drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  return train_dataset, test_dataset, max_len, max_bfs_len


def load_2d_datasets(load_dir,
                     train_seeds,
                     test_seeds,
                     batch_size,
                     env_name,
                     num_trajectory,
                     max_trajectory_length,
                     full_sequence=True):
  pool = multiprocessing.Pool(100)

  def load_dataset_env(seed):
    name, wall_type = env_name.split('-')
    size = int(name.split(':')[-1])
    env = maze.Maze(size, wall_type, maze_seed=seed)
    hparam_str = ('{ENV_NAME}_tabular{TAB}_alpha{ALPHA}_seed{SEED}_'
                  'numtraj{NUM_TRAJ}_maxtraj{MAX_TRAJ}').format(
                      ENV_NAME=env_name,
                      TAB=False,
                      ALPHA=1.0,
                      SEED=seed,
                      NUM_TRAJ=num_trajectory,
                      MAX_TRAJ=max_trajectory_length)
    directory = os.path.join(load_dir, hparam_str)
    dataset = Dataset.load(directory)
    return dataset, env

  datasets_envs = pool.map(load_dataset_env, range(train_seeds + test_seeds))

  observations_train = []
  observations_test = []
  maze_maps_train = []
  maze_maps_test = []
  bfs_input_maps_train = []
  bfs_input_maps_test = []
  bfs_output_maps_train = []
  bfs_output_maps_test = []
  for idx, (dataset, env) in enumerate(datasets_envs):
    if idx < train_seeds:
      observations = observations_train
      maze_maps = maze_maps_train
      bfs_input_maps = bfs_input_maps_train
      bfs_output_maps = bfs_output_maps_train
    else:
      observations = observations_test
      maze_maps = maze_maps_test
      bfs_input_maps = bfs_input_maps_test
      bfs_output_maps = bfs_output_maps_test

    episodes, valid_steps = dataset.get_all_episodes()
    env_steps = dataset.get_all_steps(num_steps=1)
    env_observations = tf.squeeze(env_steps.observation, axis=1)
    maze_map = env.get_maze_map(stacked=True)
    for i in range(env_observations.shape[0]):
      bfs_sequence = utils.get_vi_sequence(
          env, env_observations[i].numpy().astype(np.int32))  # [L, W, W]
      bfs_input_map = env.n_action * tf.ones([env.size, env.size],
                                             dtype=tf.int32)
      if full_sequence:
        for j in range(bfs_sequence.shape[0]):
          bfs_input_maps.append(bfs_input_map)
          bfs_output_maps.append(bfs_sequence[j])
          observations.append(env_observations[i])
          maze_maps.append(maze_map)
          bfs_input_map = bfs_sequence[j]
      else:
        bfs_input_maps.append(bfs_input_map)
        bfs_output_maps.append(bfs_sequence[-1])
        observations.append(env_observations[i])
        maze_maps.append(maze_map)

  train_data = (
      tf.stack(observations_train, axis=0),
      tf.stack(maze_maps_train, axis=0),
      tf.stack(bfs_input_maps_train, axis=0),
      tf.stack(bfs_output_maps_train, axis=0),
  )
  test_data = (
      tf.stack(observations_test, axis=0),
      tf.stack(maze_maps_test, axis=0),
      tf.stack(bfs_input_maps_test, axis=0),
      tf.stack(bfs_output_maps_test, axis=0),
  )
  train_dataset = tf.data.Dataset.from_tensor_slices(
      train_data).cache().shuffle(
          train_data[0].shape[0], reshuffle_each_iteration=True).repeat().batch(
              batch_size,
              drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_data).cache().shuffle(
      test_data[0].shape[0], reshuffle_each_iteration=True).repeat().batch(
          batch_size,
          drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  return train_dataset, test_dataset
