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

# Lint as: python3
"""Helpers for evaluating an agent on Jumpy World."""

import io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow.compat.v2 as tf

sns.set_style('white')


def create_evaluation_grid(nn_model,
                           imitation_data,
                           mc_samples=1,
                           color_name='WHITE'):
  """Evaluates an agent on all environments in imitation_data."""
  obstacle_positions = sorted(imitation_data.keys())
  floor_heights = sorted(imitation_data[obstacle_positions[0]].keys())
  evaluation_grid = np.zeros((len(obstacle_positions), len(floor_heights)))
  for i, pos in enumerate(obstacle_positions):
    for j, height in enumerate(floor_heights):
      input_observations, optimal_actions, _ = imitation_data[pos][height]
      predictions = tf.nn.softmax(
          nn_model(input_observations, training=False), axis=-1)
      # MC Averaging if using RandConv
      for _ in range(mc_samples - 1):
        predictions += tf.nn.softmax(
            nn_model(input_observations, training=False), axis=-1)
      predictions /= mc_samples
      greedy_actions = np.array(
          [1 if pi[1] > pi[0] else 0 for pi in predictions])
      action_diff = greedy_actions - np.array(optimal_actions)
      if color_name == 'GREEN':
        # The collision happens when the agent touches the block
        argmax_val = pos - 5
      elif color_name in ['WHITE', 'RED']:
        argmax_val = np.argmax(optimal_actions)
      else:
        raise ValueError(f'{color_name} is not a valid obstacle color.')
      binary_mask = np.arange(len(optimal_actions)) <= argmax_val
      is_optimal = sum(binary_mask * np.abs(action_diff)) == 0
      evaluation_grid[i][j] = is_optimal
  return evaluation_grid


def neigbhour_indices(x, y, max_x, max_y):
  valid_indices = []
  for index in [(x - 1, y), (x+1, y), (x, y-1), (x, y+1)]:
    is_x_valid = (0 <= index[0]) and (index[0] < max_x)
    is_y_valid = (0 <= index[1]) and (index[1] < max_y)
    if is_x_valid and is_y_valid:
      valid_indices.append(index)
  return valid_indices


def generate_validation_positions(training_positions, min_obs_position,
                                  min_floor_height, num_positions, num_heights):
  """Generate validation positions."""
  val_pos = []
  for (obstacle_pos, floor_height) in training_positions:
    pos_index = obstacle_pos - min_obs_position
    height_index = floor_height - min_floor_height
    validation_indices = neigbhour_indices(
        pos_index, height_index, num_positions, num_heights)
    for val_pos_index, val_height_index in validation_indices:
      val_pos.append((val_pos_index + min_obs_position,
                      val_height_index + min_floor_height))
  return list(set(val_pos))


def num_solved_tasks(evaluation_grid, training_positions, validation_positions,
                     min_obs_position, min_floor_height):
  """Calculates number of tasks solved in training, validation and test sets."""
  solved_envs = {'train': 0, 'test': 0}
  if validation_positions:
    solved_envs['validation'] = 0

  num_positions, num_heights = evaluation_grid.shape
  is_train_or_validation = np.zeros_like(evaluation_grid, dtype=np.int32)

  for (obstacle_pos, floor_height) in training_positions:
    pos_index = obstacle_pos - min_obs_position
    height_index = floor_height - min_floor_height
    is_train_or_validation[pos_index][height_index] = 1

  for (obstacle_pos, floor_height) in validation_positions:
    pos_index = obstacle_pos - min_obs_position
    height_index = floor_height - min_floor_height
    is_train_or_validation[pos_index][height_index] = 2

  for pos_index in range(num_positions):
    for height_index in range(num_heights):
      if is_train_or_validation[pos_index][height_index] == 1:
        solved_envs['train'] += evaluation_grid[pos_index][height_index]
      elif is_train_or_validation[pos_index][height_index] == 2:
        solved_envs['validation'] += evaluation_grid[pos_index][height_index]
      else:
        solved_envs['test'] += evaluation_grid[pos_index][height_index]
  return solved_envs


def plot_evaluation_grid(grid, training_positions, min_obs_position,
                         min_floor_height):
  """Plots the evaluation grid."""
  fig, ax = plt.subplots(figsize=(7, 9))
  grid_x, grid_y = grid.shape
  extent = (0, grid_x, grid_y, 0)
  ax.imshow(grid.T, extent=extent, origin='lower')

  x_ticks = np.arange(grid_x)
  y_ticks = np.arange(grid_y)
  ax.set_xticks(x_ticks)
  ax.set_yticks(y_ticks)

  ax.tick_params(labelbottom=False, labelleft=False)

  # Loop over data dimensions and create text annotations.
  for (obstacle_pos, floor_height) in training_positions:
    pos_index = obstacle_pos - min_obs_position
    height_index = floor_height - min_floor_height
    ax.text(
        pos_index + 0.5,
        height_index + 0.5,
        'T',
        ha='center',
        va='center',
        color='r')

  ax.grid(color='w', linewidth=1)
  fig.tight_layout()
  return fig


def plot_to_image(figure):
  """Converts the plot specified by 'figure' to a PNG image and returns it."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  figure.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def np_array_figure(arr):
  fig, ax = plt.subplots(figsize=(6, 6))
  im = ax.imshow(arr.T, origin='lower', cmap='hot', interpolation='nearest')
  fig.colorbar(im, ax=ax)
  return plot_to_image(fig)


def sinkhorn_logspace(logits_rows, logits_cols, costs, n_steps,
                      entropy_strength):
  """Sinkhorn algorithm for (unbalanced) entropy-regularized optimal transport.

  The updates are computed in log-space and are thus more stable.

  Args:
    logits_rows: (..., n) tensor with the logits of the row-sum constraint
    logits_cols: (..., m) tensor with the logits of the column-sum constraint
    costs: (..., n, m) tensor holding the transportation costs
    n_steps: How many Sinkhorn iterations to perform.
    entropy_strength: The strength of the entropic regularizer

  Returns:
    (..., n, m) tensor with the computation optimal transportation matrices
  """
  assert n_steps > 0
  assert entropy_strength > 0

  logits_rows = tf.expand_dims(logits_rows, axis=-1)
  logits_cols = tf.expand_dims(logits_cols, axis=-2)
  log_kernel = -costs / entropy_strength + logits_rows + logits_cols

  log_lbd_cols = tf.zeros_like(logits_cols)
  for _ in range(n_steps):
    log_lbd_rows = logits_rows - tf.reduce_logsumexp(
        log_kernel + log_lbd_cols, axis=-1, keepdims=True)
    log_lbd_cols = logits_cols - tf.reduce_logsumexp(
        log_kernel + log_lbd_rows, axis=-2, keepdims=True)
  return tf.exp(log_lbd_cols + log_kernel + log_lbd_rows)


@tf.function
def induced_coupling(similarity_matrix, n_steps=3, entropy_strength=0.0001):
  """Calculates the coupling induced by the similarity matrix."""
  dist_v = tf.ones(similarity_matrix.shape[0])
  dist_v /= tf.reduce_sum(dist_v)
  dist_v = tf.math.log(dist_v)
  coupling = tf.stop_gradient(sinkhorn_logspace(
      dist_v,
      dist_v,
      1 - similarity_matrix,
      n_steps=n_steps,
      entropy_strength=entropy_strength))
  return coupling
