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

"""Utilities."""

import numpy as np
import typing
import tensorflow.compat.v2 as tf

def create_mlp(
    input_dim,
    output_dim,
    hidden_dims = (256, 256),
    activation = tf.nn.relu,
    near_zero_last_layer = False,
):
  """Creates an MLP.

  Args:
    input_dim: input dimensionaloty.
    output_dim: output dimensionality.
    hidden_dims: hidden layers dimensionality.
    activation: activations after hidden units.

  Returns:
    An MLP model.
  """
  initialization = tf.keras.initializers.VarianceScaling(
      scale=0.333, mode='fan_in', distribution='uniform')
  near_zero_initialization = tf.keras.initializers.VarianceScaling(
      scale=1e-2, mode='fan_in', distribution='uniform')

  layers = []
  for hidden_dim in hidden_dims:
    layers.append(
        tf.keras.layers.Dense(
            hidden_dim,
            activation=activation,
            kernel_initializer=initialization))
  layers += [
      tf.keras.layers.Dense(
          output_dim,
          kernel_initializer=near_zero_initialization
          if near_zero_last_layer else initialization)
  ]

  inputs = tf.keras.Input(shape=(input_dim,))
  outputs = tf.keras.Sequential(layers)(inputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.call = tf.function(model.call)
  return model


def create_conv(
    input_shape,
    kernel_sizes = (2, 2, 3),
    stride_sizes = (1, 1, 2),
    pool_sizes = None,
    num_filters = 64,
    activation = tf.nn.relu,
    activation_last_layer = True,
    output_dim = None,
    padding = 'same',
    residual=False,
):
  """Creates an MLP.

  Args:
    input_shape: input shape.
    hidden_dims: hidden layers dimensionality.
    activation: activations after hidden units.

  Returns:
    An MLP model.
  """
  if not hasattr(num_filters, '__len__'):
    num_filters = (num_filters,) * len(kernel_sizes)
  if not hasattr(pool_sizes, '__len__'):
    pool_sizes = (pool_sizes,) * len(kernel_sizes)

  layers = []
  for i, (kernel_size, stride, filters, pool_size) in enumerate(
      zip(kernel_sizes, stride_sizes, num_filters, pool_sizes)):
    if i == len(kernel_sizes) - 1 and not output_dim:
      activation = activation if activation_last_layer else None
    layers.append(
        tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=(stride, stride),
            activation=activation,
            padding=padding))
    if pool_size:
      layers.append(
          tf.keras.layers.MaxPool2D(pool_size=pool_size, padding=padding))

  if output_dim:
    layers += [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            output_dim,
            activation=activation if activation_last_layer else None)
    ]

  inputs = tf.keras.Input(shape=input_shape)
  if residual:
    outputs = layers[0](inputs)
    for layer in layers[1:-1]:
      outputs = outputs + layer(outputs)
    outputs = layers[-1](outputs)
  else:
    outputs = tf.keras.Sequential(layers)(inputs)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.call = tf.function(model.call)
  return model


def get_vi_sequence(env, observation):
  """Returns [L, W, W] optimal actions."""
  start_x, start_y = observation
  target_location = env.target_location
  nav_map = env.nav_map
  current_points = [target_location]
  chosen_actions = {target_location: 0}
  visited_points = {target_location: True}
  vi_sequence = []
  vi_map = np.full((env.size, env.size),
                   fill_value=env.n_action,
                   dtype=np.int32)

  found_start = False
  while current_points and not found_start:
    next_points = []
    for point_x, point_y in current_points:
      for (action, (next_point_x,
                    next_point_y)) in [(0, (point_x - 1, point_y)),
                                       (1, (point_x, point_y - 1)),
                                       (2, (point_x + 1, point_y)),
                                       (3, (point_x, point_y + 1))]:

        if (next_point_x, next_point_y) in visited_points:
          continue

        if not (next_point_x >= 0 and next_point_y >= 0 and
                next_point_x < len(nav_map) and
                next_point_y < len(nav_map[next_point_x])):
          continue

        if nav_map[next_point_x][next_point_y] == 'x':
          continue

        next_points.append((next_point_x, next_point_y))
        visited_points[(next_point_x, next_point_y)] = True
        chosen_actions[(next_point_x, next_point_y)] = action
        vi_map[next_point_x, next_point_y] = action

        if next_point_x == start_x and next_point_y == start_y:
          found_start = True
    vi_sequence.append(vi_map.copy())
    current_points = next_points

  return np.array(vi_sequence)
