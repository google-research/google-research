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

"""Creates actor and critic network designed to work with MultiGrid.

Note: Social influence hparams: conv kernel size=3, conv output channels=6,
      LSTM=128
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf

from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.utils import nest_utils


def one_hot_layer(class_dim=None):
  """A Teras Sequential layer for making one-hot inputs."""
  # Check if inputs were supplied correctly
  if class_dim is None:
    raise TypeError('class_dim is not set')

  # Helper method (not inlined for clarity)
  def _one_hot(x, num_classes):
    x = tf.keras.backend.one_hot(x, num_classes=num_classes)
    return tf.reshape(x, [-1, num_classes])

  # Final layer representation as a Lambda layer
  return tf.keras.layers.Lambda(_one_hot, arguments={'num_classes': class_dim})


def cast_and_scale(scale_by=10.0):

  def _cast_and_scale(x):
    x = tf.keras.backend.cast(x, 'float32')
    return x / scale_by

  return tf.keras.layers.Lambda(_cast_and_scale)


def construct_multigrid_networks(observation_spec,
                                 action_spec,
                                 use_rnns=True,
                                 actor_fc_layers=(200, 100),
                                 value_fc_layers=(200, 100),
                                 lstm_size=(128,),
                                 conv_filters=8,
                                 conv_kernel=3,
                                 scalar_fc=5,
                                 scalar_name='direction',
                                 scalar_dim=4,
                                 random_z=False,
                                 xy_dim=None):
  """Creates an actor and critic network designed for use with MultiGrid.

  A convolution layer processes the image and a dense layer processes the
  direction the agent is facing. These are fed into some fully connected layers
  and an LSTM.

  Args:
    observation_spec: A tf-agents observation spec.
    action_spec: A tf-agents action spec.
    use_rnns: If True, will construct RNN networks.
    actor_fc_layers: Dimension and number of fully connected layers in actor.
    value_fc_layers: Dimension and number of fully connected layers in critic.
    lstm_size: Number of cells in each LSTM layers.
    conv_filters: Number of convolution filters.
    conv_kernel: Size of the convolution kernel.
    scalar_fc: Number of neurons in the fully connected layer processing the
      scalar input.
    scalar_name: Name of the scalar input.
    scalar_dim: Highest possible value for the scalar input. Used to convert to
      one-hot representation.
    random_z: If True, will provide an additional layer to process a randomly
      generated float input vector.
    xy_dim: If not None, will provide two additional layers to process 'x' and
      'y' inputs. The dimension provided is the maximum value of x and y, and
      is used to create one-hot representation.

  Returns:
    A tf-agents ActorDistributionRnnNetwork for the actor, and a ValueRnnNetwork
    for the critic.
  """
  preprocessing_layers = {
      'image':
          tf.keras.models.Sequential([
              cast_and_scale(),
              tf.keras.layers.Conv2D(conv_filters, conv_kernel),
              tf.keras.layers.ReLU(),
              tf.keras.layers.Flatten()
          ]),
  }
  if scalar_name in observation_spec:
    preprocessing_layers[scalar_name] = tf.keras.models.Sequential(
        [one_hot_layer(scalar_dim),
         tf.keras.layers.Dense(scalar_fc)])
  if 'position' in observation_spec:
    preprocessing_layers['position'] = tf.keras.models.Sequential(
        [cast_and_scale(), tf.keras.layers.Dense(scalar_fc)])

  if random_z:
    preprocessing_layers['random_z'] = tf.keras.models.Sequential(
        [tf.keras.layers.Lambda(lambda x: x)])  # Identity layer
  if xy_dim is not None:
    preprocessing_layers['x'] = tf.keras.models.Sequential(
        [one_hot_layer(xy_dim)])
    preprocessing_layers['y'] = tf.keras.models.Sequential(
        [one_hot_layer(xy_dim)])

  preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

  if use_rnns:
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=actor_fc_layers,
        output_fc_layer_params=None,
        lstm_size=lstm_size)
    value_net = value_rnn_network.ValueRnnNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=value_fc_layers,
        output_fc_layer_params=None)
  else:
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=actor_fc_layers,
        activation_fn=tf.keras.activations.tanh)
    value_net = value_network.ValueNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=value_fc_layers,
        activation_fn=tf.keras.activations.tanh)

  return actor_net, value_net


def get_spatial_basis(h, w, d):
  """Gets a sinusoidal position encoding for image attention."""
  half_d = d // 2
  basis = np.zeros((h, w, d), dtype=np.float32)
  div = np.exp(
      np.arange(0, half_d, 2, dtype=np.float32) * -np.log(100.0) / half_d)
  h_grid = np.expand_dims(np.arange(0, h, dtype=np.float32), 1)
  w_grid = np.expand_dims(np.arange(0, w, dtype=np.float32), 1)
  basis[:, :, 0:half_d:2] = np.sin(h_grid * div)[:, np.newaxis, :]
  basis[:, :, 1:half_d:2] = np.cos(h_grid * div)[:, np.newaxis, :]
  basis[:, :, half_d::2] = np.sin(w_grid * div)[np.newaxis, :, :]
  basis[:, :, half_d + 1::2] = np.cos(w_grid * div)[np.newaxis, :, :]
  return basis


class AttentionCombinerConv(tf.keras.layers.Layer):
  """Combiner that applies attention to input images."""

  def __init__(self,
               image_index_flat,
               network_state_index_flat,
               image_shape,
               conv_filters=64,
               n_heads=8,
               basis_dim=8):
    super(AttentionCombinerConv, self).__init__(trainable=True)
    self.combiner = tf.keras.layers.Concatenate(axis=-1)
    self.image_index_flat = image_index_flat
    self.network_state_index_flat = network_state_index_flat
    self.image_shape = image_shape
    self.conv_filters = conv_filters
    self.n_heads = n_heads
    self.basis_dim = basis_dim
    self.attention_network = tf.keras.Sequential([
        tf.keras.layers.Reshape((image_shape[0] * image_shape[1], n_heads)),
        tf.keras.layers.Softmax(axis=1)
    ])
    self.q = tf.keras.layers.Dense(conv_filters)
    self.k = tf.keras.layers.Conv2D(conv_filters, 1, padding='same')
    self.v = tf.keras.layers.Conv2D(conv_filters, 1, padding='same')
    self.spatial_basis = tf.constant(
        get_spatial_basis(image_shape[0], image_shape[1],
                          basis_dim)[np.newaxis, :, :, :])

  def __call__(self, obs):
    h, w, _ = self.image_shape
    input_copy = obs.copy()
    batch_size = tf.shape(input_copy[self.image_index_flat])[0]
    spatial_basis_tiled = tf.tile(self.spatial_basis, (batch_size, 1, 1, 1))
    image_features = tf.concat(
        (input_copy[self.image_index_flat], spatial_basis_tiled), axis=-1)
    network_state = self.combiner(input_copy[self.network_state_index_flat])
    query = self.q(network_state)
    keys = self.k(image_features)
    values = self.v(image_features)

    depth_per_head = self.conv_filters // self.n_heads
    q_heads = tf.reshape(query, (-1, 1, 1, self.n_heads, depth_per_head))
    k_heads = tf.reshape(keys, (-1, h, w, self.n_heads, depth_per_head))
    v_heads = tf.reshape(values, (-1, h * w, self.n_heads, depth_per_head))

    attention_weights = tf.reduce_sum(q_heads * k_heads, axis=-1)
    attention_weights = self.attention_network(attention_weights)
    weighted_features = tf.reshape(
        attention_weights[:, :, :, tf.newaxis] * v_heads,
        (-1, h * w, self.conv_filters))
    input_copy[self.image_index_flat] = tf.reduce_sum(weighted_features, axis=1)

    input_copy.pop(self.network_state_index_flat)
    return self.combiner(input_copy)

  def get_config(self):
    return {
        'image_index_flat': self.image_index_flat,
        'network_state_index_flat': self.network_state_index_flat,
        'image_shape': self.image_shape,
        'conv_filters': self.conv_filters,
        'n_heads': self.n_heads,
        'basis_dim': self.basis_dim
    }


def construct_attention_networks(observation_spec,
                                 action_spec,
                                 use_rnns=True,
                                 actor_fc_layers=(200, 100),
                                 value_fc_layers=(200, 100),
                                 lstm_size=(128,),
                                 conv_filters=8,
                                 conv_kernel=3,
                                 scalar_fc=5,
                                 scalar_name='direction',
                                 scalar_dim=4):
  """Creates an actor and critic network designed for use with MultiGrid.

  A convolution layer processes the image and a dense layer processes the
  direction the agent is facing. These are fed into some fully connected layers
  and an LSTM.

  Args:
    observation_spec: A tf-agents observation spec.
    action_spec: A tf-agents action spec.
    use_rnns: If True, will construct RNN networks.
    actor_fc_layers: Dimension and number of fully connected layers in actor.
    value_fc_layers: Dimension and number of fully connected layers in critic.
    lstm_size: Number of cells in each LSTM layers.
    conv_filters: Number of convolution filters.
    conv_kernel: Size of the convolution kernel.
    scalar_fc: Number of neurons in the fully connected layer processing the
      scalar input.
    scalar_name: Name of the scalar input.
    scalar_dim: Highest possible value for the scalar input. Used to convert to
      one-hot representation.

  Returns:
    A tf-agents ActorDistributionRnnNetwork for the actor, and a ValueRnnNetwork
    for the critic.
  """
  preprocessing_layers = {
      'image':
          tf.keras.models.Sequential([
              cast_and_scale(),
              tf.keras.layers.Conv2D(conv_filters, conv_kernel, padding='same'),
              tf.keras.layers.ReLU(),
          ]),
      'policy_state':
          tf.keras.layers.Lambda(lambda x: x)
  }
  if scalar_name in observation_spec:
    preprocessing_layers[scalar_name] = tf.keras.models.Sequential(
        [one_hot_layer(scalar_dim),
         tf.keras.layers.Dense(scalar_fc)])
  if 'position' in observation_spec:
    preprocessing_layers['position'] = tf.keras.models.Sequential(
        [cast_and_scale(), tf.keras.layers.Dense(scalar_fc)])

  preprocessing_nest = tf.nest.map_structure(lambda l: None,
                                             preprocessing_layers)
  flat_observation_spec = nest_utils.flatten_up_to(
      preprocessing_nest,
      observation_spec,
  )
  image_index_flat = flat_observation_spec.index(observation_spec['image'])
  network_state_index_flat = flat_observation_spec.index(
      observation_spec['policy_state'])
  image_shape = observation_spec['image'].shape  # N x H x W x D
  preprocessing_combiner = AttentionCombinerConv(image_index_flat,
                                                 network_state_index_flat,
                                                 image_shape)

  if use_rnns:
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=actor_fc_layers,
        output_fc_layer_params=None,
        lstm_size=lstm_size)
    value_net = value_rnn_network.ValueRnnNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=value_fc_layers,
        output_fc_layer_params=None)
  else:
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=actor_fc_layers,
        activation_fn=tf.keras.activations.tanh)
    value_net = value_network.ValueNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=value_fc_layers,
        activation_fn=tf.keras.activations.tanh)

  return actor_net, value_net
