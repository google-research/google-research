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

import tensorflow.compat.v2 as tf

from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network


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


def construct_multigrid_networks(observation_spec, action_spec, use_rnns=True,
                                 actor_fc_layers=(200, 100),
                                 value_fc_layers=(200, 100), lstm_size=(128,),
                                 conv_filters=8, conv_kernel=3, scalar_fc=5,
                                 scalar_name='direction', scalar_dim=4,
                                 random_z=False, xy_dim=None):
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
    scalar_fc: Number of neurons in the fully connected layer processing
      the scalar input.
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
      'image': tf.keras.models.Sequential(
          [cast_and_scale(),
           tf.keras.layers.Conv2D(conv_filters, conv_kernel),
           tf.keras.layers.Flatten()]),
      scalar_name: tf.keras.models.Sequential(
          [one_hot_layer(scalar_dim),
           tf.keras.layers.Dense(scalar_fc)])
      }
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
        output_fc_layer_params=None,
        lstm_size=lstm_size)
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

