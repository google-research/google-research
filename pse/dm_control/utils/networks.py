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

"""Networks required for DrQ and DrQ + PSEs.

Architectures, shapes, and hparams referenced from the DrQ paper:
  https://arxiv.org/pdf/2004.13649.pdf
"""
import math
import gin
import tensorflow as tf

from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.networks import network
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity


@gin.configurable
class ImageEncoder(network.Network):
  """ImageEncoder network used in CURL to embed into query/key pairs.

  Network hardcodes parameters used in the CURL paper for dm_control
  environments. Note we can generalize this down the line if we want to explore.
  """

  def __init__(self, input_tensor_spec, num_conv_layers=4, name='ImageEncoder'):
    super(ImageEncoder, self).__init__(input_tensor_spec, (), name=name)
    # Because of Keras we need to assign to self AFTER we call super.

    if (not isinstance(input_tensor_spec, dict) and
        list(input_tensor_spec[0].keys()) != ['pixels']):
      raise ValueError('ImageEncoder input is espected to be of the form: '
                       'dict(pixels=<image_data>)')

    sqrt2 = math.sqrt(2.0)
    self._conv_encoder = tf.keras.Sequential()
    self._conv_encoder.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.Orthogonal(gain=sqrt2)))

    for _ in range(num_conv_layers - 1):
      self._conv_encoder.add(
          tf.keras.layers.Conv2D(
              filters=32,
              kernel_size=(3, 3),
              strides=(1, 1),
              activation=tf.keras.activations.relu,
              kernel_initializer=tf.keras.initializers.Orthogonal(gain=sqrt2)))

    self._conv_encoder.add(tf.keras.layers.Flatten())

  def call(self, observation, step_type=None, network_state=(), training=False):
    del step_type  # unused.
    img = observation['pixels']
    if img.dtype == tf.uint8:
      img = tf.cast(img, tf.float32)
      img = img / 255.0

    states = self._conv_encoder(img, training=training)
    return states, network_state


@gin.configurable
class Critic(network.Network):
  """CriticNetwork.

  Network has an ImageEncoder for the image input, this gets concatenated with
  the action, and finally passed through 2 dense layers.
  """

  def __init__(self,
               input_tensor_spec,
               image_encoder=None,
               fc_encoder_layers=(50,),
               joint_fc_layers=(1024, 1024),
               name='Critic'):
    """Initializes a Critic network.

    Args:
      input_tensor_spec: Tuple of the form (observation_spec, action_spec)
      image_encoder: Optional ImageEncoder used on the input observation. If
        None the network assumes the observation is a vector and this network
        simply applies fc layers.
      fc_encoder_layers: Dense layers followed by LayerNorm and a tanh.
      joint_fc_layers: Iterable (list or tuple) with number of units for a dense
        layer stack applied on the joint observation and action.
      name: Name for the network.
    """
    super(Critic, self).__init__(input_tensor_spec, (), name=name)

    # Because of Keras we need to assign to self AFTER we call super.
    self._image_encoder = image_encoder

    self._fc_encoder = tf.keras.Sequential()
    for fc_layer_units in fc_encoder_layers:
      self._fc_encoder.add(
          tf.keras.layers.Dense(
              fc_layer_units,
              # Default gain of 1.0 matches Pytorch.
              kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
          ))
    # 1e-5 is default epsilon in Pytorch.
    self._fc_encoder.add(tf.keras.layers.LayerNormalization(epsilon=1e-5))
    self._fc_encoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    self._dense_layers = tf.keras.Sequential()
    for units in joint_fc_layers:
      self._dense_layers.add(
          tf.keras.layers.Dense(
              units,
              activation=tf.keras.activations.relu,
              kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0)))

    # Project to a scalar value estimate with no activation for the last layer.
    self._dense_layers.add(
        tf.keras.layers.Dense(
            1, kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0)))

  def call(self, inputs, step_type=(), network_state=(), training=False):
    del step_type  # unused.
    observations, actions = inputs

    if self._image_encoder:
      encoded_image, network_state = self._image_encoder(
          observations, training=training)
      encoded_image = self._fc_encoder(encoded_image)
      joint = tf.keras.layers.concatenate([encoded_image, actions])
    else:
      joint = tf.keras.layers.concatenate(
          [observations['position'], observations['velocity'], actions])

    value = self._dense_layers(joint)
    return tf.reshape(value, [-1]), network_state


@gin.configurable
class Actor(network.DistributionNetwork):
  """Actor for use with SAC and CURL on dm_control environments.

  The network will use an ImageEncoder as specified by the CURL paper, a stack
  of dense layers, and then returns a tanh-squashed MultivariateNormalDiag
  distribution as done in the SAC paper.
  """

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               image_encoder=None,
               fc_encoder_layers=(50,),
               fc_layers=(1024, 1024),
               log_std_bounds=(-10, 2),
               name='Actor'):
    """Initializes an Actor network.

    Args:
      input_tensor_spec: Tensor spec matching the environment's
        observation_spec.
      output_tensor_spec: Tensor spec matching the environment's action_spec.
      image_encoder: Optional ImageEncoder used on the input observation. If
        None the network assumes the observation is a vector and this network
        simply applies fc layers.
      fc_encoder_layers: Dense layers followed by LayerNorm and a tanh.
      fc_layers: Iterable (list or tuple) with number of units for a dense layer
        stack.
      log_std_bounds: Bounds for scaling the log_std in the
        TanhNormalProjectionNetwork.
      name: Name for the network.
    """

    def scale_and_exp(log_std):
      scale_min, scale_max = log_std_bounds
      log_std = tf.keras.activations.tanh(log_std)
      log_std = scale_min + 0.5 * (scale_max - scale_min) * (log_std + 1)
      return tf.exp(log_std)

    distribution_projection_network = (
        tanh_normal_projection_network.TanhNormalProjectionNetwork(
            output_tensor_spec, std_transform=scale_and_exp))

    super(Actor, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        output_spec=distribution_projection_network.output_spec,
        name=name)

    # Because of Keras we need to assign to self AFTER we call super.
    self._image_encoder = image_encoder

    self._fc_encoder = tf.keras.Sequential()
    for fc_layer_units in fc_encoder_layers:
      self._fc_encoder.add(
          tf.keras.layers.Dense(
              fc_layer_units,
              # Default gain of 1.0 matches Pytorch.
              kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0),
          ))
    # 1e-5 is default epsilon in Pytorch.
    self._fc_encoder.add(tf.keras.layers.LayerNormalization(epsilon=1e-5))
    self._fc_encoder.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

    self._dense_layers = tf.keras.Sequential()

    for units in fc_layers:
      self._dense_layers.add(
          tf.keras.layers.Dense(
              units,
              activation=tf.keras.activations.relu,
              kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0)))
    self._distribution_projection_network = distribution_projection_network

  @property
  def trainable_variables(self):
    """Override trainable_variables property to remove encoder_variables."""
    if self._image_encoder:
      encoder_variables = object_identity.ObjectIdentitySet(
          self._image_encoder.trainable_variables)
      return [
          v for v in super(Actor, self).trainable_variables
          if v not in encoder_variables
      ]
    else:
      return super(Actor, self).trainable_variables

  def call(self, observations, step_type=(), network_state=(), training=False):
    if self._image_encoder:
      encoded, network_state = self._image_encoder(
          observations, training=training)
      encoded = self._fc_encoder(encoded)
    else:
      # dm_control state observations need to be flattened as they are
      # structured as a dict(position, velocity)
      encoded = tf.keras.layers.concatenate(
          [observations['position'], observations['velocity']])

    encoded = self._dense_layers(encoded)

    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    action_distribution, network_state = self._distribution_projection_network(
        encoded, outer_rank, training=training)

    return action_distribution, network_state
