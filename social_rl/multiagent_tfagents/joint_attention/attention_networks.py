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

"""Creates actor and critic networks with attention architecture.

Also implements attention version of standard TFAgents Networks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np
from six.moves import zip
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import categorical_projection_network
from tf_agents.networks import encoding_network
from tf_agents.networks import lstm_encoding_network
from tf_agents.networks import network
from tf_agents.networks import normal_projection_network
from tf_agents.networks import utils
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step
from tf_agents.utils import nest_utils

from social_rl.multiagent_tfagents import multigrid_networks


class _Stack(tf.keras.layers.Layer):
  """Stack of pooling and convolutional blocks with residual connections."""

  def __init__(self, num_ch, num_blocks, **kwargs):
    # pylint: disable=g-complex-comprehension
    super(_Stack, self).__init__(**kwargs)
    self.num_ch = num_ch
    self.num_blocks = num_blocks
    self._conv = tf.keras.layers.Conv2D(
        num_ch, 3, strides=1, padding="same", kernel_initializer="lecun_normal")
    self._max_pool = tf.keras.layers.MaxPool2D(
        pool_size=3, padding="same", strides=2)

    self._res_convs0 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding="same", name="res_%d/conv2d_0" % i,
            kernel_initializer="lecun_normal")
        for i in range(num_blocks)
    ]
    self._res_convs1 = [
        tf.keras.layers.Conv2D(
            num_ch, 3, strides=1, padding="same", name="res_%d/conv2d_1" % i,
            kernel_initializer="lecun_normal")
        for i in range(num_blocks)
    ]

  def __call__(self, conv_out, training=False):
    # Downscale.
    conv_out = self._conv(conv_out)
    conv_out = self._max_pool(conv_out)

    # Residual block(s).
    for (res_conv0, res_conv1) in zip(self._res_convs0, self._res_convs1):
      block_input = conv_out
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv0(conv_out)
      conv_out = tf.nn.relu(conv_out)
      conv_out = res_conv1(conv_out)
      conv_out += block_input

    return conv_out

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        "num_ch": self.num_ch,
        "num_blocks": self.num_blocks
    })
    return config


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
               n_heads=4,
               basis_dim=16):
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
    self.k = tf.keras.layers.Conv2D(conv_filters, 1, padding="same")
    self.v = tf.keras.layers.Conv2D(conv_filters, 1, padding="same")
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
    mean_attention_weights = tf.reshape(
        tf.reduce_mean(attention_weights, axis=-1), (-1, h, w))
    weighted_features = tf.reshape(
        attention_weights[:, :, :, tf.newaxis] * v_heads,
        (-1, h * w, self.conv_filters))
    input_copy[self.image_index_flat] = tf.reduce_sum(weighted_features, axis=1)

    input_copy.pop(self.network_state_index_flat)
    return self.combiner(input_copy), mean_attention_weights

  def get_config(self):
    return {
        "image_index_flat": self.image_index_flat,
        "network_state_index_flat": self.network_state_index_flat,
        "image_shape": self.image_shape,
        "conv_filters": self.conv_filters,
        "n_heads": self.n_heads,
        "basis_dim": self.basis_dim
    }


@gin.configurable
def construct_attention_networks(observation_spec,
                                 action_spec,
                                 use_rnns=True,
                                 actor_fc_layers=(200, 100),
                                 value_fc_layers=(200, 100),
                                 lstm_size=(128,),
                                 conv_filters=8,
                                 conv_kernel=3,
                                 scalar_fc=5,
                                 scalar_name="direction",
                                 scalar_dim=4,
                                 use_stacks=False,
                                 ):
  """Creates an actor and critic network designed for use with MultiGrid.

  A convolution layer processes the image and a dense layer processes the
  direction the agent is facing. These are fed into some fully connected layers
  and an LSTM.

  Args:
    observation_spec: A tf-agents observation spec.
    action_spec: A tf-agents action spec.
    use_rnns: If True, will construct RNN networks. Non-recurrent networks are
      not supported currently.
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
    use_stacks: Use ResNet stacks (compresses the image).

  Returns:
    A tf-agents ActorDistributionRnnNetwork for the actor, and a ValueRnnNetwork
    for the critic.
  """
  if not use_rnns:
    raise NotImplementedError(
        "Non-recurrent attention networks are not suppported.")
  preprocessing_layers = {
      "policy_state":
          tf.keras.layers.Lambda(lambda x: x)
  }
  if use_stacks:
    preprocessing_layers["image"] = tf.keras.models.Sequential([
        multigrid_networks.cast_and_scale(),
        _Stack(conv_filters // 2, 2),
        _Stack(conv_filters, 2),
        tf.keras.layers.ReLU(),
    ])
  else:
    preprocessing_layers["image"] = tf.keras.models.Sequential([
        multigrid_networks.cast_and_scale(),
        tf.keras.layers.Conv2D(conv_filters, conv_kernel, padding="same"),
        tf.keras.layers.ReLU(),
    ])
  if scalar_name in observation_spec:
    preprocessing_layers[scalar_name] = tf.keras.models.Sequential(
        [multigrid_networks.one_hot_layer(scalar_dim),
         tf.keras.layers.Dense(scalar_fc)])
  if "position" in observation_spec:
    preprocessing_layers["position"] = tf.keras.models.Sequential(
        [multigrid_networks.cast_and_scale(), tf.keras.layers.Dense(scalar_fc)])

  preprocessing_nest = tf.nest.map_structure(lambda l: None,
                                             preprocessing_layers)
  flat_observation_spec = nest_utils.flatten_up_to(
      preprocessing_nest,
      observation_spec,
  )
  image_index_flat = flat_observation_spec.index(observation_spec["image"])
  network_state_index_flat = flat_observation_spec.index(
      observation_spec["policy_state"])
  if use_stacks:
    image_shape = [i // 4 for i in observation_spec["image"].shape]  # H x W x D
  else:
    image_shape = observation_spec["image"].shape
  preprocessing_combiner = AttentionCombinerConv(image_index_flat,
                                                 network_state_index_flat,
                                                 image_shape)

  custom_objects = {"_Stack": _Stack}
  with tf.keras.utils.custom_object_scope(custom_objects):
    actor_net = AttentionActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=actor_fc_layers,
        output_fc_layer_params=None,
        lstm_size=lstm_size)
    value_net = AttentionValueRnnNetwork(
        observation_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        input_fc_layer_params=value_fc_layers,
        output_fc_layer_params=None)

  return actor_net, value_net


@gin.configurable
def construct_multigrid_networks(observation_spec,
                                 action_spec,
                                 use_rnns=True,
                                 actor_fc_layers=(200, 100),
                                 value_fc_layers=(200, 100),
                                 lstm_size=(128,),
                                 conv_filters=8,
                                 conv_kernel=3,
                                 scalar_fc=5,
                                 scalar_name="direction",
                                 scalar_dim=4,
                                 use_stacks=False,
                                 ):
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
    use_stacks: Use ResNet stacks (compresses the image).

  Returns:
    A tf-agents ActorDistributionRnnNetwork for the actor, and a ValueRnnNetwork
    for the critic.
  """

  preprocessing_layers = {
      "policy_state":
          tf.keras.layers.Lambda(lambda x: x)
  }
  if use_stacks:
    preprocessing_layers["image"] = tf.keras.models.Sequential([
        multigrid_networks.cast_and_scale(),
        _Stack(conv_filters // 2, 2),
        _Stack(conv_filters, 2),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten()
    ])
  else:
    preprocessing_layers["image"] = tf.keras.models.Sequential([
        multigrid_networks.cast_and_scale(),
        tf.keras.layers.Conv2D(conv_filters, conv_kernel, padding="same"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten()
    ])
  if scalar_name in observation_spec:
    preprocessing_layers[scalar_name] = tf.keras.models.Sequential(
        [multigrid_networks.one_hot_layer(scalar_dim),
         tf.keras.layers.Dense(scalar_fc)])
  if "position" in observation_spec:
    preprocessing_layers["position"] = tf.keras.models.Sequential(
        [multigrid_networks.cast_and_scale(), tf.keras.layers.Dense(scalar_fc)])

  preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

  custom_objects = {"_Stack": _Stack}
  with tf.keras.utils.custom_object_scope(custom_objects):
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


def _categorical_projection_net(action_spec, logits_init_output_factor=0.1):
  return categorical_projection_network.CategoricalProjectionNetwork(
      action_spec, logits_init_output_factor=logits_init_output_factor)


def _normal_projection_net(action_spec,
                           init_action_stddev=0.35,
                           init_means_output_factor=0.1):
  std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      init_means_output_factor=init_means_output_factor,
      std_bias_initializer_value=std_bias_initializer_value)


class AttentionNetwork(network.Network):
  """A modification of tf_agents network that returns attention info."""

  def __call__(self, inputs, *args, **kwargs):
    """A wrapper around `Network.call`.

    A typical `call` method in a class subclassing `Network` will have a
    signature that accepts `inputs`, as well as other `*args` and `**kwargs`.
    `call` can optionally also accept `step_type` and `network_state`
    (if `state_spec != ()` is not trivial).  e.g.:

    ```python
    def call(self,
             inputs,
             step_type=None,
             network_state=(),
             training=False):
        ...
        return outputs, new_network_state
    ```

    We will validate the first argument (`inputs`)
    against `self.input_tensor_spec` if one is available.

    If a `network_state` kwarg is given it is also validated against
    `self.state_spec`.  Similarly, the return value of the `call` method is
    expected to be a tuple/list with 2 values:  `(output, new_state)`.
    We validate `new_state` against `self.state_spec`.

    If no `network_state` kwarg is given (or if empty `network_state = ()` is
    given, it is up to `call` to assume a proper "empty" state, and to
    emit an appropriate `output_state`.

    Args:
      inputs: The input to `self.call`, matching `self.input_tensor_spec`.
      *args: Additional arguments to `self.call`.
      **kwargs: Additional keyword arguments to `self.call`. These can include
        `network_state` and `step_type`.  `step_type` is required if the
        network"s `call` requires it. `network_state` is required if the
        underlying network"s `call` requires it.

    Returns:
      A tuple `(outputs, new_network_state)`.
    """
    if self.input_tensor_spec is not None:
      nest_utils.assert_matching_dtypes_and_inner_shapes(
          inputs,
          self.input_tensor_spec,
          allow_extra_fields=True,
          caller=self,
          tensors_name="`inputs`",
          specs_name="`input_tensor_spec`")

    call_argspec = network.tf_inspect.getargspec(self.call)

    # Convert *args, **kwargs to a canonical kwarg representation.
    normalized_kwargs = network.tf_inspect.getcallargs(self.call, inputs, *args,
                                                       **kwargs)
    network_state = normalized_kwargs.get("network_state", None)
    normalized_kwargs.pop("self", None)

    # pylint: disable=literal-comparison
    network_has_state = (
        network_state is not None and network_state is not () and
        network_state is not [])
    # pylint: enable=literal-comparison

    if network_has_state:
      nest_utils.assert_matching_dtypes_and_inner_shapes(
          network_state,
          self.state_spec,
          allow_extra_fields=True,
          caller=self,
          tensors_name="`network_state`",
          specs_name="`state_spec`")

    if "step_type" not in call_argspec.args and not call_argspec.keywords:
      normalized_kwargs.pop("step_type", None)

    if (network_state in (None, ()) and
        "network_state" not in call_argspec.args and not call_argspec.keywords):
      normalized_kwargs.pop("network_state", None)

    outputs, new_state, attention_weights = tf.keras.layers.Layer.__call__(
        self, **normalized_kwargs)

    nest_utils.assert_matching_dtypes_and_inner_shapes(
        new_state,
        self.state_spec,
        allow_extra_fields=True,
        caller=self,
        tensors_name="`new_state`",
        specs_name="`state_spec`")

    return outputs, new_state, attention_weights


class AttentionDistributionNetwork(
    AttentionNetwork,
    network.DistributionNetwork,
):

  def __call__(self, inputs, *args, **kwargs):
    return AttentionNetwork.__call__(self, inputs, *args, **kwargs)


class AttentionEncodingNetwork(AttentionNetwork,
                               encoding_network.EncodingNetwork):
  """A modification of tf_agents encoding network that returns attention info."""

  def __call__(self, inputs, *args, **kwargs):
    return AttentionNetwork.__call__(self, inputs, *args, **kwargs)

  def call(self, observation, step_type=None, network_state=(), training=False):
    del step_type  # unused.

    if self._batch_squash:
      outer_rank = nest_utils.get_outer_rank(observation,
                                             self.input_tensor_spec)
      batch_squash = utils.BatchSquash(outer_rank)
      observation = tf.nest.map_structure(batch_squash.flatten, observation)

    if self._flat_preprocessing_layers is None:
      processed = observation
    else:
      processed = []
      for obs, layer in zip(
          nest_utils.flatten_up_to(self._preprocessing_nest, observation),
          self._flat_preprocessing_layers):
        processed.append(layer(obs, training=training))
      if len(processed) == 1 and self._preprocessing_combiner is None:
        # If only one observation is passed and the preprocessing_combiner
        # is unspecified, use the preprocessed version of this observation.
        processed = processed[0]

    states = processed

    if self._preprocessing_combiner is not None:
      states, attention_weights = self._preprocessing_combiner(states)

    for layer in self._postprocessing_layers:
      states = layer(states, training=training)

    if self._batch_squash:
      states = tf.nest.map_structure(batch_squash.unflatten, states)

    return states, network_state, attention_weights


class AttentionLSTMEncodingNetwork(AttentionNetwork,
                                   lstm_encoding_network.LSTMEncodingNetwork):
  """A modification of tf_agents LSTM encoding network that returns attention info."""

  def __init__(
      self,
      input_tensor_spec,
      preprocessing_layers=None,
      preprocessing_combiner=None,
      conv_layer_params=None,
      input_fc_layer_params=(75, 40),
      lstm_size=None,
      output_fc_layer_params=(75, 40),
      activation_fn=tf.keras.activations.relu,
      rnn_construction_fn=None,
      rnn_construction_kwargs=None,
      dtype=tf.float32,
      name="LSTMEncodingNetwork",
  ):
    super(AttentionLSTMEncodingNetwork,
          self).__init__(input_tensor_spec, preprocessing_layers,
                         preprocessing_combiner, conv_layer_params,
                         input_fc_layer_params, lstm_size,
                         output_fc_layer_params, activation_fn,
                         rnn_construction_fn, rnn_construction_kwargs, dtype,
                         name)

    kernel_initializer = tf.compat.v1.variance_scaling_initializer(
        scale=2.0, mode="fan_in", distribution="truncated_normal")
    input_encoder = AttentionEncodingNetwork(
        input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=input_fc_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        dtype=dtype)
    self._input_encoder = input_encoder

  def __call__(self, inputs, *args, **kwargs):
    return AttentionNetwork.__call__(self, inputs, *args, **kwargs)

  def call(self, observation, step_type, network_state=(), training=False):
    """Apply the network.

    Args:
      observation: A tuple of tensors matching `input_tensor_spec`.
      step_type: A tensor of `StepType.
      network_state: (optional.) The network state.
      training: Whether the output is being used for training.

    Returns:
      `(outputs, network_state)` - the network output and next network state.

    Raises:
      ValueError: If observation tensors lack outer `(batch,)` or
        `(batch, time)` axes.
    """
    num_outer_dims = nest_utils.get_outer_rank(observation,
                                               self.input_tensor_spec)
    if num_outer_dims not in (1, 2):
      raise ValueError(
          "Input observation must have a batch or batch x time outer shape.")

    has_time_dim = num_outer_dims == 2
    if not has_time_dim:
      # Add a time dimension to the inputs.
      observation = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                          observation)
      step_type = tf.nest.map_structure(lambda t: tf.expand_dims(t, 1),
                                        step_type)

    state, _, attention_weights = self._input_encoder(
        observation, step_type=step_type, network_state=(), training=training)

    network_kwargs = {}
    if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
      network_kwargs["reset_mask"] = tf.equal(
          step_type, time_step.StepType.FIRST, name="mask")

    # Unroll over the time sequence.
    output = self._lstm_network(
        inputs=state,
        initial_state=network_state,
        training=training,
        **network_kwargs)

    if isinstance(self._lstm_network, dynamic_unroll_layer.DynamicUnroll):
      state, network_state = output
    else:
      state = output[0]
      network_state = tf.nest.pack_sequence_as(
          self._lstm_network.cell.state_size, tf.nest.flatten(output[1:]))

    for layer in self._output_encoder:
      state = layer(state, training=training)

    if not has_time_dim:
      # Remove time dimension from the state.
      state = tf.squeeze(state, [1])

    return state, network_state, attention_weights


@gin.configurable
class AttentionActorDistributionRnnNetwork(AttentionDistributionNetwork):
  """A modification of tf_agents rnn network that returns attention info."""

  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               input_fc_layer_params=(200, 100),
               input_dropout_layer_params=None,
               lstm_size=None,
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               dtype=tf.float32,
               discrete_projection_net=_categorical_projection_net,
               continuous_projection_net=_normal_projection_net,
               rnn_construction_fn=None,
               rnn_construction_kwargs=None,
               name="ActorDistributionRnnNetwork"):
    """Creates an instance of `ActorDistributionRnnNetwork`.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations. All of these
        layers must not be already built. For more details see the documentation
        of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include `tf.keras.layers.Add`
        and `tf.keras.layers.Concatenate(axis=-1)`. This layer must not be
        already built. For more details see the documentation of
        `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      input_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied before
        the LSTM cell.
      input_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent", if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of input_fc_layer_params, or
        be None.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      dtype: The dtype to use by the convolution and fully connected layers.
      discrete_projection_net: Callable that generates a discrete projection
        network to be called with some hidden state and the outer_rank of the
        state.
      continuous_projection_net: Callable that generates a continuous projection
        network to be called with some hidden state and the outer_rank of the
        state.
      rnn_construction_fn: (Optional.) Alternate RNN construction function, e.g.
        tf.keras.layers.LSTM, tf.keras.layers.CuDNNLSTM. It is invalid to
        provide both rnn_construction_fn and lstm_size.
      rnn_construction_kwargs: (Optional.) Dictionary or arguments to pass to
        rnn_construction_fn.
        The RNN will be constructed via:  ``` rnn_layer =
          rnn_construction_fn(**rnn_construction_kwargs) ```
      name: A string representing name of the network.

    Raises:
      ValueError: If "input_dropout_layer_params" is not None.
    """
    if input_dropout_layer_params:
      raise ValueError("Dropout layer is not supported.")

    lstm_encoder = AttentionLSTMEncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=activation_fn,
        rnn_construction_fn=rnn_construction_fn,
        rnn_construction_kwargs=rnn_construction_kwargs,
        dtype=dtype,
        name=name)

    def map_proj(spec):
      if tensor_spec.is_discrete(spec):
        return discrete_projection_net(spec)
      else:
        return continuous_projection_net(spec)

    projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
    output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                        projection_networks)

    super(AttentionActorDistributionRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=lstm_encoder.state_spec,
        output_spec=output_spec,
        name=name)

    self._lstm_encoder = lstm_encoder
    self._projection_networks = projection_networks
    self._output_tensor_spec = output_tensor_spec

  def __call__(self, inputs, *args, **kwargs):
    return AttentionDistributionNetwork.__call__(self, inputs, *args, **kwargs)

  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec

  def call(self, observation, step_type, network_state=(), training=False):
    state, network_state, attention_weights = self._lstm_encoder(
        observation,
        step_type=step_type,
        network_state=network_state,
        training=training)
    outer_rank = nest_utils.get_outer_rank(observation, self.input_tensor_spec)
    output_actions = tf.nest.map_structure(
        lambda proj_net: proj_net(state, outer_rank, training=training)[0],
        self._projection_networks)
    return output_actions, network_state, attention_weights


@gin.configurable
class AttentionValueRnnNetwork(network.Network):
  """Recurrent value network. Reduces to 1 value output per batch item."""

  def __init__(self,
               input_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               input_fc_layer_params=(75, 40),
               input_dropout_layer_params=None,
               lstm_size=(40,),
               output_fc_layer_params=(75, 40),
               activation_fn=tf.keras.activations.relu,
               dtype=tf.float32,
               name="ValueRnnNetwork"):
    """Creates an instance of `ValueRnnNetwork`.

    Network supports calls with shape outer_rank + input_tensor_shape.shape.
    Note outer_rank must be at least 1.

    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input observations.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations. All of these
        layers must not be already built. For more details see the documentation
        of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them.  Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`. This
        layer must not be already built. For more details see the documentation
        of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      input_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied before
        the LSTM cell.
      input_dropout_layer_params: Optional list of dropout layer parameters,
        where each item is the fraction of input units to drop. The dropout
        layers are interleaved with the fully connected layers; there is a
        dropout layer after each fully connected layer, except if the entry in
        the list is None. This list must have the same length of
        input_fc_layer_params, or be None.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.keras.activations.relu,.
      dtype: The dtype to use by the convolution, LSTM, and fully connected
        layers.
      name: A string representing name of the network.
    """
    del input_dropout_layer_params

    lstm_encoder = AttentionLSTMEncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=activation_fn,
        dtype=dtype,
        name=name)

    postprocessing_layers = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(
            minval=-0.03, maxval=0.03))

    super(AttentionValueRnnNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=lstm_encoder.state_spec,
        name=name)

    self._lstm_encoder = lstm_encoder
    self._postprocessing_layers = postprocessing_layers

  def call(self, observation, step_type=None, network_state=(), training=False):
    state, network_state, _ = self._lstm_encoder(
        observation,
        step_type=step_type,
        network_state=network_state,
        training=training)
    value = self._postprocessing_layers(state, training=training)
    return tf.squeeze(value, -1), network_state
