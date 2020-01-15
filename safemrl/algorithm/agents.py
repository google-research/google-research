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

"""TF-Agents policies, networks, and helpers.

Custom TF-Agents policies, networks, and helpers for Safe SAC.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

import gin
import numpy as np
import tensorflow.compat.v1 as tf
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.policies import actor_policy
from tf_agents.policies import boltzmann_policy
from tf_agents.spec import tensor_spec
from tf_agents.utils import nest_utils


def process_replay_buffer(replay_buffer, max_ep_len=500, k=1, as_tensor=False):
  """Process replay buffer to infer safety rewards with episode boundaries."""
  rb_data = replay_buffer.gather_all()
  rew = rb_data.reward

  boundary_idx = np.where(rb_data.is_boundary().numpy())[1]

  last_idx = 0
  k_labels = []

  for term_idx in boundary_idx:
    # TODO(krshna): remove +1?
    fail = 1 - int(term_idx - last_idx >= max_ep_len + 1)
    ep_rew = tf.gather(rew, np.arange(last_idx, term_idx), axis=1)
    labels = np.zeros(ep_rew.shape_as_list())  # ignore obs dim
    labels[:, Ellipsis, -k:] = fail
    k_labels.append(labels)
    last_idx = term_idx

  flat_labels = np.concatenate(k_labels, axis=-1).astype(np.float32)
  n_flat_labels = flat_labels.shape[1]
  n_rews = rb_data.reward.shape_as_list()[1]
  safe_rew_labels = np.pad(
      flat_labels, ((0, 0), (0, n_rews - n_flat_labels)), mode='constant')
  if as_tensor:
    return tf.to_float(safe_rew_labels)
  return safe_rew_labels


# Pre-processor layers to remove observation from observation dict returned by
# goal-conditioned point-mass environment.
@gin.configurable
def extract_obs_merge_w_ac_layer():
  def f(layer_input):
    return tf.keras.layers.concatenate(
        [layer_input[0]['observation'], layer_input[1]], axis=1)
  return tf.keras.layers.Lambda(f)


@gin.configurable
def extract_observation_layer():
  return tf.keras.layers.Lambda(lambda obs: obs['observation'])


# CriticNetwork constructed with EncodingNetwork
@gin.configurable
class CriticNetwork(network.Network):
  """Critic Network."""

  def __init__(
      self,
      input_tensor_spec,
      preprocessing_combiner=None,
      joint_fc_layer_params=None,
      joint_dropout_layer_params=None,
      kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1. / 3., mode='fan_in', distribution='uniform'),
      activation_fn=tf.nn.relu,
      name='CriticNetwork'):
    """Creates an instance of `CriticNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each a nest of
        `tensor_spec.TensorSpec` representing the inputs.
      preprocessing_combiner: Combiner layer for obs and action inputs
      joint_fc_layer_params: Optional list of fully connected parameters after
        merging observations and actions, where each item is the number of units
        in the layer.
      joint_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of joint_fc_layer_params, or
        be None.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        observation.
    """
    observation_spec, action_spec = input_tensor_spec

    if (len(tf.nest.flatten(observation_spec)) > 1 and
        preprocessing_combiner is None):
      raise ValueError('Only a single observation is supported by this network')

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    preprocessing_layers = None
    # combiner assumes a single batch dimension, without time

    super(CriticNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    self._encoder = encoding_network.EncodingNetwork(
        input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=joint_fc_layer_params,
        dropout_layer_params=joint_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)
    self._value_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003),
        name='value')

  def call(self, observations, step_type, network_state=()):
    state, network_state = self._encoder(
        observations, step_type=step_type, network_state=network_state)
    q_val = self._value_layer(state)
    return tf.reshape(q_val, [-1]), network_state


@gin.configurable
class SafeActorPolicyRSVar(actor_policy.ActorPolicy):
  """Returns safe actions by rejection sampling with increasing variance."""

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               safety_critic_network=None,
               safety_threshold=0.1,
               info_spec=(),
               observation_normalizer=None,
               clip=True,
               resample_metric=None,
               name=None):
    super(SafeActorPolicyRSVar,
          self).__init__(time_step_spec, action_spec, actor_network, info_spec,
                         observation_normalizer, clip, name)
    self._safety_critic_network = safety_critic_network
    self._safety_threshold = safety_threshold
    self._resample_metric = resample_metric

  def _apply_actor_network(self, time_step, policy_state):
    has_batch_dim = time_step.step_type.shape.as_list()[0] > 1
    observation = time_step.observation
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(observation)
    actions, policy_state = self._actor_network(observation,
                                                time_step.step_type,
                                                policy_state)
    if has_batch_dim:
      return actions, policy_state

    # samples "best" safe action out of 50
    sampled_ac = actions.sample(50)
    obs = nest_utils.stack_nested_tensors(
        [time_step.observation for _ in range(50)])
    q_val, _ = self._safety_critic_network((obs, sampled_ac),
                                           time_step.step_type)
    fail_prob = tf.nn.sigmoid(q_val)
    safe_ac_mask = fail_prob < self._safety_threshold
    safe_ac_idx = tf.where(safe_ac_mask)

    resample_count = 0
    while resample_count < 4 and not safe_ac_idx.shape.as_list()[0]:
      if self._resample_metric is not None:
        self._resample_metric()
      resample_count += 1
      scale = actions.scale * 1.5  # increase variance by constant 1.5
      actions = self._actor_network.output_spec.build_distribution(
          loc=actions.loc, scale=scale)
      sampled_ac = actions.sample(50)
      q_val, _ = self._safety_critic_network((obs, sampled_ac),
                                             time_step.step_type)

      fail_prob = tf.nn.sigmoid(q_val)
      safe_ac_idx = tf.where(tf.squeeze(fail_prob) < self._safety_threshold)

    if not safe_ac_idx.shape.as_list()[0]:  # return safest action
      safe_ac_idx = tf.math.argmin(fail_prob)
      return sampled_ac[safe_ac_idx], policy_state

    actions = tf.squeeze(tf.gather(sampled_ac, safe_ac_idx))
    fail_prob_safe = tf.gather(fail_prob, safe_ac_idx)
    safe_idx = tf.math.argmax(fail_prob_safe)
    return actions[safe_idx], policy_state


BoltzmannPolicyInfo = collections.namedtuple('BoltzmannPolicyInfo',
                                             ('temperature',))


@gin.configurable
class SafetyBoltzmannPolicy(boltzmann_policy.BoltzmannPolicy):
  """A policy that awares safety."""

  def __init__(self, policy, temperature=1.0, name=None):
    super(SafetyBoltzmannPolicy, self).__init__(policy, temperature, name)
    info_spec = BoltzmannPolicyInfo(
        temperature=tensor_spec.TensorSpec((), tf.float32, name='temperature'))
    self._info_spec = info_spec
    self._setup_specs()  # run again to make sure specs are correctly updated

  def _distribution(self, time_step, policy_state):
    distribution_step = super(SafetyBoltzmannPolicy,
                              self)._distribution(time_step, policy_state)
    distribution_step = distribution_step._replace(
        info=BoltzmannPolicyInfo(temperature=self._temperature))
    return distribution_step
