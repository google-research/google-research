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

"""Defines networks for residual training."""

from acme.tf import networks
from acme.tf import utils as tf_utils

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class ArmPolicyNormalDiagHead(snt.Module):
  """Policy head corresponding to the BC network structure in rrlfd/bc/bc_agent.

  If binary_grip_action is enabled, it uses the first two logits to predict a
  binary action and the remaining logits for regression.
  """

  def __init__(
      self,
      binary_grip_action,
      num_dimensions,
      init_scale=0.3,
      min_scale=1e-6,
      tanh_mean=False,
      fixed_scale=False,
      use_tfd_independent=False,
      w_init=tf.initializers.VarianceScaling(1e-4),
      b_init=tf.initializers.Zeros()):
    super().__init__(name='ArmPolicyNormalDiagHead')
    self._binary_grip_action = binary_grip_action
    self._init_scale = init_scale
    self._min_scale = min_scale
    self._tanh_mean = tanh_mean
    # Important when initializing from a BC network using binary grip action
    # (so one extra logit).
    mean_dimensions = (
        num_dimensions + 1 if binary_grip_action else num_dimensions)
    self._mean_layer = snt.Linear(
        mean_dimensions, w_init=w_init, b_init=b_init, name='mean')
    self._fixed_scale = fixed_scale

    if not fixed_scale:
      self._scale_layer = snt.Linear(
          num_dimensions, w_init=w_init, b_init=b_init, name='scale')
    self._use_tfd_independent = use_tfd_independent

  def __call__(self, inputs):
    inputs = tf_utils.batch_concat(inputs)
    zero = tf.constant(0, dtype=inputs.dtype)
    mean = self._mean_layer(inputs)
    if self._binary_grip_action:
      grip_pred = tf.gather(
          tf.constant([2.0, -2.0]), tf.math.argmax(mean[:, :2], axis=1))
      mean = tf.concat([tf.expand_dims(grip_pred, axis=1), mean[:, 2:]], axis=1)

    if self._fixed_scale:
      scale = tf.ones_like(mean) * self._init_scale
    else:
      scale = tf.nn.softplus(self._scale_layer(inputs))
      scale *= self._init_scale / tf.nn.softplus(zero)
      scale += self._min_scale

    # Maybe transform the mean.
    if self._tanh_mean:
      mean = tf.tanh(mean)

    if self._use_tfd_independent:
      dist = tfd.Independent(tfd.Normal(loc=mean, scale=scale))
    else:
      dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=scale)

    return dist


# class MultiInputSequential(snt.Module):
#   """Extend Sequential to accept multiple inputs / args in call."""

#   def __init__(self,
#                layers: Iterable[Callable[..., Any]] = None,
#                name: Optional[Text] = None):
#     super(MultiInputSequential, self).__init__(name=name)
#     self._layers = list(layers) if layers is not None else []

#     # no batch norm in net
#   def __call__(self, inputs, training=False):
#     outputs = inputs
#     for i, mod in enumerate(self._layers):
#       if i == 0:
#         # Pass additional arguments to the first layer.
#         outputs = mod(outputs, *args, **kwargs)
#       else:
#         outputs = mod(outputs)
#     return outputs


def make_bc_network(
    action_spec,
    policy_layer_sizes=(256, 256, 256),
    policy_init_std=1e-9,
    binary_grip_action=False):
  """Residual BC network in Sonnet, equivalent to residual policy network."""
  num_dimensions = np.prod(action_spec.shape, dtype=int)
  if policy_layer_sizes:
    policy_network = snt.Sequential([
        tf_utils.batch_concat,
        networks.LayerNormMLP([int(l) for l in policy_layer_sizes]),
        networks.MultivariateNormalDiagHead(
            num_dimensions,
            init_scale=policy_init_std,
            min_scale=1e-10)
    ])
  else:
    policy_network = snt.Sequential([
        tf_utils.batch_concat,
        ArmPolicyNormalDiagHead(
            binary_grip_action=binary_grip_action,
            num_dimensions=num_dimensions,
            init_scale=policy_init_std,
            min_scale=1e-10)
    ])
  return {
      # 'observation': tf_utils.batch_concat,
      'policy': policy_network,
  }


def make_mpo_networks(
    action_spec,
    policy_layer_sizes=(256, 256, 256),
    critic_layer_sizes=(512, 512, 256),
    policy_init_std=1e-9,
    obs_network=None):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  critic_layer_sizes = list(critic_layer_sizes) + [1]

  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          init_scale=policy_init_std,
          min_scale=1e-10)
  ])
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes),
      action_network=networks.ClipToSpec(action_spec))
  if obs_network is None:
    obs_network = tf_utils.batch_concat

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': obs_network,
  }


def make_d4pg_networks(
    action_spec,
    policy_layer_sizes=(256, 256, 256),
    critic_layer_sizes=(512, 512, 256),
    vmin=-150.,
    vmax=150.,
    num_atoms=51,
    policy_weights_init_scale=0.333,
    obs_network=None):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  policy_layer_sizes = list(policy_layer_sizes) + [int(num_dimensions)]

  policy_network = snt.Sequential([
      networks.LayerNormMLP(
          policy_layer_sizes,
          init_scale=policy_weights_init_scale),
      networks.TanhToSpec(action_spec)
  ])

  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = snt.Sequential([
      networks.CriticMultiplexer(
          critic_network=networks.LayerNormMLP(
              critic_layer_sizes, activate_final=True)),
      networks.DiscreteValuedHead(vmin, vmax, num_atoms)
  ])
  if obs_network is None:
    obs_network = tf_utils.batch_concat

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': obs_network,
  }


def make_dmpo_networks(
    action_spec,
    policy_layer_sizes=(256, 256, 256),
    critic_layer_sizes=(512, 512, 256),
    vmin=-150.,
    vmax=150.,
    num_atoms=51,
    policy_init_std=1e-9,
    obs_network=None,
    binary_grip_action=False):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  if policy_layer_sizes:
    policy_network = snt.Sequential([
        networks.LayerNormMLP([int(l) for l in policy_layer_sizes]),
        networks.MultivariateNormalDiagHead(
            num_dimensions,
            init_scale=policy_init_std,
            min_scale=1e-10)
    ])
  else:
    # Useful when initializing from a trained BC network.
    policy_network = snt.Sequential([
        ArmPolicyNormalDiagHead(
            binary_grip_action=binary_grip_action,
            num_dimensions=num_dimensions,
            init_scale=policy_init_std,
            min_scale=1e-10)
    ])
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes),
      action_network=networks.ClipToSpec(action_spec))
  critic_network = snt.Sequential(
      [critic_network,
       networks.DiscreteValuedHead(vmin, vmax, num_atoms)])
  if obs_network is None:
    obs_network = tf_utils.batch_concat

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': obs_network,
  }
