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

"""Implicit Quantile agent with MICo loss."""

import collections
import functools

from dopamine.jax import losses
from dopamine.jax.agents.implicit_quantile import implicit_quantile_agent
from flax import nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf

from mico.atari import metric_utils

NetworkType = collections.namedtuple(
    'network', ['quantile_values', 'quantiles', 'representation'])


def stable_scaled_log_softmax(x, tau, axis=-1):
  max_x = jnp.amax(x, axis=axis, keepdims=True)
  y = x - max_x
  tau_lse = max_x + tau * jnp.log(
      jnp.sum(jnp.exp(y / tau), axis=axis, keepdims=True))
  return x - tau_lse


def stable_softmax(x, tau, axis=-1):
  max_x = jnp.amax(x, axis=axis, keepdims=True)
  y = x - max_x
  return jax.nn.softmax(y / tau, axis=axis)


class AtariImplicitQuantileNetwork(nn.Module):
  """The Implicit Quantile Network (Dabney et al., 2018).."""

  def apply(self, x, num_actions, quantile_embedding_dim, num_quantiles, rng):
    initializer = jax.nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    # We need to add a "batch dimension" as nn.Conv expects it, yet vmap will
    # have removed the true batch dimension.
    x = x[None, Ellipsis]
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(x, features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)
    x = jax.nn.relu(x)
    representation = x.reshape((x.shape[0], -1))  # flatten
    state_vector_length = representation.shape[-1]
    state_net_tiled = jnp.tile(representation, [num_quantiles, 1])
    quantiles_shape = [num_quantiles, 1]
    quantiles = jax.random.uniform(rng, shape=quantiles_shape)
    quantile_net = jnp.tile(quantiles, [1, quantile_embedding_dim])
    quantile_net = (
        jnp.arange(1, quantile_embedding_dim + 1, 1).astype(jnp.float32)
        * onp.pi
        * quantile_net)
    quantile_net = jnp.cos(quantile_net)
    quantile_net = nn.Dense(quantile_net,
                            features=state_vector_length,
                            kernel_init=initializer)
    quantile_net = jax.nn.relu(quantile_net)
    x = state_net_tiled * quantile_net
    x = nn.Dense(x, features=512, kernel_init=initializer)
    x = jax.nn.relu(x)
    quantile_values = nn.Dense(x, features=num_actions, kernel_init=initializer)
    return NetworkType(quantile_values, quantiles, representation)


@functools.partial(
    jax.vmap,
    in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None, None, None, None),
    out_axes=(None, 0, 0, 0))
def munchausen_target_quantile_values(target_network, states, actions,
                                      next_states, rewards, terminals,
                                      num_tau_prime_samples,
                                      num_quantile_samples, cumulative_gamma,
                                      rng, tau, alpha, clip_value_min):
  """Build the munchausen target for return values at given quantiles."""
  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
  target_action = target_network(states, num_quantiles=num_quantile_samples,
                                 rng=rng1)
  curr_state_representation = target_action.representation
  curr_state_representation = jnp.squeeze(curr_state_representation)
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  gamma_with_terminal = jnp.tile(gamma_with_terminal, [num_tau_prime_samples])

  replay_net_target_outputs = target_network(
      next_states, num_quantiles=num_tau_prime_samples, rng=rng2)
  replay_quantile_values = replay_net_target_outputs.quantile_values

  target_next_action = target_network(next_states,
                                      num_quantiles=num_quantile_samples,
                                      rng=rng3)
  target_next_quantile_values_action = target_next_action.quantile_values
  replay_next_target_q_values = jnp.squeeze(
      jnp.mean(target_next_quantile_values_action, axis=0))

  q_state_values = target_action.quantile_values
  replay_target_q_values = jnp.squeeze(jnp.mean(q_state_values, axis=0))

  num_actions = q_state_values.shape[-1]
  replay_action_one_hot = jax.nn.one_hot(actions, num_actions)
  replay_next_log_policy = stable_scaled_log_softmax(
      replay_next_target_q_values, tau, axis=0)
  replay_next_policy = stable_softmax(
      replay_next_target_q_values, tau, axis=0)
  replay_log_policy = stable_scaled_log_softmax(replay_target_q_values,
                                                tau, axis=0)

  tau_log_pi_a = jnp.sum(replay_log_policy * replay_action_one_hot, axis=0)
  tau_log_pi_a = jnp.clip(tau_log_pi_a, a_min=clip_value_min, a_max=1)
  munchausen_term = alpha * tau_log_pi_a
  weighted_logits = (
      replay_next_policy * (replay_quantile_values -
                            replay_next_log_policy))

  target_quantile_vals = jnp.sum(weighted_logits, axis=1)
  rewards += munchausen_term
  rewards = jnp.tile(rewards, [num_tau_prime_samples])
  target_quantile_vals = (
      rewards + gamma_with_terminal * target_quantile_vals)
  next_state_representation = target_next_action.representation
  next_state_representation = jnp.squeeze(next_state_representation)

  return (
      rng,
      jax.lax.stop_gradient(target_quantile_vals[:, None]),
      jax.lax.stop_gradient(curr_state_representation),
      jax.lax.stop_gradient(next_state_representation))


@functools.partial(
    jax.vmap,
    in_axes=(None, None, 0, 0, 0, 0, None, None, None, None, None),
    out_axes=(None, 0, 0, 0))
def target_quantile_values(online_network, target_network, states,
                           next_states, rewards, terminals,
                           num_tau_prime_samples, num_quantile_samples,
                           cumulative_gamma, double_dqn, rng):
  """Build the target for return values at given quantiles."""
  rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
  curr_state_representation = target_network(states,
                                             num_quantiles=num_quantile_samples,
                                             rng=rng3).representation
  curr_state_representation = jnp.squeeze(curr_state_representation)
  rewards = jnp.tile(rewards, [num_tau_prime_samples])
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  gamma_with_terminal = jnp.tile(gamma_with_terminal, [num_tau_prime_samples])
  # Compute Q-values which are used for action selection for the next states
  # in the replay buffer. Compute the argmax over the Q-values.
  if double_dqn:
    outputs_action = online_network(next_states,
                                    num_quantiles=num_quantile_samples,
                                    rng=rng1)
  else:
    outputs_action = target_network(next_states,
                                    num_quantiles=num_quantile_samples,
                                    rng=rng1)
  target_quantile_values_action = outputs_action.quantile_values
  target_q_values = jnp.squeeze(
      jnp.mean(target_quantile_values_action, axis=0))
  # Shape: batch_size.
  next_qt_argmax = jnp.argmax(target_q_values)
  # Get the indices of the maximium Q-value across the action dimension.
  # Shape of next_qt_argmax: (num_tau_prime_samples x batch_size).
  next_state_target_outputs = target_network(
      next_states,
      num_quantiles=num_tau_prime_samples,
      rng=rng2)
  next_qt_argmax = jnp.tile(next_qt_argmax, [num_tau_prime_samples])
  target_quantile_vals = (
      jax.vmap(lambda x, y: x[y])(next_state_target_outputs.quantile_values,
                                  next_qt_argmax))
  target_quantile_vals = rewards + gamma_with_terminal * target_quantile_vals
  # We return with an extra dimension, which is expected by train.
  next_state_representation = next_state_target_outputs.representation
  next_state_representation = jnp.squeeze(next_state_representation)
  return (
      rng,
      jax.lax.stop_gradient(target_quantile_vals[:, None]),
      jax.lax.stop_gradient(curr_state_representation),
      jax.lax.stop_gradient(next_state_representation))


@functools.partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12, 14, 15, 16,
                                            17, 18))
def train(target_network, optimizer, states, actions, next_states, rewards,
          terminals, num_tau_samples, num_tau_prime_samples,
          num_quantile_samples, cumulative_gamma, double_dqn, kappa, rng,
          mico_weight, distance_fn, tau, alpha, clip_value_min):
  """Run a training step."""
  def loss_fn(model, rng_input, target_quantile_vals, target_r, target_next_r):
    model_output = jax.vmap(
        lambda m, x, y, z: m(x=x, num_quantiles=y, rng=z),
        in_axes=(None, 0, None, None))(
            model, states, num_tau_samples, rng_input)
    quantile_values = model_output.quantile_values
    quantiles = model_output.quantiles
    representations = model_output.representation
    representations = jnp.squeeze(representations)
    chosen_action_quantile_values = jax.vmap(lambda x, y: x[:, y][:, None])(
        quantile_values, actions)
    # Shape of bellman_erors and huber_loss:
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    bellman_errors = (target_quantile_vals[:, :, None, :] -
                      chosen_action_quantile_values[:, None, :, :])
    # The huber loss (see Section 2.3 of the paper) is defined via two cases:
    # case_one: |bellman_errors| <= kappa
    # case_two: |bellman_errors| > kappa
    huber_loss_case_one = (
        (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
        0.5 * bellman_errors ** 2)
    huber_loss_case_two = (
        (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
        kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))
    huber_loss = huber_loss_case_one + huber_loss_case_two
    # Tile by num_tau_prime_samples along a new dimension. Shape is now
    # batch_size x num_tau_prime_samples x num_tau_samples x 1.
    # These quantiles will be used for computation of the quantile huber loss
    # below (see section 2.3 of the paper).
    quantiles = jnp.tile(quantiles[:, None, :, :],
                         [1, num_tau_prime_samples, 1, 1]).astype(jnp.float32)
    # Shape: batch_size x num_tau_prime_samples x num_tau_samples x 1.
    quantile_huber_loss = (jnp.abs(quantiles - jax.lax.stop_gradient(
        (bellman_errors < 0).astype(jnp.float32))) * huber_loss) / kappa
    # Sum over current quantile value (num_tau_samples) dimension,
    # average over target quantile value (num_tau_prime_samples) dimension.
    # Shape: batch_size x num_tau_prime_samples x 1.
    quantile_huber_loss = jnp.sum(quantile_huber_loss, axis=2)
    quantile_huber_loss = jnp.mean(quantile_huber_loss, axis=1)
    online_dist = metric_utils.representation_distances(
        representations, target_r, distance_fn)
    target_dist = metric_utils.target_distances(
        target_next_r, rewards, distance_fn, cumulative_gamma)
    metric_loss = jnp.mean(jax.vmap(losses.huber_loss)(online_dist,
                                                       target_dist))
    loss = ((1. - mico_weight) * quantile_huber_loss +
            mico_weight * metric_loss)
    return jnp.mean(loss), (jnp.mean(quantile_huber_loss), metric_loss)

  if tau is None:
    rng, target_quantile_vals, target_r, target_next_r = target_quantile_values(
        optimizer.target,
        target_network,
        states,
        next_states,
        rewards,
        terminals,
        num_tau_prime_samples,
        num_quantile_samples,
        cumulative_gamma,
        double_dqn,
        rng)
  else:
    rng, target_quantile_vals, target_r, target_next_r = (
        munchausen_target_quantile_values(
            target_network,
            states,
            actions,
            next_states,
            rewards,
            terminals,
            num_tau_prime_samples,
            num_quantile_samples,
            cumulative_gamma,
            rng,
            tau,
            alpha,
            clip_value_min))
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  rng, rng_input = jax.random.split(rng)
  all_losses, grad = grad_fn(optimizer.target, rng_input, target_quantile_vals,
                             target_r, target_next_r)
  loss, component_losses = all_losses
  quantile_loss, metric_loss = component_losses
  optimizer = optimizer.apply_gradient(grad)
  return rng, optimizer, loss, quantile_loss, metric_loss


@gin.configurable
class MetricImplicitQuantileAgent(
    implicit_quantile_agent.JaxImplicitQuantileAgent):
  """Implicit Quantile Agent with the MICo loss."""

  def __init__(self, num_actions, summary_writer=None,
               mico_weight=0.5, distance_fn=metric_utils.cosine_distance,
               tau=None, alpha=0.9, clip_value_min=-1):
    self._mico_weight = mico_weight
    self._distance_fn = distance_fn
    self._tau = tau
    self._alpha = alpha
    self._clip_value_min = clip_value_min
    super().__init__(num_actions, network=AtariImplicitQuantileNetwork,
                     summary_writer=summary_writer)

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self._rng, self.optimizer, loss, quantile_loss, metric_loss = train(
            self.target_network,
            self.optimizer,
            self.replay_elements['state'],
            self.replay_elements['action'],
            self.replay_elements['next_state'],
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            self.num_tau_samples,
            self.num_tau_prime_samples,
            self.num_quantile_samples,
            self.cumulative_gamma,
            self.double_dqn,
            self.kappa,
            self._rng,
            self._mico_weight,
            self._distance_fn,
            self._tau,
            self._alpha,
            self._clip_value_min)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='Losses/Combined',
                                         simple_value=loss),
              tf.compat.v1.Summary.Value(tag='Losses/Quantile',
                                         simple_value=quantile_loss),
              tf.compat.v1.Summary.Value(tag='Losses/Metric',
                                         simple_value=metric_loss),
          ])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
