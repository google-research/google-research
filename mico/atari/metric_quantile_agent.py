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

"""Quantile regression agent with MICo loss."""

import collections
import functools

from dopamine.jax import losses
from dopamine.jax.agents.quantile import quantile_agent
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from mico.atari import metric_rainbow_agent
from mico.atari import metric_utils

NetworkType = collections.namedtuple(
    'network', ['q_values', 'logits', 'probabilities', 'representation'])


@gin.configurable
class AtariQuantileNetwork(nn.Module):
  """Convolutional network used to compute the agent's return quantiles."""
  num_actions: int
  num_atoms: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    x = x.astype(jnp.float32) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    representation = x.reshape(-1)  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(representation)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_actions * self.num_atoms,
                 kernel_init=initializer)(x)
    logits = x.reshape((self.num_actions, self.num_atoms))
    probabilities = nn.softmax(logits)
    q_values = jnp.mean(logits, axis=1)
    return metric_rainbow_agent.NetworkType(q_values, logits, probabilities,
                                            representation)


@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, None))
def target_distribution(target_network, states, next_states, rewards, terminals,
                        cumulative_gamma):
  """Builds the Quantile target distribution as per Dabney et al. (2017)."""
  curr_state_representation = target_network(states).representation
  curr_state_representation = jnp.squeeze(curr_state_representation)
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  next_state_target_outputs = target_network(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values)
  next_qt_argmax = jnp.argmax(q_values)
  logits = jnp.squeeze(next_state_target_outputs.logits)
  next_logits = logits[next_qt_argmax]
  next_state_representation = next_state_target_outputs.representation
  next_state_representation = jnp.squeeze(next_state_representation)
  return (
      jax.lax.stop_gradient(rewards + gamma_with_terminal * next_logits),
      jax.lax.stop_gradient(curr_state_representation),
      jax.lax.stop_gradient(next_state_representation))


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12, 13, 14))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, kappa, num_atoms,
          cumulative_gamma, mico_weight, distance_fn):
  """Run a training step."""
  def loss_fn(params, bellman_target, target_r, target_next_r):
    def q_online(state):
      return network_def.apply(params, state)

    model_output = jax.vmap(q_online)(states)
    logits = model_output.logits
    logits = jnp.squeeze(logits)
    representations = model_output.representation
    representations = jnp.squeeze(representations)
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    bellman_errors = (bellman_target[:, None, :] -
                      chosen_action_logits[:, :, None])  # Input `u' of Eq. 9.
    # Eq. 9 of paper.
    huber_loss = (
        (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
        0.5 * bellman_errors ** 2 +
        (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
        kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))

    tau_hat = ((jnp.arange(num_atoms, dtype=jnp.float32) + 0.5) /
               num_atoms)  # Quantile midpoints.  See Lemma 2 of paper.
    # Eq. 10 of paper.
    tau_bellman_diff = jnp.abs(
        tau_hat[None, :, None] - (bellman_errors < 0).astype(jnp.float32))
    quantile_huber_loss = tau_bellman_diff * huber_loss
    # Sum over tau dimension, average over target value dimension.
    quantile_loss = jnp.sum(jnp.mean(quantile_huber_loss, 2), 1)
    online_dist = metric_utils.representation_distances(
        representations, target_r, distance_fn)
    target_dist = metric_utils.target_distances(
        target_next_r, rewards, distance_fn, cumulative_gamma)
    metric_loss = jnp.mean(jax.vmap(losses.huber_loss)(online_dist,
                                                       target_dist))
    loss = ((1. - mico_weight) * quantile_loss +
            mico_weight * metric_loss)
    return jnp.mean(loss), (loss, jnp.mean(quantile_loss), metric_loss)

  def q_target(state):
    return network_def.apply(target_params, state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  bellman_target, target_r, target_next_r = target_distribution(
      q_target, states, next_states, rewards, terminals, cumulative_gamma)
  all_losses, grad = grad_fn(online_params, bellman_target, target_r,
                             target_next_r)
  mean_loss, component_losses = all_losses
  loss, quantile_loss, metric_loss = component_losses
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return (optimizer_state, online_params, loss, mean_loss, quantile_loss,
          metric_loss)


@gin.configurable
class MetricQuantileAgent(quantile_agent.JaxQuantileAgent):
  """Quantile Agent with the MICo loss."""

  def __init__(self, num_actions, summary_writer=None,
               mico_weight=0.5, distance_fn=metric_utils.cosine_distance):
    self._mico_weight = mico_weight
    self._distance_fn = distance_fn
    network = AtariQuantileNetwork
    super().__init__(num_actions, network=network,
                     summary_writer=summary_writer)

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        (self.optimizer_state, self.online_params,
         loss, mean_loss, quantile_loss, metric_loss) = train(
             self.network_def,
             self.online_params,
             self.target_network_params,
             self.optimizer,
             self.optimizer_state,
             self.replay_elements['state'],
             self.replay_elements['action'],
             self.replay_elements['next_state'],
             self.replay_elements['reward'],
             self.replay_elements['terminal'],
             self._kappa,
             self._num_atoms,
             self.cumulative_gamma,
             self._mico_weight,
             self._distance_fn)
        if self._replay_scheme == 'prioritized':
          probs = self.replay_elements['sampling_probabilities']
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))
          loss = loss_weights * loss
          mean_loss = jnp.mean(loss)
        if self.summary_writer is not None:
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='Losses/Combined',
                                         simple_value=mean_loss),
              tf.compat.v1.Summary.Value(tag='Losses/Quantile',
                                         simple_value=quantile_loss),
              tf.compat.v1.Summary.Value(tag='Losses/Metric',
                                         simple_value=metric_loss),
          ])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
