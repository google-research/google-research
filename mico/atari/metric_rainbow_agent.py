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

"""Rainbow Agent with the MICo loss."""

import collections
import functools

from dopamine.jax import losses
from dopamine.jax.agents.rainbow import rainbow_agent
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from mico.atari import metric_utils

NetworkType = collections.namedtuple(
    'network', ['q_values', 'logits', 'probabilities', 'representation'])


@gin.configurable
class AtariRainbowNetwork(nn.Module):
  """Convolutional network used to compute the agent's return distributions."""
  num_actions: int
  num_atoms: int

  @nn.compact
  def __call__(self, x, support):
    initializer = jax.nn.initializers.variance_scaling(
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
    q_values = jnp.sum(support * probabilities, axis=1)
    return NetworkType(q_values, logits, probabilities, representation)


@functools.partial(jax.jit, static_argnums=(0, 3, 12, 13, 14))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, loss_weights,
          support, cumulative_gamma, mico_weight, distance_fn):
  """Run a training step."""
  def loss_fn(params, bellman_target, loss_multipliers, target_r,
              target_next_r):
    def q_online(state):
      return network_def.apply(params, state, support)

    model_output = jax.vmap(q_online)(states)
    logits = model_output.logits
    logits = jnp.squeeze(logits)
    representations = model_output.representation
    representations = jnp.squeeze(representations)
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    c51_loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
        bellman_target,
        chosen_action_logits)
    c51_loss *= loss_multipliers
    online_dist, norm_average, angular_distance = (
        metric_utils.representation_distances(
            representations, target_r, distance_fn,
            return_distance_components=True))
    target_dist = metric_utils.target_distances(
        target_next_r, rewards, distance_fn, cumulative_gamma)
    metric_loss = jnp.mean(jax.vmap(losses.huber_loss)(online_dist,
                                                       target_dist))
    loss = ((1. - mico_weight) * c51_loss +
            mico_weight * metric_loss)
    aux_losses = {
        'loss': loss,
        'mean_loss': jnp.mean(loss),
        'c51_loss': jnp.mean(c51_loss),
        'metric_loss': metric_loss,
        'norm_average': jnp.mean(norm_average),
        'angular_distance': jnp.mean(angular_distance),
    }
    return jnp.mean(loss), aux_losses

  def q_target(state):
    return network_def.apply(target_params, state, support)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  bellman_target, target_r, target_next_r = target_distribution(
      q_target,
      states,
      next_states,
      rewards,
      terminals,
      support,
      cumulative_gamma)
  (_, aux_losses), grad = grad_fn(online_params, bellman_target,
                                  loss_weights, target_r, target_next_r)
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, aux_losses


@functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, None, None))
def target_distribution(target_network, states, next_states, rewards, terminals,
                        support, cumulative_gamma):
  """Builds the C51 target distribution as per Bellemare et al. (2017)."""
  curr_state_representation = target_network(states).representation
  curr_state_representation = jnp.squeeze(curr_state_representation)
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  target_support = rewards + gamma_with_terminal * support
  next_state_target_outputs = target_network(next_states)
  q_values = jnp.squeeze(next_state_target_outputs.q_values)
  next_qt_argmax = jnp.argmax(q_values)
  probabilities = jnp.squeeze(next_state_target_outputs.probabilities)
  next_probabilities = probabilities[next_qt_argmax]
  next_state_representation = next_state_target_outputs.representation
  next_state_representation = jnp.squeeze(next_state_representation)
  return (
      jax.lax.stop_gradient(rainbow_agent.project_distribution(
          target_support, next_probabilities, support)),
      jax.lax.stop_gradient(curr_state_representation),
      jax.lax.stop_gradient(next_state_representation))


@gin.configurable
class MetricRainbowAgent(rainbow_agent.JaxRainbowAgent):
  """Rainbow Agent with the MICo loss."""

  def __init__(self, num_actions, summary_writer=None,
               mico_weight=0.01, distance_fn=metric_utils.cosine_distance):
    self._mico_weight = mico_weight
    self._distance_fn = distance_fn
    network = AtariRainbowNetwork
    super().__init__(num_actions, network=network,
                     summary_writer=summary_writer)

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        if self._replay_scheme == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          # Weight the loss by the inverse priorities.
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        self.optimizer_state, self.online_params, aux_losses = train(
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
            loss_weights,
            self._support,
            self.cumulative_gamma,
            self._mico_weight,
            self._distance_fn)

        loss = aux_losses.pop('loss')
        if self._replay_scheme == 'prioritized':
          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
          # leave it as is here, using the more direct sqrt(). Taking the square
          # root "makes sense", as we are dealing with a squared loss.  Add a
          # small nonzero value to the loss to avoid 0 priority items. While
          # technically this may be okay, setting all items to 0 priority will
          # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))

        if self._replay_scheme == 'prioritized':
          probs = self.replay_elements['sampling_probabilities']
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))
          loss = loss_weights * loss
        if self.summary_writer is not None:
          values = []
          for k in aux_losses:
            values.append(
                tf.compat.v1.Summary.Value(tag=f'Losses/{k}',
                                           simple_value=aux_losses[k]))
          summary = tf.compat.v1.Summary(value=values)
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
