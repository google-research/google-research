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

"""DQN Agent with incoherent / orthogonals loss."""

import collections
import functools

from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from generalization_representations_rl_aistats22.atari import coherence_utils


NetworkType = collections.namedtuple('network', ['q_values', 'representation'])


@gin.configurable
class JAXDQNNetworkWithRepresentations(nn.Module):
  """The convolutional network used to compute the agent's Q-values."""
  num_actions: int

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
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
    x = x.reshape(-1)  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(x)
    representation = nn.relu(x)  # Use penultimate layer as representation
    q_values = nn.Dense(features=self.num_actions,
                        kernel_init=initializer)(representation)
    return NetworkType(q_values, representation)


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12, 13, 14))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, cumulative_gamma,
          coherence_weight, option, use_ortho_loss, use_cohe_loss):
  """Run the training step."""
  def loss_fn(params, bellman_target):
    def q_online(state):
      return network_def.apply(params, state)

    model_output = jax.vmap(q_online)(states)
    q_values = model_output.q_values
    q_values = jnp.squeeze(q_values)
    representations = model_output.representation
    representations = jnp.squeeze(representations)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    bellman_loss = jnp.mean(
        jax.vmap(losses.mse_loss)(bellman_target, replay_chosen_q))
    if use_cohe_loss and use_ortho_loss:
      coherence_loss = coherence_utils.orthogonal_features_coherence(
          representations, option)
      cosine_similarity = coherence_utils.orthogonality(representations)
      orthogonality_loss = jnp.mean(
          jnp.abs(cosine_similarity - jnp.eye(representations.shape[0])))
    if use_cohe_loss and not use_ortho_loss:
      coherence_loss = coherence_utils.orthogonal_features_coherence(
          representations, option)
      orthogonality_loss = 0.
    if use_ortho_loss and not use_cohe_loss:
      coherence_loss = 0.
      cosine_similarity = coherence_utils.orthogonality(representations)
      orthogonality_loss = jnp.mean(
          jnp.abs(cosine_similarity - jnp.eye(representations.shape[0])))
    loss = ((1. - coherence_weight) * bellman_loss + coherence_weight *
            (coherence_loss + orthogonality_loss))
    return jnp.mean(loss), (bellman_loss, coherence_loss, orthogonality_loss)

  def q_target(state):
    return network_def.apply(target_params, state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  bellman_target, _, _ = target_outputs(
      q_target, states, next_states, rewards, terminals, cumulative_gamma)
  (loss, component_losses), grad = grad_fn(online_params, bellman_target)
  bellman_loss, coherence_loss, orthogonality_loss = component_losses
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, bellman_loss, coherence_loss, orthogonality_loss


def target_outputs(target_network, states, next_states, rewards, terminals,
                   cumulative_gamma):
  """Compute the target Q-value."""
  curr_state_representation = jax.vmap(target_network, in_axes=(0))(
      states).representation
  curr_state_representation = jnp.squeeze(curr_state_representation)
  next_state_output = jax.vmap(target_network, in_axes=(0))(next_states)
  next_state_q_vals = next_state_output.q_values
  next_state_q_vals = jnp.squeeze(next_state_q_vals)
  next_state_representation = next_state_output.representation
  next_state_representation = jnp.squeeze(next_state_representation)
  replay_next_qt_max = jnp.max(next_state_q_vals, 1)
  return (
      jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max *
                            (1. - terminals)),
      jax.lax.stop_gradient(curr_state_representation),
      jax.lax.stop_gradient(next_state_representation))


@gin.configurable
class IncoherentDQNAgent(dqn_agent.JaxDQNAgent):
  """DQN Agent with the coherence loss."""

  def __init__(self,
               num_actions,
               summary_writer=None,
               coherence_weight=0.01,
               option='logsumexp',
               use_ortho_loss=True,
               use_cohe_loss=True):
    self._coherence_weight = coherence_weight
    self._option = option
    self._use_ortho_loss = use_ortho_loss
    self._use_cohe_loss = use_cohe_loss
    network = JAXDQNNetworkWithRepresentations
    super().__init__(num_actions, network=network,
                     summary_writer=summary_writer)

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        (self.optimizer_state, self.online_params,
         loss, bellman_loss, coherence_loss, orthogonality_loss) = train(
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
             self.cumulative_gamma,
             self._coherence_weight,
             self._option,
             self._use_ortho_loss,
             self._use_cohe_loss)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          if self._use_ortho_loss and self._use_cohe_loss:
            summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag='Losses/Aggregate',
                                           simple_value=loss),
                tf.compat.v1.Summary.Value(tag='Losses/Bellman',
                                           simple_value=bellman_loss),
                tf.compat.v1.Summary.Value(tag='Losses/Incoherence',
                                           simple_value=coherence_loss),
                tf.compat.v1.Summary.Value(
                    tag='Losses/Aggregate', simple_value=loss),
                tf.compat.v1.Summary.Value(
                    tag='Losses/Bellman', simple_value=bellman_loss),
                tf.compat.v1.Summary.Value(
                    tag='Losses/Incoherence', simple_value=coherence_loss),
                tf.compat.v1.Summary.Value(
                    tag='Losses/Orthogonality',
                    simple_value=orthogonality_loss),
            ])
          elif self._use_ortho_loss and not self._use_cohe_loss:
            summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag='Losses/Aggregate',
                                           simple_value=loss),
                tf.compat.v1.Summary.Value(tag='Losses/Bellman',
                                           simple_value=bellman_loss),
                tf.compat.v1.Summary.Value(
                    tag='Losses/Aggregate', simple_value=loss),
                tf.compat.v1.Summary.Value(
                    tag='Losses/Bellman', simple_value=bellman_loss),
                tf.compat.v1.Summary.Value(
                    tag='Losses/Orthogonality',
                    simple_value=orthogonality_loss),
            ])
          elif self._use_cohe_loss and not self._use_ortho_loss:
            summary = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag='Losses/Aggregate',
                                           simple_value=loss),
                tf.compat.v1.Summary.Value(tag='Losses/Bellman',
                                           simple_value=bellman_loss),
                tf.compat.v1.Summary.Value(tag='Losses/Incoherence',
                                           simple_value=coherence_loss),
            ])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
