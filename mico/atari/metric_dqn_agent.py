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

"""DQN Agent with MICo loss."""

import collections
import functools
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import tensorflow as tf

from mico.atari import metric_utils

NetworkType = collections.namedtuple('network', ['q_values', 'representation'])


@gin.configurable
class AtariDQNNetwork(nn.Module):
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
    representation = x.reshape(-1)  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(representation)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions,
                        kernel_init=initializer)(x)
    return NetworkType(q_values, representation)


@functools.partial(jax.jit, static_argnums=(0, 8, 9, 10))
def train(network_def, target_params, optimizer, states, actions, next_states,
          rewards, terminals, cumulative_gamma, mico_weight, distance_fn):
  """Run the training step."""
  online_params = optimizer.target
  def loss_fn(params, bellman_target, target_r, target_next_r):
    def q_online(state):
      return network_def.apply(params, state)

    model_output = jax.vmap(q_online)(states)
    q_values = model_output.q_values
    q_values = jnp.squeeze(q_values)
    representations = model_output.representation
    representations = jnp.squeeze(representations)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    bellman_loss = jnp.mean(jax.vmap(losses.mse_loss)(bellman_target,
                                                      replay_chosen_q))
    online_dist = metric_utils.representation_distances(
        representations, target_r, distance_fn)
    target_dist = metric_utils.target_distances(
        target_next_r, rewards, distance_fn, cumulative_gamma)
    metric_loss = jnp.mean(jax.vmap(losses.huber_loss)(online_dist,
                                                       target_dist))
    loss = ((1. - mico_weight) * bellman_loss +
            mico_weight * metric_loss)
    return jnp.mean(loss), (bellman_loss, metric_loss)

  def q_target(state):
    return network_def.apply(target_params, state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  bellman_target, target_r, target_next_r = target_outputs(
      q_target, states, next_states, rewards, terminals, cumulative_gamma)
  (loss, component_losses), grad = grad_fn(online_params, bellman_target,
                                           target_r, target_next_r)
  bellman_loss, metric_loss = component_losses
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, bellman_loss, metric_loss


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
class MetricDQNAgent(dqn_agent.JaxDQNAgent):
  """DQN Agent with the MICo loss."""

  def __init__(self, num_actions, summary_writer=None,
               mico_weight=0.01, distance_fn=metric_utils.cosine_distance):
    self._mico_weight = mico_weight
    self._distance_fn = distance_fn
    network = AtariDQNNetwork
    super().__init__(num_actions, network=network,
                     summary_writer=summary_writer)

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self.optimizer, loss, bellman_loss, metric_loss = train(
            self.network_def,
            self.target_network_params,
            self.optimizer,
            self.replay_elements['state'],
            self.replay_elements['action'],
            self.replay_elements['next_state'],
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            self.cumulative_gamma,
            self._mico_weight,
            self._distance_fn)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.compat.v1.Summary(value=[
              tf.compat.v1.Summary.Value(tag='Losses/Aggregate',
                                         simple_value=loss),
              tf.compat.v1.Summary.Value(tag='Losses/Bellman',
                                         simple_value=bellman_loss),
              tf.compat.v1.Summary.Value(tag='Losses/Metric',
                                         simple_value=metric_loss),
          ])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
