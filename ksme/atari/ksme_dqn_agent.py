# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""DQN Agent with KSMe loss."""

import collections
import functools
from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.metrics import statistics_instance
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ksme.atari import metric_utils

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


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12, 13))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, cumulative_gamma,
          mico_weight, distance_fn, similarity_fn):
  """Run the training step."""
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
    online_similarities = metric_utils.representation_similarities(
        representations, target_r, distance_fn, similarity_fn)
    target_similarities = metric_utils.target_similarities(
        target_next_r, rewards, distance_fn, similarity_fn, cumulative_gamma)
    kernel_loss = jnp.mean(jax.vmap(losses.huber_loss)(online_similarities,
                                                       target_similarities))
    loss = ((1. - mico_weight) * bellman_loss +
            mico_weight * kernel_loss)
    return jnp.mean(loss), (bellman_loss, kernel_loss)

  def q_target(state):
    return network_def.apply(target_params, state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  bellman_target, target_r, target_next_r = target_outputs(
      q_target, states, next_states, rewards, terminals, cumulative_gamma)
  (loss, component_losses), grad = grad_fn(online_params, bellman_target,
                                           target_r, target_next_r)
  bellman_loss, kernel_loss = component_losses
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, bellman_loss, kernel_loss


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
class KSMeDQNAgent(dqn_agent.JaxDQNAgent):
  """DQN Agent with the KSMe loss."""

  def __init__(self, num_actions, summary_writer=None,
               mico_weight=0.01, distance_fn='dot',
               similarity_fn='dot'):
    self._mico_weight = mico_weight
    if distance_fn == 'cosine':
      self._distance_fn = metric_utils.cosine_distance
    elif distance_fn == 'dot':
      self._distance_fn = metric_utils.l2
    else:
      raise ValueError(f'Unknown distance function: {distance_fn}')

    if similarity_fn == 'cosine':
      self._similarity_fn = metric_utils.cosine_similarity
    elif similarity_fn == 'dot':
      self._similarity_fn = metric_utils.dot
    else:
      raise ValueError(f'Unknown similarity function: {similarity_fn}')

    network = AtariDQNNetwork
    super().__init__(num_actions, network=network,
                     summary_writer=summary_writer)
    logging.info('\t mico_weight: %f', mico_weight)
    logging.info('\t distance_fn: %s', distance_fn)
    logging.info('\t similarity_fn: %s', similarity_fn)

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        (self.optimizer_state, self.online_params,
         loss, bellman_loss, kernel_loss) = train(
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
             self._mico_weight,
             self._distance_fn,
             self._similarity_fn)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          if hasattr(self, 'collector_dispatcher'):
            self.collector_dispatcher.write(
                [statistics_instance.StatisticsInstance(
                    'Losses/Aggregate', np.asarray(loss),
                    step=self.training_steps),
                 statistics_instance.StatisticsInstance(
                     'Losses/Bellman', np.asarray(bellman_loss),
                     step=self.training_steps),
                 statistics_instance.StatisticsInstance(
                     'Losses/Metric', np.asarray(kernel_loss),
                     step=self.training_steps),
                 ],
                collector_allowlist=self._collector_allowlist)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
