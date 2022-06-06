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

"""DQN Agent with time input."""
import collections
import functools
from typing import Tuple

from dopamine.jax import losses
from dopamine.jax import networks
from dopamine.jax.agents.dqn import dqn_agent
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf

from aux_tasks.auxiliary_mc import gammas_monte_carlo_replay_buffer as monte_carlo_rb
from aux_tasks.auxiliary_mc import networks as aux_mc_networks

AuxiliaryPredictionDQNNetworkType = collections.namedtuple(
    'dqn_network_with_random_rewards', ['q_values', 'aux_prediction'])


@gin.configurable
class DQNNetworkWithAuxiliaryPredictions(nn.Module):
  """Generates q_values with per-state auxiliary predictions.

  Attributes:
    num_actions: int, number of actions the agent can take at any state.
    num_predictions: int, number of auxiliary predictions.
    rng_key: int, Fixed rng for random reward generation.
    inputs_preprocessed: bool, Whether inputs are already preprocessed.
  """
  num_actions: int
  num_predictions: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x):

    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = networks.preprocess_atari_inputs(x)

    hidden_sizes = [32, 64, 64]
    kernel_sizes = [8, 4, 3]
    stride_sizes = [4, 2, 1]
    for hidden_size, kernel_size, stride_size in zip(hidden_sizes, kernel_sizes,
                                                     stride_sizes):
      x = nn.Conv(
          features=hidden_size,
          kernel_size=(kernel_size, kernel_size),
          strides=(stride_size, stride_size),
          kernel_init=initializer)(x)
      x = nn.relu(x)
    features = x.reshape((-1))  # flatten
    x = nn.Dense(features=512, kernel_init=initializer)(features)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)

    # MSE loss for Auxiliary task MC predictions.
    auxiliary_pred = nn.Dense(features=512, kernel_init=initializer)(features)
    auxiliary_pred = nn.relu(auxiliary_pred)
    auxiliary_pred = nn.Dense(
        features=self.num_predictions, kernel_init=initializer)(auxiliary_pred)
    return AuxiliaryPredictionDQNNetworkType(q_values, auxiliary_pred)


@gin.configurable
class ImpalaEncoderWithAuxiliaryPredictions(nn.Module):
  """Impala Network generating q_values with per-state auxiliary predictions."""
  num_actions: int
  num_predictions: int
  inputs_preprocessed: bool = False
  stack_sizes: Tuple[int, Ellipsis] = (16, 32, 32)
  num_blocks: int = 2

  def setup(self):
    self.encoder = aux_mc_networks.ImpalaEncoder()

  @nn.compact
  def __call__(self, x, key=None):
    # Generate a random number generation key if not provided
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = networks.preprocess_atari_inputs(x)

    x = self.encoder(x)
    features = x.reshape((-1))  # flatten

    x = nn.Dense(
        features=512, kernel_init=initializer)(features)
    x = nn.relu(x)
    q_values = nn.Dense(features=self.num_actions, kernel_init=initializer)(x)

    # MSE loss for Auxiliary task MC predictions.
    auxiliary_pred = nn.Dense(features=512, kernel_init=initializer)(features)
    auxiliary_pred = nn.relu(auxiliary_pred)
    auxiliary_pred = nn.Dense(
        features=self.num_predictions, kernel_init=initializer)(auxiliary_pred)

    return AuxiliaryPredictionDQNNetworkType(q_values, auxiliary_pred)


@gin.configurable
class RandomRewardNetwork(nn.Module):
  """Generates random rewards using a noisy network.

  Attributes:
    num_actions: int, number of actions the agent can take at any state.
    num_rewards: int, number of random rewards to generate.
    rng_key: int, Fixed rng for random reward generation.
    inputs_preprocessed: bool, Whether inputs are already preprocessed.
  """
  num_actions: int
  num_rewards: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x, rng_key):

    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = networks.preprocess_atari_inputs(x)

    hidden_sizes = [32, 64, 64]
    kernel_sizes = [8, 4, 3]
    stride_sizes = [4, 2, 1]
    for hidden_size, kernel_size, stride_size in zip(hidden_sizes, kernel_sizes,
                                                     stride_sizes):
      x = nn.Conv(
          features=hidden_size,
          kernel_size=(kernel_size, kernel_size),
          strides=(stride_size, stride_size),
          kernel_init=initializer)(x)
      x = nn.relu(x)
    features = x.reshape((-1))  # flatten

    # Use a fixed random seed for NoisyNetwork.
    net = networks.NoisyNetwork(rng_key=rng_key, eval_mode=False)
    # Return `self.num_rewards` random outputs.
    rewards = net(features, self.num_rewards)
    x = jax.nn.sigmoid(features)  # clip rewards between -1 and 1
    return rewards


@functools.partial(jax.jit, static_argnames=('network_def'))
def get_rewards(network_def, params, state, rng_key):
  return network_def.apply(params, state, rng_key=rng_key)


@functools.partial(
    jax.jit,
    static_argnames=('network_def', 'optimizer', 'cumulative_gamma',
                     'loss_type'))
def train(network_def,
          online_params,
          target_params,
          optimizer,
          optimizer_state,
          states,
          auxiliary_mc_returns,
          actions,
          next_states,
          rewards,
          terminals,
          cumulative_gamma,
          auxloss_weight=0.0):
  """Run the training step."""
  def loss_fn(params, target, auxiliary_target):
    def q_online(state):
      return network_def.apply(params, state)

    model_output = jax.vmap(q_online)(states)
    q_values = jnp.squeeze(model_output.q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    td_loss = jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))

    # Auxiliary task loss.
    auxiliary_predictions = jnp.squeeze(model_output.aux_prediction)
    aux_loss = jnp.mean(jax.vmap(losses.mse_loss)(
        auxiliary_predictions, auxiliary_target))
    loss = ((1. - auxloss_weight) * td_loss +
            auxloss_weight * aux_loss)
    return loss, (td_loss, aux_loss)

  def q_target(state):
    return network_def.apply(target_params, state)

  target = dqn_agent.target_q(q_target, next_states, rewards, terminals,
                              cumulative_gamma)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, component_losses), grad = grad_fn(online_params, target,
                                           auxiliary_mc_returns)
  td_loss, aux_loss = component_losses
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, td_loss, aux_loss


@gin.configurable
class DiscountedJaxDQNAgentWithAuxiliaryMC(dqn_agent.JaxDQNAgent):
  """An implementation of the DQN agent with replay buffer logging to disk."""

  def __init__(self,
               num_actions,
               network=DQNNetworkWithAuxiliaryPredictions,
               num_rewards=2,
               auxloss_weight=0.0,
               summary_writer=None,
               preprocess_fn=None,
               seed=None):
    """Initializes the agent and constructs the components of its graph.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      network: Jax network to use for training.
      num_rewards: int, Number of random rewards to generate at each step.
      auxloss_weight: float: weight for aux loss.
      summary_writer: Tensorflow summary writer for logging summaries.
      preprocess_fn: Preprocessing function.
      seed: int, Agent seed.
    """
    network = functools.partial(network, num_predictions=num_rewards)
    self.num_rewards = num_rewards
    self._auxloss_weight = auxloss_weight
    super().__init__(
        num_actions, network=network, summary_writer=summary_writer, seed=seed,
        preprocess_fn=preprocess_fn)
    # Create network for random reward generation.

  def _build_replay_buffer(self):
    """Creates a monte carlo replay buffer used by the agent."""

    return monte_carlo_rb.OutOfGraphReplayBufferdiscountedWithMC(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype,
        list_of_discounts=onp.linspace(0.1, 0.999, self.num_rewards))
    # Pass a compy of `extra_storage_types` to avoid updating it when
    # updating `extra_monte_carlo_storage_types`.
    # extra_monte_carlo_storage_types=extra_storage_types[:],
    # reverse_fill=True)

  def _train_step(self):
    """Runs a single training step."""
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        states = self.preprocess_fn(self.replay_elements['state'])
        next_states = self.preprocess_fn(self.replay_elements['next_state'])
        self.optimizer_state, self.online_params, loss, td_loss, auxloss = train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            states,
            # List of monte carlo returns for all gamma.
            self.replay_elements['monte_carlo_gamma'],
            self.replay_elements['action'],
            next_states,
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            self.cumulative_gamma,
            self._auxloss_weight)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          with self.summary_writer.as_default():
            tf.summary.scalar('Losses/Aggregate', loss, step=self.training_steps)
            tf.summary.scalar(
                'Losses/Auxiliary',
                auxloss,
                step=self.training_steps)
            tf.summary.scalar('Losses/TD', td_loss, step=self.training_steps)
          self.summary_writer.flush()
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
