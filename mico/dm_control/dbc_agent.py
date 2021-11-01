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

r"""Implementation of DBC: https://arxiv.org/abs/2006.10742."""


import collections
import functools
from typing import Any, Mapping, Optional, Tuple

from absl import logging
from dopamine.jax import continuous_networks
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax.agents.sac import sac_agent
from dopamine.labs.sac_from_pixels import continuous_networks as internal_continuous_networks
import flax
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf

from mico.atari import metric_utils


@gin.configurable
class SACConvNetwork(nn.Module):
  """A convolutional value and policy networks for SAC."""
  action_shape: Tuple[int, Ellipsis]
  num_layers: int = 2
  hidden_units: int = 256
  action_limits: Optional[Tuple[Tuple[float, Ellipsis], Tuple[float, Ellipsis]]] = None

  def setup(self):
    self._sac_network = continuous_networks.SACNetwork(
        self.action_shape, self.num_layers,
        self.hidden_units, self.action_limits)

  def __call__(self,
               z,
               key,
               mean_action = True):
    """Calls the SAC actor/critic networks."""
    actor_output = self._sac_network.actor(z, key)
    action = actor_output.mean_action if mean_action else actor_output.sampled_action
    critic_output = self._sac_network.critic(z, action)

    return continuous_networks.SacOutput(actor_output, critic_output)

  def actor(self, z,
            key):
    """Calls the SAC actor network."""
    return self._sac_network.actor(z, key)

  def critic(self, z,
             action):
    """Calls the SAC critic network."""
    return self._sac_network.critic(z, action)


@gin.configurable
class RewardModel(nn.Module):
  """A reward model from latent states."""
  hidden_units: int = 512

  @nn.compact
  def __call__(self, z):
    kernel_initializer = jax.nn.initializers.glorot_uniform()
    x = nn.Dense(features=self.hidden_units, kernel_init=kernel_initializer)(z)
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    return nn.Dense(features=1, kernel_init=kernel_initializer)(x)


DynamicsModelType = collections.namedtuple('dynamics_model_type',
                                           ['mu', 'sigma', 'sample'])


@gin.configurable
class DynamicsModel(nn.Module):
  """A dynamics model from latent states."""
  action_shape: Tuple[int, Ellipsis]
  embedding_dim: int = 50
  layer_width: int = 512
  min_sigma: float = 1e-4
  max_sigma: float = 1e1
  probabilistic: bool = True

  @nn.compact
  def __call__(self, z, a,
               key):
    kernel_initializer = jax.nn.initializers.glorot_uniform()
    x = nn.Dense(features=self.layer_width, kernel_init=kernel_initializer)(
        jnp.concatenate([z, a]))
    x = nn.LayerNorm()(x)
    x = nn.relu(x)
    mu = nn.Dense(features=self.embedding_dim,
                  kernel_init=kernel_initializer)(x)
    if self.probabilistic:
      sigma = nn.Dense(features=self.embedding_dim,
                       kernel_init=kernel_initializer)(x)
      sigma = nn.sigmoid(sigma)
      sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
      eps = jax.random.normal(key, shape=sigma.shape)
      sample = mu + sigma * eps
    else:
      sigma = jnp.zeros(self.embedding_dim)
      sample = mu
    return DynamicsModelType(mu, sigma, sample)


def l1(x, y):
  return jnp.sum(abs(x - y))


def mico_target_distances(next_z, rewards, shuffled_idx, cumulative_gamma):
  reward_diffs = jax.vmap(l1)(rewards, rewards[shuffled_idx])
  shuffled_next_z = next_z[shuffled_idx]
  next_state_distances = jax.vmap(metric_utils.cosine_distance)(next_z,
                                                                shuffled_next_z)
  return jax.lax.stop_gradient(
      reward_diffs + cumulative_gamma * next_state_distances)


def target_z_distances(dynamics_output, rewards, shuffled_idx,
                       cumulative_gamma):
  """Target distance using the metric operator."""
  # Reward difference term.
  reward_diffs = jax.vmap(l1)(rewards, rewards[shuffled_idx])

  # Mu difference term.
  mus = dynamics_output.mu
  mu_diffs = jax.vmap(metric_utils.l2)(mus, mus[shuffled_idx])

  # Sigma difference term.
  sigmas = dynamics_output.sigma
  # Paper says Frobenius norm, but since there is an assumption of a diagonal
  # covariance matrix, we can just use l2.
  sigma_diffs = jax.vmap(metric_utils.l2)(sigmas, sigmas[shuffled_idx])

  return (
      jax.lax.stop_gradient(
          reward_diffs + cumulative_gamma * (mu_diffs + sigma_diffs)))


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 30))
def train(encoder_network_def,
          network_def,
          reward_model_def,
          dynamics_model_def,
          encoder_optim,
          optim,
          reward_optim,
          dynamics_optim,
          alpha_optim,
          encoder_params,
          encoder_target_params,
          params,
          target_params,
          log_alpha,
          reward_params,
          dynamics_params,
          encoder_optim_state,
          optim_state,
          reward_optim_state,
          dynamics_optim_state,
          alpha_optim_state,
          key,
          states,
          actions,
          next_states,
          rewards,
          terminals,
          cumulative_gamma,
          target_entropy,
          reward_scale_factor,
          use_mico = False):
  """Run the training step."""
  # Get the models from all the optimizers.
  frozen_params = params

  batch_size = states.shape[0]
  actions = jnp.reshape(actions, (batch_size, -1))  # Flatten

  rng1, rng2, rng3, rng4 = jax.random.split(key, num=4)

  def encoder_online_actor(state):
    return encoder_network_def.apply(encoder_params, state).actor_z

  def encoder_target_actor(state):
    return encoder_network_def.apply(encoder_target_params, state).actor_z

  # Encode states for other losses.
  fixed_encoded_states = jax.vmap(encoder_online_actor)(states)
  encoded_next_states = jax.vmap(encoder_target_actor)(next_states)

  def bisimulation_loss_fn(
      encoder_params,
      dynamics_params):
    def encoder_online_critic(state):
      return encoder_network_def.apply(encoder_params, state).critic_z

    def dynamics_online(z, a, rng):
      return dynamics_model_def.apply(dynamics_params, z, a, rng)

    rng11, rng12 = jax.random.split(rng1)
    brng = jnp.stack(jax.random.split(rng11, num=batch_size))
    predicted_dynamics = jax.vmap(dynamics_online)(
        fixed_encoded_states, actions, brng)
    learned_z = jax.vmap(encoder_online_critic)(states)
    # We shuffle the batch element IDs.
    shuffled_idx = jnp.array(list(range(batch_size)))
    shuffled_idx = jax.random.shuffle(rng12, shuffled_idx)
    shuffled_z = learned_z[shuffled_idx]
    if use_mico:
      base_distances = jax.vmap(metric_utils.cosine_distance)(learned_z,
                                                              shuffled_z)
      norm_average = 0.5 * (jnp.sum(jnp.square(learned_z), -1) +
                            jnp.sum(jnp.square(shuffled_z), -1))
      # Using default value of beta = 0.1.
      online_distances = norm_average + 0.1 * base_distances
      learned_next_z = jax.vmap(encoder_online_critic)(next_states)
      target_distances = mico_target_distances(learned_next_z,
                                               rewards,
                                               shuffled_idx,
                                               cumulative_gamma)
      return jnp.mean(jax.vmap(losses.huber_loss)(online_distances,
                                                  target_distances))
    else:
      online_dist = jax.vmap(l1)(learned_z, shuffled_z)
      target_dist = target_z_distances(
          predicted_dynamics, rewards, shuffled_idx, cumulative_gamma)
      return jnp.mean(jax.vmap(lambda x: x**2)(online_dist - target_dist))

  def dynamics_loss_fn(
      reward_params,
      dynamics_params):
    # Compute reward loss.
    def reward_online(z):
      return reward_model_def.apply(reward_params, z)

    predicted_rewards = jax.vmap(reward_online)(fixed_encoded_states)
    reward_loss = jnp.mean(jax.vmap(losses.mse_loss)(rewards,
                                                     predicted_rewards))

    # Compute dynamics loss.
    def dynamics_online(z, a, rng):
      return dynamics_model_def.apply(dynamics_params, z, a, rng)

    brng = jnp.stack(jax.random.split(rng2, num=batch_size))
    predicted_dynamics = jax.vmap(dynamics_online)(
        fixed_encoded_states, actions, brng)
    dynamics_loss = jnp.mean(jax.vmap(losses.mse_loss)(
        predicted_dynamics.sample, encoded_next_states))

    combined_loss = reward_loss + dynamics_loss
    return combined_loss, {
        'combined_loss': combined_loss,
        'reward_loss': reward_loss,
        'dynamics_loss': dynamics_loss,
    }

  def sac_loss_fn(
      params,
      log_alpha):
    """Calculates the loss for one transition."""
    def critic_online(z, action):
      return network_def.apply(params, z, action, method=network_def.critic)

    # We use frozen_params so that gradients can flow back to the actor without
    # being used to update the critic.
    def frozen_critic_online(z, action):
      return network_def.apply(frozen_params, z, action,
                               method=network_def.critic)

    def actor_online(z, action):
      return network_def.apply(params, z, action, method=network_def.actor)

    def q_target(next_z, rng):
      return network_def.apply(target_params, next_z, rng, True)

    # J_Q(\theta) from equation (5) in paper.
    q_values_1, q_values_2 = jax.vmap(critic_online)(fixed_encoded_states,
                                                     actions)
    q_values_1 = jnp.squeeze(q_values_1)
    q_values_2 = jnp.squeeze(q_values_2)

    brng = jnp.stack(jax.random.split(rng3, num=batch_size))
    target_outputs = jax.vmap(q_target)(encoded_next_states, brng)
    target_q_values_1, target_q_values_2 = target_outputs.critic
    target_q_values_1 = jnp.squeeze(target_q_values_1)
    target_q_values_2 = jnp.squeeze(target_q_values_2)
    target_q_values = jnp.minimum(target_q_values_1, target_q_values_2)

    alpha_value = jnp.exp(log_alpha)
    log_probs = target_outputs.actor.log_probability
    targets = reward_scale_factor * rewards + cumulative_gamma * (
        target_q_values - alpha_value * log_probs) * (1. - terminals)
    targets = jax.lax.stop_gradient(targets)
    critic_loss_1 = jax.vmap(losses.mse_loss)(q_values_1, targets)
    critic_loss_2 = jax.vmap(losses.mse_loss)(q_values_2, targets)
    critic_loss = jnp.mean(critic_loss_1 + critic_loss_2)

    # J_{\pi}(\phi) from equation (9) in paper.
    brng = jnp.stack(jax.random.split(rng4, num=batch_size))
    mean_actions, sampled_actions, action_log_probs = jax.vmap(
        actor_online)(fixed_encoded_states, brng)

    q_values_no_grad_1, q_values_no_grad_2 = jax.vmap(frozen_critic_online)(
        fixed_encoded_states, sampled_actions)
    q_values_no_grad_1 = jnp.squeeze(q_values_no_grad_1)
    q_values_no_grad_2 = jnp.squeeze(q_values_no_grad_2)
    no_grad_q_values = jnp.minimum(q_values_no_grad_1, q_values_no_grad_2)
    alpha_value = jnp.exp(jax.lax.stop_gradient(log_alpha))
    policy_loss = jnp.mean(alpha_value * action_log_probs - no_grad_q_values)

    # J(\alpha) from equation (18) in paper.
    entropy_diffs = -action_log_probs - target_entropy
    alpha_loss = jnp.mean(log_alpha * jax.lax.stop_gradient(entropy_diffs))

    # Giving a smaller weight to the critic empirically gives better results
    combined_loss = 0.5 * critic_loss + 1.0 * policy_loss + 1.0 * alpha_loss
    return combined_loss, {
        'critic_loss': critic_loss,
        'policy_loss': policy_loss,
        'alpha_loss': alpha_loss,
        'critic_value_1': jnp.mean(q_values_1),
        'critic_value_2': jnp.mean(q_values_2),
        'target_value_1': jnp.mean(target_q_values_1),
        'target_value_2': jnp.mean(target_q_values_2),
        'mean_action': jnp.mean(mean_actions, axis=0)
    }

  bisim_grad_fn = jax.value_and_grad(bisimulation_loss_fn)
  bisim_loss, bisim_gradient = bisim_grad_fn(encoder_params,
                                             dynamics_params)
  dynamics_grad_fn = jax.value_and_grad(dynamics_loss_fn, argnums=(0, 1),
                                        has_aux=True)
  (_, dynamics_aux_vars), dynamics_gradients = dynamics_grad_fn(reward_params,
                                                                dynamics_params)
  reward_gradient, dynamics_gradient = dynamics_gradients
  sac_grad_fn = jax.value_and_grad(sac_loss_fn, argnums=(0, 1), has_aux=True)
  (_, sac_aux_vars), sac_gradients = sac_grad_fn(params, log_alpha)
  network_gradient, alpha_gradient = sac_gradients

  # Apply gradients to all the optimizers.
  encoder_updates, encoder_optim_state = encoder_optim.update(
      bisim_gradient, encoder_optim_state, params=encoder_params)
  encoder_params = optax.apply_updates(encoder_params, encoder_updates)
  reward_updates, reward_optim_state = reward_optim.update(
      reward_gradient, reward_optim_state, params=reward_params)
  reward_params = optax.apply_updates(reward_params, reward_updates)
  dynamics_updates, dynamics_optim_state = dynamics_optim.update(
      dynamics_gradient, dynamics_optim_state, params=dynamics_params)
  dynamics_params = optax.apply_updates(dynamics_params, dynamics_updates)
  updates, optim_state = optim.update(network_gradient, optim_state,
                                      params=params)
  params = optax.apply_updates(params, updates)
  alpha_updates, alpha_optim_state = alpha_optim.update(
      alpha_gradient, alpha_optim_state, params=log_alpha)
  log_alpha = optax.apply_updates(log_alpha, alpha_updates)

  # Compile everything in a dict.
  returns = {
      'encoder_params': encoder_params,
      'reward_params': reward_params,
      'dynamics_params': dynamics_params,
      'network_params': params,
      'log_alpha': log_alpha,
      'encoder_optim_state': encoder_optim_state,
      'reward_optim_state': reward_optim_state,
      'dynamics_optim_state': dynamics_optim_state,
      'network_optim_state': optim_state,
      'alpha_optim_state': alpha_optim_state,
      'Losses/Bisim': bisim_loss,
      'Losses/Reward': dynamics_aux_vars['reward_loss'],
      'Losses/Dynamics': dynamics_aux_vars['dynamics_loss'],
      'Losses/Critic': sac_aux_vars['critic_loss'],
      'Losses/Actor': sac_aux_vars['policy_loss'],
      'Losses/Alpha': sac_aux_vars['alpha_loss'],
      'Values/CriticValues1': sac_aux_vars['critic_value_1'],
      'Values/CriticValues2': sac_aux_vars['critic_value_2'],
      'Values/TargetValues1': sac_aux_vars['target_value_1'],
      'Values/TargetValues2': sac_aux_vars['target_value_2'],
      'Values/Alpha': jnp.squeeze(jnp.exp(log_alpha)),
  }
  for i, a in enumerate(sac_aux_vars['mean_action']):
    returns.update({f'Values/MeanActions{i}': a})
  return returns


@functools.partial(jax.jit, static_argnums=(0, 1))
def select_action(encoder_network_def, network_def, params, state, rng,
                  eval_mode=False):
  """Sample an action to take from the current networks."""
  encoder_params, actor_params = params
  rng, rng2 = jax.random.split(rng)
  z = encoder_network_def.apply(encoder_params, state).actor_z
  greedy_a, sampled_a, _ = network_def.apply(
      actor_params, z, rng2, method=network_def.actor)
  return rng, jnp.where(eval_mode, greedy_a, sampled_a)


@gin.configurable
class DBCAgent(sac_agent.SACAgent):
  """A JAX implementation of Deep Bisim for Control."""

  def __init__(self,
               action_shape,
               action_limits,
               observation_shape,
               action_dtype=jnp.float32,
               observation_dtype=jnp.float32,
               summary_writer=None,
               use_mico=False):
    assert isinstance(observation_shape, tuple)
    self.encoder_network_def = internal_continuous_networks.SACEncoderNetwork()
    self.reward_model_def = RewardModel()
    self.dynamics_model_def = DynamicsModel(action_shape)
    self._use_mico = use_mico
    super().__init__(action_shape, action_limits, observation_shape,
                     action_dtype, observation_dtype, network=SACConvNetwork,
                     summary_writer=summary_writer)
    logging.info('\tuse_mico: %s', use_mico)

  def _build_networks_and_optimizer(self):
    rngs = jax.random.split(self._rng, num=5)
    self._rng, encoder_key, network_key, reward_key, dynamics_key = rngs
    self.network_optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.encoder_optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.reward_optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.dynamics_optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.alpha_optimizer = dqn_agent.create_optimizer(self._optimizer_name)

    # Initialize encoder network.
    self.encoder_params = self.encoder_network_def.init(encoder_key, self.state)
    self.encoder_optimizer_state = self.encoder_optimizer.init(
        self.encoder_params)

    # Create a sample latent state for initializing the SAC network.
    sample_z = jnp.zeros_like(
        self.encoder_network_def.apply(
            self.encoder_params, self.state).critic_z)
    # since it is only used for shape inference during initialization.
    self.network_params = self.network_def.init(network_key, sample_z,
                                                network_key)
    self.optimizer_state = self.network_optimizer.init(self.network_params)

    # Initialize reward and dynamics models.
    self.reward_params = self.reward_model_def.init(reward_key, sample_z)
    self.reward_optimizer_state = self.reward_optimizer.init(self.reward_params)
    # Sending a dummy action and key for initialization.
    self.dynamics_params = self.dynamics_model_def.init(
        dynamics_key, sample_z, jnp.zeros(self.action_shape), dynamics_key)
    self.dynamics_optimizer_state = self.dynamics_optimizer.init(
        self.dynamics_params)

    self.encoder_target_params = self.encoder_params
    self.target_params = self.network_params

    # \alpha network
    self.log_alpha = jnp.zeros(1)
    self.alpha_optimizer_state = self.alpha_optimizer.init(self.log_alpha)

  def _maybe_sync_weights(self):
    """Syncs the target weights with the online weights."""
    if (self.target_update_type == 'hard' and
        self.training_steps % self.target_update_period != 0):
      return

    def _sync_weights(target_p, online_p):
      return (self.target_smoothing_coefficient * online_p +
              (1 - self.target_smoothing_coefficient) * target_p)

    self.target_params = jax.tree_multimap(_sync_weights, self.target_params,
                                           self.network_params)
    self.encoder_target_params = jax.tree_multimap(_sync_weights,
                                                   self.encoder_target_params,
                                                   self.encoder_params)

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode."""
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    if self._replay.add_count > self.min_replay_history:
      self._rng, self.action = select_action(
          self.encoder_network_def,
          self.network_def,
          (self.encoder_params, self.network_params),
          self.state,
          self._rng, self.eval_mode)
    else:
      self._rng, action_rng = jax.random.split(self._rng)
      self.action = jax.random.uniform(action_rng, self.action_shape,
                                       self.action_dtype, self.action_limits[0],
                                       self.action_limits[1])
    self.action = onp.asarray(self.action)
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition & returns the agent's next action."""
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    if self._replay.add_count > self.min_replay_history:
      self._rng, self.action = select_action(
          self.encoder_network_def,
          self.network_def,
          (self.encoder_params, self.network_params),
          self.state,
          self._rng, self.eval_mode)
    else:
      self._rng, action_rng = jax.random.split(self._rng)
      self.action = jax.random.uniform(action_rng, self.action_shape,
                                       self.action_dtype, self.action_limits[0],
                                       self.action_limits[1])
    self.action = onp.asarray(self.action)
    return self.action

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self._rng, key = jax.random.split(self._rng)

        train_returns = train(
            self.encoder_network_def, self.network_def,
            self.reward_model_def, self.dynamics_model_def,
            self.encoder_optimizer, self.network_optimizer,
            self.reward_optimizer, self.dynamics_optimizer,
            self.alpha_optimizer,
            self.encoder_params, self.encoder_target_params,
            self.network_params, self.target_params,
            self.log_alpha, self.reward_params,
            self.dynamics_params,
            self.encoder_optimizer_state, self.optimizer_state,
            self.reward_optimizer_state, self.dynamics_optimizer_state,
            self.alpha_optimizer_state,
            key, self.replay_elements['state'],
            self.replay_elements['action'], self.replay_elements['next_state'],
            self.replay_elements['reward'], self.replay_elements['terminal'],
            self.cumulative_gamma, self.target_entropy,
            self.reward_scale_factor,
            self._use_mico)

        self.network_params = train_returns['network_params']
        self.encoder_params = train_returns['encoder_params']
        self.reward_params = train_returns['reward_params']
        self.dynamics_params = train_returns['dynamics_params']
        self.log_alpha = train_returns['log_alpha']
        self.encoder_optimizer_state = train_returns['encoder_optim_state']
        self.optimizer_state = train_returns['network_optim_state']
        self.reward_optimizer_state = train_returns['reward_optim_state']
        self.dynamics_optimizer_state = train_returns['dynamics_optim_state']
        self.alpha_optimizer_state = train_returns['alpha_optim_state']

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):

          for k in train_returns:
            if k.startswith('Losses') or k.startswith('Values'):
              self.summary_writer.scalar(k, train_returns[k],
                                         self.training_steps)
          self.summary_writer.flush()
        self._maybe_sync_weights()
    self.training_steps += 1

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state."""
    if not tf.io.gfile.exists(checkpoint_dir):
      return None
    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {
        'state': self.state,
        'training_steps': self.training_steps,
        'encoder_optimizer_state': self.encoder_optimizer_state,
        'optimizer_state': self.optimizer_state,
        'reward_optimizer_state': self.reward_optimizer_state,
        'dynamics_optimizer_state': self.dynamics_optimizer_state,
        'alpha_optimizer_state': self.alpha_optimizer_state,
        'encoder_params': self.encoder_params,
        'network_params': self.network_params,
        'reward_params': self.reward_params,
        'dynamics_params': self.dynamics_params,
        'log_alpha': self.log_alpha,
        'target_params': self.target_params,
        'encoder_target_params': self.encoder_target_params,
    }
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint."""
    try:
      # self._replay.load() will throw a NotFoundError if it does not find all
      # the necessary files.
      self._replay.load(checkpoint_dir, iteration_number)
    except tf.errors.NotFoundError:
      if not self.allow_partial_reload:
        # If we don't allow partial reloads, we will return False.
        return False
      logging.warning('Unable to reload replay buffer!')
    if bundle_dictionary is not None:
      self.state = bundle_dictionary['state']
      self.training_steps = bundle_dictionary['training_steps']

      self.encoder_optimizer_state = bundle_dictionary[
          'encoder_optimizer_state']
      self.network_optimizer_state = bundle_dictionary['optimizer_state']
      self.reward_optimizer_state = bundle_dictionary['reward_optimizer_state']
      self.dynamics_optimizer_state = bundle_dictionary[
          'dynamics_optimizer_state']
      self.alpha_optimizer_state = bundle_dictionary['alpha_optimizer_state']
      self.encoder_params = bundle_dictionary['encoder_params']
      self.network_params = bundle_dictionary['network_params']
      self.reward_params = bundle_dictionary['reward_params']
      self.dynamics_params = bundle_dictionary['dynamics_params']
      self.log_alpha = bundle_dictionary['log_alpha']
      self.target_params = bundle_dictionary['target_params']
      self.encoder_target_params = bundle_dictionary[
          'encoder_target_params']
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    return True
