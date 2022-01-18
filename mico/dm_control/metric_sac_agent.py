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

"""SAC Agent with MICo loss.

Based on agent described in
  "Soft Actor-Critic Algorithms and Applications"
  by Tuomas Haarnoja et al.
  https://arxiv.org/abs/1812.05905
"""


import collections
import functools
from typing import Any, Mapping, Optional, Tuple

from absl import logging
from dopamine.jax import continuous_networks
from dopamine.jax import losses
from dopamine.jax.agents.sac import sac_agent
from dopamine.labs.sac_from_pixels import continuous_networks as internal_continuous_networks
import flax
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import optax
from mico.atari import metric_utils


SacCriticOutput = collections.namedtuple(
    'sac_critic_output', ['q_value1', 'q_value2', 'representation'])
SacOutput = collections.namedtuple('sac_output',
                                   ['actor', 'critic', 'representation'])


@gin.configurable
class SACConvNetwork(nn.Module):
  """A conv value and policy networks for SAC, also returns encoding."""
  action_shape: Tuple[int, Ellipsis]
  num_layers: int = 2
  hidden_units: int = 256
  action_limits: Optional[Tuple[Tuple[float, Ellipsis], Tuple[float, Ellipsis]]] = None

  def setup(self):
    self._encoder = internal_continuous_networks.SACEncoderNetwork()
    self._sac_network = continuous_networks.SACNetwork(
        self.action_shape, self.num_layers,
        self.hidden_units, self.action_limits)

  def __call__(self,
               state,
               key,
               mean_action = True):
    """Calls the SAC actor/critic networks."""
    encoding = self._encoder(state)

    actor_output = self._sac_network.actor(encoding.actor_z, key)
    action = actor_output.mean_action if mean_action else actor_output.sampled_action
    critic_output = self._sac_network.critic(encoding.critic_z, action)

    return SacOutput(actor_output, critic_output, encoding.critic_z)

  def actor(self, state,
            key):
    """Calls the SAC actor network."""
    encoding = self._encoder(state)
    return self._sac_network.actor(encoding.actor_z, key)

  def critic(self, state, action):
    """Calls the SAC critic network."""
    encoding = self._encoder(state)
    critic_out = self._sac_network.critic(encoding.critic_z, action)
    return SacCriticOutput(critic_out.q_value1, critic_out.q_value2,
                           encoding.critic_z)


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def train(network_def,
          optim,
          alpha_optim,
          optimizer_state,
          alpha_optimizer_state,
          params,
          target_params,
          log_alpha,
          key,
          states,
          actions,
          next_states,
          rewards,
          terminals,
          cumulative_gamma,
          target_entropy,
          reward_scale_factor,
          mico_weight):
  """Run the training step."""
  frozen_params = params  # For use within loss_fn without apply gradients

  batch_size = states.shape[0]
  actions = jnp.reshape(actions, (batch_size, -1))  # Flatten

  rng1, rng2 = jax.random.split(key, num=2)

  def loss_fn(
      params,
      log_alpha):
    """Calculates the loss for one transition."""
    def critic_online(state, action):
      return network_def.apply(params, state, action, method=network_def.critic)

    # We use frozen_params so that gradients can flow back to the actor without
    # being used to update the critic.
    def frozen_critic_online(state, action):
      return network_def.apply(frozen_params, state, action,
                               method=network_def.critic)

    def actor_online(state, action):
      return network_def.apply(params, state, action, method=network_def.actor)

    def q_target(next_state, rng):
      return network_def.apply(target_params, next_state, rng, True)

    # J_Q(\theta) from equation (5) in paper.
    q_values_1, q_values_2, representations = jax.vmap(critic_online)(
        states, actions)
    q_values_1 = jnp.squeeze(q_values_1)
    q_values_2 = jnp.squeeze(q_values_2)
    representations = jnp.squeeze(representations)

    brng1 = jnp.stack(jax.random.split(rng1, num=batch_size))
    target_outputs = jax.vmap(q_target)(next_states, brng1)
    target_q_values_1, target_q_values_2 = target_outputs.critic
    target_next_r = target_outputs.representation
    target_q_values_1 = jnp.squeeze(target_q_values_1)
    target_q_values_2 = jnp.squeeze(target_q_values_2)
    target_next_r = jnp.squeeze(target_next_r)
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
    brng2 = jnp.stack(jax.random.split(rng2, num=batch_size))
    mean_actions, sampled_actions, action_log_probs = jax.vmap(
        actor_online)(states, brng2)

    q_values_no_grad_1, q_values_no_grad_2, target_r = jax.vmap(
        frozen_critic_online)(states, sampled_actions)
    q_values_no_grad_1 = jnp.squeeze(q_values_no_grad_1)
    q_values_no_grad_2 = jnp.squeeze(q_values_no_grad_2)
    target_r = jnp.squeeze(target_r)
    no_grad_q_values = jnp.minimum(q_values_no_grad_1, q_values_no_grad_2)
    alpha_value = jnp.exp(jax.lax.stop_gradient(log_alpha))
    policy_loss = jnp.mean(alpha_value * action_log_probs - no_grad_q_values)

    # J(\alpha) from equation (18) in paper.
    entropy_diffs = -action_log_probs - target_entropy
    alpha_loss = jnp.mean(log_alpha * jax.lax.stop_gradient(entropy_diffs))

    # MICo loss.
    distance_fn = metric_utils.cosine_distance
    online_dist = metric_utils.representation_distances(
        representations, target_r, distance_fn)
    target_dist = metric_utils.target_distances(
        target_next_r, rewards, distance_fn, cumulative_gamma)
    mico_loss = jnp.mean(jax.vmap(losses.huber_loss)(online_dist, target_dist))

    # Giving a smaller weight to the critic empirically gives better results
    sac_loss = 0.5 * critic_loss + 1.0 * policy_loss + 1.0 * alpha_loss
    combined_loss = (1. - mico_weight) * sac_loss + mico_weight * mico_loss
    return combined_loss, {
        'mico_loss': mico_loss,
        'critic_loss': critic_loss,
        'policy_loss': policy_loss,
        'alpha_loss': alpha_loss,
        'critic_value_1': jnp.mean(q_values_1),
        'critic_value_2': jnp.mean(q_values_2),
        'target_value_1': jnp.mean(target_q_values_1),
        'target_value_2': jnp.mean(target_q_values_2),
        'mean_action': jnp.mean(mean_actions, axis=0)
    }

  grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

  (_, aux_vars), gradients = grad_fn(params, log_alpha)

  network_gradient, log_alpha_gradient = gradients

  # Apply gradients to all the optimizers.
  updates, optimizer_state = optim.update(network_gradient, optimizer_state,
                                          params=params)
  params = optax.apply_updates(params, updates)
  alpha_updates, alpha_optimizer_state = alpha_optim.update(
      log_alpha_gradient, alpha_optimizer_state, params=log_alpha)
  log_alpha = optax.apply_updates(log_alpha, alpha_updates)

  # Compile everything in a dict.
  returns = {
      'network_params': params,
      'log_alpha': log_alpha,
      'optimizer_state': optimizer_state,
      'alpha_optimizer_state': alpha_optimizer_state,
      'Losses/Mico': aux_vars['mico_loss'],
      'Losses/Critic': aux_vars['critic_loss'],
      'Losses/Actor': aux_vars['policy_loss'],
      'Losses/Alpha': aux_vars['alpha_loss'],
      'Values/CriticValues1': jnp.mean(aux_vars['critic_value_1']),
      'Values/CriticValues2': jnp.mean(aux_vars['critic_value_2']),
      'Values/TargetValues1': jnp.mean(aux_vars['target_value_1']),
      'Values/TargetValues2': jnp.mean(aux_vars['target_value_2']),
      'Values/Alpha': jnp.exp(log_alpha),
  }
  for i, a in enumerate(aux_vars['mean_action']):
    returns.update({f'Values/MeanActions{i}': a})
  return returns


@gin.configurable
class MetricSACAgent(sac_agent.SACAgent):
  """A JAX implementation of SAC with MICo."""

  def __init__(self,
               action_shape,
               action_limits,
               observation_shape,
               mico_weight=1e-5,
               action_dtype=jnp.float32,
               observation_dtype=jnp.float32,
               summary_writer=None):
    assert isinstance(observation_shape, tuple)
    logging.info('Creating MetricSACAgent')
    self._mico_weight = mico_weight
    super().__init__(action_shape, action_limits, observation_shape,
                     action_dtype, observation_dtype, network=SACConvNetwork,
                     summary_writer=summary_writer)

  def _train_step(self):
    """Runs a single training step."""
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self._rng, key = jax.random.split(self._rng)

        train_returns = train(
            self.network_def,
            self.network_optimizer,
            self.alpha_optimizer,
            self.optimizer_state,
            self.alpha_optimizer_state,
            self.network_params,
            self.target_params,
            self.log_alpha,
            key, self.replay_elements['state'],
            self.replay_elements['action'], self.replay_elements['next_state'],
            self.replay_elements['reward'], self.replay_elements['terminal'],
            self.cumulative_gamma, self.target_entropy,
            self.reward_scale_factor,
            self._mico_weight)

        self.network_params = train_returns['network_params']
        self.optimizer_state = train_returns['optimizer_state']
        self.log_alpha = train_returns['log_alpha']
        self.alpha_optimizer_state = train_returns['alpha_optimizer_state']

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

