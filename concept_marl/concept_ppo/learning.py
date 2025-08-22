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

"""Learner for the Concept PPO agent."""

from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import rlax

from concept_marl.concept_ppo import networks
from concept_marl.utils.concept_types import ConceptType


class Batch(NamedTuple):
  """A batch of data; all shapes are expected to be [B, ...]."""
  observations: types.NestedArray
  actions: jnp.ndarray
  advantages: jnp.ndarray

  # Target value estimate used to bootstrap the value function.
  target_values: jnp.ndarray

  # Value estimate and action log-prob at behavior time.
  behavior_values: jnp.ndarray
  behavior_log_probs: jnp.ndarray


class TrainingState(NamedTuple):
  """Training state consists of network parameters and optimiser state."""
  params: networks_lib.Params
  opt_state: optax.OptState
  random_key: networks_lib.PRNGKey


class ConceptPPOLearner(acme.Learner):
  """Learner for Concept PPO."""

  def __init__(
      self,
      concept_ppo_networks,
      iterator,
      optimizer,
      random_key,
      ppo_clipping_epsilon = 0.2,
      clip_value = True,
      gae_lambda = 0.95,
      discount = 0.99,
      entropy_cost = 0.,
      value_cost = 1.,
      concept_cost = 0.,
      max_abs_reward = np.inf,
      num_epochs = 4,
      num_minibatches = 1,
      counter = None,
      logger = None,
  ):

    def gae_advantages(
        rewards, discounts, values
    ):
      """Uses truncated GAE to compute advantages (same as regular PPO)."""

      # Apply reward clipping.
      rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

      advantages = rlax.truncated_generalized_advantage_estimation(
          rewards[:-1], discounts[:-1], gae_lambda, values)
      advantages = jax.lax.stop_gradient(advantages)

      # Exclude the bootstrap value
      target_values = values[:-1] + advantages
      target_values = jax.lax.stop_gradient(target_values)

      return advantages, target_values  # pytype: disable=bad-return-type  # numpy-scalars

    def loss(
        params,
        observations,
        actions,
        behaviour_log_probs,
        target_values,
        advantages,
        behavior_values,
        categorical_values = 4,
    ):
      """Surrogate loss using clipped probability ratios."""

      # extract concept observations
      scalar_concept_labels = observations['concept_observations'][
          'scalar_concepts']['concept_values']
      scalar_concept_types = observations['concept_observations'][
          'scalar_concepts']['concept_types']
      cat_concept_labels = observations['concept_observations']['cat_concepts'][
          'concept_values']

      # collect value estimates, concept predictions, etc from network.
      (distribution_params,
       values), concepts = concept_ppo_networks.network.apply(
           params, observations, is_training=True)
      log_probs = concept_ppo_networks.log_prob(distribution_params, actions)
      entropy = concept_ppo_networks.entropy(distribution_params)

      # compute importance sampling weights: current policy / behavior policy.
      rhos = jnp.exp(log_probs - behaviour_log_probs)

      # compute clipped policy loss.
      policy_loss = rlax.clipped_surrogate_pg_loss(rhos, advantages,
                                                   ppo_clipping_epsilon)

      # value function loss, excluding the bootstrap value.
      unclipped_value_error = target_values - values
      unclipped_value_loss = unclipped_value_error ** 2

      if clip_value:
        # Clip values to reduce variablility during critic training.
        clipped_values = behavior_values + jnp.clip(
            values - behavior_values, -ppo_clipping_epsilon,
            ppo_clipping_epsilon)
        clipped_value_error = target_values - clipped_values
        clipped_value_loss = clipped_value_error ** 2
        value_loss = jnp.mean(jnp.fmax(unclipped_value_loss,
                                       clipped_value_loss))
      else:
        # For Mujoco envs clipping hurts a lot. Evidenced by Figure 43 in
        # https://arxiv.org/pdf/2006.05990.pdf
        value_loss = jnp.mean(unclipped_value_loss)

      # entropy regulariser.
      entropy_loss = -jnp.mean(entropy)

      # concept losses
      scalar_idx = scalar_concept_labels.shape[1]
      scalar_concepts = concepts[:, :scalar_idx]
      cat_concepts = concepts[:, scalar_idx:]

      # binary concept loss (binary cross entropy)
      binary_masks = scalar_concept_types == ConceptType.BINARY
      concept_bin_loss = optax.sigmoid_binary_cross_entropy(
          scalar_concepts, scalar_concept_labels) * binary_masks

      # scalar concept loss (MSE)
      reg_masks = scalar_concept_types == ConceptType.SCALAR
      concept_reg_loss = (
          (scalar_concept_labels - scalar_concepts)**2) * reg_masks

      # complete scalar concept loss
      scalar_concept_loss = concept_reg_loss + concept_bin_loss

      # categorical loss (cross entropy)
      new_shape = (cat_concepts.shape[0], categorical_values,
                   cat_concepts.shape[-1] // categorical_values)

      cat_concept_preds = cat_concepts.reshape(new_shape)
      cat_concept_labels = cat_concept_labels.reshape(new_shape)
      cat_concept_loss = optax.softmax_cross_entropy(cat_concept_preds,
                                                     cat_concept_labels)

      # combine concept losses
      concept_loss = jnp.mean(scalar_concept_loss) + jnp.mean(cat_concept_loss)

      # total loss
      total_loss = (
          policy_loss + value_loss * value_cost + entropy_loss * entropy_cost +
          concept_loss * concept_cost)

      # standard losses info (for logging)
      info = {
          'loss_total': total_loss,
          'loss_policy': policy_loss,
          'loss_value': value_loss,
          'loss_entropy': entropy_loss,
          'loss_concept_total': concept_loss
      }
      return total_loss, info  # pytype: disable=bad-return-type  # numpy-scalars

    @jax.jit
    def sgd_step(
        state, sample
    ):
      """Performs a minibatch SGD step, returning new state and metrics."""

      # Extract the data.
      data = sample.data
      observations, actions, rewards, termination, extra = (data.observation,
                                                            data.action,
                                                            data.reward,
                                                            data.discount,
                                                            data.extras)
      discounts = termination * discount
      behavior_log_probs = extra['log_prob']

      def get_behavior_values(params,
                              observations):
        o = jax.tree.map(lambda x: jnp.reshape(x, [-1] + list(x.shape[2:])),
                         observations)
        (_, behavior_values), _ = concept_ppo_networks.network.apply(
            params, o, is_training=True)
        behavior_values = jnp.reshape(behavior_values, rewards.shape[0:2])
        return behavior_values

      behavior_values = get_behavior_values(state.params, observations)

      # Vmap over batch dimension
      batch_gae_advantages = jax.vmap(gae_advantages, in_axes=0)
      advantages, target_values = batch_gae_advantages(rewards, discounts,
                                                       behavior_values)
      # Exclude the last step - it was only used for bootstrapping.
      # The shape is [num_sequences, num_steps, ..]
      observations, actions, behavior_log_probs, behavior_values = jax.tree.map(
          lambda x: x[:, :-1],
          (observations, actions, behavior_log_probs, behavior_values))

      trajectories = Batch(
          observations=observations,
          actions=actions,
          advantages=advantages,
          behavior_log_probs=behavior_log_probs,
          target_values=target_values,
          behavior_values=behavior_values)

      # Concatenate all trajectories. Reshape from [num_sequences, num_steps,..]
      # to [num_sequences * num_steps,..]
      assert len(target_values.shape) > 1
      # This method of pulling out num_sequences, num_steps from the shape
      # of target_values was inherited from Acme's original PPO.
      num_sequences = target_values.shape[0]
      num_steps = target_values.shape[1]
      batch_size = num_sequences * num_steps
      assert batch_size % num_minibatches == 0, (
          'Num minibatches must divide batch size. Got batch_size={}'
          ' num_minibatches={}.'
      ).format(batch_size, num_minibatches)
      batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]),
                           trajectories)

      # Compute gradients.
      grad_fn = jax.grad(loss, has_aux=True)

      def model_update_minibatch(
          carry,
          minibatch,
      ):
        """Performs model update for a single minibatch."""
        params, opt_state = carry
        # Normalize advantages at the minibatch level before using them.
        advantages = ((minibatch.advantages -
                       jnp.mean(minibatch.advantages, axis=0)) /
                      (jnp.std(minibatch.advantages, axis=0) + 1e-8))
        gradients, metrics = grad_fn(params,
                                     minibatch.observations,
                                     minibatch.actions,
                                     minibatch.behavior_log_probs,
                                     minibatch.target_values,
                                     advantages,
                                     minibatch.behavior_values)

        # Apply updates
        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)

        metrics['norm_grad'] = optax.global_norm(gradients)
        metrics['norm_updates'] = optax.global_norm(updates)
        return (params, opt_state), metrics

      def model_update_epoch(
          carry, unused_t
      ):
        """Performs model updates based on one epoch of data."""
        key, params, opt_state, batch = carry
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, batch_size)
        shuffled_batch = jax.tree.map(
            lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(
            lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
            shuffled_batch)

        (params, opt_state), metrics = jax.lax.scan(
            model_update_minibatch, (params, opt_state), minibatches,
            length=num_minibatches)

        return (key, params, opt_state, batch), metrics

      params = state.params
      opt_state = state.opt_state
      # Repeat training for the given number of epoch, taking a random
      # permutation for every epoch.
      (key, params, opt_state, _), metrics = jax.lax.scan(
          model_update_epoch, (state.random_key, params, opt_state, batch), (),
          length=num_epochs)

      metrics = jax.tree.map(jnp.mean, metrics)
      metrics['norm_params'] = optax.global_norm(params)
      metrics['observations_mean'] = jnp.mean(
          utils.batch_concat(
              jax.tree.map(lambda x: jnp.abs(jnp.mean(x, axis=(0, 1))),
                           observations),
              num_batch_dims=0))
      metrics['observations_std'] = jnp.mean(
          utils.batch_concat(
              jax.tree.map(lambda x: jnp.std(x, axis=(0, 1)), observations),
              num_batch_dims=0))
      metrics['rewards_mean'] = jnp.mean(
          jnp.abs(jnp.mean(rewards, axis=(0, 1))))
      metrics['rewards_std'] = jnp.std(rewards, axis=(0, 1))
      new_state = TrainingState(params=params,
                                opt_state=opt_state,
                                random_key=key)
      return new_state, metrics

    def make_initial_state(key):
      """Initialises the training state (parameters and optimiser state)."""
      key_init, key_state = jax.random.split(key)
      initial_params = concept_ppo_networks.network.init(key_init)
      initial_opt_state = optimizer.init(initial_params)
      return TrainingState(
          params=initial_params,
          opt_state=initial_opt_state,
          random_key=key_state)

    # Initialise training state (parameters and optimiser state).
    self._state = make_initial_state(random_key)

    # Internalise iterator.
    self._iterator = iterator
    self._sgd_step = sgd_step

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner', steps_key=self._counter.get_steps_key())

  def step(self):
    """Does a step of SGD and logs the results."""

    # Do a batch of SGD.
    sample = next(self._iterator)
    self._state, results = self._sgd_step(self._state, sample)

    # Update our counts and record it.
    counts = self._counter.increment(steps=1)

    # Snapshot and attempt to write logs.
    self._logger.write({**results, **counts})

  def get_variables(self, names):
    return [self._state.params]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
