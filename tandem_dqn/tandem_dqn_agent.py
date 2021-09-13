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

"""Tandem DQN agent (one active, one passive agent)."""

import functools

from absl import logging
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf


@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, cumulative_gamma,
          loss_type='huber', double_dqn=True):
  """Run the training step."""
  def loss_fn(params, target):
    def q_online(state):
      return network_def.apply(params, state)

    q_values = jax.vmap(q_online)(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    if loss_type == 'huber':
      loss = jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))
    else:
      loss = jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))
    return loss

  def q_online(state):
    return network_def.apply(online_params, state)

  def q_target(state):
    return network_def.apply(target_params, state)

  target = target_q(q_online,
                    q_target,
                    next_states,
                    rewards,
                    terminals,
                    cumulative_gamma,
                    double_dqn)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grad = grad_fn(online_params, target)
  updates, optimizer_state = optimizer.update(grad, optimizer_state)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss


def target_q(online_network, target_network, next_states, rewards, terminals,
             cumulative_gamma, double_dqn):
  """Compute the target Q-value."""
  if double_dqn:
    next_state_q_vals_for_argmax = jax.vmap(
        online_network, in_axes=(0))(next_states).q_values
  else:
    next_state_q_vals_for_argmax = jax.vmap(
        target_network, in_axes=(0))(next_states).q_values
  next_state_q_vals_for_argmax = jnp.squeeze(next_state_q_vals_for_argmax)
  next_argmax = jnp.argmax(next_state_q_vals_for_argmax, axis=1)
  q_values = jax.vmap(
      target_network, in_axes=(0))(next_states).q_values
  replay_next_qt_max = jax.vmap(lambda t, u: t[u])(q_values, next_argmax)
  return jax.lax.stop_gradient(rewards + cumulative_gamma * replay_next_qt_max *
                               (1. - terminals))


@gin.configurable
class TandemDQNAgent(dqn_agent.JaxDQNAgent):
  """Tandem DQN agent."""

  def __init__(self, num_actions, double_dqn=True, summary_writer=None):
    self._double_dqn = double_dqn
    super().__init__(num_actions, summary_writer=summary_writer)

  def _build_networks_and_optimizer(self):
    self._rng, active_rng, passive_rng = jax.random.split(self._rng, 3)
    # Initialize active networks.
    self.active_online_params = self.network_def.init(active_rng,
                                                      x=self.state)
    self.active_optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.active_optimizer_state = self.active_optimizer.init(
        self.active_online_params)
    self.active_target_params = self.active_online_params
    # Initialize passive network with the regular network.
    self.passive_online_params = self.network_def.init(passive_rng,
                                                       x=self.state)
    self.passive_optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.passive_optimizer_state = self.passive_optimizer.init(
        self.passive_online_params)
    self.passive_target_params = self.passive_online_params

  def _sync_weights(self):
    """Syncs the target_params with the online_params."""
    self.active_target_params = self.active_online_params
    self.passive_target_params = self.passive_online_params

  def _select_action(self, params):
    self._rng, action = dqn_agent.select_action(self.network_def,
                                                params,
                                                self.state,
                                                self._rng,
                                                self.num_actions,
                                                self.eval_mode,
                                                self.epsilon_eval,
                                                self.epsilon_train,
                                                self.epsilon_decay_period,
                                                self.training_steps,
                                                self.min_replay_history,
                                                self.epsilon_fn)
    action = onp.asarray(action)
    return action

  def begin_episode(self, agent_type, observation):
    self._reset_state()
    self._record_observation(observation)
    if agent_type == 'passive':
      params = self.passive_online_params
    else:
      params = self.active_online_params
      if not self.eval_mode:
        self._train_step()

    action = self._select_action(params)
    if agent_type == 'active':
      self.action = action
    return action

  def step(self, agent_type, reward, observation):
    self._last_observation = self._observation
    self._record_observation(observation)
    if agent_type == 'passive':
      params = self.passive_online_params
    else:
      params = self.active_online_params
      if not self.eval_mode:
        self._store_transition(
            self._last_observation, self.action, reward, False)
        self._train_step()

    action = self._select_action(params)
    if agent_type == 'active':
      self.action = onp.asarray(self.action)
    return action

  def _train_step(self):
    """Runs a single tandem training step.

    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target if training steps is a multiple of
    target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        self.active_optimizer_state, self.active_online_params, active_loss = (
            dqn_agent.train(
                self.network_def,
                self.active_online_params,
                self.active_target_params,
                self.active_optimizer,
                self.active_optimizer_state,
                self.replay_elements['state'],
                self.replay_elements['action'],
                self.replay_elements['next_state'],
                self.replay_elements['reward'],
                self.replay_elements['terminal'],
                self.cumulative_gamma,
                self._double_dqn))
        (self.passive_optimizer_state, self.passive_online_params,
         passive_loss) = dqn_agent.train(
             self.network_def,
             self.passive_online_params,
             self.passive_target_params,
             self.passive_optimizer,
             self.passive_optimizer_state,
             self.replay_elements['state'],
             self.replay_elements['action'],
             self.replay_elements['next_state'],
             self.replay_elements['reward'],
             self.replay_elements['terminal'],
             self.cumulative_gamma,
             self._double_dqn)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          values = [tf.compat.v1.Summary.Value(tag='Losses/Active',
                                               simple_value=active_loss),
                    tf.compat.v1.Summary.Value(tag='Losses/Passive',
                                               simple_value=passive_loss)]
          summary = tf.compat.v1.Summary(value=values)
          self.summary_writer.add_summary(summary, self.training_steps)
          self.summary_writer.flush()
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1

  def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
    """Returns a self-contained bundle of the agent's state.

    This is used for checkpointing. It will return a dictionary containing all
    non-TensorFlow objects (to be saved into a file by the caller), and it saves
    all TensorFlow objects into a checkpoint file.

    Args:
      checkpoint_dir: str, directory where TensorFlow objects will be saved.
      iteration_number: int, iteration number to use for naming the checkpoint
        file.

    Returns:
      A dict containing additional Python objects to be checkpointed by the
        experiment. If the checkpoint directory does not exist, returns None.
    """
    if not tf.io.gfile.exists(checkpoint_dir):
      return None
    # Checkpoint the out-of-graph replay buffer.
    self._replay.save(checkpoint_dir, iteration_number)
    bundle_dictionary = {
        'state': self.state,
        'training_steps': self.training_steps,
        'active_online_params': self.active_online_params,
        'active_target_params': self.active_target_params,
        'active_optimizer_state': self.active_optimizer_state,
        'passive_online_params': self.passive_online_params,
        'passive_target_params': self.passive_target_params,
        'passive_optimizer_state': self.passive_optimizer_state,
    }
    return bundle_dictionary

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved.
      iteration_number: int, checkpoint version, used when restoring the replay
        buffer.
      bundle_dictionary: dict, containing additional Python objects owned by
        the agent.

    Returns:
      bool, True if unbundling was successful.
    """
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
      self.active_online_params = bundle_dictionary['active_online_params']
      self.passive_online_params = bundle_dictionary['passive_online_params']
      self.active_target_params = bundle_dictionary['active_target_params']
      self.passive_target_params = bundle_dictionary['passive_target_params']
      # We recreate the optimizer with the new online weights.
      self.active_optimizer_state = bundle_dictionary['active_optimizer_state']
      self.passive_optimizer_state = (
          bundle_dictionary['passive_optimizer_state'])
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    return True
