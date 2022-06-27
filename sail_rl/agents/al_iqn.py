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

"""The implicit quantile networks agent with Advantage Learning (AL-IQN).

Advantage Learning is an action-gap increasing algorithm
(Baird, 1999; Bellemare et al., 2016).
The AL-IQN agent follows the description given in
"Self-Imitation Advantage Learning" (Ferret et al., 2020).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
from dopamine.agents.implicit_quantile import implicit_quantile_agent
import gin
import tensorflow.compat.v2 as tf

from sail_rl.common import replay_buffer


@gin.configurable
class ALImplicitQuantileAgent(implicit_quantile_agent.ImplicitQuantileAgent):
  """An implementation of the AL-IQN agent."""

  def __init__(self,
               sess,
               num_actions,
               kappa,
               num_tau_samples,
               num_tau_prime_samples,
               num_quantile_samples,
               alpha,
               clip,
               tf_device='/cpu:*',
               **kwargs):
    """Initializes the agent and constructs the Graph.

    Most of this constructor's parameters are IQN-specific hyperparameters whose
    values are taken from Dabney et al. (2018).

    Args:
      sess: `tf.compat.v1.Session` object for running associated ops.
      num_actions: int, number of actions the agent can take at any state.
      kappa: float, Huber loss cutoff.
      num_tau_samples: int, number of online quantile samples for loss
        estimation.
      num_tau_prime_samples: int, number of target quantile samples for loss
        estimation.
      num_quantile_samples: int, number of quantile samples for computing
        Q-values.
      alpha: float, value of the alpha parameter of AL.
      clip: float, if v > 0, (G_t - Q_target(s, a)) is clipped to (-v, v).
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      **kwargs: additional parameters, see ImplicitQuantileAgent.
    """
    self._alpha = alpha
    self._clip = clip
    self._curr_episode = 0

    super(ALImplicitQuantileAgent, self).__init__(
        sess, num_actions,
        kappa=kappa, num_tau_samples=num_tau_samples,
        num_tau_prime_samples=num_tau_prime_samples,
        num_quantile_samples=num_quantile_samples,
        **kwargs)

  def _build_networks(self):
    """Builds the IQN computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's quantile values.
      self.target_convnet: For computing the next state's target quantile
        values.
      self._net_outputs: The actual quantile values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' quantile values.
      self._replay_next_target_net_outputs: The replayed next states' target
        quantile values.
    """
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')

    # Compute the Q-values which are used for action selection in the current
    # state.
    self._net_outputs = self.online_convnet(self.state_ph,
                                            self.num_quantile_samples)
    # Shape of self._net_outputs.quantile_values:
    # num_quantile_samples x num_actions.
    # e.g. if num_actions is 2, it might look something like this:
    # Vals for Quantile .2  Vals for Quantile .4  Vals for Quantile .6
    #    [[0.1, 0.5],         [0.15, -0.3],          [0.15, -0.2]]
    # Q-values = [(0.1 + 0.15 + 0.15)/3, (0.5 + 0.15 + -0.2)/3].
    self._q_values = tf.reduce_mean(self._net_outputs.quantile_values, axis=0)
    self._q_argmax = tf.argmax(self._q_values, axis=0)

    self._replay_net_outputs = self.online_convnet(self._replay.states,
                                                   self.num_tau_samples)
    # Shape: (num_tau_samples x batch_size) x num_actions.
    self._replay_net_quantile_values = self._replay_net_outputs.quantile_values
    self._replay_net_quantiles = self._replay_net_outputs.quantiles

    # Do the same for next states in the replay buffer.
    self._replay_net_target_outputs = self.target_convnet(
        self._replay.next_states, self.num_tau_prime_samples)
    # Shape: (num_tau_prime_samples x batch_size) x num_actions.
    vals = self._replay_net_target_outputs.quantile_values
    self._replay_net_target_quantile_values = vals

    # Compute Q-values which are used for action selection for the next states
    # in the replay buffer. Compute the argmax over the Q-values.
    outputs_action = self.target_convnet(self._replay.next_states,
                                         self.num_quantile_samples)

    # Shape: (num_quantile_samples x batch_size) x num_actions.
    target_quantile_values_action = outputs_action.quantile_values
    # Shape: num_quantile_samples x batch_size x num_actions.
    target_quantile_values_action = tf.reshape(target_quantile_values_action,
                                               [self.num_quantile_samples,
                                                self._replay.batch_size,
                                                self.num_actions])
    # Shape: batch_size x num_actions.
    self._replay_net_target_q_values = tf.squeeze(tf.reduce_mean(
        target_quantile_values_action, axis=0))
    self._replay_next_qt_argmax = tf.argmax(
        self._replay_net_target_q_values, axis=1)

    # SAIL addition.
    # Calculates **current state** action-values.
    # Shape: num_tau_samples x batch_size x num_actions.
    replay_target_net_outputs = self.target_convnet(self._replay.states,
                                                    self.num_quantile_samples)
    replay_target_net_quantiles = tf.reshape(
        replay_target_net_outputs.quantile_values,
        [self.num_quantile_samples,
         self._replay.batch_size,
         self.num_actions])
    # Shape: batch_size x num_actions.
    self._replay_target_q_values = tf.squeeze(tf.reduce_mean(
        replay_target_net_quantiles, axis=0))

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A SAILWrappedPrioritizedReplayBuffer object.
    """
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    # Both replay schemes use the same data structure, but the 'uniform' scheme
    # sets all priorities to the same value (which yields uniform sampling).
    return replay_buffer.SAILWrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_target_quantile_values_op(self):
    """Build an op used as a target for return values at given quantiles.

    Returns:
      An op calculating the target quantile return.
    """
    batch_size = tf.shape(self._replay.rewards)[0]

    # Calculate AL modified rewards.
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_target_q = tf.reduce_max(
        self._replay_target_q_values,
        axis=1,
        name='replay_chosen_target_q')
    replay_target_q_al = tf.reduce_sum(
        replay_action_one_hot * self._replay_target_q_values,
        axis=1,
        name='replay_chosen_target_q_al')

    if self._clip > 0.:
      al_bonus = self._alpha * tf.clip_by_value(
          (replay_target_q_al - replay_target_q),
          -self._clip, self._clip)
    else:
      al_bonus = self._alpha * (
          replay_target_q_al - replay_target_q)

    # Shape of rewards: (num_tau_prime_samples x batch_size) x 1.
    rewards = (self._replay.rewards + al_bonus)[:, None]
    rewards = tf.tile(rewards, [self.num_tau_prime_samples, 1])

    is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: (num_tau_prime_samples x batch_size) x 1.
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = tf.tile(gamma_with_terminal[:, None],
                                  [self.num_tau_prime_samples, 1])

    # Get the indices of the maximum Q-value across the action dimension.
    # Shape of replay_next_qt_argmax: (num_tau_prime_samples x batch_size) x 1.

    replay_next_qt_argmax = tf.tile(
        self._replay_next_qt_argmax[:, None], [self.num_tau_prime_samples, 1])

    # Shape of batch_indices: (num_tau_prime_samples x batch_size) x 1.
    batch_indices = tf.cast(tf.range(
        self.num_tau_prime_samples * batch_size)[:, None], tf.int64)

    # Shape of batch_indexed_target_values:
    # (num_tau_prime_samples x batch_size) x 2.
    batch_indexed_target_values = tf.concat(
        [batch_indices, replay_next_qt_argmax], axis=1)

    # Shape of next_target_values: (num_tau_prime_samples x batch_size) x 1.
    target_quantile_values = tf.gather_nd(
        self._replay_net_target_quantile_values,
        batch_indexed_target_values)[:, None]

    return rewards + gamma_with_terminal * target_quantile_values

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(
          self._last_observation, self.action, reward, False,
          replay_buffer.PLACEHOLDER_RETURN_VALUE, self._curr_episode)
      self._train_step()

    self.action = self._select_action()
    return self.action

  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    if not self.eval_mode:
      self._store_transition(
          self._observation, self.action, reward, True,
          replay_buffer.PLACEHOLDER_RETURN_VALUE, self._curr_episode)
      self._update_episodic_return(self._curr_episode)
      self._curr_episode += 1

  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sess.run(self._train_op)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1

  def _update_episodic_return(self, num_episode):
    """Calculates and sets the episodic return for the last episode completed.

    Executes a tf session and executes replay buffer ops in order to store the
    episodic returns.

    Args:
      num_episode: int, identifier of the episode to be updated.
    """
    self._replay.calculate_and_store_return(num_episode)

  def _store_transition(self, last_observation, action, reward, is_terminal,
                        episode_return, episode_num, priority=None):
    """Stores an experienced transition.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer:
      (last_observation, action, reward, is_terminal, return, episode_num).

    Pedantically speaking, this does not actually store an entire transition
    since the next state is recorded on the following time step.

    Args:
      last_observation: numpy array, last observation.
      action: int, the action taken.
      reward: float, the reward.
      is_terminal: bool, indicating if the current state is a terminal state.
      episode_return: float, sum of discounted rewards from the current state.
      episode_num: int, episode identifier.
      priority: float, priority of the transition.
    """
    if priority is None:
      priority = 1.

    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal,
                       episode_return, episode_num, priority)

  def unbundle(self, checkpoint_dir, iteration_number, bundle_dictionary):
    """Restores the agent from a checkpoint.

    Restores the agent's Python objects to those specified in bundle_dictionary,
    and restores the TensorFlow objects to those specified in the
    checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
      agent's state.

    Args:
      checkpoint_dir: str, path to the checkpoint saved by tf.Save.
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
    # Update current episode number via info from the buffer.
    self._curr_episode = self._replay.episode_num_last_completed + 1
    if bundle_dictionary is not None:
      for key in self.__dict__:
        if key in bundle_dictionary:
          self.__dict__[key] = bundle_dictionary[key]
    elif not self.allow_partial_reload:
      return False
    else:
      logging.warning("Unable to reload the agent's parameters!")
    # Restore the agent's TensorFlow graph.
    self._saver.restore(self._sess,
                        os.path.join(checkpoint_dir,
                                     'tf_ckpt-{}'.format(iteration_number)))
    return True
