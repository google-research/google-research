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

"""Compact implementation of a DQN agent with Advantage Learning (AL-DQN).

Advantage Learning is an action-gap increasing algorithm
(Baird, 1999; Bellemare et al., 2016).
The AL-DQN agent follows the description given in
"Self-Imitation Advantage Learning" (Ferret et al., 2020).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
from absl import logging
from dopamine.agents.dqn import dqn_agent
import gin
import tensorflow.compat.v2 as tf

from sail_rl.common import replay_buffer

FLAGS = flags.FLAGS


@gin.configurable
class ALDQNAgent(dqn_agent.DQNAgent):
  """An implementation of the AL-DQN agent."""

  def __init__(self,
               sess,
               num_actions,
               alpha,
               clip,
               tf_device='/cpu:*',
               **kwargs):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.compat.v1.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      alpha: float, value of the alpha parameter of AL.
      clip: float, if v > 0, (G_t - Q_target(s, a)) is clipped to (-v, v).
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      **kwargs: DQNAgent optional params, see the constructor of DQNAgent.
    """
    self._alpha = alpha
    self._clip = clip
    self._curr_episode = 0

    super(ALDQNAgent, self).__init__(
        sess, num_actions, tf_device=tf_device, **kwargs)

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph)
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_net_outputs = self.online_convnet(
        self._replay.next_states)
    self._replay_target_net_outputs = self.target_convnet(
        self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A SAILWrapperReplayBuffer object.
    """
    return replay_buffer.SAILWrappedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_target_q_op(self):
    """Build an op used as a target for the Q-value.

    Returns:
      target_q_op: An op calculating the Q-value.
    """
    # Get the maximum Q-value across the actions dimension.
    replay_next_qt_max = tf.reduce_max(
        self._replay_next_target_net_outputs.q_values, 1)

    # Calculate the Bellman target value.
    #   Q_t = R_t + \gamma^N * Q'_t+1
    # where,
    #   Q'_t+1 = \argmax_a Q(S_t+1, a)
    #          (or) 0 if S_t is a terminal state,
    # and
    #   N is the update horizon (by default, N=1).
    # Here, R_t is augmented for AL:
    #   R_t = R_t + Q_target(s_t, a_t) - max_a Q_target(s_t, a),
    replay_target_q = tf.reduce_max(
        self._replay_target_net_outputs.q_values,
        axis=1,
        name='replay_max_target_q')

    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_target_q_al = tf.reduce_sum(
        self._replay_target_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_target_q_al')

    if self._clip > 0.:
      al_bonus = self._alpha * tf.clip_by_value(
          tf.nn.relu(replay_target_q_al - replay_target_q),
          0., self._clip)
    else:
      al_bonus = self._alpha * tf.nn.relu(
          replay_target_q_al - replay_target_q)

    rewards = self._replay.rewards + al_bonus

    update_target = rewards + self.cumulative_gamma * replay_next_qt_max * (
        1. - tf.cast(self._replay.terminals, tf.float32))

    return (al_bonus, update_target)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    _, target = self._build_target_q_op()
    target = tf.stop_gradient(target)

    loss = tf.compat.v1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)

    if self.summary_writer is not None:
      with tf.compat.v1.variable_scope('Losses'):
        tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))

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
                        episode_return, episode_num):
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
    """
    self._replay.add(last_observation, action, reward, is_terminal,
                     episode_return, episode_num)

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
