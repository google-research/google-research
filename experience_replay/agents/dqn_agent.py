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

"""Elephant DQN agent with adjustable replay ratios."""


from dopamine.agents.dqn import dqn_agent

import gin
import tensorflow.compat.v1 as tf
from experience_replay.replay_memory import prioritized_replay_buffer


def statistics_summaries(name, var):
  """Attach additional statistical summaries to the variable."""
  var = tf.to_float(var)
  with tf.variable_scope(name):
    tf.summary.scalar('mean', tf.reduce_mean(var))
    tf.summary.scalar('stddev', tf.math.reduce_std(var))
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
  tf.summary.histogram(name, var)


@gin.configurable
class ElephantDQNAgent(dqn_agent.DQNAgent):
  """A compact implementation of an Elephant DQN agent."""

  def __init__(self,
               replay_scheme='uniform',
               oldest_policy_in_buffer=250000,
               **kwargs):
    """Initializes the agent and constructs the components of its graph."""
    self._replay_scheme = replay_scheme
    self._oldest_policy_in_buffer = oldest_policy_in_buffer

    dqn_agent.DQNAgent.__init__(self, **kwargs)
    tf.logging.info('\t replay_scheme: %s', replay_scheme)
    tf.logging.info('\t oldest_policy_in_buffer: %s', oldest_policy_in_buffer)

    # We maintain attributes to record online and target network updates which
    # is later used for non-integer logic.
    self._online_network_updates = 0
    self._target_network_updates = 0

    # pylint: disable=protected-access
    buffer_to_oldest_policy_ratio = (
        float(self._replay.memory._replay_capacity) /
        float(self._oldest_policy_in_buffer))
    # pylint: enable=protected-access

    # This ratio is used to adjust other attributes that are explicitly tied to
    # agent steps.  When designed, the Dopamine agents assumed that the replay
    # ratio remain fixed and therefore elements such as epsilon_decay_period
    # will not be set appropriately without adjustment.
    self._gin_param_multiplier = (
        buffer_to_oldest_policy_ratio / self.update_period)
    tf.logging.info('\t self._gin_param_multiplier: %f',
                    self._gin_param_multiplier)

    # Adjust agent attributes that are tied to the agent steps.
    self.update_period *= self._gin_param_multiplier
    self.target_update_period *= self._gin_param_multiplier
    self.epsilon_decay_period *= self._gin_param_multiplier

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A `WrappedPrioritizedReplayBuffer` object.

    Raises:
      ValueError: if given an invalid replay scheme.
    """
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    # Both replay schemes use the same data structure, but the 'uniform' scheme
    # sets all priorities to the same value (which yields uniform sampling).

    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype,
        replay_forgetting='default',
        sample_newest_immediately=False)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        reduction_indices=1,
        name='replay_chosen_q')

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)

    if self._replay_scheme == 'prioritized':
      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
      # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
      # a fixed exponent actually performs better, except on Pong.
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.math.pow(probs + 1e-10, 0.5)
      loss_weights /= tf.reduce_max(loss_weights)

      # Rainbow and prioritized replay are parametrized by an exponent alpha,
      # but in both cases it is set to 0.5 - for simplicity's sake we leave it
      # as is here, using the more direct tf.sqrt(). Taking the square root
      # "makes sense", as we are dealing with a squared loss.
      # Add a small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will cause
      # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, tf.math.pow(loss + 1e-10, 0.5))

      # Weight the loss by the inverse priorities.
      loss = loss_weights * loss
    else:
      update_priorities_op = tf.no_op()

    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar('HuberLoss', tf.reduce_mean(loss))
    with tf.control_dependencies([update_priorities_op]):
      # Schaul et al. reports a slightly different rule, where 1/N is also
      # exponentiated by beta. Not doing so seems more reasonable, and did not
      # impact performance in our experiments.
      return self.optimizer.minimize(tf.reduce_mean(loss))

  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train_op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    # We maintain training_steps as a measure of genuine training steps, not
    # tied to environment interactions. This is used to control the online and
    # target network updates.
    if self._replay.memory.add_count > self.min_replay_history:
      while self._online_network_updates * self.update_period < self.training_steps:
        self._sess.run(self._train_op)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, self.training_steps)
        self._online_network_updates += 1

      while self._target_network_updates * self.target_update_period < self.training_steps:
        self._sess.run(self._sync_qt_ops)
        self._target_network_updates += 1

    self.training_steps += 1

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    """Stores a transition when in training mode.

    Executes a tf session and executes replay buffer ops in order to store the
    following tuple in the replay buffer (last_observation, action, reward,
    is_terminal, priority).

    Args:
      last_observation: Last observation, type determined via observation_type
        parameter in the replay_memory constructor.
      action: An integer, the action taken.
      reward: A float, the reward.
      is_terminal: Boolean indicating if the current state is a terminal state.
      priority: Float. Priority of sampling the transition. If None, the default
        priority will be used. If replay scheme is uniform, the default priority
        is 1. If the replay scheme is prioritized, the default priority is the
        maximum ever seen [Schaul et al., 2015].
    """
    if priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.0
      else:
        priority = self._replay.memory.sum_tree.max_recorded_priority

    if not self.eval_mode:
      self._replay.add(last_observation,
                       action,
                       reward,
                       is_terminal,
                       priority)

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
    bundle_dictionary = super(ElephantDQNAgent, self).bundle_and_checkpoint(
        checkpoint_dir, iteration_number)
    bundle_dictionary['_online_network_updates'] = self._online_network_updates
    bundle_dictionary['_target_network_updates'] = self._target_network_updates
    return bundle_dictionary
