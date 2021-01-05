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

"""Elephant Rainbow agent with adjustable replay ratios."""

import collections


from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import legacy_networks

import gin
import numpy as np
import tensorflow.compat.v1 as tf

from experience_replay.replay_memory import prioritized_replay_buffer
from experience_replay.replay_memory.circular_replay_buffer import ReplayElement


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
class ElephantRainbowAgent(dqn_agent.DQNAgent):
  """A compact implementation of an Elephant Rainbow agent."""

  def __init__(self,
               sess,
               num_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=legacy_networks.rainbow_network,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               alpha_exponent=0.5,
               beta_exponent=0.5,
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=2500,
               replay_forgetting='default',
               sample_newest_immediately=False,
               oldest_policy_in_buffer=250000):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: function expecting three parameters:
        (num_actions, network_type, state). This function will return the
        network_type object containing the tensors output by the network.
        See dopamine.discrete_domains.legacy_networks.rainbow_network as
        an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      alpha_exponent: float, alpha hparam in prioritized experience replay.
      beta_exponent: float, beta hparam in prioritized experience replay.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
      replay_forgetting:  str, What strategy to employ for forgetting old
        trajectories.  One of ['default', 'elephant'].
      sample_newest_immediately: bool, when True, immediately trains on the
        newest transition instead of using the max_priority hack.
      oldest_policy_in_buffer: int, the number of gradient updates of the oldest
        policy that has added data to the replay buffer.
    """
    # We need this because some tools convert round floats into ints.
    vmax = float(vmax)
    self._num_atoms = num_atoms
    self._support = tf.linspace(-vmax, vmax, num_atoms)
    self._replay_scheme = replay_scheme
    self._alpha_exponent = alpha_exponent
    self._beta_exponent = beta_exponent
    self._replay_forgetting = replay_forgetting
    self._sample_newest_immediately = sample_newest_immediately
    self._oldest_policy_in_buffer = oldest_policy_in_buffer
    # TODO(b/110897128): Make agent optimizer attribute private.
    self.optimizer = optimizer

    dqn_agent.DQNAgent.__init__(
        self,
        sess=sess,
        num_actions=num_actions,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        gamma=gamma,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=self.optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)
    tf.logging.info('\t replay_scheme: %s', replay_scheme)
    tf.logging.info('\t alpha_exponent: %f', alpha_exponent)
    tf.logging.info('\t beta_exponent: %f', beta_exponent)
    tf.logging.info('\t replay_forgetting: %s', replay_forgetting)
    tf.logging.info('\t oldest_policy_in_buffer: %s', oldest_policy_in_buffer)
    self.episode_return = 0.0

    # We maintain attributes to record online and target network updates which
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
    self.update_period = self.update_period * self._gin_param_multiplier
    self.target_update_period = (
        self.target_update_period * self._gin_param_multiplier)
    self.epsilon_decay_period = int(self.epsilon_decay_period *
                                    self._gin_param_multiplier)

    if self._replay_scheme == 'prioritized':
      if self._replay_forgetting == 'elephant':
        raise NotImplementedError

  def _get_network_type(self):
    """Returns the type of the outputs of a value distribution network.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('c51_network',
                                  ['q_values', 'logits', 'probabilities'])

  def _network_template(self, state):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    return self.network(self.num_actions, self._num_atoms, self._support,
                        self._get_network_type(), state)

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
    extra_elements = [ReplayElement('return', (), np.float32)]

    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype,
        extra_storage_types=extra_elements,
        replay_forgetting=self._replay_forgetting,
        sample_newest_immediately=self._sample_newest_immediately)

  def _build_target_distribution(self):
    """Builds the C51 target distribution as per Bellemare et al. (2017).

    First, we compute the support of the Bellman target, r + gamma Z'. Where Z'
    is the support of the next state distribution:

      * Evenly spaced in [-vmax, vmax] if the current state is nonterminal;
      * 0 otherwise (duplicated num_atoms times).

    Second, we compute the next-state probabilities, corresponding to the action
    with highest expected value.

    Finally we project the Bellman target (support + probabilities) onto the
    original support.

    Returns:
      target_distribution: tf.tensor, the target distribution from the replay.
    """
    batch_size = self._replay.batch_size

    # size of rewards: batch_size x 1
    rewards = self._replay.rewards[:, None]

    # size of tiled_support: batch_size x num_atoms
    tiled_support = tf.tile(self._support, [batch_size])
    tiled_support = tf.reshape(tiled_support, [batch_size, self._num_atoms])

    # size of target_support: batch_size x num_atoms

    is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
    # Incorporate terminal state to discount factor.
    # size of gamma_with_terminal: batch_size x 1
    gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier
    gamma_with_terminal = gamma_with_terminal[:, None]

    target_support = rewards + gamma_with_terminal * tiled_support

    # size of next_qt_argmax: 1 x batch_size
    next_qt_argmax = tf.argmax(
        self._replay_next_target_net_outputs.q_values, axis=1)[:, None]
    batch_indices = tf.range(tf.to_int64(batch_size))[:, None]
    # size of next_qt_argmax: batch_size x 2
    batch_indexed_next_qt_argmax = tf.concat(
        [batch_indices, next_qt_argmax], axis=1)

    # size of next_probabilities: batch_size x num_atoms
    next_probabilities = tf.gather_nd(
        self._replay_next_target_net_outputs.probabilities,
        batch_indexed_next_qt_argmax)

    return rainbow_agent.project_distribution(target_support,
                                              next_probabilities, self._support)

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    target_distribution = tf.stop_gradient(
        self._build_target_distribution())

    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_net_outputs.logits)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = tf.gather_nd(self._replay_net_outputs.logits,
                                        reshaped_actions)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_distribution,
        logits=chosen_action_logits)

    # Record returns encountered in the sampled training batches.
    returns = self._replay.transition['return']
    statistics_summaries('returns', returns)
    train_counts = self._replay.transition['train_counts']
    statistics_summaries('train_counts', train_counts)
    steps_until_first_train = self._replay.transition['steps_until_first_train']
    statistics_summaries('steps_until_first_train', steps_until_first_train)
    age = self._replay.transition['age']
    statistics_summaries('age', age)

    if self._replay_scheme == 'prioritized':
      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
      # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
      # a fixed exponent actually performs better, except on Pong.
      probs = self._replay.transition['sampling_probabilities']
      beta = self._beta_exponent
      tf.summary.histogram('probs', probs)
      loss_weights = 1.0 / tf.math.pow(probs + 1e-10, beta)
      tf.summary.histogram('loss_weights', loss_weights)
      loss_weights /= tf.reduce_max(loss_weights)

      # Rainbow and prioritized replay are parametrized by an exponent alpha,
      # but in both cases it is set to 0.5 - for simplicity's sake we leave it
      # as is here, using the more direct tf.sqrt(). Taking the square root
      # "makes sense", as we are dealing with a squared loss.
      # Add a small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will cause
      # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      alpha = self._alpha_exponent
      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, tf.math.pow(loss + 1e-10, alpha))

      # Weight the loss by the inverse priorities.
      loss = loss_weights * loss
    else:
      update_priorities_op = tf.no_op()

    update_train_counts_op = self._replay.tf_update_train_counts(
        self._replay.indices)

    with tf.control_dependencies([update_priorities_op,
                                  update_train_counts_op]):
      if self.summary_writer is not None:
        with tf.variable_scope('Losses'):
          tf.summary.scalar('CrossEntropyLoss', tf.reduce_mean(loss))
      # Schaul et al. reports a slightly different rule, where 1/N is also
      # exponentiated by beta. Not doing so seems more reasonable, and did not
      # impact performance in our experiments.
      return self.optimizer.minimize(tf.reduce_mean(loss)), loss

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_state()
    self._reset_return()

    if self._replay_forgetting == 'elephant':
      self._replay.memory.sort_replay_buffer_trajectories()

    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self.action = self._select_action()
    return self.action

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
      self._update_return(reward)
      self._store_transition(self._last_observation,
                             self.action,
                             reward,
                             False,
                             self.episode_return)
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
      self._update_return(reward)
      self._store_transition(self._observation,
                             self.action,
                             reward,
                             True,
                             self.episode_return)

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
                        episode_return,
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
      episode_return: A float, the episode undiscounted return so far.
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

    # TODO(liamfedus): This storage mechanism is brittle depending on order.
    # The internal replay buffers should be added via **kwargs not *args.
    if not self.eval_mode:
      self._replay.add(last_observation,
                       action,
                       reward,
                       is_terminal,
                       episode_return,
                       priority)

  def _update_return(self, reward):
    """Updates the current context based on the reward."""
    if self.episode_return != self.episode_return + reward:
      self.episode_return += reward

  def _reset_return(self):
    """Reset the episode return."""
    self.episode_return = 0.0

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
    bundle_dictionary = super(ElephantRainbowAgent, self).bundle_and_checkpoint(
        checkpoint_dir, iteration_number)
    bundle_dictionary['_online_network_updates'] = self._online_network_updates
    bundle_dictionary['_target_network_updates'] = self._target_network_updates
    return bundle_dictionary
