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

"""Rainbow agent used for learning the bisimulation metric."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

from dopamine.agents.rainbow import rainbow_agent
import gin
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim


@gin.configurable
def atari_network(num_actions, num_atoms, support, network_type, state,
                  representation_layer=10):
  """The convolutional network used to compute agent's Q-value distributions.

  Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    representation_layer: int, the layer which will be used as the
      representation for computing the bisimulation distances. Defaults to
      a high value, which defaults to the penultimate layer.

  Returns:
    net: _network_type object containing the tensors output by the network.
  """
  weights_initializer = contrib_slim.variance_scaling_initializer(
      factor=1.0 / np.sqrt(3.0), mode='FAN_IN', uniform=True)

  curr_layer = 1
  net = tf.cast(state, tf.float32)
  net = tf.div(net, 255.)
  representation = None
  if representation_layer <= curr_layer:
    representation = contrib_slim.flatten(net)
  net = contrib_slim.conv2d(
      net,
      32, [8, 8],
      stride=4,
      weights_initializer=weights_initializer,
      trainable=False)
  curr_layer += 1
  if representation is None and representation_layer <= curr_layer:
    representation = contrib_slim.flatten(net)
  net = contrib_slim.conv2d(
      net,
      64, [4, 4],
      stride=2,
      weights_initializer=weights_initializer,
      trainable=False)
  curr_layer += 1
  if representation is None and representation_layer <= curr_layer:
    representation = contrib_slim.flatten(net)
  net = contrib_slim.conv2d(
      net,
      64, [3, 3],
      stride=1,
      weights_initializer=weights_initializer,
      trainable=False)
  net = contrib_slim.flatten(net)
  curr_layer += 1
  if representation is None and representation_layer <= curr_layer:
    representation = net
  net = contrib_slim.fully_connected(
      net, 512, weights_initializer=weights_initializer, trainable=False)
  curr_layer += 1
  if representation is None:
    representation = net
  net = contrib_slim.fully_connected(
      net,
      num_actions * num_atoms,
      activation_fn=None,
      weights_initializer=weights_initializer,
      trainable=False)

  logits = tf.reshape(net, [-1, num_actions, num_atoms])
  probabilities = contrib_layers.softmax(logits)
  q_values = tf.reduce_sum(support * probabilities, axis=2)
  return network_type(q_values, logits, probabilities, representation)


@gin.configurable
def bisimulation_network(states, hidden_dimension=512, num_layers=1,
                         trainable=True):
  """Creates the network for approximating the bisimulation distances.

  Args:
    states: Tensor, concatentation of two state representations.
    hidden_dimension: int, dimensionality of hidden_layers.
    num_layers: int, the number of layers to use for the approximant.
    trainable: bool, whether this network will be trainable.

  Returns:
    Network to approximate bisimulation metric.
  """
  net = tf.cast(states, tf.float32)
  net = contrib_slim.flatten(net)
  for _ in range(num_layers):
    net = contrib_slim.fully_connected(
        net, hidden_dimension, trainable=trainable)
  return contrib_slim.fully_connected(net, 1, trainable=trainable)


SequentialDistances = (
    collections.namedtuple('sequential_distances', ['bisimulation', 'value']))


@gin.configurable
class BisimulationRainbowAgent(rainbow_agent.RainbowAgent):
  """A subclass of Rainbow which learns the on-policy bisimulation metric."""

  def __init__(self,
               sess,
               num_actions,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.000075, epsilon=0.00015),
               bisim_horizon_discount=0.99,
               evaluate_metric_only=False,
               summary_writer=None):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: tf.Session, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      optimizer: `tf.train.Optimizer`, for training the bisimulation estimator.
      bisim_horizon_discount: float, amount by which to increase the horizon for
        estimating the distance.
      evaluate_metric_only: bool, if set, will evaluate the loaded metric
        approximant.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
    """
    self.bisim_optimizer = optimizer
    self.bisim_horizon_discount = bisim_horizon_discount
    self.bisim_horizon_discount_value = 1.0
    self.bisim_horizon = 0.0
    self.evaluate_metric_only = evaluate_metric_only
    self.start_recording = False
    super(BisimulationRainbowAgent, self).__init__(
        sess, num_actions, network=atari_network, summary_writer=summary_writer)
    self._source_state = np.copy(self.state)
    self._evaluation_steps = 0
    self.eval_distances = SequentialDistances([], [])

  def reload_checkpoint(self, checkpoint_path):
    """Reload variables from a fully specified checkpoint.

    Args:
      checkpoint_path: string, full path to a checkpoint to reload.
    """
    assert checkpoint_path
    global_vars = set([x.name for x in tf.global_variables()])
    ckpt_vars = [
        '{}:0'.format(name)
        for name, _ in tf.train.list_variables(checkpoint_path)
    ]
    # Only include non trainable variables which are also present in the
    # checkpoint to restore
    include_vars = list(global_vars.intersection(set(ckpt_vars)))
    variables_to_restore = contrib_slim.get_variables_to_restore(
        include=include_vars)
    if variables_to_restore:
      reloader = tf.train.Saver(var_list=variables_to_restore)
      reloader.restore(self._sess, checkpoint_path)
    tf.logging.info('Done restoring from %s!', checkpoint_path)

  def _get_network_type(self):
    """Returns the type of the outputs of a Q value network.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('c51_network',
                                  ['q_values', 'logits', 'probabilities',
                                   'representation'])

  def _concat_states(self, states, transpose=False):
    """Concatenate all pairs of states in a batch.

    Args:
      states: Tensor, batch of states from which we will concatenate
        batch_size^2 pairs of states.
      transpose: bool, whether to concatenate states in transpose order.

    Returns:
      A batch_size^2 Tensor containing the concatenation of all elements in
        `states`.
    """
    # tiled_states will have shape
    # [batch_size, batch_size, representation_dimension] and will be of the
    # following form (where \phi_1 is the representation of the state of the
    # first batch_element):
    # [ \phi_1 \phi_2 ... \phi_batch_size ]
    # [ \phi_1 \phi_2 ... \phi_batch_size ]
    # ...
    # [ \phi_1 \phi_2 ... \phi_batch_size ]
    tiled_states = tf.tile([states], [self.batch_size, 1, 1])
    # transpose_tiled_states will have shape
    # [batch_size, batch_size, representation_dimension] and will be of the
    # following form (where \phi_1 is the representation of the state of the
    # first batch_element):
    # [ \phi_1 \phi_1 ... \phi_1 ]
    # [ \phi_2 \phi_2 ... \phi_2 ]
    # ...
    # [ \phi_batch_size \phi_batch_size ... \phi_batch_size ]
    transpose_tiled_states = tf.keras.backend.repeat(states, self.batch_size)
    # concat_states will be a
    # [batch_size, batch_size, representation_dimension*2] matrix containing the
    # concatenation of all pairs of states in the batch.
    if transpose:
      concat_states = tf.concat([transpose_tiled_states, tiled_states], 2)
    else:
      concat_states = tf.concat([tiled_states, transpose_tiled_states], 2)
    # We return a reshaped matrix which results in a new batch of size
    # batch_size ** 2. Resulting matrix will have shape
    # [batch_size**2, representation_dimension].
    representation_dimension = tf.shape(states)[1]
    return tf.reshape(concat_states,
                      (self.batch_size**2, representation_dimension * 2))

  def _build_bisimulation_target(self):
    """Build the bisimulation target."""
    r1 = tf.tile([self._replay.rewards], [self.batch_size, 1])
    r2 = tf.transpose(r1)
    reward_differences = tf.abs(r1 - r2)
    reward_differences = tf.reshape(reward_differences, (self.batch_size**2, 1))
    if self.summary_writer is not None:
      mean_reward_diff = tf.reduce_mean(reward_differences)
      tf.summary.scalar('Training/AverageRewardDiff', mean_reward_diff)
    self.next_state_distances = self.bisim_horizon_ph * self.s2_target_distances
    return reward_differences + self.gamma * self.next_state_distances

  def _build_train_op(self):
    return tf.no_op()

  def _sync_qt_ops(self):
    return tf.no_op()

  def _network_template(self, state):
    return self.network(self.num_actions, self._num_atoms, self._support,
                        self._get_network_type(), state)

  def _create_network(self, name):
    return tf.make_template('Online', self._network_template)

  def _build_networks(self):
    super(BisimulationRainbowAgent, self)._build_networks()
    self._build_all_bisimulation_parts()

  def _build_all_bisimulation_parts(self):
    """Builds the bisimulation networks and ops."""
    self.batch_size = tf.shape(self._replay.rewards)[0]
    self._replay_target_outputs = self.target_convnet(self._replay.states)
    self.bisim_horizon_ph = tf.placeholder(tf.float32, ())
    self.online_bisimulation = tf.make_template('OnlineBisim',
                                                bisimulation_network)
    self.target_bisimulation = tf.make_template('TargetBisim',
                                                bisimulation_network,
                                                trainable=False)
    # For evaluating the metric from an episode's first state.
    self.source_state_ph = tf.placeholder(self.observation_dtype,
                                          self.state_ph.shape,
                                          name='source_state_ph')
    self._initial_state_net = self.online_convnet(self.source_state_ph)
    concat_states = tf.concat(
        [self._initial_state_net.representation,
         self._net_outputs.representation], 1)
    self.state_distances = tf.squeeze(self.online_bisimulation(concat_states))
    self.state_value = tf.reduce_max(self._net_outputs.q_values, axis=1)[0]
    if self.summary_writer is not None:
      tf.summary.scalar('Eval/StateDistances', self.state_distances)
    if self.evaluate_metric_only:
      return

    self.s1_online_distances = self.online_bisimulation(
        self._concat_states(self._replay_net_outputs.representation))
    self.s2_target_distances = self.target_bisimulation(
        self._concat_states(
            self._replay_next_target_net_outputs.representation))
    # bisimulation_target = rew_diff + gamma * next_distance.
    bisimulation_target = tf.stop_gradient(self._build_bisimulation_target())
    # We zero-out diagonal entries, since those are estimating the distance
    # between a state and itself, which we know to be 0.
    diagonal_mask = 1.0 - tf.diag(tf.ones(self.batch_size, dtype=tf.float32))
    diagonal_mask = tf.reshape(diagonal_mask, (self.batch_size**2, 1))
    bisimulation_target *= diagonal_mask
    bisimulation_estimate = self.s1_online_distances
    bisimulation_loss = tf.losses.mean_squared_error(
        bisimulation_target,
        bisimulation_estimate)
    if self.summary_writer is not None:
      average_distance = tf.reduce_mean(bisimulation_estimate)
      average_target = tf.reduce_mean(bisimulation_target)
      average_next_state_dists = tf.reduce_mean(self.next_state_distances)
      tf.summary.scalar('Training/loss', bisimulation_loss)
      tf.summary.scalar('Training/AverageDistance', average_distance)
      tf.summary.scalar('Training/AverageTargetDistance', average_target)
      tf.summary.scalar('Training/AverageNextStateDistance',
                        average_next_state_dists)
      tf.summary.scalar('Training/BisimHorizon', self.bisim_horizon_ph)
      tf.summary.histogram('Training/OnlineDistance', bisimulation_estimate)
      tf.summary.histogram('Training/TargetDistance', bisimulation_target)
    self._train_bisim_op = self.bisim_optimizer.minimize(bisimulation_loss)
    self._bisim_sync_op = self._build_sync_op(online_scope='OnlineBisim',
                                              target_scope='TargetBisim')

  def _build_sync_op(self, online_scope='Online', target_scope='Target'):
    # Get trainable variables from online and target Rainbow
    sync_qt_ops = []
    trainables_online = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=online_scope)
    trainables_target = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope)
    for (w_online, w_target) in zip(trainables_online, trainables_target):
      # Assign weights from online to target network.
      sync_qt_ops.append(w_target.assign(w_online, use_locking=True))
    return sync_qt_ops

  def _train_bisimulation(self):
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sess.run(self._train_bisim_op,
                       {self.bisim_horizon_ph: self.bisim_horizon})
        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries,
                                   {self.source_state_ph: self._source_state,
                                    self.state_ph: self.state,
                                    self.bisim_horizon_ph: self.bisim_horizon})
          self.summary_writer.add_summary(summary, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._bisim_sync_op)
        self.bisim_horizon = 1.0 - self.bisim_horizon_discount_value
        self.bisim_horizon_discount_value *= self.bisim_horizon_discount

    self.training_steps += 1

  def _evaluate_bisimulation(self):
    if self.start_recording:
      current_distance = self._sess.run(
          self.state_distances,
          {self.source_state_ph: self._source_state,
           self.state_ph: self.state,
           self.bisim_horizon_ph: self.bisim_horizon})
      self.eval_distances.bisimulation.append(current_distance)
      source_v = self._sess.run(self.state_value, {self.state_ph: self.state})
      target_v = self._sess.run(self.state_value,
                                {self.state_ph: self._source_state})
      self.eval_distances.value.append(abs(source_v - target_v))
    if self.evaluate_metric_only and self.summary_writer is not None:
      summary = self._sess.run(self._merged_summaries,
                               {self.source_state_ph: self._source_state,
                                self.state_ph: self.state,
                                self.bisim_horizon_ph: self.bisim_horizon})
      self.summary_writer.add_summary(summary, self._evaluation_steps)
    self._evaluation_steps += 1

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    action = super(BisimulationRainbowAgent, self).begin_episode(observation)
    if not self.evaluate_metric_only:
      self._train_bisimulation()
    self._evaluate_bisimulation()
    return action

  def step(self, reward, observation, set_source_state=False):
    """Records the most recent transition and returns the agent's next action.

    We store the observation of the last time step since we want to store it
    with the reward.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.
      set_source_state: bool, whether to set the current state as the source
        state.

    Returns:
      int, the selected action.
    """
    _ = super(BisimulationRainbowAgent, self).step(reward, observation)
    self._store_transition(self._last_observation, self.action, reward, False)
    if set_source_state and not self.start_recording:
      # We only want to set the source state once.
      self.start_recording = True
      self._source_state = np.copy(self.state)
    if not self.evaluate_metric_only:
      self._train_bisimulation()
    self._evaluate_bisimulation()
    return self.action

  def end_episode(self, reward):
    """Signals the end of the episode to the agent.

    We store the observation of the current time step, which is the last
    observation of the episode.

    Args:
      reward: float, the last reward from the environment.
    """
    self._store_transition(self._observation, self.action, reward, True)

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    if priority is None:
      if self._replay_scheme == 'uniform':
        priority = 1.
      else:
        priority = self._replay.memory.sum_tree.max_recorded_priority
    self._replay.add(last_observation, action, reward, is_terminal, priority)

  def get_distances(self):
    return self.eval_distances
