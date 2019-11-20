# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Class for computing bisimulation metrics on deterministic grid worlds.

The grid worlds created will have the form:
  *****
  *  g*
  *   *
  *****
where a reward of `self.reward_value` (set to `1.`) is received upon
entering the cell marked with 'g', and a reward of `-self.reward_value` is
received upon taking an action that would drive the agent to a wall.

One can also specify a deterministic policy by using '^', 'v', '<', and '>'
characters instead of spaces. The 'g' cell will default to the down action.

The class supports:
- Reading a grid specification from a file or creating a square room from a
  specified wall length.
- Computing the exact bisimulation metric up to a desired tolerance using the
  standard dynamic programming method.
- Computing the exact bisimulation metric using sampled pairs of trajectories.
- Computing the approximate bisimulation distance using the bisimulation loss
  function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import gin.tf


@gin.configurable
class GridWorld(object):
  """Class defining deterministic grid world MDPs."""

  def __init__(self,
               base_dir,
               wall_length=2,
               grid_file=None,
               gamma=0.99,
               representation_dimension=64,
               batch_size=64,
               target_update_period=100,
               num_iterations=10000,
               starting_learning_rate=0.01,
               use_decayed_learning_rate=False,
               learning_rate_decay=0.96,
               epsilon=1e-8,
               staircase=False,
               add_noise=True,
               bisim_horizon_discount=0.9,
               double_period_halfway=True,
               total_final_samples=1000,
               debug=False):
    """Initialize a deterministic GridWorld from file.

    Args:
      base_dir: str, directory where to store exact metric and event files.
      wall_length: int, length of walls for constructing a 1-room MDP. Ignored
        if grid_file is not None.
      grid_file: str, path to file defining GridWorld MDP.
      gamma: float, discount factor.
      representation_dimension: int, dimension of each state representation.
      batch_size: int, size of sample batches for the learned metric.
      target_update_period: int, period at which target network weights are
        synced from the online network.
      num_iterations: int, number of iterations to run learning procedure.
      starting_learning_rate: float, starting learning rate for AdamOptimizer.
      use_decayed_learning_rate: bool, whether to use decayed learning rate.
      learning_rate_decay: float, amount by which to decay learning rate.
      epsilon: float, epsilon for AdamOptimizer.
      staircase: bool, whether to decay learning rate at discrete intervals.
      add_noise: bool, whether to add noise to grid points (thereby making
        them continuous) when learning the metric.
      bisim_horizon_discount: float, amount by which to increase the horizon for
        estimating the distance.
      double_period_halfway: bool, whether to double the update period halfway
        through training.
      total_final_samples: int, number of samples to draw at the end of training
        (if noise is on).
      debug: bool, whether we are in debug mode.
    """
    self.base_dir = base_dir
    if not tf.gfile.Exists(self.base_dir):
      tf.gfile.MakeDirs(self.base_dir)
    self.exact_bisim_filename = 'exact_bisim_metric.pkl'
    self.wall_length = wall_length
    self.goal_reward = 1.
    self.wall_penalty = -1.
    self.gamma = gamma
    self.representation_dimension = representation_dimension
    self.batch_size = batch_size
    self.target_update_period = target_update_period
    self.num_iterations = num_iterations
    self.starting_learning_rate = starting_learning_rate
    self.use_decayed_learning_rate = use_decayed_learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.epsilon = epsilon
    self.staircase = staircase
    self.add_noise = add_noise
    self.double_period_halfway = double_period_halfway
    self.bisim_horizon_discount = bisim_horizon_discount
    self.total_final_samples = total_final_samples
    self.debug = debug
    self.raw_grid = []
    # Assume no policy by default. If there is a policy, we will compute
    # the on-policy bisimulation metric.
    self.has_policy = False
    # Either read in from file or build a room using wall_length.
    if grid_file is not None:
      tf.logging.info('Will try to open: {}'.format(grid_file))
      assert tf.gfile.Exists(grid_file)
      with tf.gfile.Open(grid_file) as f:
        for l in f:
          self.raw_grid.append(list(l)[:-1])
          # If we see a policy character we assume there is a policy.
          if (not self.has_policy and
              ('<' in l or '>' in l or '^' in l or 'v' in l)):
            self.has_policy = True
      self.raw_grid = np.array(self.raw_grid)
    else:
      self.raw_grid.append(['*'] * (wall_length + 2))
      for row in range(wall_length):
        self.raw_grid.append(['*'] + [' '] * wall_length + ['*'])
      self.raw_grid.append(['*'] * (wall_length + 2))
      self.raw_grid = np.array(self.raw_grid)
      self.raw_grid[1, wall_length] = 'g'
    # First make walls 0s and cells 1s.
    self.indexed_states = [
        [0 if x == '*' else 1 for x in y] for y in self.raw_grid]
    # Now do a cumsum to get unique IDs for each cell.
    self.indexed_states = (
        np.reshape(np.cumsum(self.indexed_states),
                   np.shape(self.indexed_states)) * self.indexed_states)
    # Subtract 1 so we're 0-indexed, and walls are now -1.
    self.indexed_states -= 1
    self.num_rows = np.shape(self.indexed_states)[0]
    self.num_cols = np.shape(self.indexed_states)[1]
    # The maximum value on any dimension, used for normalizing the inputs.
    self.max_val = max(self.num_rows - 2., self.num_cols - 2.)
    self.num_states = np.max(self.indexed_states) + 1
    # State-action rewards.
    self.rewards = {}
    # Set up the next state transitions.
    self.actions = ['up', 'down', 'left', 'right']
    # Policy to action mapping.
    policy_to_action = {'^': 'up',
                        'v': 'down',
                        '<': 'left',
                        '>': 'right',
                        'g': 'down'}  # Assume down from goal state.
    # Deltas for 4 actions.
    self.action_deltas = {'up': (-1, 0),
                          'down': (1, 0),
                          'left': (0, -1),
                          'right': (0, 1)}
    self.next_states_grid = {}
    self.next_states = {}
    self.inverse_index_states = np.zeros((self.num_states, 2))
    self.policy = [''] * self.num_states if self.has_policy else None
    # So we only update inverse_index_states once.
    first_pass = True
    # Set up the transition and reward dynamics.
    for action in self.actions:
      self.next_states_grid[action] = np.copy(self.indexed_states)
      self.next_states[action] = np.arange(self.num_states)
      self.rewards[action] = np.zeros(self.num_states)
      for row in range(self.num_rows):
        for col in range(self.num_cols):
          state = self.indexed_states[row, col]
          if self.has_policy and state >= 0:
            self.policy[state] = policy_to_action[self.raw_grid[row, col]]
          if first_pass and state >= 0:
            self.inverse_index_states[state] = [row, col]
          next_row = row + self.action_deltas[action][0]
          next_col = col + self.action_deltas[action][1]
          if (next_row < 1 or next_row > self.num_rows or
              next_col < 1 or next_col > self.num_cols or
              state < 0 or self.indexed_states[next_row, next_col] < 0):
            if state >= 0:
              self.rewards[action][state] = self.wall_penalty
            continue
          if self.raw_grid[next_row, next_col] == 'g':
            self.rewards[action][state] = self.goal_reward
          else:
            self.rewards[action][state] = 0.
          next_state = self.indexed_states[next_row, next_col]
          self.next_states_grid[action][row, col] = next_state
          self.next_states[action][state] = next_state
      first_pass = False
    # Initial metric is undefined.
    self.bisim_metric = None
    # Dictionary for collecting statistics.
    self.statistics = {}

  def compute_exact_metric(self, tolerance=0.001, verbose=False):
    """Compute the exact bisimulation metric up to the specified tolerance.

    Args:
      tolerance: float, maximum difference in metric estimate between successive
        iterations. Once this threshold is past, computation stops.
      verbose: bool, whether to print verbose messages.
    """
    # If we've saved the exact metric to file load it.
    bisim_path = os.path.join(self.base_dir, self.exact_bisim_filename)
    if tf.gfile.Exists(bisim_path):
      with tf.gfile.GFile(bisim_path, 'rb') as f:
        self.bisim_metric = pickle.load(f)
      print('Successfully reloaded exact bisimulation metric from file.')
      return
    # Initial metric is all zeros.
    self.bisim_metric = np.zeros((self.num_states, self.num_states))
    metric_difference = tolerance * 2.
    i = 1
    exact_metric_differences = []
    start_time = time.time()
    while metric_difference > tolerance:
      new_metric = np.zeros((self.num_states, self.num_states))
      for s1 in range(self.num_states):
        for s2 in range(self.num_states):
          if self.has_policy:
            action1 = self.policy[s1]
            action2 = self.policy[s2]
            next_state1 = self.next_states[action1][s1]
            next_state2 = self.next_states[action2][s2]
            rew1 = self.rewards[action1][s1]
            rew2 = self.rewards[action2][s2]
            new_metric[s1, s2] = (
                abs(rew1 - rew2) +
                self.gamma * self.bisim_metric[next_state1, next_state2])
          else:
            for action in self.actions:
              next_state1 = self.next_states[action][s1]
              next_state2 = self.next_states[action][s2]
              rew1 = self.rewards[action][s1]
              rew2 = self.rewards[action][s2]
              act_distance = (
                  abs(rew1 - rew2) +
                  self.gamma * self.bisim_metric[next_state1, next_state2])
              if act_distance > new_metric[s1, s2]:
                new_metric[s1, s2] = act_distance
      metric_difference = np.max(abs(new_metric - self.bisim_metric))
      exact_metric_differences.append(metric_difference)
      if i % 1000 == 0 and verbose:
        print('Iteration {}: {}'.format(i, metric_difference))
      self.bisim_metric = np.copy(new_metric)
      i += 1
    total_time = time.time() - start_time
    exact_statistics = {
        'tolerance': tolerance,
        'time': total_time,
        'num_iterations': i,
        'metric_differences': exact_metric_differences,
        'metric': self.bisim_metric,
    }
    self.statistics['exact'] = exact_statistics
    print('**** Exact statistics ***')
    print('Number of states: {}'.format(self.num_states))
    print('Total number of iterations: {}'.format(i))
    print('Total time: {}'.format(total_time))
    print('*************************')
    if verbose:
      self.pretty_print_metric()
    # Save the metric to file so we save this step next time.
    with tf.gfile.GFile(bisim_path, 'w') as f:
      pickle.dump(self.bisim_metric, f)

  def compute_sampled_metric(self, tolerance=0.001, verbose=False):
    """Use trajectory sampling to compute the exact bisimulation metric.

    Will compute until the difference with the exact metric is within tolerance.
    Will try for a maximum of self.num_iterations.

    Args:
      tolerance: float, required accuracy before stopping.
      verbose: bool, whether to print verbose messages.
    """
    self.sampled_bisim_metric = np.zeros((self.num_states, self.num_states))
    exact_metric_errors = []
    metric_differences = []
    start_time = time.time()
    exact_metric_error = tolerance * 10
    i = 0
    while exact_metric_error > tolerance:
      new_metric = np.zeros((self.num_states, self.num_states))
      # Generate a pair of sampled trajectories.
      s1 = np.random.randint(self.num_states)
      s2 = np.random.randint(self.num_states)
      if self.has_policy:
        action1 = self.policy[s1]
        action2 = self.policy[s2]
      else:
        action1 = self.actions[np.random.randint(4)]
        action2 = action1
      next_s1 = self.next_states[action1][s1]
      next_s2 = self.next_states[action2][s2]
      rew1 = self.rewards[action1][s1]
      rew2 = self.rewards[action2][s2]
      if self.has_policy:
        new_metric[s1, s2] = (
            abs(rew1 - rew2) +
            self.gamma * self.sampled_bisim_metric[next_s1, next_s2])
      else:
        new_metric[s1, s2] = max(
            self.sampled_bisim_metric[s1, s2],
            abs(rew1 - rew2) +
            self.gamma * self.sampled_bisim_metric[next_s1, next_s2])
      metric_difference = np.max(
          abs(new_metric - self.sampled_bisim_metric))
      metric_differences.append(metric_difference)
      exact_metric_error = np.max(
          abs(self.bisim_metric - self.sampled_bisim_metric))
      exact_metric_errors.append(exact_metric_error)
      if i % 10000 == 0 and verbose:
        print('Iteration {}: {}'.format(i, metric_difference))
      self.sampled_bisim_metric = np.copy(new_metric)
      i += 1
      if i > self.num_iterations:
        break
    total_time = time.time() - start_time
    exact_sampling_statistics = {
        'time': total_time,
        'tolerance': tolerance,
        'num_iterations': i,
        'sampled_metric_differences': metric_differences,
        'exact_metric_errors': exact_metric_errors,
    }
    self.statistics['exact_sampling'] = exact_sampling_statistics
    print('**** Exact sampled statistics ***')
    print('Number of states: {}'.format(self.num_states))
    print('Total number of iterations: {}'.format(i))
    print('Total time: {}'.format(total_time))
    print('*************************')
    if verbose:
      self.pretty_print_metric()

  def _network_template(self, states):
    """Create the network to approximate the bisimulation metric.

    Args:
      states: Tensor, concatenation of two state representations.

    Returns:
      Network to approximate bisimulation metric.
    """
    net = tf.cast(states, tf.float64)
    net = slim.flatten(net)
    # Normalize and rescale in range [-1, 1].
    net /= self.max_val
    net = 2.0 * net - 1.0
    net = slim.fully_connected(net, self.representation_dimension)
    return slim.fully_connected(net, 1)

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
    batch_size = tf.shape(states)[0]
    tiled_states = tf.tile([states], [batch_size, 1, 1])
    # transpose_tiled_states will have shape
    # [batch_size, batch_size, representation_dimension] and will be of the
    # following form (where \phi_1 is the representation of the state of the
    # first batch_element):
    # [ \phi_1 \phi_1 ... \phi_1 ]
    # [ \phi_2 \phi_2 ... \phi_2 ]
    # ...
    # [ \phi_batch_size \phi_batch_size ... \phi_batch_size ]
    transpose_tiled_states = tf.keras.backend.repeat(states, batch_size)
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
    return tf.reshape(concat_states, (batch_size**2, 4))

  def _build_bisimulation_target(self):
    """Build the bisimulation target."""
    batch_size = tf.shape(self.rewards_ph)[0]
    r1 = tf.tile([self.rewards_ph], [batch_size, 1])
    r2 = tf.transpose(r1)
    reward_differences = tf.abs(r1 - r2)
    reward_differences = tf.reshape(reward_differences, (batch_size**2, 1))
    next_state_distances = self.bisim_horizon_ph * self.s2_target_distances
    return reward_differences + self.gamma * next_state_distances

  def _build_train_op(self, optimizer):
    """Build the TensorFlow graph used to learn the bisimulation metric.

    Args:
      optimizer: a tf.train optimizer.
    Returns:
      A TensorFlow op to minimize the bisimulation loss.
    """
    self.online_network = tf.make_template('Online',
                                           self._network_template)
    self.target_network = tf.make_template('Target',
                                           self._network_template)
    self.s1_ph = tf.placeholder(tf.float64, (self.batch_size, 2),
                                name='s1_ph')
    self.s2_ph = tf.placeholder(tf.float64, (self.batch_size, 2),
                                name='s2_ph')
    self.s1_online_distances = self.online_network(
        self._concat_states(self.s1_ph))
    self.s1_target_distances = self.target_network(
        self._concat_states(self.s1_ph))
    self.s2_target_distances = self.target_network(
        self._concat_states(self.s2_ph))
    self.action_ph = tf.placeholder(tf.int32, (self.batch_size,))
    self.rewards_ph = tf.placeholder(tf.float64, (self.batch_size,))
    # We use an expanding horizon for computing the distances.
    self.bisim_horizon_ph = tf.placeholder(tf.float64, ())
    # bisimulation_target_1 = rew_diff + gamma * next_distance.
    bisimulation_target_1 = tf.stop_gradient(self._build_bisimulation_target())
    # bisimulation_target_2 = curr_distance.
    bisimulation_target_2 = tf.stop_gradient(self.s1_target_distances)
    # We slowly taper in the maximum according to the bisim horizon.
    bisimulation_target = tf.maximum(
        bisimulation_target_1, bisimulation_target_2 * self.bisim_horizon_ph)
    # We zero-out diagonal entries, since those are estimating the distance
    # between a state and itself, which we know to be 0.
    diagonal_mask = 1.0 - tf.diag(tf.ones(self.batch_size, dtype=tf.float64))
    diagonal_mask = tf.reshape(diagonal_mask, (self.batch_size**2, 1))
    bisimulation_target *= diagonal_mask
    bisimulation_estimate = self.s1_online_distances
    # We start with a mask that includes everything.
    loss_mask = tf.ones(tf.shape(bisimulation_estimate))
    # We have to enforce that states being compared are done only using the same
    # action.
    indicators = self.action_ph
    indicators = tf.cast(indicators, tf.float64)
    # indicators will initially have shape [batch_size], we first tile it:
    square_ids = tf.tile([indicators], [self.batch_size, 1])
    # We subtract square_ids from its transpose:
    square_ids = square_ids - tf.transpose(square_ids)
    # At this point all zero-entries are the ones with equal IDs.
    # Now we would like to convert the zeros in this matrix to 1s, and make
    # everything else a 0. We can do this with the following operation:
    loss_mask = 1 - tf.abs(tf.sign(square_ids))
    # Now reshape to match the shapes of the estimate and target.
    loss_mask = tf.reshape(loss_mask, (self.batch_size**2, 1))
    larger_targets = bisimulation_target - bisimulation_estimate
    larger_targets_count = tf.reduce_sum(
        tf.cast(larger_targets > 0., tf.float64))
    tf.summary.scalar('Learning/LargerTargets', larger_targets_count)
    tf.summary.scalar('Learning/NumUpdates', tf.count_nonzero(loss_mask))
    tf.summary.scalar('Learning/BisimHorizon', self.bisim_horizon_ph)
    bisimulation_loss = tf.losses.mean_squared_error(
        bisimulation_target,
        bisimulation_estimate,
        weights=loss_mask)
    tf.summary.scalar('Learning/loss', bisimulation_loss)
    # Plot average distance between sampled representations.
    average_distance = tf.reduce_mean(bisimulation_estimate)
    tf.summary.scalar('Approx/AverageDistance', average_distance)
    return optimizer.minimize(bisimulation_loss)

  def _build_sync_op(self):
    """Build the sync op."""
    sync_count = tf.Variable(0, trainable=False)
    sync_ops = [tf.assign_add(sync_count, 1)]
    trainables_online = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='Online')
    trainables_target = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='Target')
    for (w_online, w_target) in zip(trainables_online, trainables_target):
      sync_ops.append(w_target.assign(w_online, use_locking=True))
    tf.summary.scalar('Learning/SyncCount', sync_count)
    return sync_ops

  def _build_eval_metric(self):
    """Build a network to evaluate the metric between all prototypical states.

    For each pair of states (s, t) we return max(d(s, t), d(t, s)), since the
    approximant cannot in general guarantee symmetry.

    Returns:
      An op computing the euclidean distance between the representations of all
        pairs of states in self.eval_states_ph.
    """
    self.eval_states_ph = tf.placeholder(tf.float64, (self.num_states, 2),
                                         name='eval_states_ph')
    distances = tf.maximum(
        self.online_network(self._concat_states(self.eval_states_ph)),
        self.online_network(self._concat_states(self.eval_states_ph,
                                                transpose=True)))
    return distances

  def learn_metric(self, verbose=False):
    """Approximate the bisimulation metric by learning.

    Args:
      verbose: bool, whether to print verbose messages.
    """
    summary_writer = tf.summary.FileWriter(self.base_dir)
    global_step = tf.Variable(0, trainable=False)
    inc_global_step_op = tf.assign_add(global_step, 1)
    bisim_horizon = 0.0
    bisim_horizon_discount_value = 1.0
    if self.use_decayed_learning_rate:
      learning_rate = tf.train.exponential_decay(self.starting_learning_rate,
                                                 global_step,
                                                 self.num_iterations,
                                                 self.learning_rate_decay,
                                                 staircase=self.staircase)
    else:
      learning_rate = self.starting_learning_rate
    tf.summary.scalar('Learning/LearningRate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       epsilon=self.epsilon)
    train_op = self._build_train_op(optimizer)
    sync_op = self._build_sync_op()
    eval_op = tf.stop_gradient(self._build_eval_metric())
    eval_states = []
    # Build the evaluation tensor.
    for state in range(self.num_states):
      row, col = self.inverse_index_states[state]
      # We make the evaluation states at the center of each grid cell.
      eval_states.append([row + 0.5, col + 0.5])
    eval_states = np.array(eval_states, dtype=np.float64)
    normalized_bisim_metric = (
        self.bisim_metric / np.linalg.norm(self.bisim_metric))
    metric_errors = []
    average_metric_errors = []
    normalized_metric_errors = []
    average_normalized_metric_errors = []
    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session() as sess:
      summary_writer.add_graph(graph=tf.get_default_graph())
      sess.run(tf.global_variables_initializer())
      merged_summaries = tf.summary.merge_all()
      for i in range(self.num_iterations):
        sampled_states = np.random.randint(self.num_states,
                                           size=(self.batch_size,))
        sampled_actions = np.random.randint(4,
                                            size=(self.batch_size,))
        if self.add_noise:
          sampled_noise = np.clip(
              np.random.normal(0, 0.1, size=(self.batch_size, 2)),
              -0.3, 0.3)
        sampled_action_names = [self.actions[x] for x in sampled_actions]
        next_states = [self.next_states[a][s]
                       for s, a in zip(sampled_states, sampled_action_names)]
        rewards = np.array([self.rewards[a][s]
                            for s, a in zip(sampled_states,
                                            sampled_action_names)])
        states = np.array(
            [self.inverse_index_states[x] for x in sampled_states])
        next_states = np.array([self.inverse_index_states[x]
                                for x in next_states])
        states = states.astype(np.float64)
        states += 0.5  # Place points in center of grid.
        next_states = next_states.astype(np.float64)
        next_states += 0.5
        if self.add_noise:
          states += sampled_noise
          next_states += sampled_noise

        _, summary = sess.run(
            [train_op, merged_summaries],
            feed_dict={self.s1_ph: states,
                       self.s2_ph: next_states,
                       self.action_ph: sampled_actions,
                       self.rewards_ph: rewards,
                       self.bisim_horizon_ph: bisim_horizon,
                       self.eval_states_ph: eval_states})
        summary_writer.add_summary(summary, i)
        if self.double_period_halfway and i > self.num_iterations / 2.:
          self.target_update_period *= 2
          self.double_period_halfway = False
        if i % self.target_update_period == 0:
          bisim_horizon = 1.0 - bisim_horizon_discount_value
          bisim_horizon_discount_value *= self.bisim_horizon_discount
          sess.run(sync_op)
        # Now compute difference with exact metric.
        self.learned_distance = sess.run(
            eval_op, feed_dict={self.eval_states_ph: eval_states})
        self.learned_distance = np.reshape(self.learned_distance,
                                           (self.num_states, self.num_states))
        metric_difference = np.max(
            abs(self.learned_distance - self.bisim_metric))
        average_metric_difference = np.mean(
            abs(self.learned_distance - self.bisim_metric))
        normalized_learned_distance = (
            self.learned_distance / np.linalg.norm(self.learned_distance))
        normalized_metric_difference = np.max(
            abs(normalized_learned_distance - normalized_bisim_metric))
        average_normalized_metric_difference = np.mean(
            abs(normalized_learned_distance - normalized_bisim_metric))
        error_summary = tf.Summary(value=[
            tf.Summary.Value(tag='Approx/Error',
                             simple_value=metric_difference),
            tf.Summary.Value(tag='Approx/AvgError',
                             simple_value=average_metric_difference),
            tf.Summary.Value(tag='Approx/NormalizedError',
                             simple_value=normalized_metric_difference),
            tf.Summary.Value(tag='Approx/AvgNormalizedError',
                             simple_value=average_normalized_metric_difference),
        ])
        summary_writer.add_summary(error_summary, i)
        sess.run(inc_global_step_op)
        if i % 100 == 0:
          # Collect statistics every 100 steps.
          metric_errors.append(metric_difference)
          average_metric_errors.append(average_metric_difference)
          normalized_metric_errors.append(normalized_metric_difference)
          average_normalized_metric_errors.append(
              average_normalized_metric_difference)
          saver.save(sess, os.path.join(self.base_dir, 'tf_ckpt'),
                     global_step=i)
        if self.debug and i % 100 == 0:
          self.pretty_print_metric(metric_type='learned')
          print('Iteration: {}'.format(i))
          print('Metric difference: {}'.format(metric_difference))
          print('Normalized metric difference: {}'.format(
              normalized_metric_difference))
      if self.add_noise:
        # Finally, if we have noise, we draw a bunch of samples to get estimates
        # of the distances between states.
        sampled_distances = {}
        for _ in range(self.total_final_samples):
          eval_states = []
          for state in range(self.num_states):
            row, col = self.inverse_index_states[state]
            # We make the evaluation states at the center of each grid cell.
            eval_states.append([row + 0.5, col + 0.5])
          eval_states = np.array(eval_states, dtype=np.float64)
          eval_noise = np.clip(
              np.random.normal(0, 0.1, size=(self.num_states, 2)),
              -0.3, 0.3)
          eval_states += eval_noise
          distance_samples = sess.run(
              eval_op, feed_dict={self.eval_states_ph: eval_states})
          distance_samples = np.reshape(distance_samples,
                                        (self.num_states, self.num_states))
          for s1 in range(self.num_states):
            for s2 in range(self.num_states):
              sampled_distances[(tuple(eval_states[s1]),
                                 tuple(eval_states[s2]))] = (
                                     distance_samples[s1, s2])
      else:
        # Otherwise we just use the last evaluation metric.
        sampled_distances = self.learned_distance
    learned_statistics = {
        'num_iterations': self.num_iterations,
        'metric_errors': metric_errors,
        'average_metric_errors': average_metric_errors,
        'normalized_metric_errors': normalized_metric_errors,
        'average_normalized_metric_errors': average_normalized_metric_errors,
        'learned_distances': sampled_distances,
    }
    self.statistics['learned'] = learned_statistics
    if verbose:
      self.pretty_print_metric(metric_type='learned')

  def pretty_print_metric(self, metric_type='exact', print_side_by_side=True):
    """Print out a nice grid version of metric.

    Args:
      metric_type: str, which of the metrics to print, possible values:
        ('exact', 'sampled', 'learned').
      print_side_by_side: bool, whether to print side-by-side with the exact
        metric.
    """
    for s1 in range(self.num_states):
      print('From state {}'.format(s1))
      for row in range(self.num_rows):
        for col in range(self.num_cols):
          if self.indexed_states[row, col] < 0:
            sys.stdout.write('**********')
            continue
          s2 = self.indexed_states[row, col]
          if metric_type == 'exact' or print_side_by_side:
            val = self.bisim_metric[s1, s2]
          elif metric_type == 'sampled':
            val = self.sampled_bisim_metric[s1, s2]
          elif metric_type == 'learned':
            val = self.learned_distance[s1, s2]
          else:
            raise ValueError('Unknown metric type: {}'.format(metric_type))
          sys.stdout.write('{:10.4}'.format(val))
        if metric_type != 'exact' and print_side_by_side:
          sys.stdout.write('    ||    ')
          for col in range(self.num_cols):
            if self.indexed_states[row, col] < 0:
              sys.stdout.write('**********')
              continue
            s2 = self.indexed_states[row, col]
            if metric_type == 'sampled':
              val = self.sampled_bisim_metric[s1, s2]
            elif metric_type == 'learned':
              val = self.learned_distance[s1, s2]
            sys.stdout.write('{:10.4}'.format(val))
        sys.stdout.write('\n')
        sys.stdout.flush()
      print('')

  def save_statistics(self):
    stats_path = os.path.join(self.base_dir, 'statistics.pkl')
    with tf.gfile.GFile(stats_path, 'w') as f:
      pickle.dump(self.statistics, f)

  def sample_distance_pairs(self, num_samples_per_cell=2, verbose=False):
    """Sample a set of points from each cell and compute all pairwise distances.

    This method also writes the resulting distances to disk.

    Args:
      num_samples_per_cell: int, number of samples to draw per cell.
      verbose: bool, whether to print verbose messages.
    """
    paired_states_ph = tf.placeholder(tf.float64, (1, 4),
                                      name='paired_states_ph')
    online_network = tf.make_template('Online', self._network_template)
    distance = online_network(paired_states_ph)
    saver = tf.train.Saver()
    if not self.add_noise:
      num_samples_per_cell = 1
    with tf.Session() as sess:
      saver.restore(sess, os.path.join(self.base_dir, 'tf_ckpt-239900'))
      total_samples = None
      for s_idx in range(self.num_states):
        s = self.inverse_index_states[s_idx]
        s = s.astype(np.float32)
        s += 0.5  # Place in center of cell.
        s = np.tile([s], (num_samples_per_cell, 1))
        if self.add_noise:
          sampled_noise = np.clip(
              np.random.normal(0, 0.1, size=(num_samples_per_cell, 2)),
              -0.3, 0.3)
          s += sampled_noise
        if total_samples is None:
          total_samples = s
        else:
          total_samples = np.concatenate([total_samples, s])
      num_total_samples = len(total_samples)
      distances = np.zeros((num_total_samples, num_total_samples))
      if verbose:
        tf.logging.info('Will compute distances for %d samples',
                        num_total_samples)
      for i in range(num_total_samples):
        s1 = total_samples[i]
        if verbose:
          tf.logging.info('Will compute distances from sample %d', i)
        for j in range(num_total_samples):
          s2 = total_samples[j]
          paired_states_1 = np.reshape(np.append(s1, s2), (1, 4))
          paired_states_2 = np.reshape(np.append(s2, s1), (1, 4))
          distance_np_1 = sess.run(
              distance, feed_dict={paired_states_ph: paired_states_1})
          distance_np_2 = sess.run(
              distance, feed_dict={paired_states_ph: paired_states_2})
          max_dist = max(distance_np_1, distance_np_2)
          distances[i, j] = max_dist
          distances[j, i] = max_dist
    sampled_distances = {
        'samples_per_cell': num_samples_per_cell,
        'samples': total_samples,
        'distances': distances,
    }
    file_path = os.path.join(self.base_dir, 'sampled_distances.pkl')
    with tf.gfile.GFile(file_path, 'w') as f:
      pickle.dump(sampled_distances, f)
