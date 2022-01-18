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

"""Generate reset goals for an environment."""

import numpy as np
import tensorflow as tf

from tf_agents.trajectories import time_step as ts


class ResetGoalGenerator(tf.Module):

  def __init__(self,
               goal_dim,
               compute_value_fn,
               distance_fn,
               use_minimum=True,
               value_threshold=None,
               lagrange_variable_max=None,
               normalize_values=False,
               optimizer=None,
               name=None):
    super(ResetGoalGenerator, self).__init__(name=name)
    self._goal_dim = goal_dim
    self._distance_fn = distance_fn
    self._compute_value_fn = compute_value_fn
    self._normalize_values = normalize_values
    self._lagrange_multiplier = tf.Variable(
        initial_value=0.0,
        trainable=True,
        dtype=tf.float32,
        name='lagrange_multiplier')
    self._value_threshold = None
    if value_threshold is not None:
      self._value_threshold = tf.constant(
          value_threshold, dtype=tf.float32, name='value_threshold')
    self._optimizer = optimizer
    self._use_minimum = use_minimum
    if lagrange_variable_max is None:
      self._lagrange_variable_upper_clip = np.inf
    else:
      self._lagrange_variable_upper_clip = lagrange_variable_max
    self._values = []

  def get_value_function(self, sampled_experience, cur_state, next_goal):
    # reaching the potential reset-state from the current state
    reset_goal_candidates = sampled_experience.observation[:, :self._goal_dim]
    cur_state = tf.expand_dims(cur_state[:-self._goal_dim], 0)
    num_sampled_reset_candidates = tf.shape(sampled_experience.observation)[0]
    cur_state_without_goal = tf.tile(cur_state,
                                     [num_sampled_reset_candidates, 1])
    cur_state_with_reset_goals = tf.concat(
        [cur_state_without_goal, reset_goal_candidates], axis=1)
    cur_state_time_steps = ts.TimeStep(
        step_type=sampled_experience.step_type,
        reward=tf.nest.map_structure(tf.zeros_like, sampled_experience.reward),
        discount=tf.zeros_like(sampled_experience.discount),
        observation=cur_state_with_reset_goals)
    cur_to_reset_reachability_value = self._compute_value_fn(
        cur_state_time_steps)

    # reaching from potential reset-state to task-goal
    next_goal = tf.expand_dims(next_goal, 0)
    next_goal = tf.tile(next_goal, [num_sampled_reset_candidates, 1])
    reset_states_with_task_goal = tf.concat(
        [sampled_experience.observation[:, :-self._goal_dim], next_goal],
        axis=1)
    reset_to_task_reachability_value = self._compute_value_fn(
        cur_state_time_steps._replace(observation=reset_states_with_task_goal))

    if self._use_minimum:
      reachability_value = tf.minimum(cur_to_reset_reachability_value,
                                      reset_to_task_reachability_value)
    else:
      reachability_value = reset_to_task_reachability_value

    return reachability_value

  def get_reset_goal(self, sampled_experience, cur_state, next_goal):
    # value function
    self._reset_goal_values = self.get_value_function(sampled_experience,
                                                      cur_state, next_goal)
    self._reset_goal_candidates = sampled_experience.observation[:, :self
                                                                 ._goal_dim]

    # distance to the initial state distribution
    self._distance_to_initial_state = tf.cast(
        self._distance_fn.distance(self._reset_goal_candidates),
        dtype=tf.float32)

    reset_goal_idx = tf.argmin(self._distance_to_initial_state -
                               self._lagrange_multiplier *
                               self._reset_goal_values)
    reset_goal = self._reset_goal_candidates[reset_goal_idx]
    self._values.append(self._reset_goal_values[reset_goal_idx])
    return reset_goal

  def update_lagrange_multipliers(self):
    assert self._optimizer is not None, 'no optimizer found'
    assert self._value_threshold is not None, 'value threshold not given'

    if self._values:
      grad_variable = [self._lagrange_multiplier]
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        assert grad_variable, 'No variable to optimize.'
        tape.watch(grad_variable)
        self._values = tf.concat(self._values, axis=0)
        self._diff_from_threshold = tf.stop_gradient(
            self._values - self._value_threshold
        )  # signs reversed because want to maximize dual
        lagrange_loss = tf.reduce_mean(self._lagrange_multiplier *
                                       self._diff_from_threshold)

      tf.debugging.check_numerics(lagrange_loss, 'lagrange loss is inf or nan.')
      lagrange_grads = tape.gradient(lagrange_loss, grad_variable)
      grads_and_vars = list(zip(lagrange_grads, grad_variable))
      self._optimizer.apply_gradients(grads_and_vars)
      self._lagrange_multiplier.assign(
          tf.clip_by_value(
              self._lagrange_multiplier,
              clip_value_min=0.,
              clip_value_max=self._lagrange_variable_upper_clip)
      )  # lagrange multipliers should be non-negative
      self._prev_values = self._values
      self._values = []

  def update_summaries(self, step_counter):
    with tf.name_scope('ResetGoalGenerator'):
      tf.compat.v2.summary.scalar(
          name='subgoal_value_function',
          data=tf.reduce_mean(self._prev_values),
          step=step_counter)
      tf.compat.v2.summary.scalar(
          name='reset_lagrange_diff',
          data=tf.reduce_mean(self._diff_from_threshold),
          step=step_counter)
      tf.compat.v2.summary.scalar(
          name='reset_lagrange_multiplier',
          data=self._lagrange_multiplier,
          step=step_counter)
    return self._reset_goal_candidates, self._reset_goal_values, self._distance_to_initial_state, self._lagrange_multiplier


class FixedResetGoal(tf.Module):

  def __init__(self, distance_fn, name=None, **kwargs):
    super(FixedResetGoal, self).__init__(name=name)
    self._distance_fn = distance_fn  # assuming this is a L2 distance function

  def get_reset_goal(self, sampled_experience, cur_state, next_goal):
    return tf.convert_to_tensor(self._distance_fn.initial_state)

  def update_summaries(self, step_counter):
    return


# hard-coded for reduced playpen with rc_o (that is single goal), with original data
class ScheduledResetGoal(tf.Module):

  def __init__(self,
               goal_dim,
               num_success_for_switch=10,
               num_chunks=10,
               name=None,
               **kwargs):
    super(ScheduledResetGoal, self).__init__(name=None)
    self._goal_dim = goal_dim
    self._reset_goal_candidates = {}
    self._success_counts = {}
    self._last_reset_idx = {}
    self._prev_goal = None
    self._num_success_for_switch = num_success_for_switch
    self._num_chunks = num_chunks

  def get_reset_goal(self, sampled_experience, cur_state, next_goal):
    goal_y_coord = next_goal.numpy()[3]
    if goal_y_coord not in self._reset_goal_candidates.keys():
      self._reset_goal_candidates[goal_y_coord] = sampled_experience.observation
      # goal-specific reset goal candidates
      print(next_goal)
      self._reset_goal_candidates[goal_y_coord] = self._reset_goal_candidates[
          goal_y_coord][tf.reduce_all(
              tf.equal(sampled_experience.observation[:, self._goal_dim:],
                       next_goal), 1)]
      print(self._reset_goal_candidates)
      self._reset_goal_idxs = tf.range(
          0, self._reset_goal_candidates[goal_y_coord].shape[0],
          self._reset_goal_candidates[goal_y_coord].shape[0] //
          self._num_chunks)  # 10 intermediate goals
      print(self._reset_goal_idxs)
      # TODO(architsh): remove this
      # self._reset_goal_idxs = tf.reverse(
      #     tf.concat([
      #         self._reset_goal_idxs,
      #         [self._reset_goal_candidates[goal_y_coord].shape[0] - 1]
      #     ],
      #               axis=0),
      #     axis=[0])[:-1]
      # print('after adding and reversing:', self._reset_goal_idxs)
      # --------------------------
      self._reset_goal_candidates[goal_y_coord] = tf.gather(
          self._reset_goal_candidates[goal_y_coord], self._reset_goal_idxs)
      # TODO(architsh): remove this
      # next_goal_dummy_copy = tf.concat([next_goal, next_goal], axis=0)
      # self._reset_goal_candidates[goal_y_coord] = tf.concat(
      #     [[next_goal_dummy_copy], self._reset_goal_candidates[goal_y_coord][1:]
      #     ],
      #     axis=0)
      # ---------------------------
      print('check first goal:', self._reset_goal_candidates)
      self._success_counts[goal_y_coord] = [0] * self._reset_goal_idxs.shape[
          0]  # goals achievement count from intermediate state
      self._last_reset_idx[goal_y_coord] = self._reset_goal_idxs.shape[0] - 1

    reset_goal = self._reset_goal_candidates[goal_y_coord][
        self._last_reset_idx[goal_y_coord]][:self._goal_dim]

    print('cur reset goal:', reset_goal)
    if tf.norm(cur_state[:4] - cur_state[6:-2]) <= 0.2:
      print('was successful')
      if self._prev_goal is not None:
        prev_goal_y_coord = self._prev_goal.numpy()[3]
        self._success_counts[prev_goal_y_coord][
            self._last_reset_idx[prev_goal_y_coord]] += 1
        print(self._success_counts[prev_goal_y_coord])
        print(self._last_reset_idx)
        if self._success_counts[prev_goal_y_coord][self._last_reset_idx[
            prev_goal_y_coord]] >= self._num_success_for_switch and self._last_reset_idx[
                prev_goal_y_coord] > 0:

          self._last_reset_idx[prev_goal_y_coord] -= 1

    # to update the right counts
    self._prev_goal = next_goal
    return reset_goal

  def update_summaries(self, step_counter):
    return
