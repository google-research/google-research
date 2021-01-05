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

"""Additional metrics for reset-free learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import seaborn as sns

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.metrics import tf_metric
from tf_agents.metrics.tf_metrics import TFDeque
from tf_agents.utils import common


class FailedEpisodes(tf_metric.TFStepMetric):
  """Counts the number of episodes ending in failure / requiring human intervention."""

  def __init__(self,
               failure_function,
               name='FailedEpisodes',
               prefix='Metrics',
               dtype=tf.int64):
    super(FailedEpisodes, self).__init__(name=name, prefix=prefix)
    self.dtype = dtype
    self._failure_function = failure_function
    self.number_failed_episodes = common.create_variable(
        initial_value=0,
        dtype=self.dtype,
        shape=(),
        name='number_failed_episodes')

  def call(self, trajectory):
    """Increase the number of number_failed_episodes according to trajectory's final reward.

    It would increase for all trajectory.is_last().

    Args:
      trajectory: A tf_agents.trajectory.Trajectory

    Returns:
      The arguments, for easy chaining.
    """
    # The __call__ will execute this.
    num_failed_episodes = tf.cast(
        self._failure_function(trajectory), self.dtype)
    num_failed_episodes = tf.reduce_sum(input_tensor=num_failed_episodes)
    self.number_failed_episodes.assign_add(num_failed_episodes)
    return trajectory

  def result(self):
    return tf.identity(self.number_failed_episodes, name=self.name)

  @common.function
  def reset(self):
    self.number_failed_episodes.assign(0)


class AnyStepGoalMetric(tf_metric.TFStepMetric):
  """Counts the number of episodes ending in failure / requiring human intervention."""

  def __init__(self,
               goal_success_fn,
               name='GoalSuccess',
               prefix='Metrics',
               batch_size=1,
               dtype=tf.int64):
    super(AnyStepGoalMetric, self).__init__(name=name, prefix=prefix)
    self.dtype = dtype
    self._goal_success_fn = goal_success_fn
    self._new_ep_and_goal_not_seen = tf.constant(False)
    self.number_successful_episodes = common.create_variable(
        initial_value=0,
        dtype=self.dtype,
        shape=(),
        name='num_successful_episodes')
    self._success_accumulator = common.create_variable(
        initial_value=0,
        dtype=self.dtype,
        shape=(batch_size,),
        name='num_successful_episodes')

  def call(self, trajectory):
    """If the agent is successful at any step in the episode, the metric increases by 1, else 0.

    Args:
      trajectory: A tf_agents.trajectory.Trajectory

    Returns:
      The arguments, for easy chaining.
    """

    self._success_accumulator.assign(
        tf.where(trajectory.is_first(),
                 tf.zeros_like(self._success_accumulator),
                 self._success_accumulator))

    self._success_accumulator.assign_add(
        tf.cast(self._goal_success_fn(trajectory), self.dtype))

    self.number_successful_episodes.assign_add(
        tf.cast(
            tf.reduce_sum(
                tf.where(
                    tf.logical_and(
                        tf.greater(self._success_accumulator, 0),
                        trajectory.is_last()), 1, 0)), self.dtype))

    return trajectory

  def result(self):
    return tf.identity(self.number_successful_episodes, name=self.name)

  @common.function
  def reset(self):
    self.number_successful_episodes.assign(0)
    self._success_accumulator.assign([0])


class StateVisitationHistogram(tf_metric.TFHistogramStepMetric):
  """Metric to compute the frequency of states visited."""

  def __init__(self,
               state_selection_function,
               state_shape=(),
               name='StateVisitationHistogram',
               dtype=tf.float64,
               buffer_size=100):
    super(StateVisitationHistogram, self).__init__(name=name)
    self._buffer = TFDeque(buffer_size, dtype, shape=state_shape)
    self._dtype = dtype
    self._state_selection_function = state_selection_function

  @common.function
  def call(self, trajectory):
    self._buffer.extend(self._state_selection_function(trajectory.observation))
    return trajectory

  @common.function
  def result(self):
    return self._buffer.data

  @common.function
  def reset(self):
    self._buffer.clear()


class StateVisitationHeatmap(tf_metric.TFStepMetric):

  def __init__(
      self,
      trajectory_to_xypos,  # acts on trajectory.observation
      state_max=None,
      x_range=None,
      y_range=None,
      num_bins=10,  # per axis
      name='StateVisitationHeatmap',
      prefix='Metrics',
      dtype=tf.int64):
    super(StateVisitationHeatmap, self).__init__(name=name, prefix=prefix)
    self.dtype = dtype
    self._conversion_function = trajectory_to_xypos
    self._state_max = state_max  # either state_max is None or x,y range are None
    self._x_range = x_range
    self._y_range = y_range
    self._num_bins = num_bins
    self._create_state_visitation_variables()

  def _find_heatmap_key(self, pos):
    if self._state_max is not None:
      x_key = tf.cast(
          tf.clip_by_value(
              tf.math.floor((pos[0, 0] + self._state_max) / self._state_delta),
              0, self._num_bins - 1),
          dtype=tf.int64)
      y_key = tf.cast(
          tf.clip_by_value(
              tf.math.floor((pos[0, 1] + self._state_max) / self._state_delta),
              0, self._num_bins - 1),
          dtype=tf.int64)
    else:
      x_key = tf.cast(
          tf.clip_by_value(
              tf.math.floor((pos[0, 0] - self._x_low) / self._x_delta), 0,
              self._num_bins - 1),
          dtype=tf.int64)
      y_key = tf.cast(
          tf.clip_by_value(
              tf.math.floor((pos[0, 1] - self._y_low) / self._y_delta), 0,
              self._num_bins - 1),
          dtype=tf.int64)
    return (x_key, y_key)

  def _create_state_visitation_variables(self, reinitialize=False):
    if not reinitialize:
      # self._state_visit_dict = {}
      self._state_visit_tf_array = common.create_variable(
          initial_value=np.zeros((self._num_bins, self._num_bins)),
          dtype=self.dtype,
          shape=(self._num_bins, self._num_bins),
          name='state_visit_count')
      if self._state_max is not None:
        self._state_delta = 2 * self._state_max / self._num_bins
        self._xticks = [
            round(-self._state_max + (2 * x_idx + 1) * self._state_delta / 2, 2)  # pylint: disable=invalid-unary-operand-type
            for x_idx in range(self._num_bins)
        ]
        self._yticks = [
            round(-self._state_max + (2 * y_idx + 1) * self._state_delta / 2, 2)  # pylint: disable=invalid-unary-operand-type
            for y_idx in range(self._num_bins)
        ]
      else:
        self._x_low, self._x_high = self._x_range
        self._y_low, self._y_high = self._y_range
        self._x_delta = (self._x_high - self._x_low) / self._num_bins
        self._y_delta = (self._y_high - self._y_low) / self._num_bins
        self._xticks = [
            round(self._x_low + (2 * x_idx + 1) * self._x_delta / 2, 2)
            for x_idx in range(self._num_bins)
        ]
        self._yticks = [
            round(self._y_low + (2 * y_idx + 1) * self._y_delta / 2, 2)
            for y_idx in range(self._num_bins)
        ]

      # for x_idx in range(self._num_bins):
      #   x_low = -self._state_max + x_idx * self._state_delta
      #   x_high = -self._state_max + (x_idx + 1) * self._state_delta
      #   for y_idx in range(self._num_bins):
      #     y_low = -self._state_max + y_idx * self._state_delta
      #     y_high = -self._state_max + (y_idx + 1) * self._state_delta
      #     self._state_visit_dict[(x_idx, y_idx)] = [
      #         (x_low, x_high),
      #         (y_low, y_high),
      #     ]
    else:
      self._state_visit_tf_array = common.create_variable(
          initial_value=np.zeros((self._num_bins, self._num_bins)),
          dtype=self.dtype,
          shape=(self._num_bins, self._num_bins),
          name='state_visit_count')

  def call(self, trajectory):
    pos = self._conversion_function(trajectory.observation)
    key = self._find_heatmap_key(pos)
    cur_val = self._state_visit_tf_array[key[0], key[1]]
    self._state_visit_tf_array[key[0], key[1]].assign(cur_val + 1)
    return trajectory

  def result(self):
    figure = plt.figure(figsize=(10, 10))
    image_array = self._state_visit_tf_array.numpy()

    sns.heatmap(
        image_array,
        xticklabels=self._yticks,
        yticklabels=self._xticks,
        linewidth=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

  def tf_summaries(self, train_step=None, step_metrics=()):
    """Generates summaries against train_step and all step_metrics.

    Args:
      train_step: (Optional) Step counter for training iterations. If None, no
        metric is generated against the global step.
      step_metrics: (Optional) Iterable of step metrics to generate summaries
        against.

    Returns:
      A list of summaries.
    """
    summaries = []
    prefix = self._prefix
    tag = common.join_scope(prefix, self.name)
    result = self.result()
    if train_step is not None:
      summaries.append(
          tf.compat.v2.summary.image(name=tag, data=result, step=train_step))
    if prefix:
      prefix += '_'
    for step_metric in step_metrics:
      # Skip plotting the metrics against itself.
      if self.name == step_metric.name:
        continue
      step_tag = '{}vs_{}/{}'.format(prefix, step_metric.name, self.name)
      # Summaries expect the step value to be an int64.
      step = tf.cast(step_metric.result(), tf.int64)
      summaries.append(
          tf.compat.v2.summary.image(name=step_tag, data=result, step=step))
    return summaries

  @common.function
  def reset(self):
    self._create_state_visitation_variables(reinitialize=True)


class ValueFunctionHeatmap(tf_metric.TFStepMetric):

  def __init__(
      self,
      trajectory_to_xypos,  # acts on trajectory.observation
      state_max=None,
      x_range=None,
      y_range=None,
      num_bins=10,  # per axis
      name='ValueFunctionHeatmap',
      prefix='ResetMetrics',
      dtype=tf.int64):
    super(ValueFunctionHeatmap, self).__init__(name=name, prefix=prefix)
    self.dtype = dtype
    self._conversion_function = trajectory_to_xypos
    self._state_max = state_max  # either state_max is None or x,y range are None
    self._x_range = x_range
    self._y_range = y_range
    self._num_bins = num_bins
    self._create_state_visitation_variables()

  def _find_heatmap_key(self, pos):
    if self._state_max is not None:
      x_key = np.clip(
          np.floor((pos[0, 0] + self._state_max) / self._state_delta), 0,
          self._num_bins - 1).astype(dtype=np.int64)
      y_key = np.clip(
          np.floor((pos[0, 1] + self._state_max) / self._state_delta), 0,
          self._num_bins - 1).astype(dtype=np.int64)
    else:
      x_key = np.clip(
          np.floor((pos[0, 0] - self._x_low) / self._x_delta), 0,
          self._num_bins - 1).astype(dtype=np.int64)
      y_key = np.clip(
          np.floor((pos[0, 1] - self._y_low) / self._y_delta), 0,
          self._num_bins - 1).astype(dtype=np.int64)
    return (x_key, y_key)

  def _create_state_visitation_variables(self, reinitialize=False):
    self._state_val_array = np.zeros((self._num_bins, self._num_bins))
    if not reinitialize:
      if self._state_max is not None:
        self._state_delta = 2 * self._state_max / self._num_bins
        self._xticks = [
            round(-self._state_max + (2 * x_idx + 1) * self._state_delta / 2, 2)  # pylint: disable=invalid-unary-operand-type
            for x_idx in range(self._num_bins)
        ]
        self._yticks = [
            round(-self._state_max + (2 * y_idx + 1) * self._state_delta / 2, 2)  # pylint: disable=invalid-unary-operand-type
            for y_idx in range(self._num_bins)
        ]
      else:
        self._x_low, self._x_high = self._x_range
        self._y_low, self._y_high = self._y_range
        self._x_delta = (self._x_high - self._x_low) / self._num_bins
        self._y_delta = (self._y_high - self._y_low) / self._num_bins
        self._xticks = [
            round(self._x_low + (2 * x_idx + 1) * self._x_delta / 2, 2)
            for x_idx in range(self._num_bins)
        ]
        self._yticks = [
            round(self._y_low + (2 * y_idx + 1) * self._y_delta / 2, 2)
            for y_idx in range(self._num_bins)
        ]

  def result(self, reset_states, values):
    figure = plt.figure(figsize=(10, 10))
    reset_states = reset_states.numpy()
    xy_pos = self._conversion_function(reset_states)
    values = values.numpy()
    discretized_state_value_lists = [
        [] for _ in range(self._num_bins * self._num_bins)
    ]
    for idx in range(xy_pos.shape[0]):
      x_key, y_key = self._find_heatmap_key(xy_pos[idx:idx + 1])
      discretized_state_value_lists[x_key * self._num_bins + y_key].append(
          values[idx])
    val_min = np.inf
    val_max = -np.inf
    for x_idx in range(self._num_bins):
      for y_idx in range(self._num_bins):
        cur_val_list = discretized_state_value_lists[x_idx * self._num_bins +
                                                     y_idx]
        if cur_val_list:
          mean_value = np.mean(cur_val_list)
          self._state_val_array[x_idx, y_idx] = mean_value
          val_min = min(val_min, mean_value)
          val_max = max(val_max, mean_value)
        else:
          self._state_val_array[x_idx, y_idx] = -np.inf

    # ticklabels and matrix indices are reversed
    sns.heatmap(
        self._state_val_array,
        vmin=val_min,
        vmax=val_max,
        xticklabels=self._yticks,
        yticklabels=self._xticks,
        linewidth=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

  def tf_summaries(self,
                   reset_states,
                   values,
                   train_step=None,
                   step_metrics=()):
    """Generates summaries against train_step and all step_metrics.

    Args:
      reset_states: candidate states for reset
      values: values assigned by our function
      train_step: (Optional) Step counter for training iterations. If None, no
        metric is generated against the global step.
      step_metrics: (Optional) Iterable of step metrics to generate summaries
        against.

    Returns:
      A list of summaries.
    """
    summaries = []
    prefix = self._prefix
    tag = common.join_scope(prefix, self.name)
    result = self.result(reset_states, values)
    if train_step is not None:
      summaries.append(
          tf.compat.v2.summary.image(name=tag, data=result, step=train_step))
    if prefix:
      prefix += '_'
    for step_metric in step_metrics:
      # Skip plotting the metrics against itself.
      if self.name == step_metric.name:
        continue
      step_tag = '{}vs_{}/{}'.format(prefix, step_metric.name, self.name)
      # Summaries expect the step value to be an int64.
      step = tf.cast(step_metric.result(), tf.int64)
      summaries.append(
          tf.compat.v2.summary.image(name=step_tag, data=result, step=step))
    return summaries

  @common.function
  def reset(self):
    self._create_state_visitation_variables(reinitialize=True)
