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

"""Implementation of various python metrics."""

import gin
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.metrics import py_metric
from tf_agents.metrics import py_metrics
from tf_agents.utils import numpy_storage


@gin.configurable
class PolicyUsageFrequency(py_metric.PyStepMetric):
  """Class for policy usage metrics.

  Policy usage metrics keep track of the last (upto) K values of binary
  indicator of the selection of behavior policy in a Deque buffer of size K.
  Calling result() will return the frequency of the policy being used in recent
  K selections.
  """

  def __init__(self, name='PolicyUsageFrequency', buffer_size=10):
    super(PolicyUsageFrequency, self).__init__(name)
    self._buffer = py_metrics.NumpyDeque(maxlen=buffer_size, dtype=np.float64)
    self.reset()

  def reset(self):
    self._buffer.clear()

  def add_to_buffer(self, values):
    """Appends new values to the buffer."""
    self._buffer.extend(values)

  def result(self):
    """Returns the value of this metric."""
    if self._buffer:
      return self._buffer.mean(dtype=np.float32)
    return np.array(0.0, dtype=np.float32)

  def _batched_call(self, is_selected):
    self.add_to_buffer(is_selected)

  def call(self, is_selected=0):
    self._batched_call(is_selected)


@gin.configurable
class StatsNumpyDeque(py_metrics.NumpyDeque):
  """Deque implementation using a numpy array as a circular buffer."""

  def __init__(self, maxlen, dtype):
    super(StatsNumpyDeque, self).__init__(maxlen=maxlen, dtype=dtype)

  def resize_buffer(self, new_size=20, dtype=np.float64):
    new_buffer = np.zeros(shape=(new_size,), dtype=dtype)
    if self._len == self._buffer.shape[0]:
      data = np.append(self._buffer[self._start_index:],
                       self._buffer[0:self._start_index])
    else:
      assert self._start_index == 0
      data = self._buffer[:self._len]
    new_buffer[0:self._len] = data
    self._start_index = 0
    self._maxlen = np.array(new_size)
    self._buffer = new_buffer

  def std(self, dtype=None):
    if self._len == self._buffer.shape[0]:
      return np.std(self._buffer, dtype=dtype)
    assert self._start_index == 0
    return np.std(self._buffer[:self._len], dtype=dtype)

  def ucb(self, coeff=0.0, dtype=None):
    if self._len == self._buffer.shape[0]:
      array = self._buffer
    else:
      assert self._start_index == 0
      array = self._buffer[:self._len]
    return np.mean(array, dtype=dtype) + coeff * np.std(array, dtype=dtype)

  def sum(self, dtype=None):
    if self._len == self._buffer.shape[0]:
      array = np.append(self._buffer[self._start_index:],
                        self._buffer[0:self._start_index])
    else:
      assert self._start_index == 0
      array = self._buffer[:self._len]
    result = np.sum(array[array != 0], dtype=dtype)
    return np.nan if result == 0 else result

  # def rolling_q(self, way='q', coeff=0, dtype=None):
  #   if self._len == self._buffer.shape[0]:
  #     array = np.append(self._buffer[self._start_index:],
  # .                     self._buffer[0:self._start_index])
  #   else:
  #     assert self._start_index == 0
  #     array = self._buffer[:self._len]
  #   q = np.mean(array[array != 0],dtype=dtype)
  #   if way == 'ucb':
  #     ucb = q + np.sqrt(coeff*np.log(len(array))/sum(array != 0))
  #     return np.inf if np.isnan(ucb) else ucb
  #   elif way == 'lcb':
  #     lcb = q - np.sqrt(coeff*np.log(len(array))/sum(array != 0))
  #     return -np.inf if np.isnan(lcb) else lcb
  #   else:
  #     return np.inf if np.isnan(q) else q
  def rolling_most_recent(self, dtype=None):
    if self._len == self._buffer.shape[0]:
      array = np.append(self._buffer[self._start_index:],
                        self._buffer[0:self._start_index])
    else:
      assert self._start_index == 0
      array = self._buffer[:self._len]
    valid = array[array != 0]
    return valid[-1] if valid else np.nan

  def replace_last(self, value):
    if self._len == self._buffer.shape[0]:
      self._buffer[self._start_index - 1] = value
    else:
      assert self._start_index == 0
      self._buffer[self._len - 1] = value


@gin.configurable
class DistributionReturnMetric(py_metrics.StreamingMetric):
  """Computes the mean and variance of batched undiscounted rewards."""

  def __init__(self,
               name='DistributionReturn',
               buffer_size=10,
               batch_size=None):
    """Creates an DistributionReturnMetric."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.episode_return so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.episode_return = np.float64(0)
    self._np_state.episode_end_mask = np.float64(0)
    # self.count_episode = 0
    super(DistributionReturnMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)
    # overwrite buffer to enable more statistics computation
    self._buffer = StatsNumpyDeque(maxlen=buffer_size, dtype=np.float64)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.episode_return = np.zeros(
        shape=(batch_size,), dtype=np.float64)
    self._np_state.episode_end_mask = np.zeros(
        shape=(batch_size,), dtype=np.float64)
    # self.count_episode = 0

  def set_mask(self, mask_item):
    self._np_state.episode_end_mask[mask_item] = 1
    return sum(self._np_state.episode_end_mask)

  def get_buffer_size(self):
    return len(self._buffer)

  # overwrite result to output statistics
  def result(self, way='mean', coeff=0.0):
    """Returns the value of this metric."""
    if self._buffer:
      if way == 'mean':
        return self._buffer.mean(dtype=np.float32)
      elif way == 'std':
        return self._buffer.std(dtype=np.float32)
      elif way == 'ucb':
        return self._buffer.ucb(coeff=coeff, dtype=np.float32)
    return np.array(0.0, dtype=np.float32)

  def _batched_call(self, traj):
    """Processes the trajectory to update the metric.

    Args:
      traj: a tf_agents.trajectory.Trajectory.
    """
    episode_return = self._np_state.episode_return

    is_first = np.where(traj.is_first())[0]
    episode_return[is_first] = 0

    episode_return += traj.reward

    is_last = np.where(traj.is_last())[0]
    is_masked = np.where(self._np_state.episode_end_mask > 0)[0]
    new_last = np.setdiff1d(is_last, is_masked)
    self.add_to_buffer(episode_return[new_last])


@gin.configurable
class DistributionEpisodeLengthMetric(py_metrics.StreamingMetric):
  """Computes the average episode length."""

  def __init__(self,
               name='DistributionEpisodeLength',
               buffer_size=10,
               batch_size=None):
    """Creates an AverageEpisodeLengthMetric."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.episode_return so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.episode_steps = np.float64(0)
    self._np_state.episode_end_mask = np.float64(0)
    super(DistributionEpisodeLengthMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)
    self._buffer = StatsNumpyDeque(maxlen=buffer_size, dtype=np.float64)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.episode_steps = np.zeros(
        shape=(batch_size,), dtype=np.float64)
    self._np_state.episode_end_mask = np.zeros(
        shape=(batch_size,), dtype=np.float64)

  def set_mask(self, mask_item):
    self._np_state.episode_end_mask[mask_item] = 1
    return sum(self._np_state.episode_end_mask)

  def get_buffer_size(self):
    return len(self._buffer)

  def result(self, way='mean', coeff=0.0):
    """Returns the value of this metric."""
    if self._buffer:
      if way == 'mean':
        return self._buffer.mean(dtype=np.float32)
      elif way == 'std':
        return self._buffer.std(dtype=np.float32)
      elif way == 'ucb':
        return self._buffer.ucb(coeff=coeff, dtype=np.float32)
      elif way == '95ucb':
        return self._buffer.mean(dtype=np.float32) + 1.96 * self._buffer.std(
            dtype=np.float32) / self.get_buffer_size()
    return np.array(0.0, dtype=np.float32)

  def _batched_call(self, traj):
    """Processes the trajectory to update the metric.

    Args:
      traj: a tf_agents.trajectory.Trajectory.
    """
    episode_steps = self._np_state.episode_steps

    # Each non-boundary trajectory (first, mid or last) represents a step.
    episode_steps[np.where(~traj.is_boundary())[0]] += 1
    is_last = np.where(traj.is_last())[0]
    is_masked = np.where(self._np_state.episode_end_mask > 0)[0]
    new_last = np.setdiff1d(is_last, is_masked)
    self.add_to_buffer(episode_steps[new_last])
    episode_steps[new_last] = 0


@gin.configurable
class QMetric(py_metric.PyStepMetric):
  """Class for policy usage metrics.

  Policy usage metrics keep track of the last (upto) K values of binary
  indicator of the selection of behavior policy in a Deque buffer of size K.
  Calling result() will return the frequency of the policy being used in recent
  K selections.
  """

  def __init__(self, name='QMetric', buffer_size=10):
    super(QMetric, self).__init__(name)
    self._buffer = StatsNumpyDeque(maxlen=buffer_size, dtype=np.float64)
    self._count = StatsNumpyDeque(maxlen=buffer_size, dtype=np.float64)
    self._sumcount = StatsNumpyDeque(maxlen=buffer_size, dtype=np.float64)
    self._np_state = numpy_storage.NumpyState()
    self._np_state._most_recent_q = np.float64(-100)  # pylint: disable=protected-access
    self._np_state._most_recent_time = np.int64(0)  # pylint: disable=protected-access
    self.reset()

  def rename(self, name):
    self._name = name
    with tf.name_scope(name) as scope_name:
      self._scope_name = scope_name

  def get_buffer_size(self):
    return len(self._buffer)

  # overwrite result to output q
  def result(self, way='q', coeff=0.0):
    """Returns the value of specified metric."""
    if self._buffer:
      if way == 'rolling_most_recent':
        return self._buffer.rolling_most_recent(dtype=np.float32)
      elif way == 'q':
        q = self._buffer.sum(dtype=np.float32) / self._count.sum()
        return np.inf if np.isnan(q) else q
      elif way == 'ucb':
        ucb = self._buffer.sum(dtype=np.float32) / self._count.sum() + np.sqrt(
            coeff * np.log(self._sumcount.sum()) / self._count.sum())
        return np.inf if np.isnan(ucb) else ucb
      elif way == 'lcb':
        lcb = self._buffer.sum(dtype=np.float32) / self._count.sum() - np.sqrt(
            coeff * np.log(self._sumcount.sum()) / self._count.sum())
        return -np.inf if np.isnan(lcb) else lcb
      elif way == 'most_recent':
        return self._np_state._most_recent_q  # pylint: disable=protected-access
      elif way == 'most_recent_time':
        return self._np_state._most_recent_time  # pylint: disable=protected-access
    return np.nan

  def reset(self):
    self._buffer.clear()
    self._np_state._most_recent_q = np.float64(-100)  # pylint: disable=protected-access
    self._np_state._most_recent_time = np.int64(0)  # pylint: disable=protected-access
    self._count.clear()
    self._sumcount.clear()

  def add_to_buffer(self, value, discount=1.0, update_time=True):
    """Appends new values to the buffer."""
    self._buffer._buffer *= discount
    self._count._buffer *= discount
    self._sumcount._buffer *= discount
    self._buffer.extend([value])
    if value != 0:
      self._np_state._most_recent_q = value  # pylint: disable=protected-access
      self._np_state._most_recent_time = np.int64(0)  # pylint: disable=protected-access
      self._count.extend([1.0])
    else:
      if update_time:
        self._np_state._most_recent_time += 1
      self._count.extend([0.0])
    self._sumcount.extend([1.0])

  def modify_last_buffer(self, value, update_time=True):
    """Modify the last element of the buffer."""
    self._buffer.replace_last(value)
    if value != 0:
      self._np_state._most_recent_q = np.float64(value)  # pylint: disable=protected-access
      self._np_state._most_recent_time = np.int64(0)  # pylint: disable=protected-access
      self._count.replace_last(1.0)
    elif update_time:
      self._np_state._most_recent_time += 1

  def is_recent(self, update_time=10):
    return ~np.isinf(
        self.result()) and self._np_state._most_recent_time <= update_time  # pylint: disable=protected-access

  def resize_buffer(self, new_size=20):
    self._buffer.resize_buffer(new_size=new_size)
    self._count.resize_buffer(new_size=new_size)
    self._sumcount.resize_buffer(new_size=new_size)

  def _batched_call(self, reward):
    self.add_to_buffer(reward)

  def call(self, reward=0):
    self._batched_call(reward)
