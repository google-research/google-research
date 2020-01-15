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

# Lint as: python3
"""Custom TFAgent PyMetric for minitaur and point-mass environments.

AverageEarlyFailureMetric used for detecting fall count for minitaur env, and
AverageFallenMetric and AverageSuccessMetric used for poit-mass envs.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import numpy as np

from tf_agents.metrics import py_metrics
from tf_agents.utils import numpy_storage


@gin.configurable
class AverageEarlyFailureMetric(py_metrics.StreamingMetric):
  """Computes average early failure rate in buffer_size episodes."""

  def __init__(self,
               max_episode_len=500,
               dtype=np.bool,
               name='AverageEarlyFailure',
               buffer_size=10,
               batch_size=None):
    """Creates an AverageEnvObsDict."""
    self._np_state = numpy_storage.NumpyState()
    self._max_episode_len = max_episode_len
    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.episode_steps = np.array(0, dtype=np.int32)
    super(AverageEarlyFailureMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.episode_steps = np.zeros(shape=(batch_size,), dtype=np.int32)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    episode_steps = self._np_state.episode_steps
    is_last = np.where(trajectory.is_last())

    episode_steps[np.where(~trajectory.is_boundary())] += 1
    self.add_to_buffer(episode_steps[is_last] < self._max_episode_len)
    episode_steps[is_last] = 0


@gin.configurable
class AverageFallenMetric(py_metrics.StreamingMetric):
  """Computes average fallen rate for PointMass envs in buffer_size episodes."""

  def __init__(self,
               dtype=np.bool,
               name='AverageFallen',
               buffer_size=10,
               batch_size=None):
    """Creates an AverageFallenMetric."""
    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    self._dtype = dtype
    super(AverageFallenMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    return

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    is_last = np.where(trajectory.is_boundary())

    if is_last:
      self.add_to_buffer(trajectory.observation['fallen'][is_last])


@gin.configurable
class AverageSuccessMetric(py_metrics.StreamingMetric):
  """Computes average success rate for PointMass env in buffer_size episodes."""

  def __init__(self, name='AverageSuccess', buffer_size=10, batch_size=None):
    """Creates an AverageSuccessMetric."""
    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    super(AverageSuccessMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    return

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    is_last = np.where(trajectory.is_last())

    if is_last:
      succ = np.logical_and(
          np.logical_not(trajectory.observation['fallen'][is_last]),
          trajectory.reward[is_last] > 0.)
      self.add_to_buffer(succ)
