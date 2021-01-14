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

"""Calculate running mean and std for an array."""
import numpy as np


class RunningMeanStd(object):
  """Running mean from OpenAI.

  The implementation is based on the following link:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm


  Attributes:
    mean: An array of running mean values with a shape.
    set during initialization.
    var: An array of running variance values with a shape
            set during initialization.
    count: A scalar indicating number of population.
  """

  def __init__(self, epsilon=1e-4, shape=()):
    """Initialize mean, var, and cound."""
    self._mean = np.zeros(shape, 'float64')
    self._var = np.ones(shape, 'float64')
    self._count = epsilon

  @property
  def mean(self):
    """getter property for mean value."""
    return np.copy(self._mean)

  @property
  def var(self):
    """getter property for var value."""
    return np.copy(self._var)

  @property
  def count(self):
    """getter property for count value."""
    copy_count = self._count
    return copy_count

  def update(self, x):
    """Update mean, var, and count based on the current vector."""
    batch_mean = np.mean(x, axis=0)
    batch_var = np.var(x, axis=0)
    batch_count = x.shape[0]
    self._update_from_moments(batch_mean, batch_var, batch_count)

  def _update_from_moments(
      self,
      batch_mean,
      batch_var,
      batch_count):
    """A wrapper function to update running mean, variance, and count."""
    updated_values = self._update_mean_var_count_from_moments(
        mean=self._mean,
        var=self._var,
        count=self._count,
        batch_mean=batch_mean,
        batch_var=batch_var,
        batch_count=batch_count)

    self._mean, self._var, self._count = updated_values

  def _update_mean_var_count_from_moments(
      self,
      mean,
      var,
      count,
      batch_mean,
      batch_var,
      batch_count):
    """Update mean and std."""

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = m2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
