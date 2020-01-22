# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Generic annealing schedules in Python.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin


@gin.configurable
class LinearSchedule(object):
  """Linear interpolation between initial_p and final_p.

    After `schedule_timesteps`, final_p is returned. Implementation derived from
    OpenAI Baselines V1.
  """

  def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
    """Constructor.

    Args:
      schedule_timesteps: (int) Number of timesteps for which to linearly anneal
        initial_p to final_p. Must be > 0.
      final_p: (float) Final output value.
      initial_p: (float) Initial output value.

    Raises:
      ValueError: If schedule_timesteps is not positive.
    """
    if schedule_timesteps <= 0:
      raise ValueError('schedule_timesteps must be positive.')
    self.schedule_timesteps = schedule_timesteps
    self.final_p = final_p
    self.initial_p = initial_p

  def value(self, timestep):
    """Computes the schedule value at the timestep.

    Args:
      timestep: (int/float) Unitless timestep parameter.
    Returns:
      Schedule scalar (float).
    Raises:
      ValueError: If timesteps is negative.
    """
    if timestep < 0:
      raise ValueError('timestep must be non-negative.')
    fraction = min(float(timestep) / self.schedule_timesteps, 1.0)
    return self.initial_p + fraction * (self.final_p - self.initial_p)
