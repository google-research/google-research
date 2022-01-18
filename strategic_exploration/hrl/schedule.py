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

class LinearSchedule(object):

  @classmethod
  def from_config(cls, config):
    """Constructs a LinearSchedule from the config.

        Args:
            config (Config): should contain all of the arguments in the
              constructor
    """
    return cls(config.begin, config.end, config.total_steps)

  def __init__(self, begin, end, total_steps):
    """
        Args:
            begin: initial value
            end: final value
            nsteps: number of steps between begin and end
    """
    self._value = float(begin)
    self._begin = float(begin)
    self._end = float(end)
    self._total_steps = total_steps

  def step(self, take_step=True):
    """Updates the value by a single step and returns the new value.

    If
        take_step is False, then the value is not updated, and the current
        value is returned.

        Args:
            take_step (bool): controls whether or not the value is updated

        Returns:
            float
        """
    step_size = float(self._begin - self._end) / self._total_steps
    self._value = max(self._end, self._value - step_size)
    return self._value

  def get_value(self, step_number):
    """Returns the value that this schedule would give after step_number

        steps. Ignores any previous step calls.

        Args:
            step_number (int): the number of steps from the beginning

        Returns:
            float
        """
    step_size = float(self._begin - self._end) / self._total_steps
    value = max(self._begin - step_size * step_number, self._end)
    return value
