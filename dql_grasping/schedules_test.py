# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Tests for dql_grasping.schedule."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from dql_grasping import schedules
from tensorflow.python.platform import test


class SchedulesTest(parameterized.TestCase, test.TestCase):

  @parameterized.named_parameters(
      ('linear1', 1000, 1.0, 0., 500, 0.5),
      ('linear2', 1000, 1.0, 0.5, 500, 0.75),
      ('linear3', 1000, 1.0, 0.5, 500, 0.75))
  def testLinear(self, schedule_timesteps, initial_p, final_p, step, expected):
    schedule = schedules.LinearSchedule(schedule_timesteps, initial_p, final_p)
    self.assertEqual(schedule.value(step), expected)

  @parameterized.named_parameters(
      ('badlinear1', -1000, 1.0, 0., 500, 0.5),
      ('badlinear2', 1000, 1.0, 0.5, -500, 0.75))
  def testInvalid(self, schedule_timesteps, initial_p, final_p, step, expected):
    with self.assertRaises(ValueError):
      schedule = schedules.LinearSchedule(
          schedule_timesteps, initial_p, final_p)
      schedule.value(step)


if __name__ == '__main__':
  test.main()
