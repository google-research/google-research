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

"""Tests for aqt.jax.train_utils."""

from absl.testing import absltest
from absl.testing import parameterized

from aqt.jax import train_utils


class UpdateBoundsUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      # No updates before the start step of 5
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 0,
          'should_update': False
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 3,
          'should_update': False
      },

      # Updates expected every 10 steps after step 5 and no time else
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 5,
          'should_update': True
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 15,
          'should_update': True
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 25,
          'should_update': True
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 13,
          'should_update': False
      },
      {
          'frequency': 10,
          'start_step': 5,
          'current_step': 18,
          'should_update': False
      },

      # Test an update frequency of -1, which indicates no bounds update
      {
          'frequency': -1,
          'start_step': 5,
          'current_step': 5,
          'should_update': True
      },
      {
          'frequency': -1,
          'start_step': 5,
          'current_step': 6,
          'should_update': False
      },
      {
          'frequency': -1,
          'start_step': 5,
          'current_step': 4,
          'should_update': False
      },

      # Test a start step of -1, which indicates no bounds update at any step.
      {
          'frequency': -1,
          'start_step': -1,
          'current_step': 3,
          'should_update': False
      },
      {
          'frequency': 5,
          'start_step': -1,
          'current_step': 12,
          'should_update': False
      })
  def test_should_update_bounds(self, frequency, start_step, current_step,
                                should_update):
    self.assertEqual(
        train_utils.should_update_bounds(
            activation_bound_update_freq=frequency,
            activation_bound_start_step=start_step,
            step=current_step), should_update)


if __name__ == '__main__':
  absltest.main()
