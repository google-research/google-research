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

"""Tests for halfcheetah_motor_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf

from norml.envs import halfcheetah_motor_env


class HalfcheetahMotorEnvTest(tf.test.TestCase):

  def test_compare_swapped_action_to_original(self):
    """Compares the behavior of swapped_action environment with original.

    The swapped action environment performs by swapping the action dimensions,
    and should behave exactly the same as the original environment when the
    action dimensions are manually switched.
    """
    original_env = halfcheetah_motor_env.HalfcheetahMotorEnv(swap_action=False)
    swapped_env = halfcheetah_motor_env.HalfcheetahMotorEnv(swap_action=True)
    original_env.seed(0)
    swapped_env.seed(0)

    obs_original = original_env.reset()
    obs_swapped = swapped_env.reset()
    self.assertAllClose(
        obs_original, obs_swapped)

    for _ in range(10):
      action_swapped = swapped_env.action_space.sample()
      action_original = action_swapped.copy()
      action_original[0], action_original[3] = action_original[
          3], action_original[0]
      obs_original, reward_original, done_original, _ = original_env.step(
          action_original)
      obs_swapped, reward_swapped, done_swapped, _ = swapped_env.step(
          action_swapped)

      self.assertAllClose(
          obs_original, obs_swapped)
      self.assertAlmostEqual(reward_original, reward_swapped)
      self.assertEqual(done_original, done_swapped)


if __name__ == "__main__":
  tf.test.main()
