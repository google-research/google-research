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

"""Tests for mon_minigrid."""
import os

from absl.testing import absltest
from absl.testing import parameterized
import gin
import gym
import gym_minigrid
import numpy as np

from minigrid_basics.envs import mon_minigrid


class MonMiniGridEnvTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(MonMiniGridEnvTest, cls).setUpClass()
    gin.parse_config_files_and_bindings([
        os.path.join(mon_minigrid.GIN_FILES_PREFIX,
                     '{}.gin'.format('classic_fourrooms'))
    ],
                                        bindings=[],
                                        skip_unknown=False)
    cls.env_id = mon_minigrid.register_environment()

  @parameterized.named_parameters(
      ('action_0', 0, np.array(0, dtype=int)),
      ('action_1', 1, np.array(1, dtype=int)),
      ('action_2', 2, np.array(2, dtype=int)),
      ('action_3', 3, np.array(3, dtype=int)),
  )
  def test_step_with_np_array_action_obs_renders_correctly(
      self, int_action, array_action):
    test_env = gym.make(self.env_id)
    test_env = gym_minigrid.wrappers.RGBImgObsWrapper(test_env)
    test_env = gym_minigrid.wrappers.ImgObsWrapper(test_env)
    test_env.reset()
    test_obs, test_reward, _, _ = test_env.step(array_action)

    reference_env = gym.make(self.env_id)
    reference_env = gym_minigrid.wrappers.RGBImgObsWrapper(reference_env)
    reference_env = gym_minigrid.wrappers.ImgObsWrapper(reference_env)
    reference_env.reset()
    reference_obs, reference_reward, _, _ = reference_env.step(int_action)

    np.testing.assert_array_equal(test_obs, reference_obs)
    self.assertEqual(test_reward, reference_reward)


if __name__ == '__main__':
  absltest.main()
