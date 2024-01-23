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

from minigrid_basics.custom_wrappers import tabular_wrapper
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
      ('use_rgb_true', True),
      ('use_rgb_false', False),
  )
  def test_observation_space_matches_observation(self, use_rgb):
    env = gym.make(self.env_id)
    env = gym_minigrid.wrappers.RGBImgObsWrapper(env)
    env = tabular_wrapper.TabularWrapper(env, get_rgb=use_rgb)
    env.reset()
    obs, _, _, _ = env.step(0)
    for key in env.observation_space:
      obs_space_shape = env.observation_space.spaces[key].shape
      if obs_space_shape == tuple():
        self.assertEqual(type(obs[key]), int)
      else:
        self.assertEqual(obs[key].shape, obs_space_shape)


if __name__ == '__main__':
  absltest.main()
