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

"""Tests for move_point_env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

from norml.envs import move_point_env


class MovePointEnvTest(tf.test.TestCase):

  def test_step(self):
    env = move_point_env.MovePointEnv((-.5, -.3), (.6, .7))

    observation = env.reset()
    self.assertEqual(observation[0, 0], -.5)
    self.assertEqual(observation[0, 1], -.3)
    act = np.array([.1, .3])
    observation, _, _, _ = env.step(act)
    self.assertAlmostEqual(observation[0, 0], -.4)
    self.assertAlmostEqual(observation[0, 1], .0)

  def test_reward(self):
    env = move_point_env.MovePointEnv((0, 0), (1, 1))
    act = np.array([1, 1])
    _, reward_pos, _, _ = env.step(act)
    env.reset()
    act = np.array([-1, -1])
    _, reward_neg, _, _ = env.step(act)
    # moving towards the target should increase the reward
    self.assertGreater(reward_pos, reward_neg)

  def test_reset(self):
    env = move_point_env.MovePointEnv((.3, .5), (.2, .1))

    observation = env.reset()
    observation_2 = env.reset()

    # observation shouldn't change if reset is called multiple times
    self.assertAlmostEqual(np.sum((observation - observation_2)**2), 0)

  def test_trial_length(self):
    np.random.seed(12345)
    trial_length = np.random.randint(10, 200)
    env = move_point_env.MovePointEnv((.3, .5), (.2, .1),
                                      trial_length=trial_length)
    ctr = 0
    stop = False
    while not stop:
      act = np.random.randn(2)
      _, _, stop, _ = env.step(act)
      ctr += 1
      self.assertLessEqual(ctr, trial_length)
    self.assertEqual(ctr, trial_length)

  def test_render_mode(self):
    env = move_point_env.MovePointEnv((.1, .2), (.4, .5))

    with self.assertRaises(ValueError):
      env.render(mode='unsupported')

  def test_render(self):
    env = move_point_env.MovePointEnv((.1, .2), (.4, .5))
    stop = False
    while not stop:
      act = np.random.randn(2)
      _, _, stop, _ = env.step(act)
    render = env.render()
    # image should have 3 color channels
    self.assertEqual(len(render.shape), 3)
    self.assertGreaterEqual(render.shape[2], 3)
    # images should be decently sized
    self.assertGreaterEqual(render.shape[0], 100)
    self.assertGreaterEqual(render.shape[1], 100)

  def test_move_in_right_direction(self):
    env = move_point_env.MovePointEnv((.1, .2), (.4, .5))

    observation = env.reset()
    stop = False
    while not stop:
      act = np.random.randn(2)
      old_observation = observation
      observation, reward, stop, _ = env.step(act)
      new_dist = np.sum((observation - (old_observation + act))**2)
      old_dist = np.sum(act**2)
      self.assertLessEqual(new_dist, old_dist + 1e-6)
      self.assertFalse(np.isnan(reward))
      self.assertEqual(observation.shape[0], 1)
      self.assertEqual(observation.shape[1], 2)
      self.assertFalse(np.isnan(observation).any())

  def test_range(self):
    with self.assertRaises(ValueError):
      #  [-2, 2] range for all locations
      _ = move_point_env.MovePointEnv((3, .2), (.4, .5))
    with self.assertRaises(ValueError):
      #  [-2, 2] range for all locations
      _ = move_point_env.MovePointEnv((.3, 5), (.4, .5))
    with self.assertRaises(ValueError):
      #  [-2, 2] range for all locations
      _ = move_point_env.MovePointEnv((.3, .5), (4, .5))
    with self.assertRaises(ValueError):
      #  [-2, 2] range for all locations
      _ = move_point_env.MovePointEnv((.3, .5), (.4, 5))

  def test_bounds(self):
    env = move_point_env.MovePointEnv((0, 0), (0, 0))

    observation = env.reset()
    for _ in range(1000):
      act = np.array([.1, 0])
      observation, _, _, _ = env.step(act)
      self.assertLessEqual(np.abs(observation).max(), 2)


if __name__ == '__main__':
  tf.test.main()
