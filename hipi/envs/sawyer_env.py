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

"""Implements tasks on the Sawyer manipulation environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from envs import multitask
import gin
import gym
from metaworld.envs.mujoco.sawyer_xyz import sawyer_reach_push_pick_place
import numpy as np
import tensorflow as tf


### Reach and Push
@gin.configurable
class SawyerDynamics(multitask.Dynamics):
  """Implements the dynamics for the Sawyer manipulation environment."""

  def __init__(self, random_init=False):
    env = sawyer_reach_push_pick_place.SawyerReachPushPickPlaceEnv(
        random_init=random_init)
    self._env = env
    self._observation_space = env.observation_space
    self._action_space = env.action_space

  def reset(self):
    return self._env.reset().astype(np.float32)

  def step(self, action):
    obs, rew, done, info = self._env.step(action)
    del rew, done, info
    return obs.astype(np.float32)


@gin.configurable
class SawyerReachTaskDistribution(multitask.TaskDistribution):
  """Implements the Sawyer reaching task."""

  def __init__(self, dynamics, margin=0.3, max_margin=0.1):
    self._dynamics = dynamics
    self._margin = margin
    self._max_margin = max_margin
    dim = 4 if margin == "tune" else 3
    self._task_space = gym.spaces.Box(
        low=np.full((dim,), -np.inf),
        high=np.full((dim,), np.inf),
        dtype=np.float32)

  def sample(self):
    goal_pos = self._dynamics._env.sample_goals_(1)[0]  # pylint: disable=protected-access
    if self._margin == "tune":
      margin = np.random.uniform(0, self._max_margin)
      task = np.concatenate([goal_pos, [margin]])
    else:
      task = goal_pos
    return task.astype(np.float32)

  def _evaluate(self, states, actions, tasks):
    hand_pos = states[:, :3]
    goal_pos = tasks[:, :3]
    if self._margin == "tune":
      margin = tasks[:, -1]
    else:
      margin = self._margin
    hand_goal_dist = tf.norm(hand_pos - goal_pos, axis=1)
    dones = hand_goal_dist < margin
    rew = tf.cast(dones, tf.float32) - 1.0
    return rew, dones

  def state_to_task(self, states):
    hand_pos = states[:, :3]
    return hand_pos
