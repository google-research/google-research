# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Utility for loading the OpenAI Gym Fetch robotics environments."""

import gym
from gym.envs.robotics.fetch import push
from gym.envs.robotics.fetch import reach
import numpy as np


class FetchReachEnv(reach.FetchReachEnv):
  """Wrapper for the FetchReach environment."""

  def __init__(self):
    super(FetchReachEnv, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((20,), -np.inf),
        high=np.full((20,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space

  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchReachEnv, self).reset()
    self.observation_space = self._new_observation_space
    return self.observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchReachEnv, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)  # Default from Fetch environment.
    info = {}
    return self.observation(s), r, done, info

  def observation(self, observation):
    start_index = 0
    end_index = 3
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)
    g[start_index:end_index] = observation['desired_goal']
    return np.concatenate([s, g]).astype(np.float32)


class FetchPushEnv(push.FetchPushEnv):
  """Wrapper for the FetchPush environment."""

  def __init__(self):
    super(FetchPushEnv, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((50,), -np.inf),
        high=np.full((50,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space

  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchPushEnv, self).reset()
    self.observation_space = self._new_observation_space
    return self.observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchPushEnv, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)
    info = {}
    return self.observation(s), r, done, info

  def observation(self, observation):
    start_index = 3
    end_index = 6
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)
    g[:start_index] = observation['desired_goal']
    g[start_index:end_index] = observation['desired_goal']
    return np.concatenate([s, g]).astype(np.float32)


class FetchReachImage(reach.FetchReachEnv):
  """Wrapper for the FetchReach environment with image observations."""

  def __init__(self):
    self._dist = []
    self._dist_vec = []
    super(FetchReachImage, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)
    self.observation_space = self._new_observation_space
    self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def reset(self):
    if self._dist:  # if len(self._dist) > 0, ...
      self._dist_vec.append(self._dist)
    self._dist = []

    # generate the new goal image
    self.observation_space = self._old_observation_space
    s = super(FetchReachImage, self).reset()
    self.observation_space = self._new_observation_space
    self._goal = s['desired_goal'].copy()

    for _ in range(10):
      hand = s['achieved_goal']
      obj = s['desired_goal']
      delta = obj - hand
      a = np.concatenate([np.clip(10 * delta, -1, 1), [0.0]])
      s, _, _, _ = super(FetchReachImage, self).step(a)

    self._goal_img = self.observation(s)

    self.observation_space = self._old_observation_space
    s = super(FetchReachImage, self).reset()
    self.observation_space = self._new_observation_space
    img = self.observation(s)
    dist = np.linalg.norm(s['achieved_goal'] - self._goal)
    self._dist.append(dist)
    return np.concatenate([img, self._goal_img])

  def step(self, action):
    s, _, _, _ = super(FetchReachImage, self).step(action)
    dist = np.linalg.norm(s['achieved_goal'] - self._goal)
    self._dist.append(dist)
    done = False
    r = float(dist < 0.05)
    info = {}
    img = self.observation(s)
    return np.concatenate([img, self._goal_img]), r, done, info

  def observation(self, observation):
    self.sim.data.site_xpos[0] = 1_000_000
    img = self.render(mode='rgb_array', height=64, width=64)
    return img.flatten()

  def _viewer_setup(self):
    super(FetchReachImage, self)._viewer_setup()
    self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.5])
    self.viewer.cam.distance = 0.8
    self.viewer.cam.azimuth = 180
    self.viewer.cam.elevation = -30


class FetchPushImage(push.FetchPushEnv):
  """Wrapper for the FetchPush environment with image observations."""

  def __init__(self, camera='camera2', start_at_obj=True, rand_y=False):
    self._start_at_obj = start_at_obj
    self._rand_y = rand_y
    self._camera_name = camera
    self._dist = []
    self._dist_vec = []
    super(FetchPushImage, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)
    self.observation_space = self._new_observation_space
    self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def _move_hand_to_obj(self):
    s = super(FetchPushImage, self)._get_obs()
    for _ in range(100):
      hand = s['observation'][:3]
      obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
      delta = obj - hand
      if np.linalg.norm(delta) < 0.06:
        break
      a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
      s, _, _, _ = super(FetchPushImage, self).step(a)

  def reset(self):
    if self._dist:  # if len(self._dist) > 0 ...
      self._dist_vec.append(self._dist)
    self._dist = []

    # generate the new goal image
    self.observation_space = self._old_observation_space
    s = super(FetchPushImage, self).reset()
    self.observation_space = self._new_observation_space
    # Randomize object position
    for _ in range(8):
      super(FetchPushImage, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    if not self._rand_y:
      object_qpos[1] = 0.75
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    self._move_hand_to_obj()
    self._goal_img = self.observation(s)
    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()
    self._goal = block_xyz[:2].copy()

    self.observation_space = self._old_observation_space
    s = super(FetchPushImage, self).reset()
    self.observation_space = self._new_observation_space
    for _ in range(8):
      super(FetchPushImage, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    object_qpos[:2] = np.array([1.15, 0.75])
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    if self._start_at_obj:
      self._move_hand_to_obj()
    else:
      for _ in range(5):
        super(FetchPushImage, self).step(self.action_space.sample())

    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
    img = self.observation(s)
    dist = np.linalg.norm(block_xyz[:2] - self._goal)
    self._dist.append(dist)
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()
    return np.concatenate([img, self._goal_img])

  def step(self, action):
    s, _, _, _ = super(FetchPushImage, self).step(action)
    block_xy = self.sim.data.get_joint_qpos('object0:joint')[:2]
    dist = np.linalg.norm(block_xy - self._goal)
    self._dist.append(dist)
    done = False
    r = float(dist < 0.05)  # Taken from the original task code.
    info = {}
    img = self.observation(s)
    return np.concatenate([img, self._goal_img]), r, done, info

  def observation(self, observation):
    self.sim.data.site_xpos[0] = 1_000_000
    img = self.render(mode='rgb_array', height=64, width=64)
    return img.flatten()

  def _viewer_setup(self):
    super(FetchPushImage, self)._viewer_setup()
    if self._camera_name == 'camera1':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 180
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera2':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.65
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    else:
      raise NotImplementedError
