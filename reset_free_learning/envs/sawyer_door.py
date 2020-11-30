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

"""Sawyer environment for opening and closing."""

from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
import metaworld.envs.mujoco.cameras as camera_configs
from metaworld.envs.mujoco.sawyer_xyz import SawyerDoorCloseEnv

import mujoco_py
import numpy as np

sideview_cam = camera_configs.create_sawyer_camera_init(
    lookat=(0.2, 0.6, 0.4),
    distance=1.2,
    elevation=-10,
    azimuth=150,
    trackbodyid=-1,
)

topview_cam = camera_configs.create_sawyer_camera_init(
    lookat=(0., 1.0, 0.5),
    distance=0.6,
    elevation=-45,
    azimuth=270,
    trackbodyid=-1,
)


class SawyerDoor(SawyerDoorCloseEnv):

  def __init__(self,
               random_init=False,
               obs_type='plain',
               goal_low=None,
               goal_high=None,
               rotMode='fixed',
               **kwargs):
    self.custom_reset_flag = False
    SawyerDoorCloseEnv.__init__(
        self,
        random_init=random_init,
        obs_type=obs_type,
        goal_low=goal_low,
        goal_high=goal_high,
        rotMode=rotMode,
        **kwargs)

    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    obj_low = (0.1, 0.95, 0.1)
    obj_high = (0.1, 0.95, 0.1)

    self.obj_and_goal_space = Box(
        np.array(obj_low),
        np.array(obj_high),
    )

    if self.obs_type == 'plain':
      self.observation_space = Box(
          np.hstack((
              self.hand_low,
              obj_low,
          )),
          np.hstack((
              self.hand_high,
              obj_high,
          )),
      )
    elif self.obs_type == 'with_goal':
      self.observation_space = Box(
          np.hstack((self.hand_low, obj_low, self.hand_low, obj_low)),
          np.hstack((self.hand_high, obj_high, self.hand_high, obj_high)),
      )
    else:
      raise NotImplementedError

    # angle -> handle mapping
    # -pi / 2 -> [-0.21, 0.69, 0.15]
    # 0 -> 0.2, 0.8, 0.15
    self.init_config = {
        'obj_init_angle': -np.pi / 2,  # default initial angle
        # 'obj_init_angle': 0,  # reset initial angle
        'obj_init_pos': np.array([0.1, 0.95, 0.1], dtype=np.float32),
        'hand_init_pos': np.array(
            [0, 0.5, 0.2],
            dtype=np.float32),  # [-0.00356643, 0.4132358, 0.2534339]
    }

    self.goal = np.array([0.2, 0.8, 0.15, 0.2, 0.8, 0.15])  # default goal
    # self.goal = np.array([-0.21, 0.69, 0.15, -0.21, 0.69, 0.15])  # reset goal
    self.obj_init_pos = self.init_config['obj_init_pos']
    self.obj_init_angle = self.init_config['obj_init_angle']
    self.hand_init_pos = self.init_config['hand_init_pos']

    # create the variable
    self.set_reward_type(reward_type='dense')

  def set_max_path_length(self, length):
    self.max_path_length = length

  def set_camera_view(self, view):
    self._camera_view = view

  def set_reward_type(self, reward_type):
    self._reward_type = reward_type

  def _get_viewer(self, mode):
    self.viewer = self._viewers.get(mode)
    if self.viewer is None:
      if 'rgb_array' in mode:
        self.viewer = mujoco_py.MjRenderContextOffscreen(
            self.sim, device_id=self.device_id)
        self.viewer_setup()
        self._viewers[mode] = self.viewer
    return super()._get_viewer(mode)

  def viewer_setup(self):
    if self._camera_view == 'topview':
      topview_cam(self.viewer.cam)
    elif self._camera_view == 'sideview':
      sideview_cam(self.viewer.cam)
    else:
      camera_configs.init_sawyer_camera_v1(self.viewer.cam)

  @property
  def model_name(self):
    return get_asset_full_path('sawyer_xyz/sawyer_door_pull_custom.xml')

  def _get_obs(self):
    hand = self.get_endeff_pos()
    objPos = self.data.get_geom_xpos('handle').copy()
    flat_obs = np.concatenate((hand, objPos))
    if self.obs_type == 'with_goal_and_id':
      return np.concatenate([flat_obs, self._state_goal, self._state_goal_idx])
    elif self.obs_type == 'with_goal':
      obs = np.concatenate([
          flat_obs,
          self._end_effector_goal,
          self._state_goal,
      ])
      return obs
    elif self.obs_type == 'plain':
      return np.concatenate([
          flat_obs,
      ])  # TODO ZP do we need the concat?
    else:
      return np.concatenate([flat_obs, self._state_goal_idx])

  # need to expose the default goal, useful for multi-goal settings
  def get_next_goal(self):
    return self.goal

  def reset_goal(self, goal=None):
    if goal is None:
      goal = self.get_next_goal()
      if goal.shape == (3,):
        goal = np.concatenate([goal, goal])

    self._state_goal = goal[3:]
    self._end_effector_goal = goal[:3]
    self._set_goal_marker(self._state_goal)
    self.sim.model.site_pos[self.model.site_name2id('goal')] = self._state_goal

  def reset_model(self):
    if not self.custom_reset_flag:
      self._reset_hand()
      self._state_goal = self.goal[3:]
      self.objHeight = self.data.get_geom_xpos('handle')[2]
      if self.random_init:
        # add noise to the initial position of the door
        initial_position = self.obj_init_angle
        initial_position += np.random.uniform(0, np.pi / 20)  # default noise
        # initial_position += np.random.uniform(-np.pi / 10, 0)  # reset noise
        self.sim.model.body_pos[self.model.body_name2id(
            'door')] = self.obj_init_pos
        self._set_obj_xyz(initial_position)

      self.reset_goal()

      self.curr_path_length = 0
      self.maxPullDist = np.linalg.norm(
          self.data.get_geom_xpos('handle')[:-1] - self._state_goal[:-1])
      self.target_reward = 10 * self.maxPullDist + 10 * 2

      return self._get_obs()
    else:
      self.custom_reset_flag = False
      return self.custom_reset()

  def compute_reward(self, actions, obs):
    if isinstance(obs, dict):
      obs = obs['state_observation']

    objPos = obs[3:6]
    rightFinger, leftFinger = self.get_site_pos(
        'rightEndEffector'), self.get_site_pos('leftEndEffector')
    fingerCOM = (rightFinger + leftFinger) / 2
    pullGoal = self._state_goal
    fingerGoal = self._end_effector_goal

    pullDist = np.linalg.norm(objPos - pullGoal)
    reachDist = np.linalg.norm(objPos - fingerCOM)
    fingerGoalDist = np.linalg.norm(fingerGoal - fingerCOM)

    if self._reward_type == 'dense':
      reachRew = -reachDist

      def reachCompleted():
        if reachDist < 0.05:
          return True
        else:
          return False

      if reachCompleted():
        self.reachCompleted = True
      else:
        self.reachCompleted = False

      def pullReward():
        c1 = 10
        c2 = 0.01
        c3 = 0.001
        if self.reachCompleted:
          pullRew = 10 * (self.maxPullDist - pullDist) + c1 * (
              np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3))
          fingerPullRew = 10 * (self.maxPullDist - fingerGoalDist) + c1 * (
              np.exp(-(fingerGoalDist**2) / c2) +
              np.exp(-(fingerGoalDist**2) / c3))
          pullRew = max(pullRew, 0)
          fingerPullRew = max(fingerPullRew, 0)
          return pullRew + fingerPullRew
        else:
          return 0

      pullRew = pullReward()
      reward = reachRew + pullRew

    if self._reward_type == 'sparse':
      reward = float(self.is_successful(obs=obs))

    return [reward, reachDist, pullDist]

  def is_successful(self, obs=None):
    if obs is None:
      obs = self._get_obs()[:6]

    return np.linalg.norm(
        obs -
        np.concatenate([self._end_effector_goal, self._state_goal])) <= 0.2

  def do_custom_reset(self, pos):
    self.custom_reset_flag = True
    self.custom_reset_pos = pos

  def custom_reset(self, pos=None):
    if pos is None:
      pos = self.custom_reset_pos

    hand_target_pos = pos[:3]
    handle_target_pos = pos[3:6]
    rotation_center = np.array([-0.06, 0.95,
                                0.1])  # the position of the joint is the center

    # door_mid_correction = np.array([0.1, -0.15, 0.05])
    # door_closed_pos = np.array([0.2, 0.8, 0.15]) - door_mid_correction
    # door_mid_pos = np.array([0.01778175, 0.6600862, 0.15]) - door_mid_correction
    # door_open_pos = np.array([-0.21, 0.69, 0.15]) - door_mid_correction

    # set end effector position
    for _ in range(10):
      self.data.set_mocap_pos('mocap', hand_target_pos)
      self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
      self.do_simulation([-1, 1], self.frame_skip)
    rightFinger, leftFinger = self.get_site_pos(
        'rightEndEffector'), self.get_site_pos('leftEndEffector')
    self.init_fingerCOM = (rightFinger + leftFinger) / 2
    self.reachCompleted = False

    # set door angle
    handle_target_pos[:2] -= rotation_center[:2]
    a, b = 0.26, -0.15  # vector from joint to handle
    x, y = handle_target_pos[0], handle_target_pos[1]
    initial_position = np.arctan2(y * a - x * b, x * a + y * b)
    initial_position += np.random.uniform(-np.pi / 20, np.pi /
                                          20)  # add noise for robustness?
    initial_position = np.clip(initial_position, -np.pi / 2, 0)
    self._set_obj_xyz(initial_position)

    # generic reset stuff
    self.reset_goal()
    self.sim.model.body_pos[self.model.body_name2id('door')] = self.obj_init_pos

    self.curr_path_length = 0
    self.maxPullDist = np.linalg.norm(
        self.data.get_geom_xpos('handle')[:-1] - self._state_goal[:-1])
    self.target_reward = 10 * self.maxPullDist + 10 * 2

    return self._get_obs()
