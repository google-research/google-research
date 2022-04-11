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

"""Sawyer environment for pushing objects."""

import metaworld.envs.mujoco.cameras as camera_configs
from metaworld.google import glfw

import mujoco_py
import numpy as np

from collections import OrderedDict
from gym.spaces import Dict, Box

from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from metaworld.envs.mujoco.utils.rotation import euler2quat
from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE

sideview_cam = camera_configs.create_sawyer_camera_init(
    lookat=(0.2, 0.75, 0.4),
    distance=0.8,
    elevation=-55,
    azimuth=180,
    trackbodyid=-1,
)

topview_cam = camera_configs.create_sawyer_camera_init(
    lookat=(0., 1.0, 0.5),
    distance=0.6,
    elevation=-45,
    azimuth=270,
    trackbodyid=-1,
)


# list of changes
# object position has been changed to have lower variance
# the constant for pushing reward has been changed from 1000 -> 10
# added reset_goal function
# the observation "with_goal" has been changed
class SawyerReachPushPickPlaceEnv(SawyerXYZEnv):

  def __init__(
      self,
      random_init=False,
      task_types=['pick_place', 'reach', 'push'],
      task_type='pick_place',
      obs_type='plain',
      goal_low=(-0.1, 0.8, 0.05),
      goal_high=(0.1, 0.9, 0.3),
      liftThresh=0.04,
      sampleMode='equal',
      rotMode='fixed',  #'fixed',
      **kwargs):
    self.quick_init(locals())

    hand_low = (-0.5, 0.40, 0.05)
    hand_high = (0.5, 1, 0.5)
    obj_low = (-0.02, 0.58, 0.02)
    obj_high = (0.02, 0.62, 0.02)

    SawyerXYZEnv.__init__(
        self,
        frame_skip=5,
        action_scale=1. / 100,
        hand_low=hand_low,
        hand_high=hand_high,
        model_name=self.model_name,
        **kwargs)
    self.task_type = task_type
    self.init_config = {
        'obj_init_angle': .3,
        'obj_init_pos': np.array([0, 0.6, 0.02]),
        'hand_init_pos': np.array([0, .6, .2]),
    }
    # we only do one task from [pick_place, reach, push]
    # per instance of SawyerReachPushPickPlaceEnv.
    # Please only set task_type from constructor.
    if self.task_type == 'pick_place':
      self.goal = np.array([0.1, 0.8, 0.2])
    elif self.task_type == 'reach':
      self.goal = np.array([-0.1, 0.8, 0.2])
    elif self.task_type == 'push':
      self.goal = np.array([0.1, 0.8, 0.02])
    else:
      raise NotImplementedError
    self.obj_init_angle = self.init_config['obj_init_angle']
    self.obj_init_pos = self.init_config['obj_init_pos']
    self.hand_init_pos = self.init_config['hand_init_pos']

    assert obs_type in OBS_TYPE
    self.obs_type = obs_type

    if goal_low is None:
      goal_low = self.hand_low

    if goal_high is None:
      goal_high = self.hand_high

    self.random_init = random_init
    self.liftThresh = liftThresh
    self.max_path_length = 150
    self.rotMode = rotMode
    self.sampleMode = sampleMode
    self.task_types = task_types
    if rotMode == 'fixed':
      self.action_space = Box(
          np.array([-1, -1, -1, -1]),
          np.array([1, 1, 1, 1]),
      )
    elif rotMode == 'rotz':
      self.action_rot_scale = 1. / 50
      self.action_space = Box(
          np.array([-1, -1, -1, -np.pi, -1]),
          np.array([1, 1, 1, np.pi, 1]),
      )
    elif rotMode == 'quat':
      self.action_space = Box(
          np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
          np.array([1, 1, 1, 2 * np.pi, 1, 1, 1, 1]),
      )
    else:
      self.action_space = Box(
          np.array([-1, -1, -1, -np.pi / 2, -np.pi / 2, 0, -1]),
          np.array([1, 1, 1, np.pi / 2, np.pi / 2, np.pi * 2, 1]),
      )
    self.obj_and_goal_space = Box(
        np.hstack((obj_low, goal_low)),
        np.hstack((obj_high, goal_high)),
    )
    self.goal_space = Box(np.array(goal_low), np.array(goal_high))
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
          np.hstack((self.hand_low, obj_low, goal_low)),
          np.hstack((self.hand_high, obj_high, goal_high)),
      )
    else:
      raise NotImplementedError('If you want to use an observation\
                with_obs_idx, please discretize the goal space after instantiate an environment.'
                               )
    self.num_resets = 0
    self.reset()

  def get_goal(self):
    return {
        'state_desired_goal': self._state_goal,
    }

  @property
  def model_name(self):
    return get_asset_full_path(
        'sawyer_xyz/sawyer_reach_push_pick_and_place.xml')

  def step(self, action):
    if self.rotMode == 'euler':
      action_ = np.zeros(7)
      action_[:3] = action[:3]
      action_[3:] = euler2quat(action[3:6])
      self.set_xyz_action_rot(action_)
    elif self.rotMode == 'fixed':
      self.set_xyz_action(action[:3])
    elif self.rotMode == 'rotz':
      self.set_xyz_action_rotz(action[:4])
    else:
      self.set_xyz_action_rot(action[:7])
    self.do_simulation([action[-1], -action[-1]])
    # The marker seems to get reset every time you do a simulation
    self._set_goal_marker(self._state_goal)
    ob = self._get_obs()
    obs_dict = self._get_obs_dict()
    reward, reachRew, reachDist, pushRew, pushDist, pickRew, placeRew, placingDist = self.compute_reward(
        action, obs_dict, mode=self.rewMode, task_type=self.task_type)
    self.curr_path_length += 1

    #info = self._get_info()
    if self.curr_path_length == self.max_path_length:
      done = True
    else:
      done = False

    goal_dist = placingDist if self.task_type == 'pick_place' else pushDist
    if self.task_type == 'reach':
      success = float(reachDist <= 0.05)
    else:
      success = float(goal_dist <= 0.07)
    info = {
        'reachDist': reachDist,
        'pickRew': pickRew,
        'epRew': reward,
        'goalDist': goal_dist,
        'success': success
    }
    info['goal'] = self._state_goal
    return ob, reward, done, info

  def _get_obs(self):
    hand = self.get_endeff_pos()
    objPos = self.data.get_geom_xpos('objGeom')
    flat_obs = np.concatenate(
        (hand, objPos - hand))  # delta position from the hand
    if self.obs_type == 'with_goal_and_id':
      return np.concatenate([flat_obs, self._state_goal, self._state_goal_idx])
    elif self.obs_type == 'with_goal':
      return np.concatenate([flat_obs, self._state_goal - objPos
                            ])  # delta position of the goal from the object
    elif self.obs_type == 'plain':
      return np.concatenate([
          flat_obs,
      ])  # TODO ZP do we need the concat?
    else:
      return np.concatenate([flat_obs, self._state_goal_idx])

  def _get_obs_dict(self):
    hand = self.get_endeff_pos()
    objPos = self.data.get_geom_xpos('objGeom')
    flat_obs = np.concatenate((hand, objPos))
    return dict(
        state_observation=flat_obs,
        state_desired_goal=self._state_goal,
        state_achieved_goal=objPos,
    )

  def _get_info(self):
    pass

  def _set_goal_marker(self, goal):
    """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
    self.data.site_xpos[self.model.site_name2id('goal_{}'.format(
        self.task_type))] = (
            goal[:3])
    for task_type in self.task_types:
      if task_type != self.task_type:
        self.data.site_xpos[self.model.site_name2id(
            'goal_{}'.format(task_type))] = (
                np.array([10.0, 10.0, 10.0]))

  def _set_objCOM_marker(self):
    """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
    objPos = self.data.get_geom_xpos('objGeom')
    self.data.site_xpos[self.model.site_name2id('objSite')] = (objPos)

  def _set_obj_xyz(self, pos):
    qpos = self.data.qpos.flat.copy()
    qvel = self.data.qvel.flat.copy()
    qpos[9:12] = pos.copy()
    qvel[9:15] = 0
    self.set_state(qpos, qvel)

  def sample_goals(self, batch_size):
    # Required by HER-TD3
    goals = self.sample_goals_(batch_size)
    if self.discrete_goal_space is not None:
      goals = [self.discrete_goals[g].copy() for g in goals]
    return {
        'state_desired_goal': goals,
    }

  def sample_task(self):
    idx = self.sample_goals_(1)
    return self.discrete_goals[idx]

  def adjust_initObjPos(self, orig_init_pos):
    #This is to account for meshes for the geom and object are not aligned
    #If this is not done, the object could be initialized in an extreme position
    diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
    adjustedPos = orig_init_pos[:2] + diff

    #The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
    return [
        adjustedPos[0], adjustedPos[1],
        self.data.get_geom_xpos('objGeom')[-1]
    ]

  def reset_goal(self, goal=None):
    # choose one of the discrete ends for a goal if none is given
    if goal is None:
      discrete_goal_list = np.array([[0.0, 0.9, 0.02], [0.0, 0.3, 0.02],
                                     [-0.3, 0.6, 0.02], [0.3, 0.6, 0.02]])
      goal_idx = [
          0, 2, 3
      ][np.random.choice(3)]  # skip the back goal as it is not reachable
      goal = discrete_goal_list[goal_idx]

      solve_reverse_task = np.random.choice(
          1)  # goal-object reversal to simulate potential reset problems
      if solve_reverse_task:
        goal = np.concatenate([goal[:2], [self.obj_init_pos[-1]]])
        self.obj_init_pos, goal = goal, self.obj_init_pos
        self._set_obj_xyz(self.obj_init_pos)

    # update the chosen goal in environment
    self._state_goal = np.concatenate((goal[:2], [self.obj_init_pos[-1]]))
    self._set_goal_marker(self._state_goal)

    # update quantities for reward calculation
    rightFinger, leftFinger = self.get_site_pos(
        'rightEndEffector'), self.get_site_pos('leftEndEffector')
    self.init_fingerCOM = (rightFinger + leftFinger) / 2
    self.maxReachDist = np.linalg.norm(self.init_fingerCOM -
                                       np.array(self._state_goal))
    self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] -
                                      np.array(self._state_goal)[:2])
    self.maxPlacingDist = np.linalg.norm(
        np.array(
            [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) -
        np.array(self._state_goal)) + self.heightTarget
    self.target_rewards = [
        1000 * self.maxPlacingDist + 1000 * 2,
        1000 * self.maxReachDist + 1000 * 2, 10 * self.maxPushDist + 10 * 2
    ]

    if self.task_type == 'reach':
      idx = 1
    elif self.task_type == 'push':
      idx = 2
    else:
      idx = 0
    self.target_reward = self.target_rewards[idx]

  def reset_model(self):
    self._reset_hand()
    self._state_goal = self.goal.copy()
    self.obj_init_pos = self.adjust_initObjPos(self.init_config['obj_init_pos'])
    self.obj_init_angle = self.init_config['obj_init_angle']
    self.objHeight = self.data.get_geom_xpos('objGeom')[2]
    self.heightTarget = self.objHeight + self.liftThresh
    if self.random_init:
      goal_pos = np.random.uniform(
          self.obj_and_goal_space.low,
          self.obj_and_goal_space.high,
          size=(self.obj_and_goal_space.low.size),
      )

      if self.task_type == 'push':
        self.obj_init_pos = np.concatenate(
            (goal_pos[:2], [self.obj_init_pos[-1]]))
      else:
        self.obj_init_pos = goal_pos[:3]

    self._set_obj_xyz(self.obj_init_pos)
    self.reset_goal()  # segregate the call to goal resetting for reuse later

    #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
    self.curr_path_length = 0
    self.num_resets += 1
    return self._get_obs()

  def reset_model_to_idx(self, idx):
    raise NotImplementedError('This API is deprecated! Please explicitly\
            call `set_goal_` then reset the environment.')

  def _reset_hand(self):
    for _ in range(10):
      self.data.set_mocap_pos('mocap', self.hand_init_pos)
      self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
      self.do_simulation([-1, 1], self.frame_skip)
    rightFinger, leftFinger = self.get_site_pos(
        'rightEndEffector'), self.get_site_pos('leftEndEffector')
    self.init_fingerCOM = (rightFinger + leftFinger) / 2
    self.pickCompleted = False

  def get_site_pos(self, siteName):
    _id = self.model.site_names.index(siteName)
    return self.data.site_xpos[_id].copy()

  def compute_rewards(self, actions, obsBatch):
    #Required by HER-TD3
    assert isinstance(obsBatch, dict) == True
    obsList = obsBatch['state_observation']
    rewards = [
        self.compute_reward(action, obs, task_type=self.task_type)[0]
        for action, obs in zip(actions, obsList)
    ]
    return np.array(rewards)

  def compute_reward(self, actions, obs, mode='general', task_type='reach'):
    if isinstance(obs, dict):
      obs = obs['state_observation']

    objPos = obs[3:6]

    rightFinger, leftFinger = self.get_site_pos(
        'rightEndEffector'), self.get_site_pos('leftEndEffector')
    fingerCOM = (rightFinger + leftFinger) / 2

    heightTarget = self.heightTarget
    goal = self._state_goal

    def compute_reward_reach(actions, obs, mode):
      c1 = 1000
      c2 = 0.01
      c3 = 0.001
      reachDist = np.linalg.norm(fingerCOM - goal)
      # reachRew = -reachDist
      # if reachDist < 0.1:
      #     reachNearRew = 1000*(self.maxReachDist - reachDist) + c1*(np.exp(-(reachDist**2)/c2) + np.exp(-(reachDist**2)/c3))
      # else:
      #     reachNearRew = 0.
      reachRew = c1 * (self.maxReachDist - reachDist) + c1 * (
          np.exp(-(reachDist**2) / c2) + np.exp(-(reachDist**2) / c3))
      reachRew = max(reachRew, 0)
      # reachNearRew = max(reachNearRew,0)
      # reachRew = -reachDist
      reward = reachRew  # + reachNearRew
      return [reward, reachRew, reachDist, None, None, None, None, None]

    def compute_reward_push(actions, obs, mode):
      c1 = 10
      c2 = 0.01
      c3 = 0.001
      assert np.all(goal == self.get_site_pos('goal_push'))
      reachDist = np.linalg.norm(fingerCOM - objPos)
      pushDist = np.linalg.norm(objPos[:2] - goal[:2])
      reachRew = -reachDist
      if reachDist < 0.05:
        # pushRew = -pushDist
        pushRew = c1 * (self.maxPushDist - pushDist) + c1 * (
            np.exp(-(pushDist**2) / c2) + np.exp(-(pushDist**2) / c3))
        pushRew = max(pushRew, 0)
      else:
        pushRew = 0
      reward = self.reach_reward_scale * reachRew + pushRew
      return [reward, reachRew, reachDist, pushRew, pushDist, None, None, None]

    def compute_reward_pick_place(actions, obs, mode):
      reachDist = np.linalg.norm(objPos - fingerCOM)
      placingDist = np.linalg.norm(objPos - goal)
      assert np.all(goal == self.get_site_pos('goal_pick_place'))

      def reachReward():
        reachRew = -reachDist  # + min(actions[-1], -1)/50
        reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
        zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
        if reachDistxy < 0.05:  #0.02
          reachRew = -reachDist
        else:
          reachRew = -reachDistxy - 2 * zRew
        #incentive to close fingers when reachDist is small
        if reachDist < 0.05:
          reachRew = -reachDist + max(actions[-1], 0) / 50
        return reachRew, reachDist

      def pickCompletionCriteria():
        tolerance = 0.01
        if objPos[2] >= (heightTarget - tolerance):
          return True
        else:
          return False

      if pickCompletionCriteria():
        self.pickCompleted = True

      def objDropped():
        return (objPos[2] <
                (self.objHeight + 0.005)) and (placingDist >
                                               0.02) and (reachDist > 0.02)
        # Object on the ground, far away from the goal, and from the gripper
        #Can tweak the margin limits

      def objGrasped(thresh=0):
        sensorData = self.data.sensordata
        return (sensorData[0] > thresh) and (sensorData[1] > thresh)

      def orig_pickReward():
        # hScale = 50
        hScale = 100
        # hScale = 1000
        if self.pickCompleted and not (objDropped()):
          return hScale * heightTarget
        # elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
        elif (reachDist < 0.1) and (objPos[2] > (self.objHeight + 0.005)):
          return hScale * min(heightTarget, objPos[2])
        else:
          return 0

      def general_pickReward():
        hScale = 50
        if self.pickCompleted and objGrasped():
          return hScale * heightTarget
        elif objGrasped() and (objPos[2] > (self.objHeight + 0.005)):
          return hScale * min(heightTarget, objPos[2])
        else:
          return 0

      def placeReward():
        # c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if mode == 'general':
          cond = self.pickCompleted and objGrasped()
        else:
          cond = self.pickCompleted and (reachDist < 0.1) and not (objDropped())
        if cond:
          placeRew = 1000 * (self.maxPlacingDist - placingDist) + c1 * (
              np.exp(-(placingDist**2) / c2) + np.exp(-(placingDist**2) / c3))
          placeRew = max(placeRew, 0)
          return [placeRew, placingDist]
        else:
          return [0, placingDist]

      reachRew, reachDist = reachReward()
      if mode == 'general':
        pickRew = general_pickReward()
      else:
        pickRew = orig_pickReward()
      placeRew, placingDist = placeReward()
      assert ((placeRew >= 0) and (pickRew >= 0))
      reward = self.reach_reward_scale * reachRew + pickRew + placeRew
      return [
          reward, reachRew, reachDist, None, None, pickRew, placeRew,
          placingDist
      ]

    if task_type == 'reach':
      return compute_reward_reach(actions, obs, mode)
    elif task_type == 'push':
      return compute_reward_push(actions, obs, mode)
    else:
      return compute_reward_pick_place(actions, obs, mode)

  def get_diagnostics(self, paths, prefix=''):
    statistics = OrderedDict()
    return statistics

  def log_diagnostics(self, paths=None, logger=None):
    pass


class SawyerObject(SawyerReachPushPickPlaceEnv):

  def set_max_path_length(self, length):
    self.max_path_length = length

  def set_camera_view(self, view):
    self._camera_view = view

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

  def close(self):
    if self.viewer is not None and not isinstance(
        self.viewer, mujoco_py.MjRenderContextOffscreen):
      glfw.destroy_window(self.viewer.window)
    self.viewer = None
