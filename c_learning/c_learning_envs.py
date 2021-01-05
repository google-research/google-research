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

"""Load and wrap the d4rl environments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
import gin
import gym
from metaworld.envs.mujoco import sawyer_xyz
import mujoco_py
import numpy as np
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

os.environ['SDL_VIDEODRIVER'] = 'dummy'


# When collecting trajectory snippets for training, we use discount = 0 to
# decide when to break a trajectory; we don't use the step_type. For data
# collection, we therefore should set done=True only when the environment truly
# terminates, not when we've reached the goal.
# Eventually, we want to create the train_env by taking any gym_env or py_env,
# putting a learned goal-sampling wrapper around it, and then using that.


def load_sawyer_reach():
  gym_env = SawyerReach()
  env = suite_gym.wrap_env(
      gym_env,
      max_episode_steps=50,
  )
  return tf_py_environment.TFPyEnvironment(env)


@gin.configurable
def load_sawyer_push(random_init=False, wide_goals=False,
                     include_gripper=False):
  """Load the sawyer pushing (and picking) environment.

  Args:
    random_init: (bool) Whether to randomize the initial arm position.
    wide_goals: (bool) Whether to use a wider range of Y positions for goals.
      The Y axis parallels the ground, pointing from the robot to the table.
    include_gripper: (bool) Whether to include the gripper open/close state in
      the observation.
  Returns:
    tf_env: An environment.
  """
  if wide_goals:
    goal_low = (-0.1, 0.5, 0.05)
  else:
    goal_low = (-0.1, 0.8, 0.05)
  if include_gripper:
    gym_env = SawyerPushGripper(random_init=random_init, goal_low=goal_low)
  else:
    gym_env = SawyerPush(random_init=random_init, goal_low=goal_low)
  env = suite_gym.wrap_env(
      gym_env,
      max_episode_steps=151,
  )
  return tf_py_environment.TFPyEnvironment(env)


@gin.configurable
def load_sawyer_drawer(random_init=False):
  gym_env = SawyerDrawer(random_init=random_init)
  env = suite_gym.wrap_env(
      gym_env,
      max_episode_steps=151,
  )
  return tf_py_environment.TFPyEnvironment(env)


@gin.configurable
def load_sawyer_window(rotMode='fixed'):  # pylint: disable=invalid-name
  gym_env = SawyerWindow(rotMode=rotMode)
  env = suite_gym.wrap_env(
      gym_env,
      max_episode_steps=151,
  )
  return tf_py_environment.TFPyEnvironment(env)


def load_sawyer_faucet():
  gym_env = SawyerFaucet()
  env = suite_gym.wrap_env(
      gym_env,
      max_episode_steps=151,
  )
  return tf_py_environment.TFPyEnvironment(env)


def load(env_name):
  """Creates the training and evaluation environment.

  This method automatically detects whether we are using a subset of the
  observation for the goal and modifies the observation space to include the
  full state + partial goal.

  Args:
    env_name: (str) Name of the environment.
  Returns:
    tf_env, eval_tf_env, obs_dim: The training and evaluation environments.
  """
  if env_name == 'sawyer_reach':
    tf_env = load_sawyer_reach()
    eval_tf_env = load_sawyer_reach()
  elif env_name == 'sawyer_push':
    tf_env = load_sawyer_push()
    eval_tf_env = load_sawyer_push()
    eval_tf_env.envs[0]._env.gym.MODE = 'eval'  # pylint: disable=protected-access
  elif env_name == 'sawyer_drawer':
    tf_env = load_sawyer_drawer()
    eval_tf_env = load_sawyer_drawer()
  elif env_name == 'sawyer_window':
    tf_env = load_sawyer_window()
    eval_tf_env = load_sawyer_window()
  elif env_name == 'sawyer_faucet':
    tf_env = load_sawyer_faucet()
    eval_tf_env = load_sawyer_faucet()
  else:
    raise NotImplementedError('Unsupported environment: %s' % env_name)
  assert len(tf_env.envs) == 1
  assert len(eval_tf_env.envs) == 1

  # By default, the environment observation contains the current state and goal
  # state. By setting the obs_to_goal parameters, the use can specify that the
  # agent should only look at certain subsets of the goal state. The following
  # code modifies the environment observation to include the full state but only
  # the user-specified dimensions of the goal state.
  obs_dim = tf_env.observation_spec().shape[0] // 2
  try:
    start_index = gin.query_parameter('obs_to_goal.start_index')
  except ValueError:
    start_index = 0
  try:
    end_index = gin.query_parameter('obs_to_goal.end_index')
  except ValueError:
    end_index = None
  if end_index is None:
    end_index = obs_dim

  indices = np.concatenate([
      np.arange(obs_dim),
      np.arange(obs_dim + start_index, obs_dim + end_index)
  ])
  tf_env = tf_py_environment.TFPyEnvironment(
      wrappers.ObservationFilterWrapper(tf_env.envs[0], indices))
  eval_tf_env = tf_py_environment.TFPyEnvironment(
      wrappers.ObservationFilterWrapper(eval_tf_env.envs[0], indices))
  return (tf_env, eval_tf_env, obs_dim)


class SawyerReach(sawyer_xyz.SawyerReachPushPickPlaceEnv):
  """Wrapper for the sawyer_reach task."""

  def __init__(self):
    super(SawyerReach, self).__init__(task_type='reach')
    self.observation_space = gym.spaces.Box(
        low=np.full(12, -np.inf),
        high=np.full(12, np.inf),
        dtype=np.float32)

  def reset(self):
    goal = self.sample_goals(1)['state_desired_goal'][0]
    self.goal = goal
    self._state_goal = goal
    return self.reset_model()

  def step(self, action):
    s, r, done, info = super(SawyerReach, self).step(action)
    r = 0.0
    done = False
    return s, r, done, info

  def _get_obs(self):
    obs = super(SawyerReach, self)._get_obs()
    return np.concatenate([obs, self.goal, np.zeros(3)])


class SawyerPush(sawyer_xyz.SawyerReachPushPickPlaceEnv):
  """Wrapper for the sawyer_push task."""

  def __init__(self, random_init=False, goal_low=None):
    assert goal_low is not None

    super(SawyerPush, self).__init__(
        task_type='push', random_init=random_init, goal_low=goal_low)
    self.observation_space = gym.spaces.Box(
        low=np.full(12, -np.inf),
        high=np.full(12, np.inf),
        dtype=np.float32)

  @gin.configurable(module='SawyerPush')
  def reset(self,
            arm_goal_type='random',
            fix_z=False,
            fix_xy=False,
            fix_g=False,
            reset_puck=False,
            in_hand_prob=0,
            custom_eval=False,
            reset_to_puck_prob=0.0):
    assert arm_goal_type in ['random', 'puck', 'goal']
    if custom_eval and self.MODE == 'eval':
      arm_goal_type = 'goal'
      in_hand_prob = 0
      reset_to_puck_prob = 0.0
    self._arm_goal_type = arm_goal_type
    # The arm_goal seems to be set to some (dummy) value before we can reset
    # the environment.
    self._arm_goal = np.zeros(3)
    if fix_g:
      self._gripper_goal = np.array([0.016])
    else:
      self._gripper_goal = np.random.uniform(0, 0.04, (1,))
    obs = super(SawyerPush, self).reset()
    if reset_puck:
      puck_pos = self.sample_goals(1)['state_desired_goal'][0]
      puck_pos[2] = 0.015
    else:
      puck_pos = obs[3:6]
    # The following line ensures that the puck starts face-up, not on edge.
    self._set_obj_xyz_quat(puck_pos, 0.0)
    if np.random.random() < reset_to_puck_prob:
      obs = self._get_obs()
      self.data.set_mocap_pos('mocap', obs[3:6])
      self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
      for _ in range(10):
        self.do_simulation([-1, 1], self.frame_skip)

    if np.random.random() < in_hand_prob:
      for _ in range(10):
        obs, _, _, _ = self.step(np.array([0, 0, 0, 1]))
      self._set_obj_xyz_quat(obs[:3], 0.0)
    obs = self._get_obs()

    self.goal = self.sample_goals(1)['state_desired_goal'][0]
    if fix_z:
      self.goal[2] = 0.015
    if fix_xy:
      self.goal[:2] = obs[3:5]

    self._set_goal_marker(self.goal)
    self._state_goal = self.goal.copy()

    if arm_goal_type == 'random':
      self._arm_goal = self.sample_goals(1)['state_desired_goal'][0]
      if fix_z:
        self._arm_goal[2] = 0.015
    elif arm_goal_type == 'puck':
      self._arm_goal = obs[3:6]
    elif arm_goal_type == 'goal':
      self._arm_goal = self.goal.copy()
    else:
      raise NotImplementedError
    return self._get_obs()

  def step(self, action):
    try:
      s, r, done, info = super(SawyerPush, self).step(action)
    except mujoco_py.MujocoException as me:
      logging.info('MujocoException: %s', me)
      s = self.reset()
      info = {}

    r = 0.0
    done = False
    return s, r, done, info

  def _get_obs(self):
    obs = super(SawyerPush, self)._get_obs()
    obs = np.concatenate([obs, self._arm_goal, self.goal])
    return obs


class SawyerPushGripper(SawyerPush):
  """Wrapper for the sawyer_push task, including the gripper in the state."""

  MODE = 'train'

  def __init__(self, random_init=False, goal_low=None):
    assert goal_low is not None

    super(SawyerPushGripper, self).__init__(
        random_init=random_init, goal_low=goal_low)
    self.observation_space = gym.spaces.Box(
        low=np.full(14, -np.inf), high=np.full(14, np.inf), dtype=np.float32)

  def _get_obs(self):
    obs = super(SawyerPushGripper, self)._get_obs()
    gripper = self.get_gripper_pos()
    obs = np.concatenate(
        [obs, gripper, self._arm_goal, self.goal, self._gripper_goal])
    return obs


class SawyerWindow(sawyer_xyz.SawyerWindowCloseEnv):
  """Wrapper for the sawyer_window task."""

  def __init__(self, rotMode='fixed'):  # pylint: disable=invalid-name
    super(SawyerWindow, self).__init__(random_init=False, rotMode=rotMode)
    self.observation_space = gym.spaces.Box(
        low=np.full(12, -np.inf), high=np.full(12, np.inf), dtype=np.float32)

  def sample_goal(self):
    low = np.array([-0.09, 0.73, 0.15])
    high = np.array([0.09, 0.73, 0.15])
    return np.random.uniform(low, high)

  @gin.configurable(module='SawyerWindow')
  def reset(self, arm_goal_type='random', reset_puck=True):
    assert arm_goal_type in ['random', 'puck', 'goal']
    self.goal = self.sample_goal()
    self._state_goal = self.goal.copy()
    self._arm_goal = np.zeros(3)
    super(SawyerWindow, self).reset()
    # Randomize the window position
    pos = self.sim.model.body_pos[self.model.body_name2id('window')]
    if reset_puck:
      pos[0] = self.sample_goal()[0]
    else:
      pos[0] = 0.0
    self.sim.model.body_pos[self.model.body_name2id('window')] = pos
    another_pos = pos.copy()
    another_pos[1] += 0.03
    self.sim.model.body_pos[self.model.body_name2id(
        'window_another')] = another_pos

    # We have set the desired state of the window above. We have to step the
    # environment once (using a null-op action) for these changes to take
    # effect.
    obs, _, _, _ = self.step(np.zeros(4))
    if arm_goal_type == 'random':
      self._arm_goal = self.sample_goal()
    elif arm_goal_type == 'puck':
      self._arm_goal = obs[3:6]
    elif arm_goal_type == 'goal':
      self._arm_goal = self.goal.copy()
    else:
      raise NotImplementedError
    return self._get_obs()

  def step(self, action):
    try:
      s, r, done, info = super(SawyerWindow, self).step(action)
    except mujoco_py.MujocoException as me:
      logging.info('MujocoException: %s', me)
      s = self.reset()
      info = {}
    r = 0.0
    done = False
    return s, r, done, info

  def _get_obs(self):
    obs = super(SawyerWindow, self)._get_obs()
    return np.concatenate([obs, self._arm_goal, self.goal])


class SawyerDrawer(sawyer_xyz.SawyerDrawerOpenEnv):
  """Wrapper for the sawyer_drawer task."""

  def __init__(self, random_init=False):
    super(SawyerDrawer, self).__init__(random_init=random_init)
    self.observation_space = gym.spaces.Box(
        low=np.full(12, -np.inf), high=np.full(12, np.inf), dtype=np.float32)

  @gin.configurable(module='SawyerDrawer')
  def reset(self, arm_goal_type='puck'):
    assert arm_goal_type in ['puck', 'goal']
    self._arm_goal = np.zeros(3)
    self.goal = np.zeros(3)
    self._state_goal = np.zeros(3)
    obs = super(SawyerDrawer, self).reset()
    offset = np.random.uniform(-0.2, 0)
    self._set_obj_xyz(offset)

    self.goal = obs[3:6]
    self.goal[1] = np.random.uniform(0.5, 0.7)
    if arm_goal_type == 'puck':
      self._arm_goal = obs[3:6]
    elif arm_goal_type == 'goal':
      self._arm_goal = self.goal.copy()
    else:
      raise NotImplementedError
    return self._get_obs()

  def step(self, action):
    s, r, done, info = super(SawyerDrawer, self).step(action)
    r = 0.0
    done = False
    return s, r, done, info

  def _get_obs(self):
    obs = super(SawyerDrawer, self)._get_obs()
    return np.concatenate([obs, self._arm_goal, self.goal])


class SawyerFaucet(sawyer_xyz.SawyerFaucetOpenEnv):
  """Wrapper for the sawyer_faucet task."""

  def __init__(self):
    super(SawyerFaucet, self).__init__()
    self.observation_space = gym.spaces.Box(
        low=np.full(12, -np.inf), high=np.full(12, np.inf), dtype=np.float32)

  @gin.configurable(module='SawyerFaucet')
  def reset(self, arm_goal_type='goal', init_width=np.pi / 2,
            goal_width=np.pi / 2):
    assert arm_goal_type in ['puck', 'goal']
    self._arm_goal = np.zeros(3)
    self.goal = np.zeros(3)
    self._state_goal = np.zeros(3)
    obs = super(SawyerFaucet, self).reset()

    offset = np.random.uniform(-goal_width, goal_width)
    self._set_obj_xyz(offset)
    self.goal = self._get_obs()[3:6]

    offset = np.random.uniform(-init_width, init_width)
    self._set_obj_xyz(offset)
    obs = self._get_obs()

    if arm_goal_type == 'puck':
      self._arm_goal = obs[3:6]
    elif arm_goal_type == 'goal':
      self._arm_goal = self.goal.copy()
    else:
      raise NotImplementedError
    return self._get_obs()

  def step(self, action):
    s, r, done, info = super(SawyerFaucet, self).step(action)
    r = 0.0
    done = False
    return s, r, done, info

  def _get_obs(self):
    obs = super(SawyerFaucet, self)._get_obs()
    return np.concatenate([obs, self._arm_goal, self.goal])
