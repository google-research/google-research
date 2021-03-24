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

"""Environments for experiments with RCE.
"""

import inspect
import os

from absl import logging
import d4rl  # pylint: disable=unused-import
import gin
import gym
from metaworld.envs.mujoco import sawyer_xyz
import numpy as np
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
import tqdm

# We need to import d4rl so that gym registers the environments.
os.environ['SDL_VIDEODRIVER'] = 'dummy'


def _get_image_obs(self):
  # The observation returned here should be in [0, 255].
  obs = self.get_image(width=84, height=84)
  return obs[::-1]


@gin.configurable
def load_env(env_name, max_episode_steps=None):
  """Loads an environment.

  Args:
    env_name: Name of the environment.
    max_episode_steps: Maximum number of steps per episode.
  Returns:
    tf_env: A TFPyEnvironment.
  """
  if env_name == 'sawyer_reach':
    gym_env = SawyerReach()
    max_episode_steps = 51
  elif env_name == 'sawyer_push':
    gym_env = SawyerPush()
    max_episode_steps = 151
  elif env_name == 'sawyer_lift':
    gym_env = SawyerLift()
    max_episode_steps = 151
  elif env_name == 'sawyer_drawer_open':
    gym_env = SawyerDrawerOpen()
    max_episode_steps = 151
  elif env_name == 'sawyer_drawer_close':
    gym_env = SawyerDrawerClose()
    max_episode_steps = 151
  elif env_name == 'sawyer_box_close':
    gym_env = SawyerBoxClose()
    max_episode_steps = 151
  elif env_name == 'sawyer_bin_picking':
    gym_env = SawyerBinPicking()
    max_episode_steps = 151
  else:
    gym_spec = gym.spec(env_name)
    gym_env = gym_spec.make()
    max_episode_steps = gym_spec.max_episode_steps

  env = suite_gym.wrap_env(
      gym_env,
      max_episode_steps=max_episode_steps)
  tf_env = tf_py_environment.TFPyEnvironment(env)
  return tf_env


@gin.configurable(denylist=['env', 'env_name'])
def get_data(env, env_name, num_expert_obs=200, terminal_offset=50):
  """Loads the success examples.

  Args:
    env: A PyEnvironment for which we want to generate success examples.
    env_name: The name of the environment.
    num_expert_obs: The number of success examples to generate.
    terminal_offset: For the d4rl datasets, we randomly subsample the last N
      steps to use as success examples. The terminal_offset parameter is N.
  Returns:
    expert_obs: Array with the success examples.
  """
  if env_name in ['hammer-human-v0', 'door-human-v0', 'relocate-human-v0']:
    dataset = env.get_dataset()
    terminals = np.where(dataset['terminals'])[0]
    expert_obs = np.concatenate(
        [dataset['observations'][t - terminal_offset:t] for t in terminals],
        axis=0)
    indices = np.random.choice(
        len(expert_obs), size=num_expert_obs, replace=False)
    expert_obs = expert_obs[indices]
  else:
    # For environments where we generate the expert dataset on the fly, we can
    # improve performance but only generating the number of expert
    # observations that we'll actually use. Not all environments support this
    # function, so we first have to check whether the environment's
    # get_dataset method accepts a num_obs kwarg.
    get_dataset_args = inspect.getfullargspec(env.get_dataset).args
    if 'num_obs' in get_dataset_args:
      dataset = env.get_dataset(num_obs=num_expert_obs)
    else:
      dataset = env.get_dataset()
    indices = np.random.choice(
        dataset['observations'].shape[0], size=num_expert_obs, replace=False)
    expert_obs = dataset['observations'][indices]
  if 'image' in env_name:
    expert_obs = expert_obs.astype(np.uint8)
  logging.info('Done loading expert observations')
  return expert_obs


class SawyerReach(sawyer_xyz.SawyerReachPushPickPlaceEnv):
  """A simple reaching environment."""

  def __init__(self):
    super(SawyerReach, self).__init__(task_type='reach')
    self.initialize_camera(self.init_camera)

  def step(self, action):
    obs = self._get_obs()
    d_before = np.linalg.norm(obs[:3] - obs[3:])
    s, r, done, info = super(SawyerReach, self).step(action)
    d_after = np.linalg.norm(s[:3] - s[3:])
    r = d_before - d_after
    done = False
    return s, r, done, info

  @gin.configurable(module='SawyerReach')
  def reset(self, random=False, width=1.0, random_color=False,
            random_size=False):
    if random_color:
      geom_id = self.model.geom_name2id('objGeom')
      rgb = np.random.uniform(np.zeros(3), np.ones(3))
      rgba = np.concatenate([rgb, [1.0]])
      self.model.geom_rgba[geom_id, :] = rgba
    if random_size:
      geom_id = self.model.geom_name2id('objGeom')
      low = np.array([0.01, 0.005, 0.0])
      high = np.array([0.05, 0.045, 0.0])
      size = np.random.uniform(low, high)
      self.model.geom_size[geom_id, :] = size
    super(SawyerReach, self).reset()

    if random:
      low = np.array([-0.2, 0.4, 0.02])
      high = np.array([0.2, 0.8, 0.02])
      if width == 1:
        scaled_low = low
        scaled_high = high
      else:
        mean = (low + high) / 2.0
        scaled_low = mean - width * (mean - low)
        scaled_high = mean + width * (high - mean)
      puck_pos = np.random.uniform(low=scaled_low, high=scaled_high)
      self._set_obj_xyz_quat(puck_pos, 0.0)

    # Hide the default goals and other markers. We use the puck position as
    # the goal. This must happen after self._set_obj_xyz_quat(...).
    self._state_goal = 10 * np.ones(3)
    self._set_goal_marker(self._state_goal)
    return self._get_obs()

  def _get_expert_obs(self):
    self.reset()
    # Don't use the observation returned from self.reset because this will be
    # an image for SawyerReachImage.
    obs = self._get_obs()
    self.data.set_mocap_pos('mocap', obs[3:6])
    self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    for _ in range(10):
      self.do_simulation([-1, 1], self.frame_skip)
    # Hide the markers, which get reset after every simulation step.
    self._set_goal_marker(self._state_goal)
    return self._get_obs()

  @gin.configurable(module='SawyerReach')
  def init_camera(self, camera, mode='default'):
    if mode == 'human':
      camera.distance = 0.5
      camera.lookat[0] = 0.6
      camera.lookat[1] = 1.0
      camera.lookat[2] = 0.5
      camera.elevation = -20
      camera.azimuth = 230
      camera.trackbodyid = -1
    elif mode == 'default':
      camera.lookat[0] = 0
      camera.lookat[1] = 0.85
      camera.lookat[2] = 0.3
      camera.distance = 0.4
      camera.elevation = -35
      camera.azimuth = 270
      camera.trackbodyid = -1
    elif mode == 'v2':
      camera.lookat[0] = 0
      camera.lookat[1] = 0.6
      camera.lookat[2] = 0.0
      camera.distance = 0.7
      camera.elevation = -35
      camera.azimuth = 180
      camera.trackbodyid = -1
    else:
      raise NotImplementedError

  def get_dataset(self, num_obs=256):
    # This generates examples at ~145 observations / sec. When using image
    # observations is slows down to ~17 FPS.
    action_vec = [self.action_space.sample() for _ in range(num_obs)]
    obs_vec = [self._get_expert_obs() for _ in tqdm.trange(num_obs)]
    dataset = {
        'observations': np.array(obs_vec, dtype=np.float32),
        'actions': np.array(action_vec, dtype=np.float32),
        'rewards': np.zeros(num_obs, dtype=np.float32),
    }
    return dataset


class SawyerPush(sawyer_xyz.SawyerReachPushPickPlaceEnv):
  """A pushing environment."""

  def __init__(self):
    super(SawyerPush, self).__init__(task_type='push')
    self.initialize_camera(self.init_camera)
    self._goal = np.array([0.1, 0.85, 0.02])

  def step(self, action):
    obs = self._get_obs()
    d_before = np.linalg.norm(obs[3:] - self._goal)
    s, _, done, info = super(SawyerPush, self).step(action)
    d_after = np.linalg.norm(s[3:] - self._goal)
    r = d_before - d_after
    done = False
    return s, r, done, info

  @gin.configurable(module='SawyerPush')
  def _get_expert_obs(self, hand_at_puck=True, wide=False, off_table=False):
    self.reset()
    if wide:
      puck_pos = np.random.uniform(low=[-0.15, 0.8, 0.02],
                                   high=[0.15, 0.9, 0.02])
    else:
      puck_pos = np.random.uniform(low=[0.05, 0.8, 0.02],
                                   high=[0.15, 0.9, 0.02])
    if off_table:
      assert not wide
      assert not hand_at_puck
      puck_pos = 10 * np.ones(3,)
    self._set_obj_xyz_quat(puck_pos, 0.0)
    if hand_at_puck:
      hand_goal = puck_pos
    else:
      hand_goal = np.random.uniform(low=[-0.2, 0.4, 0.02],
                                    high=[0.2, 0.8, 0.3])
    self.data.set_mocap_pos('mocap', hand_goal)
    self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    for _ in range(10):
      self.do_simulation([-1, 1], self.frame_skip)
    return self._get_obs()

  @gin.configurable(module='SawyerPush')
  def init_camera(self, camera, mode='default'):
    if mode == 'human':
      camera.distance = 0.5
      camera.lookat[0] = 0.6
      camera.lookat[1] = 1.0
      camera.lookat[2] = 0.5
      camera.elevation = -20
      camera.azimuth = 230
      camera.trackbodyid = -1
    elif mode == 'default':
      camera.lookat[0] = 0
      camera.lookat[1] = 0.9
      camera.lookat[2] = 0.3
      camera.distance = 0.4
      camera.elevation = -45
      camera.azimuth = 270
      camera.trackbodyid = -1
    elif mode == 'front':
      camera.lookat[0] = 0
      camera.lookat[1] = 0.85
      camera.lookat[2] = 0.05
      camera.distance = 0.4
      camera.elevation = 0
      camera.azimuth = 270
      camera.trackbodyid = -1
    elif mode == 'side':
      camera.lookat[0] = 0
      camera.lookat[1] = 0.7
      camera.lookat[2] = 0.05
      camera.distance = 0.6
      camera.elevation = 0
      camera.azimuth = 180
      camera.trackbodyid = -1
    else:
      raise NotImplementedError

  def get_dataset(self, num_obs=256):
    # This generates examples at ~145 observations / sec.
    action_vec = [self.action_space.sample() for _ in range(num_obs)]
    obs_vec = [self._get_expert_obs() for _ in tqdm.trange(num_obs)]
    dataset = {
        'observations': np.array(obs_vec, dtype=np.float32),
        'actions': np.array(action_vec, dtype=np.float32),
        'rewards': np.zeros(num_obs, dtype=np.float32),
    }
    return dataset


class SawyerLift(sawyer_xyz.SawyerReachPushPickPlaceEnv):
  """A task of lifting up an object."""

  MODE = 'train'

  def __init__(self):
    super(SawyerLift, self).__init__(task_type='reach')
    self.initialize_camera(self.init_camera)

  def _get_dist(self, z):
    min_height, max_height = self.target_height()
    d_above = abs(z - max_height)
    d_below = abs(z - min_height)
    if min_height <= z <= max_height:
      return 0.0
    else:
      return min(d_above, d_below)

  @gin.configurable(module='SawyerLift')
  def target_height(self, target_height=0.1):
    """Values 0.1 through 0.3 are reasonable."""
    if isinstance(target_height, tuple) or isinstance(target_height, list):
      min_height, max_height = target_height
    else:
      min_height = target_height - 0.02
      max_height = target_height + 0.02
    return (min_height, max_height)

  def step(self, action):
    obs = self._get_obs()
    d_before = self._get_dist(obs[5])
    # d_before = abs(obs[5] - self.target_height())
    s, r, done, info = super(SawyerLift, self).step(action)
    d_after = self._get_dist(s[5])
    # d_after = abs(s[5] - self.target_height())

    r = d_before - d_after
    done = False
    return s, r, done, info

  def init_camera(self, camera):
    camera.distance = 0.5
    camera.lookat[0] = 0.6
    camera.lookat[1] = 1.0
    camera.lookat[2] = 0.5
    camera.elevation = -20
    camera.azimuth = 230
    camera.trackbodyid = -1

  @gin.configurable(module='SawyerLift')
  def reset(self, reset_to_goal=False):
    super(SawyerLift, self).reset()
    if reset_to_goal and self.MODE == 'train':
      self._get_expert_obs(reset=False)
    return self._get_obs()

  @gin.configurable(module='SawyerLift')
  def _get_expert_obs(self, reset=True):
    if reset:
      self.reset()
    obs = self._get_obs()
    puck_pos = obs[3:6]
    min_height, max_height = self.target_height()
    puck_pos[-1] = np.random.uniform(min_height, max_height)
    self.data.set_mocap_pos('mocap', puck_pos)
    self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    for _ in range(10):
      self.do_simulation([-1, 1], self.frame_skip)
    # We have to set the puck position after moving the arm. Otherwise
    # the puck will fall while setting the arm position.
    self._set_obj_xyz_quat(puck_pos, 0.0)
    return self._get_obs()

  def get_dataset(self, num_obs=256):
    # This generates examples at ~145 observations / sec.
    action_vec = [self.action_space.sample() for _ in range(num_obs)]
    obs_vec = [self._get_expert_obs() for _ in tqdm.trange(num_obs)]
    dataset = {
        'observations': np.array(obs_vec, dtype=np.float32),
        'actions': np.array(action_vec, dtype=np.float32),
        'rewards': np.zeros(num_obs, dtype=np.float32),
    }
    return dataset


class SawyerDrawerOpen(sawyer_xyz.SawyerDrawerOpenEnv):
  """A drawer opening task."""

  def __init__(self):
    super(SawyerDrawerOpen, self).__init__()
    self.initialize_camera(self.init_camera)

  def step(self, action):
    obs = self._get_obs()
    d_before = np.linalg.norm(obs[4] - self.goal[1])
    s, r, done, info = super(SawyerDrawerOpen, self).step(action)
    d_after = np.linalg.norm(s[4] - self.goal[1])
    r = d_before - d_after
    done = False
    return s, r, done, info

  def init_camera(self, camera):
    camera.distance = 1.
    camera.lookat[0] = 0.0
    camera.lookat[1] = 0.4
    camera.lookat[2] = 0.3
    camera.elevation = -20
    camera.azimuth = 160
    camera.trackbodyid = -1

  @gin.configurable(module='SawyerDrawerOpen')
  def _get_expert_obs(self, hand_at_goal=True):
    self.reset()
    pos = np.random.uniform(-0.25, -0.15)
    self._set_obj_xyz(pos)
    if hand_at_goal:
      hand_goal = self._get_obs()[3:]
    else:
      hand_goal = np.random.uniform(low=[-0.2, 0.4, 0.02],
                                    high=[0.2, 0.8, 0.3])

    self.data.set_mocap_pos('mocap', hand_goal)
    self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    for _ in range(10):
      self.do_simulation([-1, 1], self.frame_skip)
    return self._get_obs()

  def get_dataset(self, num_obs=256):
    # This generates examples at ~145 observations / sec.
    action_vec = [self.action_space.sample() for _ in range(num_obs)]
    obs_vec = [self._get_expert_obs() for _ in tqdm.trange(num_obs)]
    dataset = {
        'observations': np.array(obs_vec, dtype=np.float32),
        'actions': np.array(action_vec, dtype=np.float32),
        'rewards': np.zeros(num_obs, dtype=np.float32),
    }
    return dataset


@gin.configurable
class SawyerDrawerClose(sawyer_xyz.SawyerDrawerCloseEnv):
  """A drawer closing task."""

  def __init__(self, random_init=False):
    super(SawyerDrawerClose, self).__init__(random_init=random_init)
    self.initialize_camera(self.init_camera)

  def step(self, action):
    obs = self._get_obs()
    d_before = np.linalg.norm(obs[4] - self.goal[1])
    s, r, done, info = super(SawyerDrawerClose, self).step(action)
    d_after = np.linalg.norm(s[4] - self.goal[1])
    r = d_before - d_after
    done = False
    return s, r, done, info

  def init_camera(self, camera):
    camera.distance = 1.
    camera.lookat[0] = 0.0
    camera.lookat[1] = 0.4
    camera.lookat[2] = 0.3
    camera.elevation = -20
    camera.azimuth = 160
    camera.trackbodyid = -1

  @gin.configurable(module='SawyerDrawerClose')
  def _get_expert_obs(self, hand_at_goal=True):
    self.reset()
    pos = np.random.uniform(0.0, 0.05)
    self._set_obj_xyz(pos)
    if hand_at_goal:
      hand_goal = self._get_obs()[3:]
    else:
      hand_goal = np.random.uniform(low=[-0.2, 0.4, 0.02],
                                    high=[0.2, 0.8, 0.3])

    self.data.set_mocap_pos('mocap', hand_goal)
    self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    for _ in range(10):
      self.do_simulation([-1, 1], self.frame_skip)
    return self._get_obs()

  def get_dataset(self, num_obs=256):
    # This generates examples at ~145 observations / sec.
    action_vec = [self.action_space.sample() for _ in range(num_obs)]
    obs_vec = [self._get_expert_obs() for _ in tqdm.trange(num_obs)]
    dataset = {
        'observations': np.array(obs_vec, dtype=np.float32),
        'actions': np.array(action_vec, dtype=np.float32),
        'rewards': np.zeros(num_obs, dtype=np.float32),
    }
    return dataset


class SawyerBoxClose(sawyer_xyz.SawyerBoxCloseEnv):
  """The task is to put a lid on a box.

  The observation dimension is 9: 3 for the hand, 3 for the lid, 3 for the
  goal.
  """

  def __init__(self):
    super(SawyerBoxClose, self).__init__()
    self.initialize_camera(self.init_camera)

  def _get_goal_pos(self, obs):
    goal_pos = obs[-3:]
    goal_pos[-1] -= 0.085
    return goal_pos

  def step(self, action):
    obs = self._get_obs()
    goal_pos = self._get_goal_pos(obs)
    d_before = np.linalg.norm(obs[3:6] - goal_pos)
    s, _, _, info = super(SawyerBoxClose, self).step(action)
    d_after = np.linalg.norm(s[3:6] - goal_pos)
    r = d_before - d_after
    done = False
    return s, r, done, info

  def _get_expert_obs(self):
    self.reset()
    obs = self._get_obs()
    goal_pos = obs[-3:]
    goal_pos[-1] -= 0.085
    self._set_obj_xyz_quat(goal_pos, self.obj_init_angle)

    self.data.set_mocap_pos('mocap', obs[-3:])
    self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    for _ in range(10):
      self.do_simulation([-1, 1], self.frame_skip)
    return self._get_obs()

  def init_camera(self, camera):
    camera.distance = 1.
    camera.lookat[0] = 0.0
    camera.lookat[1] = 1.0
    camera.lookat[2] = 0.1
    camera.elevation = -10
    camera.azimuth = 270
    camera.trackbodyid = -1

  def get_dataset(self, num_obs=256):
    # This generates examples at ~273 observations / sec.
    action_vec = [self.action_space.sample() for _ in range(num_obs)]
    obs_vec = [self._get_expert_obs() for _ in tqdm.trange(num_obs)]
    dataset = {
        'observations': np.array(obs_vec, dtype=np.float32),
        'actions': np.array(action_vec, dtype=np.float32),
        'rewards': np.zeros(num_obs, dtype=np.float32),
    }
    return dataset


class SawyerBinPicking(sawyer_xyz.SawyerBinPickingEnv):
  """A pick and place task."""

  def __init__(self):
    super(SawyerBinPicking, self).__init__()
    self.initialize_camera(self.init_camera)

  def step(self, action):
    obs = self._get_obs()
    goal_pos = np.array([0.12, 0.7, 0.046])
    d_before = np.linalg.norm(obs[3:6] - goal_pos)
    s, _, _, info = super(SawyerBinPicking, self).step(action)
    d_after = np.linalg.norm(s[3:6] - goal_pos)
    r = d_before - d_after
    done = False
    return s, r, done, info

  def _get_expert_obs(self):
    self.reset()
    goal_pos = np.random.uniform(
        low=np.array([0.06, 0.64, 0.046]), high=np.array([0.18, 0.76, 0.046]))
    self._set_obj_xyz_quat(goal_pos, self.obj_init_angle)
    self.data.set_mocap_pos('mocap', goal_pos)
    self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
    for _ in range(10):
      self.do_simulation([-1, 1], self.frame_skip)
    return self._get_obs()

  @gin.configurable(module='SawyerBinPicking')
  def init_camera(self, camera, mode='default'):
    if mode == 'default':
      camera.distance = 0.5
      camera.lookat[0] = 0.6
      camera.lookat[1] = 1.0
      camera.lookat[2] = 0.5
      camera.elevation = -20
      camera.azimuth = 230
      camera.trackbodyid = -1
    elif mode == 'side':
      camera.lookat[0] = 0.2
      camera.lookat[1] = 0.7
      camera.lookat[2] = 0.2
      camera.distance = 0.3
      camera.elevation = -30
      camera.azimuth = 180
      camera.trackbodyid = -1
    elif mode == 'front':
      camera.lookat[0] = 0.0
      camera.lookat[1] = 0.9
      camera.lookat[2] = 0.2
      camera.distance = 0.3
      camera.elevation = -30
      camera.azimuth = 270
      camera.trackbodyid = -1
    else:
      raise NotImplementedError

  def get_dataset(self, num_obs=256):
    # This generates examples at ~95 observations / sec.
    action_vec = [self.action_space.sample() for _ in range(num_obs)]
    obs_vec = [self._get_expert_obs() for _ in tqdm.trange(num_obs)]
    dataset = {
        'observations': np.array(obs_vec, dtype=np.float32),
        'actions': np.array(action_vec, dtype=np.float32),
        'rewards': np.zeros(num_obs, dtype=np.float32),
    }
    return dataset
