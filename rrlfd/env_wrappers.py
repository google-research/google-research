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

# Lint as: python3
"""Defines dm_env-like wrapper around environments."""

import collections
import copy

from absl import flags
from acme import wrappers
from acme.wrappers import video
import dm_env
from dm_env import specs
import gym
import mime
import numpy as np
import tree

flags.DEFINE_integer('dense_reward_type', 1,
                     'Dense reward signal to use, out of the options defined '
                     'for the task.')

FLAGS = flags.FLAGS


class KwargWrapper(wrappers.SinglePrecisionWrapper):
  """Single precision env wrapper that passes custom args to env step and reset."""

  def reset(self, *args, **kwargs):
    reset_returns = self._environment.reset(*args, **kwargs)
    if isinstance(reset_returns, dm_env.TimeStep):
      reset_returns = self._convert_timestep(reset_returns)
    else:
      reset_returns = (
          (self._convert_timestep(reset_returns[0]),) + reset_returns[1:])
    return reset_returns

  def step(self, action, *args, **kwargs):
    step_returns = self._environment.step(action, *args, **kwargs)
    if isinstance(step_returns, dm_env.TimeStep):
      step_returns = self._convert_timestep(step_returns)
    else:
      step_returns = (
          (self._convert_timestep(step_returns[0]),) + step_returns[1:])
    return step_returns


class FlatActionWrapper(wrappers.EnvironmentWrapper):
  """Wrapper that flattens actions from dict to array, sorted by key.

  As a consequence, dtype of all action components should be identical.
  """

  def __init__(self, environment):
    super().__init__(environment)
    self._environment = environment

  def action_spec(self):
    spec = self._environment.action_spec()
    sorted_keys = sorted(spec.keys())
    sorted_values = [spec[k] for k in sorted_keys]
    shape = (np.sum([np.prod(v.shape) for v in sorted_values]),)
    dtype = sorted_values[0].dtype
    low = np.concatenate([v.minimum.flatten() for v in sorted_values])
    high = np.concatenate([v.maximum.flatten() for v in sorted_values])
    return specs.BoundedArray(
        shape=shape,
        dtype=dtype,
        minimum=low,
        maximum=high,
        name=','.join(sorted_keys))

  def step(self, action):
    dict_action = {}
    i = 0
    for k, v in sorted(self._environment.action_spec().items()):
      dict_action[k] = action[i:i + np.prod(v.shape)]
      i += np.prod(v.shape)
    return self._environment.step(dict_action)


class StackedHistoryWrapper(wrappers.GymWrapper):
  """Wrapper around a gym environment that maintains a history of frames."""

  def __init__(self, environment):

    self.obs_hist = collections.OrderedDict()

  def add_observation(self, obs):
    """Add observation obs to buffer of self.num_input_frames past observations.

    Args:
      obs: Observation to add to buffer.
    """
    if self.input_type == 'full_state':
      return
    # TODO(minttu): History for position inputs.
    if self.num_input_frames == 1:
      self.obs_hist = obs
    else:
      k = self.input_type
      v = np.copy(obs[k])
      if k in self.obs_hist:
        self.obs_hist[k] = np.concatenate(
            [self.obs_hist[k], np.expand_dims(v, axis=2)], axis=2)
      else:
        self.obs_hist[k] = np.expand_dims(v, axis=2)
      num_channels_per_obs = v.shape[2] if len(v.shape) > 2 else 1
      channels_to_keep = self.num_input_frames * num_channels_per_obs
      self.obs_hist[k] = self.obs_hist[k][:, :, -channels_to_keep:]


class CustomStackingWrapper(wrappers.FrameStackingWrapper):
  """Frame stacking wrapper that allows stacks of different lengths per key."""

  def __init__(self, environment, stack_length):
    self._environment = environment
    original_spec = self._environment.observation_spec()
    self._stackers = {}
    for k in self._environment.observation_spec().keys():
      if k in stack_length:
        self._stackers[k] = RepeatingStacker(stack_length[k])
      else:
        self._stackers[k] = NoOpStacker()
    self._observation_spec = tree.map_structure(
        lambda stacker, spec: stacker.update_spec(spec),
        self._stackers, original_spec)


class RepeatingStacker:
  """Frame stacker which repeats the first frame until the buffer is filled."""

  def __init__(self, num_frames, flatten = False):
    self._num_frames = num_frames
    self._flatten = flatten
    self.reset()

  @property
  def num_frames(self):
    return self._num_frames

  def reset(self):
    self._stack = collections.deque(maxlen=self._num_frames)

  def step(self, frame):
    """Append frame to stack and return the stack."""
    if not self._stack:
      # Fill stack with copies of first frame if empty.
      self._stack.extend([frame] * (self._num_frames - 1))
    self._stack.append(frame)
    # Match BCAgent's stacking along axis 2.
    stacked_frames = np.stack(self._stack, axis=2)

    if not self._flatten:
      return stacked_frames
    else:
      new_shape = stacked_frames.shape[:-2] + (-1,)
      return stacked_frames.reshape(*new_shape)

  def update_spec(self, spec):
    if not self._flatten:
      new_shape = spec.shape + (self._num_frames,)
    else:
      new_shape = spec.shape[:-1] + (self._num_frames * spec.shape[-1],)
    return specs.Array(shape=new_shape, dtype=spec.dtype, name=spec.name)


class NoOpStacker:

  def reset(self):
    pass

  def step(self, frame):
    return frame

  def update_spec(self, spec):
    return spec


class EarlyFusionImageWrapper(wrappers.EnvironmentWrapper):
  """Convert from [h, w, history, c] to [h, w, history * c]."""

  def __init__(self, environment, image_key):
    super().__init__(environment)
    self.image_key = image_key
    original_spec = self._environment.observation_spec()
    new_spec = original_spec.copy()
    original_shape = original_spec[self.image_key].shape
    new_spec[self.image_key] = (
        *original_shape[:2], np.prod(original_shape[2:]))
    self._observation_spec = new_spec

  def _wrap_observation(self, observation):
    obs = observation[self.image_key]
    observation[self.image_key] = (
        np.reshape(obs, [*obs.shape[:2], np.prod(obs.shape[2:])]))
    return observation

  def step(self, action):
    timestep = self._environment.step(action)
    observation = self._wrap_observation(timestep.observation)
    timestep._replace(observation=observation)
    return timestep

  def reset(self):
    timestep = self._environment.reset()
    observation = self._wrap_observation(timestep.observation)
    timestep._replace(observation=observation)
    return timestep


class TransposeImageWrapper(wrappers.EnvironmentWrapper):
  """Convert from [h, w, history, c] to [history, h, w, c]."""

  def __init__(self, environment, image_key):
    super().__init__(environment)
    self.image_key = image_key
    original_spec = self._environment.observation_spec()
    new_spec = original_spec.copy()
    original_shape = original_spec[self.image_key].shape
    new_spec[self.image_key] = (
        original_shape[2], *original_shape[:2], *original_shape[3:])
    self._observation_spec = new_spec

  def _wrap_observation(self, observation):
    observation[self.image_key] = np.moveaxis(observation[self.image_key], 2, 0)
    return observation

  def step(self, action):
    timestep = self._environment.step(action)
    observation = self._wrap_observation(timestep.observation)
    timestep._replace(observation=observation)
    return timestep

  def reset(self):
    timestep = self._environment.reset()
    observation = self._wrap_observation(timestep.observation)
    timestep._replace(observation=observation)
    return timestep


class MimeVisibleKeysWrapper(wrappers.EnvironmentWrapper):
  """Mime environment wrapper that exposes a subset of state keys."""

  def __init__(self, environment, visible_keys):
    super().__init__(environment)
    self.visible_keys = visible_keys
    self._environment.observation_space = (
        self._environment.unwrapped._make_dict_space(*visible_keys))

  def _strip_digit_suffix(self, string):
    while string and string[-1].isdigit():
      string = string[:-1]
    return string

  def _filter_observation(self, obs):
    obs = {self._strip_digit_suffix(k): v for k, v in obs.items()
           if self._strip_digit_suffix(k) in self.visible_keys}
    obs = {k: np.array(v) for k, v in obs.items()}
    return obs

  def step(self, action):
    obs, reward, done, info = self._environment.step(action)
    # Mime envs do not filter observation keys according to observation space.
    obs = self._filter_observation(obs)
    return obs, reward, done, info

  def reset(self):
    obs = self._environment.reset()
    # Mime envs do not filter observation keys according to observation space.
    obs = self._filter_observation(obs)
    return obs


class AdroitRewardWrapper(wrappers.EnvironmentWrapper):
  """Define sparse rewards for Adroit environments."""

  def __init__(self, environment, sparse=False, dense_reward_multiplier=1.0):
    self._environment = environment
    self._sparse_reward = sparse
    self._dense_reward_multiplier = dense_reward_multiplier
    self._goals_achieved = []

  def reset(self):
    self._goals_achieved = []
    return self._environment.reset()

  def step(self, action):
    obs, reward, done, info = self._environment.step(action)
    self._goals_achieved.append(info['goal_achieved'])
    success_percentage = self._environment.evaluate_success(
        [{'env_infos': {'goal_achieved': self._goals_achieved}}])
    if self._sparse_reward:
      # TODO(minttu): also return original reward for logging?
      reward = success_percentage / 100.0
    else:
      reward *= self._dense_reward_multiplier
      reward += float(bool(success_percentage))
    return obs, reward, done, info


class MimeRewardWrapper(wrappers.EnvironmentWrapper):
  """Define sparse and dense rewards for Mime environments."""

  def __init__(self, environment, sparse=False, dense_reward_multiplier=1.0):
    self._environment = environment
    self._sparse_reward = sparse
    self._dense_reward_multiplier = dense_reward_multiplier

  def get_dense_reward(self, state, scene):
    if isinstance(scene, mime.envs.table_envs.PickScene):
      xyz_dist_cube_gripper = state['distance_to_goal']
      dist_cube_gripper = np.linalg.norm(xyz_dist_cube_gripper)
      if FLAGS.dense_reward_type == 1:
        dense = -dist_cube_gripper
      else:
        # Grip state is 0 when open; 1.0490666666666801 when fully closed.
        norm_grip_state = state['grip_state'] / 1.0490666666666801
        near_cube = np.all(xyz_dist_cube_gripper < [0.02, 0.02, 0.02])
        holding_object = (
            self.env.unwrapped.scene.robot.gripper.controller._object_grasped)  # pylint: disable=protected-access
        if holding_object:  # -d(c, g) - openness(g) + height(g)
          table_height = self.env.unwrapped.scene.workspace[0][2]
          dense = (
              -dist_cube_gripper - (1 - norm_grip_state)
              + state['tool_position'][2] - table_height)
        elif near_cube:  # -d(c, g) - openness(g)
          dense = -dist_cube_gripper - (1 - norm_grip_state)
        else:  # -d(c, g)
          dense = -dist_cube_gripper
    if isinstance(scene, mime.envs.table_envs.PushScene):
      middle_of_goal = np.mean(scene.goal_workspace, axis=0)
      cube_position = state['target_position']
      cube_distance_to_goal = np.linalg.norm(cube_position - middle_of_goal)
      dense = (-np.linalg.norm(state['distance_to_goal'])
               -cube_distance_to_goal)
    elif isinstance(scene, mime.envs.table_envs.BowlScene):
      xyz_dist_cube_gripper = state['distance_to_cube']
      dist_cube_gripper = np.linalg.norm(xyz_dist_cube_gripper)
      cube_position = np.array(state['cube_position'])
      bowl_position = np.array(state['bowl_position'])
      dist_cube_bowl = np.linalg.norm(cube_position - bowl_position)
      if FLAGS.dense_reward_type == 1:
        dense = -dist_cube_gripper - dist_cube_bowl
      else:
        norm_grip_state = state['grip_state'] / 1.0490666666666801
        near_cube = np.all(xyz_dist_cube_gripper < [0.02, 0.02, 0.02])
        holding_object = (
            self.env.unwrapped.scene.robot.gripper.controller._object_grasped)  # pylint: disable=protected-access
        if holding_object:  # -d(c, g) - openness(g) + height(g) - d(c, b)
          table_height = self.env.unwrapped.scene.workspace[0][2]
          dense = (
              -dist_cube_gripper - (1 - norm_grip_state)
              + state['tool_position'][2] - table_height  # set cap for height
              - dist_cube_bowl)
        elif near_cube:  # -d(c, g) - openness(g)
          dense = -dist_cube_gripper - (1 - norm_grip_state)
        else:  # -d(c, g)
          dense = -dist_cube_gripper
    elif 'distance_to_goal' in state:
      dense = -np.linalg.norm(state['distance_to_goal'])
    else:
      raise NotImplementedError(
          f'Dense reward signal not defined for {self.env_name}.')
    return dense

  def step(self, action):
    obs, reward, done, info = self._environment.step(action)
    if self._sparse_reward:
      # reward = float(reward)
      # Some Mime environments do not define a reward.
      reward = float(int(info['success']))
    else:
      reward = self.get_dense_reward(obs, self._environment.unwrapped.scene)
      reward *= self._dense_reward_multiplier
      reward += float(int(info['success']))
    return obs, reward, done, info


class GymInfoAdapter(wrappers.GymWrapper):
  """Extends GymWrapper to add relevant environment info to the observation."""

  def __init__(self, environment, info_defaults):
    super().__init__(environment)
    self._info_defaults = info_defaults

  def observation_spec(self):
    # self.observation_spec() is DM env, self._observation_spec Gym interface.
    observation_spec = copy.deepcopy(self._observation_spec)
    for k, v in self._info_defaults.items():
      if isinstance(v, bool):
        observation_spec[k] = specs.Array((), bool, k)
      elif isinstance(v, str):
        observation_spec[k] = specs.StringArray((), str, k)
      else:
        raise NotImplementedError(
            f'Info field of type {type(v)} not mapped to acme spec type.')
    return observation_spec

  def info_from_observation(self, observation):
    return {k: observation[k].item() for k in self._info_defaults.keys()}

  def _wrap_observation(self, observation, env_info=None):
    # Default values.
    info = dict(self._info_defaults.items())
    if env_info is not None:
      info.update(env_info)
    for key in self.observation_spec():
      if key in info:
        observation[key] = info[key]
    return observation

  def reset(self):
    self._reset_next_step = False
    observation = self._environment.reset()
    observation = self._wrap_observation(observation)
    return dm_env.restart(observation)

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    observation, reward, done, info = self._environment.step(action)
    for k in info:
      assert k in self._info_defaults.keys() | {'TimeLimit.truncated'}
    observation = self._wrap_observation(observation, info)
    self._reset_next_step = done

    if done:
      truncated = info.get('TimeLimit.truncated', False)
      if truncated:
        return dm_env.truncation(reward, observation)
      return dm_env.termination(reward, observation)
    return dm_env.transition(reward, observation)


class GymAdroitAdapter(GymInfoAdapter):
  """Wrapper exposing a Gym Adroit environment using DM env interface.

  This wraps the Gym Adroit environment in the same way as GymWrapper, but also
  exposes relevant environment info as fields in the observation.
  """

  def __init__(self, environment, end_on_success=False):
    super().__init__(
        environment,
        info_defaults={'goal_achieved': False, 'success': False})
    self._end_on_success = end_on_success

  def reset(self):
    self._goals_achieved = []
    return super().reset()

  def step(self, action):
    if self._reset_next_step:
      return self.reset()

    observation, reward, done, info = self._environment.step(action)
    self._goals_achieved.append(info['goal_achieved'])
    success = self._environment.evaluate_success(
        [{'env_infos': {'goal_achieved': self._goals_achieved}}])
    info['success'] = bool(success)
    if self._end_on_success:
      done = done or success
    for k in info:
      assert k in self._info_defaults.keys() | {'TimeLimit.truncated'}
    observation = self._wrap_observation(observation, info)
    self._reset_next_step = done

    if done:
      truncated = info.get('TimeLimit.truncated', False)
      if truncated:
        return dm_env.truncation(reward, observation)
      return dm_env.termination(reward, observation)
    return dm_env.transition(reward, observation)


class GymMimeAdapter(GymInfoAdapter):
  """Wrapper exposing a Gym Mime environment using DM env interface.

  This wraps the Gym Mime environment in the same way as GymWrapper, but also
  exposes relevant environment info as fields in the observation.
  """

  def __init__(self, environment):
    super().__init__(
        environment,
        info_defaults={'success': False, 'failure_message': ''})


class MimeWrapper(wrappers.EnvironmentWrapper):
  """Class responsible for initializing mime environments."""

  def __init__(
      self,
      task,
      use_egl=False,
      seed=None,
      input_type='depth',
      image_size=None,
      render=False,
      lateral_friction=0.5,
      max_episode_steps=None):
    cam_str = 'Cam' if input_type in ['depth', 'rgb', 'rgbd'] else ''
    egl_str = '-EGL' if use_egl and cam_str else ''
    env_name = f'UR5{egl_str}-{task}{cam_str}Env-v0'
    print('Creating env', env_name)
    self.env_name = env_name
    self._seed = seed
    self._image_size = image_size
    self._render = render
    self.task = task
    self._lateral_friction = lateral_friction
    self.max_episode_steps = max_episode_steps
    self.create_env()
    self.default_max_episode_steps = self._environment._max_episode_steps

  def create_env(self):
    env = gym.make(self.env_name)
    if self._seed is not None:
      env.seed(self._seed)
    if self._image_size is not None:
      env.env._cam_resolution = (self._image_size, self._image_size)  # pylint: disable=protected-access
    env.unwrapped.scene.renders(self._render)
    if self.task == 'Push':  # TODO(minttu): Apply to all tasks
      env.unwrapped.scene.lateral_friction = self._lateral_friction
    if self.max_episode_steps is not None:
      env._max_episode_steps = self.max_episode_steps  # pylint: disable=protected-access
    self._environment = env

  def step(self, action):
    for k, v in action.items():
      action[k] = np.clip(v,
                          self._environment.action_space[k].low,
                          self._environment.action_space[k].high)
    return self._environment.step(action)


class AdroitWrapper(wrappers.EnvironmentWrapper):
  """Class responsible for initializing Adroit environments."""

  def __init__(self, task, image_size, max_episode_steps):
    environment = gym.make(f'visual-{task}-v0')
    if image_size is not None:
      environment.env.im_size = image_size
    im_size = environment.env.im_size
    self.observation_space = copy.deepcopy(environment.observation_space)
    self.observation_space['rgb'].shape = (im_size, im_size, 3)
    self.observation_space['rgb'].dtype = np.uint8
    self.observation_space['rgb'].low = (
        np.zeros([im_size, im_size, 3], np.uint8))
    self.observation_space['rgb'].high = (
        255 * np.ones([im_size, im_size, 3], np.uint8))

    self.default_max_episode_steps = environment._max_episode_steps
    if max_episode_steps is not None:
      environment._max_episode_steps = max_episode_steps
    self._environment = environment
    self.task = task


class KeyedVideoWrapper(video.VideoWrapper):
  """Video wrapper which records only the visual part of the observation.

  The observation frame to record is defined by a key (in the observation
  dictionary).
  """

  def __init__(self, environment, visual_key, **kwargs):
    self.visual_key = visual_key
    super().__init__(environment, **kwargs)

  def _append_frame(self, observation):
    """Appends a frame to the sequence of frames."""
    if self._counter % self._record_every == 0:
      self._frames.append(self._render_frame(observation[self.visual_key]))
