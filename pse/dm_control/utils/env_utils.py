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

# Lint as: python3
"""Utility library for combining tf-agents and dm-control."""
import collections
import copy
import functools
from typing import Sequence, Text
from absl import logging
import gin
import numpy as np
from tf_agents.environments import dm_control_wrapper
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from distracting_control import suite


@gin.configurable
def load_pixels(
    domain_name,
    task_name,
    observation_key = 'pixels',
    pixels_only = True,
    task_kwargs=None,
    environment_kwargs=None,
    visualize_reward = False,
    render_kwargs=None,
    env_wrappers = (),
    camera_kwargs=None,
    background_kwargs=None,
    color_kwargs=None,
):
  """Returns an environment from a domain name, task name and optional settings.

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    observation_key: Optional custom string specifying the pixel observation's
      key in the `OrderedDict` of observations. Defaults to 'pixels'.
    pixels_only: If True (default), the original set of 'state' observations
      returned by the wrapped environment will be discarded, and the
      `OrderedDict` of observations will only contain pixels. If False, the
      `OrderedDict` will contain the original observations as well as the pixel
      observations.
    task_kwargs: Optional `dict` of keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.
    render_kwargs: Optional `dict` of keyword arguments for rendering.
    env_wrappers: Iterable with references to wrapper classes to use on the
      wrapped environment.
    camera_kwargs: optional dict of camera distraction arguments
    background_kwargs: optional dict of background distraction arguments
    color_kwargs: optional dict of color distraction arguments

  Returns:
    The requested environment.

  Raises:
    ImportError: if dm_control module was not available.
  """

  dm_env = suite.load(
      domain_name,
      task_name,
      task_kwargs=task_kwargs,
      environment_kwargs=environment_kwargs,
      visualize_reward=visualize_reward,
      camera_kwargs=camera_kwargs,
      background_kwargs=background_kwargs,
      color_kwargs=color_kwargs,
      pixels_only=pixels_only,
      render_kwargs=render_kwargs,
      pixels_observation_key=observation_key)

  env = dm_control_wrapper.DmControlWrapper(dm_env, render_kwargs)

  for wrapper in env_wrappers:
    env = wrapper(env)

  return env


@gin.configurable
def load_dm_env_for_training(
    env_name,
    frame_shape=(84, 84, 3),
    episode_length=1000,
    action_repeat=4,
    frame_stack=3,
    task_kwargs=None,
    render_kwargs=None,
    # Camera args
    camera_camera_id=None,
    camera_horizontal_delta=None,
    camera_vertical_delta=None,
    camera_max_vel=None,
    camera_vel_std=None,
    camera_roll_delta=None,
    camera_max_roll_vel=None,
    camera_roll_std=None,
    camera_max_zoom_in_percent=None,
    camera_max_zoom_out_percent=None,
    camera_limit_to_upper_quadrant=None,
    camera_seed=None,
    # Background args
    background_dynamic=None,
    background_ground_plane_alpha=None,
    background_video_alpha=None,
    background_dataset_path=None,
    background_dataset_videos=None,
    background_num_videos=None,
    background_shuffle_buffer_size=None,
    # Color args
    color_max_delta=None,
    color_step_std=None,
    # Other args
    stack_within_repeat=False):
  """Gin-configurable builder of training environment with augmentations."""

  camera_kwargs = {}
  if camera_camera_id is not None:
    camera_kwargs['camera_id'] = camera_camera_id
  if camera_horizontal_delta is not None:
    camera_kwargs['horizontal_delta'] = camera_horizontal_delta
  if camera_vertical_delta is not None:
    camera_kwargs['vertical_delta'] = camera_vertical_delta
  if camera_max_vel is not None:
    camera_kwargs['max_vel'] = camera_max_vel
  if camera_vel_std is not None:
    camera_kwargs['vel_std'] = camera_vel_std
  if camera_roll_delta is not None:
    camera_kwargs['roll_delta'] = camera_roll_delta
  if camera_max_roll_vel is not None:
    camera_kwargs['max_roll_vel'] = camera_max_roll_vel
  if camera_roll_std is not None:
    camera_kwargs['roll_std'] = camera_roll_std
  if camera_max_zoom_in_percent is not None:
    camera_kwargs['max_zoom_in_percent'] = camera_max_zoom_in_percent
  if camera_max_zoom_out_percent is not None:
    camera_kwargs['max_zoom_out_percent'] = camera_max_zoom_out_percent
  if camera_limit_to_upper_quadrant is not None:
    camera_kwargs['limit_to_upper_quadrant'] = camera_limit_to_upper_quadrant
  if camera_seed is not None:
    camera_kwargs['seed'] = camera_seed

  camera_kwargs = camera_kwargs if camera_kwargs else None

  background_kwargs = {}

  if background_dynamic is not None:
    background_kwargs['dynamic'] = background_dynamic
  if background_ground_plane_alpha is not None:
    background_kwargs['ground_plane_alpha'] = background_ground_plane_alpha
  if background_video_alpha is not None:
    background_kwargs['video_alpha'] = background_video_alpha
  if background_dataset_path is not None:
    background_kwargs['dataset_path'] = background_dataset_path
  if background_dataset_videos is not None:
    background_kwargs['dataset_videos'] = background_dataset_videos
  if background_num_videos is not None:
    background_kwargs['num_videos'] = background_num_videos
  if background_shuffle_buffer_size is not None:
    background_kwargs['shuffle_buffer_size'] = background_shuffle_buffer_size

  background_kwargs = background_kwargs if background_kwargs else None

  color_kwargs = {}

  if color_max_delta is not None:
    color_kwargs['max_delta'] = color_max_delta
  if color_step_std is not None:
    color_kwargs['step_std'] = color_step_std

  color_kwargs = color_kwargs if color_kwargs else None

  return load_dm_env(env_name, frame_shape, episode_length, action_repeat,
                     frame_stack, task_kwargs, render_kwargs, camera_kwargs,
                     background_kwargs, color_kwargs, stack_within_repeat)


@gin.configurable
def load_dm_env_for_eval(
    env_name,
    frame_shape=(84, 84, 3),
    episode_length=1000,
    action_repeat=4,
    frame_stack=3,
    task_kwargs=None,
    render_kwargs=None,
    # Camera args
    camera_camera_id=None,
    camera_horizontal_delta=None,
    camera_vertical_delta=None,
    camera_max_vel=None,
    camera_vel_std=None,
    camera_roll_delta=None,
    camera_max_roll_vel=None,
    camera_roll_std=None,
    camera_max_zoom_in_percent=None,
    camera_max_zoom_out_percent=None,
    camera_limit_to_upper_quadrant=None,
    camera_seed=None,
    # Background args
    background_dynamic=None,
    background_ground_plane_alpha=None,
    background_video_alpha=None,
    background_dataset_path=None,
    background_dataset_videos=None,
    background_num_videos=None,
    background_shuffle_buffer_size=None,
    # Color args
    color_max_delta=None,
    color_step_std=None,
    # Other args
    stack_within_repeat=False):
  """Gin-configurable builder of eval environment with augmentations."""

  camera_kwargs = {}
  if camera_camera_id is not None:
    camera_kwargs['camera_id'] = camera_camera_id
  if camera_horizontal_delta is not None:
    camera_kwargs['horizontal_delta'] = camera_horizontal_delta
  if camera_vertical_delta is not None:
    camera_kwargs['vertical_delta'] = camera_vertical_delta
  if camera_max_vel is not None:
    camera_kwargs['max_vel'] = camera_max_vel
  if camera_vel_std is not None:
    camera_kwargs['vel_std'] = camera_vel_std
  if camera_roll_delta is not None:
    camera_kwargs['roll_delta'] = camera_roll_delta
  if camera_max_roll_vel is not None:
    camera_kwargs['max_roll_vel'] = camera_max_roll_vel
  if camera_roll_std is not None:
    camera_kwargs['roll_std'] = camera_roll_std
  if camera_max_zoom_in_percent is not None:
    camera_kwargs['max_zoom_in_percent'] = camera_max_zoom_in_percent
  if camera_max_zoom_out_percent is not None:
    camera_kwargs['max_zoom_out_percent'] = camera_max_zoom_out_percent
  if camera_limit_to_upper_quadrant is not None:
    camera_kwargs['limit_to_upper_quadrant'] = camera_limit_to_upper_quadrant
  if camera_seed is not None:
    camera_kwargs['seed'] = camera_seed

  camera_kwargs = camera_kwargs if camera_kwargs else None

  background_kwargs = {}

  if background_dynamic is not None:
    background_kwargs['dynamic'] = background_dynamic
  if background_ground_plane_alpha is not None:
    background_kwargs['ground_plane_alpha'] = background_ground_plane_alpha
  if background_video_alpha is not None:
    background_kwargs['video_alpha'] = background_video_alpha
  if background_dataset_path is not None:
    background_kwargs['dataset_path'] = background_dataset_path
  if background_dataset_videos is not None:
    background_kwargs['dataset_videos'] = background_dataset_videos
  if background_num_videos is not None:
    background_kwargs['num_videos'] = background_num_videos
  if background_shuffle_buffer_size is not None:
    background_kwargs['shuffle_buffer_size'] = background_shuffle_buffer_size

  background_kwargs = background_kwargs if background_kwargs else None

  color_kwargs = {}

  if color_max_delta is not None:
    color_kwargs['max_delta'] = color_max_delta
  if color_step_std is not None:
    color_kwargs['step_std'] = color_step_std

  color_kwargs = color_kwargs if color_kwargs else None

  logging.info('camera_kwargs are: %s', str(camera_kwargs))
  logging.info('background_kwargs are: %s', str(background_kwargs))
  logging.info('color_kwargs are: %s', str(color_kwargs))

  return load_dm_env(env_name, frame_shape, episode_length, action_repeat,
                     frame_stack, task_kwargs, render_kwargs, camera_kwargs,
                     background_kwargs, color_kwargs, stack_within_repeat)


def load_dm_env(env_name,
                frame_shape=(84, 84, 3),
                episode_length=1000,
                action_repeat=4,
                frame_stack=3,
                task_kwargs=None,
                render_kwargs=None,
                camera_kwargs=None,
                background_kwargs=None,
                color_kwargs=None,
                stack_within_repeat=False):
  """Returns an environment from a domain name, task name."""
  domain_name, task_name = env_name.split('-')
  logging.info('Loading environment.')
  render_kwargs = render_kwargs or {}
  render_kwargs['width'] = frame_shape[0]
  render_kwargs['height'] = frame_shape[1]

  if 'camera_id' not in render_kwargs:
    render_kwargs['camera_id'] = 2 if domain_name == 'quadruped' else 0
  if camera_kwargs and 'camera_id' not in camera_kwargs:
    camera_kwargs['camera_id'] = 2 if domain_name == 'quadruped' else 0

  env = load_pixels(
      domain_name,
      task_name,
      task_kwargs=task_kwargs,
      render_kwargs=render_kwargs,
      camera_kwargs=camera_kwargs,
      background_kwargs=background_kwargs,
      color_kwargs=color_kwargs)

  env = FrameStackActionRepeatWrapper(
      env,
      action_repeat=action_repeat,
      stack_size=frame_stack,
      stack_within_repeat=stack_within_repeat)

  # Shorten episode length
  max_episode_steps = (episode_length + action_repeat - 1) // action_repeat
  env = wrappers.TimeLimit(env, max_episode_steps)
  return env


@gin.configurable
class FrameStackActionRepeatWrapper(wrappers.PyEnvironmentBaseWrapper):
  """Environment wrapper for stacking and action repeat."""

  def __init__(self,
               env,
               action_repeat=None,
               stack_size=None,
               stack_within_repeat=False):
    super(FrameStackActionRepeatWrapper, self).__init__(env)
    self._action_repeat = action_repeat or 1
    assert self._action_repeat >= 1
    self._stack_size = stack_size
    self._stack_within_repeat = stack_within_repeat

    # Create a copy of the observation spec as we might have to modify it.
    observation_spec = env.observation_spec()
    self._new_observation_spec = copy.copy(observation_spec)

    frame_shape = observation_spec['pixels'].shape

    # Setup frame stacking
    self._frames = None
    if stack_size:
      self._frames = collections.deque(maxlen=stack_size)
      # Update the observation spec in the environment.

      # Redefine pixels spec
      stacked_frame_shape = frame_shape[:2] + (frame_shape[2] * stack_size,)
      self._new_observation_spec['pixels'] = array_spec.ArraySpec(
          shape=stacked_frame_shape,
          dtype=observation_spec['pixels'].dtype,
          name='stacked_pixels')

  def _step(self, action):
    """Steps the environment."""
    if self.current_time_step().is_last():
      return self.reset()

    total_reward = 0

    for _ in range(self._action_repeat):
      time_step = self._env.step(action)

      if self._frames is not None and self._stack_within_repeat:
        self._frames.append(time_step.observation['pixels'])

      total_reward += time_step.reward
      if time_step.is_first() or time_step.is_last():
        break

    # Only add the last frame of the action repeat if we don't stack within.
    if self._frames is not None and not self._stack_within_repeat:
      self._frames.append(time_step.observation['pixels'])

    total_reward = np.asarray(
        total_reward, dtype=np.asarray(time_step.reward).dtype)

    # Stack frames.
    if self._frames is not None:
      time_step.observation['pixels'] = np.concatenate(self._frames, axis=2)

    return ts.TimeStep(time_step.step_type, total_reward, time_step.discount,
                       time_step.observation)

  def _reset(self):
    """Starts a new sequence and returns the first `TimeStep`."""

    time_step = self._env.reset()

    # Initial frame stacks
    if self._frames is not None:
      for _ in range(self._stack_size):
        self._frames.append(time_step.observation['pixels'])

    if self._frames:
      time_step.observation['pixels'] = np.concatenate(self._frames, axis=2)

    return ts.TimeStep(time_step.step_type, time_step.reward,
                       time_step.discount, time_step.observation)

  def observation_spec(self):
    """Defines the observations provided by the environment."""
    return self._new_observation_spec

  def render(self, mode='rgb_array'):
    if mode != 'rgb_array':
      raise ValueError('Only `rgb_array` mode is supported.')
    if self._current_time_step is None:
      raise ValueError('Step or reset the environment first.')

    # Make sure we only show the first image.
    output_img = self._current_time_step.observation['pixels'][:, :, :3]
    output_img = np.asarray(output_img, dtype=np.float32)

    if self._frames is not None:
      stacked_frames = np.asarray(self._frames[0], dtype=np.float32)
      stacked_frames = functools.reduce(np.add,
                                        list(self._frames)[1:], stacked_frames)
      stacked_frames = stacked_frames / self._stack_size
      output_img = np.concatenate([output_img, stacked_frames], axis=1)

    output_img = np.asarray(output_img, dtype=np.uint8)
    return output_img
