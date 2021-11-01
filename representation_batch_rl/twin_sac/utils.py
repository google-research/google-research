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

# python3
"""Utilities used for this twin sac implementation."""
import json

from absl import flags
from absl import logging
from dm_control import manipulation
from dm_control import suite
from tf_agents.environments import dm_control_wrapper
from tf_agents.environments import suite_dm_control
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers

from distracting_control import suite as distractor_suite
from representation_batch_rl.gym.wrappers.action_repeat_dm_wrapper import ActionRepeatDMWrapper
from representation_batch_rl.gym.wrappers.flatten_image_observations_wrapper import FlattenImageObservationsWrapper
from representation_batch_rl.gym.wrappers.frame_stack_wrapper import FrameStackWrapperTfAgents
from representation_batch_rl.representation_batch_rl.tf_utils import JointImageObservationsWrapper


FLAGS = flags.FLAGS


def make_hparam_string(xm_parameters, **hparam_str_dict):
  if xm_parameters:
    for key, value in json.loads(xm_parameters).items():
      if key not in hparam_str_dict:
        hparam_str_dict[key] = value
  return ','.join([
      '%s=%s' % (k, str(hparam_str_dict[k]))
      for k in sorted(hparam_str_dict.keys())
  ])


def _load_dm_env(domain_name,
                 task_name,
                 pixels,
                 action_repeat,
                 max_episode_steps=None,
                 obs_type='pixels',
                 distractor=False):
  """Load a Deepmind control suite environment."""
  try:
    if not pixels:
      env = suite_dm_control.load(domain_name=domain_name, task_name=task_name)
      if action_repeat > 1:
        env = wrappers.ActionRepeat(env, action_repeat)

    else:
      def wrap_repeat(env):
        return ActionRepeatDMWrapper(env, action_repeat)
      camera_id = 2 if domain_name == 'quadruped' else 0

      pixels_only = obs_type == 'pixels'
      if distractor:
        render_kwargs = dict(width=84, height=84, camera_id=camera_id)

        env = distractor_suite.load(
            domain_name,
            task_name,
            difficulty='hard',
            dynamic=False,
            background_dataset_path='DAVIS/JPEGImages/480p/',
            task_kwargs={},
            environment_kwargs={},
            render_kwargs=render_kwargs,
            visualize_reward=False,
            env_state_wrappers=[wrap_repeat])

        # env = wrap_repeat(env)

        # env = suite.wrappers.pixels.Wrapper(
        #     env,
        #     pixels_only=pixels_only,
        #     render_kwargs=render_kwargs,
        #     observation_key=obs_type)

        env = dm_control_wrapper.DmControlWrapper(env, render_kwargs)

      else:
        env = suite_dm_control.load_pixels(
            domain_name=domain_name, task_name=task_name,
            render_kwargs=dict(width=84, height=84, camera_id=camera_id),
            env_state_wrappers=[wrap_repeat],
            observation_key=obs_type,
            pixels_only=pixels_only)

    if action_repeat > 1 and max_episode_steps is not None:
      # Shorten episode length.
      max_episode_steps = (max_episode_steps +
                           action_repeat - 1) // action_repeat
      env = wrappers.TimeLimit(env, max_episode_steps)

    return env

  except ValueError as e:
    logging.warning('cannot instantiate dm env: domain_name=%s, task_name=%s',
                    domain_name, task_name)
    logging.warning('Supported domains and tasks: %s', str(
        {key: list(val.SUITE.keys()) for key, val in suite._DOMAINS.items()}))  # pylint: disable=protected-access
    raise e


def load_env(env_name, seed,
             action_repeat = 0,
             frame_stack = 1,
             obs_type='pixels'):
  """Loads a learning environment.

  Args:
    env_name: Name of the environment.
    seed: Random seed.
    action_repeat: (optional) action repeat multiplier. Useful for DM control
      suite tasks.
    frame_stack: (optional) frame stack.
    obs_type: `pixels` or `state`
  Returns:
    Learning environment.
  """

  action_repeat_applied = False
  state_env = None

  if env_name.startswith('dm'):
    _, domain_name, task_name = env_name.split('-')
    if 'manipulation' in domain_name:
      env = manipulation.load(task_name)
      env = dm_control_wrapper.DmControlWrapper(env)
    else:
      env = _load_dm_env(domain_name, task_name, pixels=False,
                         action_repeat=action_repeat)
      action_repeat_applied = True
    env = wrappers.FlattenObservationsWrapper(env)

  elif env_name.startswith('pixels-dm'):
    if 'distractor' in env_name:
      _, _, domain_name, task_name, _ = env_name.split('-')
      distractor = True
    else:
      _, _, domain_name, task_name = env_name.split('-')
      distractor = False
    # TODO(tompson): Are there DMC environments that have other
    # max_episode_steps?
    env = _load_dm_env(domain_name, task_name, pixels=True,
                       action_repeat=action_repeat,
                       max_episode_steps=1000,
                       obs_type=obs_type,
                       distractor=distractor)
    action_repeat_applied = True
    if obs_type == 'pixels':
      env = FlattenImageObservationsWrapper(env)
      state_env = None
    else:
      env = JointImageObservationsWrapper(env)
      state_env = tf_py_environment.TFPyEnvironment(
          wrappers.FlattenObservationsWrapper(
              _load_dm_env(
                  domain_name,
                  task_name,
                  pixels=False,
                  action_repeat=action_repeat)))

  else:
    env = suite_mujoco.load(env_name)
    env.seed(seed)

  if action_repeat > 1 and not action_repeat_applied:
    env = wrappers.ActionRepeat(env, action_repeat)
  if frame_stack > 1:
    env = FrameStackWrapperTfAgents(env, frame_stack)

  env = tf_py_environment.TFPyEnvironment(env)

  return env, state_env

