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

"""Goal-conditioned PointMassEnv implementation with well and drunk spider envs.

Implementation of point-mass environment with well and drunk spider layouts.
Includes goal-conditioned and time limit bonus wrappers.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import gym
import numpy as np
from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

# Implementation of PointMass environment from Benjamin Eyesenbach:
#    https://github.com/google-research/google-research/tree/master/sorb

WALLS = {
    'IndianWell':
        np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]),
    'IndianWell2':
        np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]),
    'IndianWell3':
        np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]),
    'DrunkSpider':
        np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]),
    'DrunkSpiderShort':
        np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]),
}


def resize_walls(walls, factor):
  """Increase the environment by rescaling.

  Args:
    walls: 0/1 array indicating obstacle locations.
    factor: (int) factor by which to rescale the environment.

  Returns:
    walls: rescaled walls
  """
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  col_indices = np.array([i for i in range(width) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls


@gin.configurable
class PointMassEnv(gym.Env):
  """Class for 2D navigation in PointMass environment."""

  def __init__(self,
               env_name='IndianWell',
               start=None,
               resize_factor=1,
               action_noise=0.3,
               start_bounds=None,
               alive_bonus=0.,
               action_pen=False):
    """Initialize the point environment.

    Args:
      env_name: environment name
      start: starting position
      resize_factor: (int) Scale the map by this factor.
      action_noise: (float) Standard deviation of noise to add to actions. Use 0
        to add no noise.
      start_bounds: starting bound
      alive_bonus: a live bonus
      action_pen: penlaty for taking actions
    """
    walls = env_name
    self._start = start if start is None else np.array(start, dtype=float)

    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._height = height
    self._width = width

    if start_bounds is not None:
      self._start_space = gym.spaces.Box(
          low=start_bounds[0], high=start_bounds[1], dtype=np.float32)
    else:
      self._start_space = gym.spaces.Box(
          low=np.zeros(2), high=np.array([height, width]), dtype=np.float32)

    self._action_noise = action_noise
    self._alive_bonus = alive_bonus
    self._action_pen = float(action_pen)
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
    self.observation_space = gym.spaces.Dict({
        'observation':
            gym.spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([self._height, self._width]),
                dtype=np.float32),
        'fallen':
            gym.spaces.Discrete(2),
        'task_agn_reward':
            gym.spaces.Box(low=np.array(0), high=np.array(1))
    })
    self._validate_start_goal()
    self.reset()

  def _discretize_state(self, state, resolution=1.0):
    (i, j) = np.floor(resolution * state).astype(np.int)
    # Round down to the nearest cell if at the boundary.
    if i == self._height:
      i -= 1
    if j == self._width:
      j -= 1
    return (i, j)

  def _is_blocked(self, state):
    if not self.observation_space.contains(state):
      return True
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  def _validate_start_goal(self):
    if self._start is not None:
      assert not self._is_out_of_bounds(self._start), \
          'start must be in bounds of env'
      assert not self._is_in_well(self._start), 'start must not be in well'

  def _sample_empty_state(self):
    if self._start is not None:
      state = self._start.copy()
    else:
      state = self._start_space.sample()
    assert not self._is_in_well(state)
    return state

  def reset(self, reset_args=None):
    self.state = reset_args or self._sample_empty_state()
    obs = dict(observation=self.state.copy(), fallen=False, task_agn_reward=0.)
    return obs

  def _is_out_of_bounds(self, state):
    if not self.observation_space['observation'].contains(state):
      return True
    return False

  def _is_in_well(self, state):
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  def step(self, action):
    if self._action_noise > 0:
      action += np.random.normal(0, self._action_noise)
    action = np.clip(action, self.action_space.low, self.action_space.high)
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    rew = 0
    fallen = False
    task_agn = 0.
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if self._is_out_of_bounds(new_state):
          obs = dict(observation=new_state, fallen=True, task_agn_reward=1.)
          return obs, -1., True, {}
        elif self._is_in_well(new_state):
          # rew adds -1 if in well for one full step
          rew -= 1.0 / num_substeps
          fallen = True
          task_agn = 1.
        self.state = new_state
    # control rew discourages large actions, in range [-1.414*c, 0]
    rew += -1.0 * np.linalg.norm(action) * self._action_pen
    obs = dict(
        observation=self.state.copy(), fallen=fallen, task_agn_reward=task_agn)
    return obs, rew + self._alive_bonus, False, {}

  @property
  def walls(self):
    return self._walls


@gin.configurable
class GoalConditionedPointWrapper(gym.Wrapper):
  """Wrapper that appends goal to state produced by environment."""

  def __init__(self,
               env,
               goal=None,
               normalize_obs=False,
               task_rew_type='l2',
               reset_on_fail=True,
               goal_bounds=None,
               threshold_distance=1.0,
               fall_penalty=0.):
    """Initialize the environment.

    Args:
      env: an environment.
      goal: default goal to use, instead of sampling
      normalize_obs: returns normalized observations
      task_rew_type: string in ('l1', '-1', 'l2') indicating task reward to use
      reset_on_fail: boolean indicating if episode finishes when fallen == True
      goal_bounds: bounds for goal
      threshold_distance: (float) States are considered equivalent if they are
        at most this far away from one another.
      fall_penalty: penalty for falls
    """
    self._default_goal = goal if goal is None else np.array(goal)
    self._task_rew_type = task_rew_type
    self._reset_on_fail = reset_on_fail
    self._threshold_distance = threshold_distance
    self._fall_penalty = fall_penalty
    self._norm_obs = normalize_obs

    if normalize_obs:
      obs_space = self.observation_space = gym.spaces.Box(
          low=np.array([0., 0.]), high=np.array([1., 1.]), dtype=np.float32)
    else:
      obs_space = env.observation_space['observation']
    if goal_bounds:
      goal_space = gym.spaces.Box(
          low=goal_bounds[0], high=goal_bounds[1], dtype=np.float32)
    else:
      goal_space = env.observation_space['observation']
    if goal:
      assert goal_space.contains(np.array(goal)), 'goal not in goal space'
    super(GoalConditionedPointWrapper, self).__init__(env)

    # overwrites observation space to include goals
    self.observation_space = gym.spaces.Dict({
        'observation': obs_space,
        'goal': goal_space,
        'fallen': env.observation_space['fallen'],
        'task_agn_reward': env.observation_space['task_agn_reward']
    })

  def _normalize_obs(self, obs):
    return np.array(
        [obs[0] / float(self.env._height), obs[1] / float(self.env._width)])  # pylint: disable=protected-access

  def reset(self):
    """Resets environment, sampling goal if self._sample_goal == True."""
    obs = self.env.reset()
    if self._default_goal is not None:
      (obs, goal) = (obs, self._default_goal.copy())
    else:
      (obs, goal) = self._sample_goal_well(obs)

    obs['goal'] = self._goal = goal
    if self._norm_obs:
      obs['observation'] = self._normalize_obs(obs['observation'])
    return obs

  def _sample_goal_well(self, obs):
    goal_dist = 0
    state = obs['observation']
    while goal_dist < 4:
      goal = self.observation_space['goal'].sample()
      goal_dist = np.abs(state - goal).sum()
    return (obs, goal)

  def step(self, action):
    obs, rew, done, _ = self.env.step(action)
    obs['goal'] = goal = self._goal
    if self._norm_obs:
      obs['observation'] = self._normalize_obs(obs['observation'])
    state = obs['observation']
    if done:  # this means point mass fell outside bounds of env
      return obs, self._fall_penalty, done, {}

    task_rew = rew  # includes alive bonus and fallen in well penalty..
    if self._task_rew_type == 'l1':
      # task_rew range: [-1, 0]
      max_dist = self.env._height + self.env._width    # pylint: disable=protected-access
      task_rew += -np.abs(state - goal).sum() / max_dist
    elif self._task_rew_type == 'l2':
      max_dist = np.sqrt(self.env._height**2 + self.env._width**2)  # pylint: disable=protected-access
      task_rew += -np.linalg.norm(state - goal) / max_dist
    elif self._task_rew_type == '-1':  # alive penalty
      task_rew += -.1

    if self._is_done(state, goal):
      task_rew = 1.
      done = True
    elif obs['fallen']:  # if fallen into well
      done = True if self._reset_on_fail else False
      task_rew += self._fall_penalty

    return obs, task_rew, done, {}

  def set_sample_goal_args(self, default_goal=None, sample_goal=None):
    assert default_goal is not None
    assert sample_goal is not None
    self._default_goal = np.array(default_goal)
    self._sample_goal = sample_goal

  def _is_done(self, obs, goal):
    """Determines whether observation equals goal."""
    return np.linalg.norm(obs - goal) < self._threshold_distance


class NonTerminatingTimeLimit(wrappers.PyEnvironmentBaseWrapper):
  """Resets the environment without setting done = True.

  Resets the environment if either these conditions holds:
    1. The base environment returns done = True
    2. The time limit is exceeded.
  """

  def __init__(self, env, duration):
    super(NonTerminatingTimeLimit, self).__init__(env)
    self._duration = duration
    self._step_count = None

  def _reset(self):
    self._step_count = 0
    return self._env.reset()

  @property
  def duration(self):
    return self._duration

  def _step(self, action):
    if self._step_count is None:
      return self.reset()

    timestep = self._env.step(action)  # pylint: disable=protected-access

    self._step_count += 1
    if self._step_count >= self._duration or timestep.is_last():
      self._step_count = None

    return timestep


@gin.configurable
class TimeLimitBonus(wrappers.PyEnvironmentBaseWrapper):
  """End episodes after specified steps, adding early bonus/penalty."""

  def __init__(self, env, duration, early_term_bonus=1., early_term_penalty=1.):
    super(TimeLimitBonus, self).__init__(env)
    self._duration = duration
    self._num_steps = None
    self._early_term_bonus = early_term_bonus
    self._early_term_penalty = early_term_penalty

  def _reset(self):
    self._num_steps = 0
    return self._env.reset()

  def _step(self, action):
    if self._num_steps is None:
      return self.reset()

    time_step = self._env.step(action)

    self._num_steps += 1
    if self._num_steps >= self._duration:
      time_step = time_step._replace(step_type=ts.StepType.LAST)

    if time_step.is_last():
      if not time_step.observation['fallen']:
        early_bonus = (
            time_step.reward + self._early_term_bonus *
            (self._duration - self._num_steps))
      else:
        early_bonus = (
            time_step.reward - self._early_term_penalty *
            (self._duration - self._num_steps))
      time_step = time_step._replace(
          reward=early_bonus.astype(time_step.reward.dtype))

      self._num_steps = None

    return time_step

  @property
  def duration(self):
    return self._duration


@gin.configurable
def env_load_fn(environment_name,  # pylint: disable=dangerous-default-value
                max_episode_steps=50,
                resize_factor=1,
                env_kwargs=dict(action_noise=0., start=(0, 3)),
                goal_env_kwargs=dict(goal=(7, 3)),
                terminate_on_timeout=True):
  """Loads the selected environment and wraps it with the specified wrappers.

  Args:
    environment_name: Name for the environment to load.
    max_episode_steps: If None the max_episode_steps will be set to the default
      step limit defined in the environment's spec. No limit is applied if set
      to 0 or if there is no timestep_limit set in the environment's spec.
    resize_factor: A factor for resizing.
    env_kwargs: Arguments for envs.
    goal_env_kwargs: Arguments for goal envs.
    terminate_on_timeout: Whether to set done = True when the max episode steps
      is reached.

  Returns:
    A PyEnvironmentBase instance.
  """
  gym_env = PointMassEnv(
      environment_name, resize_factor=resize_factor, **env_kwargs)

  gym_env = GoalConditionedPointWrapper(gym_env, **goal_env_kwargs)
  env = gym_wrapper.GymWrapper(
      gym_env, discount=1.0, auto_reset=True, simplify_box_bounds=False)

  if max_episode_steps > 0:
    if terminate_on_timeout:
      env = TimeLimitBonus(env, max_episode_steps)
    else:
      env = NonTerminatingTimeLimit(env, max_episode_steps)

  return env
