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

"""Utility for loading the AntMaze environments."""
import d4rl
import gym
import numpy as np


R = 'r'
G = 'g'
U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]

BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, G, 1, 1, G, G, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, 1, G, G, G, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, G, G, G, 1, G, G, G, G, G, 1],
                [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
                [1, G, G, G, G, G, G, 1, G, G, G, 1],
                [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
                [1, G, G, 1, G, 1, G, G, G, G, G, 1],
                [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
                [1, G, G, 1, G, G, G, 1, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


class AntMaze(d4rl.locomotion.ant.AntMazeEnv):
  """Utility wrapper for the AntMaze environments.

  For comparisons in the offline RL setting, we used unmodified AntMaze tasks,
  without this wrapper.
  """

  def __init__(self, map_name, non_zero_reset=False):
    self._goal_obs = np.zeros(29)
    if map_name == 'umaze':
      maze_map = U_MAZE
    elif map_name == 'medium':
      maze_map = BIG_MAZE
    elif map_name == 'large':
      maze_map = HARDEST_MAZE
    else:
      raise NotImplementedError
    super(AntMaze, self).__init__(maze_map=maze_map,
                                  reward_type='sparse',
                                  non_zero_reset=non_zero_reset,
                                  eval=True,
                                  maze_size_scaling=4.0,
                                  ref_min_score=0.0,
                                  ref_max_score=1.0)
    self.observation_space = gym.spaces.Box(
        low=np.full((58,), -np.inf),
        high=np.full((58,), np.inf),
        dtype=np.float32)

  def reset(self):
    super(AntMaze, self).reset()
    goal_xy = self._goal_sampler(np.random)
    state = self.sim.get_state()
    state = state._replace(
        qpos=np.concatenate([goal_xy, state.qpos[2:]]))
    self.sim.set_state(state)
    for _ in range(50):
      self.do_simulation(np.zeros(8), self.frame_skip)
    self._goal_obs = self.BASE_ENV._get_obs(self).copy()  # pylint: disable=protected-access
    super(AntMaze, self).reset()
    return self._get_obs()

  def step(self, action):
    super(AntMaze, self).step(action)
    s = self._get_obs()
    dist = np.linalg.norm(self._goal_obs[:2] - s[:2])
    # Distance threshold from [RIS, Chane-Sane '21] and [UPN, Srinivas '18].
    r = (dist <= 0.5)
    done = False
    info = {}
    return s, r, done, info

  def _get_obs(self):
    assert self._expose_all_qpos  # pylint: disable=protected-access
    s = self.BASE_ENV._get_obs(self)  # pylint: disable=protected-access
    return np.concatenate([s, self._goal_obs]).astype(np.float32)

  def _get_reset_location(self):
    if np.random.random() < 0.5:
      return super(AntMaze, self)._get_reset_location()
    else:
      return self._goal_sampler(np.random)


class OfflineAntWrapper(gym.ObservationWrapper):
  """Wrapper for exposing the goals of the AntMaze environments."""

  def __init__(self, env):
    env.observation_space = gym.spaces.Box(
        low=-1.0 * np.ones(58),
        high=np.ones(58),
        dtype=np.float32,
    )
    super(OfflineAntWrapper, self).__init__(env)

  def observation(self, observation):
    goal_obs = np.zeros_like(observation)
    goal_obs[:2] = self.env.target_goal
    return np.concatenate([observation, goal_obs])

  @property
  def max_episode_steps(self):
    return self.env.max_episode_steps


def make_offline_ant(env_name):
  """Loads the D4RL AntMaze environments."""
  if env_name == 'offline_ant_umaze':
    env = gym.make('antmaze-umaze-v2')
  elif env_name == 'offline_ant_umaze_diverse':
    env = gym.make('antmaze-umaze-diverse-v2')
  elif env_name == 'offline_ant_medium_play':
    env = gym.make('antmaze-medium-play-v2')
  elif env_name == 'offline_ant_medium_diverse':
    env = gym.make('antmaze-medium-diverse-v2')
  elif env_name == 'offline_ant_large_play':
    env = gym.make('antmaze-large-play-v2')
  elif env_name == 'offline_ant_large_diverse':
    env = gym.make('antmaze-large-diverse-v2')
  else:
    raise NotImplementedError
  return OfflineAntWrapper(env.env)
