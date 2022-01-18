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

"""Implements the multi-agent task list environement.

Agents must complete a sequence of tasks. The current tasks are:
0) Picking up a key.
1) Opening a door, consuming the key.
2) Picking up a ball.
3) Opening (toggling) a box.
4) Dropping the ball.
5) Reaching the goal.

The agents are optionally rewarded for completing, or penalized for
performing the task early.
"""
import gym
import gym_minigrid.minigrid as minigrid
import numpy as np
from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid.register import register


class TaskListEnv(multigrid.MultiGridEnv):
  """Environment with a list of tasks, sparse reward."""

  def __init__(self,
               size=8,
               n_agents=3,
               max_steps=250,
               reward_shaping=0.0,
               mistake_penalty=0.0,
               **kwargs):
    """Constructor for multi-agent gridworld environment generator.

    Args:
      size: Number of tiles for the width and height of the square grid.
      n_agents: The number of agents.
      max_steps: Number of environment steps before the episode end (max episode
        length).
      reward_shaping: Reward given for completing subtasks. 0 for sparse reward.
      mistake_penalty: Penalty for completing subtasks out of order.
      **kwargs: See superclass.
    """
    self.doors = [None] * n_agents
    self.keys = [None] * n_agents
    self.boxes = [None] * n_agents
    self.balls = [None] * n_agents
    self.task_idx = [0] * n_agents
    self.last_carrying = [None] * n_agents
    self.reward_shaping = reward_shaping
    self.mistake_penalty = mistake_penalty

    super().__init__(
        grid_size=size,
        max_steps=max_steps,
        n_agents=n_agents,
        fully_observed=True,
        **kwargs)
    self.metrics = {'keys_picked': 0,
                    'doors_opened': 0,
                    'balls_picked': 0,
                    'boxes_opened': 0,
                    'balls_dropped': 0,
                    'goals_reached': 0
                    }

    if self.minigrid_mode:
      self.position_obs_space = gym.spaces.Box(
          low=0, high=max(size, 10), shape=(12,), dtype='uint8')
    else:
      self.position_obs_space = gym.spaces.Box(
          low=0,
          high=max(size, 10),
          shape=(self.n_agents, 12),
          dtype='uint8')

    self.observation_space = gym.spaces.Dict({
        'image': self.image_obs_space,
        'direction': self.direction_obs_space,
        'position': self.position_obs_space
    })

  def _gen_grid(self, width, height):
    self.height = height

    # Create an empty grid
    self.grid = multigrid.Grid(width, height)

    # Generate the surrounding walls
    self.grid.wall_rect(0, 0, width, height)

    # Place a goal in the bottom-right corner
    self.place_obj(minigrid.Goal(), max_tries=100)
    self.place_agent()

    for i in range(self.n_agents):
      self.doors[i] = multigrid.Door('grey', is_locked=True)
      self.place_obj(self.doors[i], max_tries=100)
      self.keys[i] = minigrid.Key('grey')
      self.place_obj(self.keys[i], max_tries=100)
      self.balls[i] = minigrid.Ball('purple')
      self.place_obj(self.balls[i], max_tries=100)
      self.boxes[i] = minigrid.Box('green')
      self.place_obj(self.boxes[i], max_tries=100)

    self.task_idx = [0] * self.n_agents

    self.mission = 'Do some random tasks'

  def add_extra_info(self, obs):
    for i in range(self.n_agents):
      carried_encoding = np.zeros(3)
      if self.carrying[i]:
        carried_encoding = self.carrying[i].encode()
      task_encoding = np.zeros(7)
      task_encoding[self.task_idx[i]] = 1
      extra_info = np.concatenate([task_encoding, carried_encoding])
      if self.minigrid_mode:
        obs['position'] = np.concatenate((obs['position'], extra_info))
      else:
        obs['position'][i] = np.concatenate((obs['position'][i], extra_info))
    return obs

  def step(self, action):
    obs, reward, done, info = multigrid.MultiGridEnv.step(self, action)
    if all([idx == 6 for idx in self.task_idx]):
      done = True
    obs = self.add_extra_info(obs)
    return obs, reward, done, info

  def reset(self):
    obs = super(TaskListEnv, self).reset()
    obs = self.add_extra_info(obs)
    return obs

  def step_one_agent(self, action, agent_id):
    reward = 0

    # Get the position in front of the agent
    fwd_pos = self.front_pos[agent_id]

    # Get the contents of the cell in front of the agent
    fwd_cell = self.grid.get(*fwd_pos)

    # Rotate left
    if action == self.actions.left:
      self.agent_dir[agent_id] -= 1
      if self.agent_dir[agent_id] < 0:
        self.agent_dir[agent_id] += 4
      self.rotate_agent(agent_id)

    # Rotate right
    elif action == self.actions.right:
      self.agent_dir[agent_id] = (self.agent_dir[agent_id] + 1) % 4
      self.rotate_agent(agent_id)

    # Move forward
    elif action == self.actions.forward:
      successful_forward = self._forward(agent_id, fwd_pos)
      # Note: agent doesn't actually move forward onto a goal.
      if successful_forward and fwd_cell and fwd_cell.type == 'goal':
        # Task 5 is to reach the goal.
        if self.task_idx[agent_id] == 5:
          self.task_idx[agent_id] += 1
          self.agent_is_done(agent_id)
          reward = 1

    # Pick up an object
    elif action == self.actions.pickup:
      successful_pickup = self._pickup(agent_id, fwd_pos)
      if successful_pickup:
        if self.carrying[agent_id].type == 'key':
          # Task 0 is the pick up the key
          if self.task_idx[agent_id] == 0:
            self.task_idx[agent_id] += 1
            self.metrics['keys_picked'] += 1
            reward += self.reward_shaping
          else:
            reward -= self.mistake_penalty
        elif self.carrying[agent_id].type == 'ball':
          # Task 2 is to pick up a ball.
          if self.task_idx[agent_id] == 2:
            self.task_idx[agent_id] += 1
            reward += self.reward_shaping
            self.metrics['balls_picked'] += 1
          else:
            reward -= self.mistake_penalty
        else:
          reward -= self.mistake_penalty

    # Drop an object
    elif action == self.actions.drop:
      current_item = self.carrying[agent_id]
      successful_drop = self._drop(agent_id, fwd_pos)
      # Task 4 is to drop a ball.
      if (successful_drop and current_item and
          current_item.type == 'ball' and
          self.task_idx[agent_id] == 4):
        self.task_idx[agent_id] += 1
        reward += self.reward_shaping
        self.metrics['balls_dropped'] += 1

    # Toggle/activate an object
    elif action == self.actions.toggle:
      successful_toggle = self._toggle(agent_id, fwd_pos)
      if successful_toggle:
        if fwd_cell.type == 'door':
          # Task 1 is to open a door.
          if self.task_idx[agent_id] == 1:
            self.task_idx[agent_id] += 1
            reward += self.reward_shaping
            self.metrics['doors_opened'] += 1
            self.carrying[agent_id] = None
          else:
            reward -= self.mistake_penalty
        elif fwd_cell.type == 'box':
          # Task 3 is to open a box.
          if self.task_idx[agent_id] == 3:
            self.task_idx[agent_id] += 1
            reward += self.reward_shaping
            self.metrics['boxes_opened'] += 1
          else:
            reward -= self.mistake_penalty

    # Done action -- by default acts as no-op.
    elif action == self.actions.done:
      pass

    else:
      assert False, 'unknown action'

    return reward


class TaskListEnv8x8(TaskListEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=2, reward_shaping=1, **kwargs)


class TaskListEnvSparse8x8(TaskListEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=2, reward_shaping=0, **kwargs)


class TaskListEnv8x8Minigrid(TaskListEnv):

  def __init__(self, **kwargs):
    super().__init__(size=8, n_agents=1, reward_shaping=1, minigrid_mode=True,
                     **kwargs)

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname


register(
    env_id='MultiGrid-TaskList-8x8-v0',
    entry_point=module_path + ':TaskListEnv8x8'
)


register(
    env_id='MultiGrid-TaskList-Sparse-8x8-v0',
    entry_point=module_path + ':TaskListEnvSparse8x8'
)


register(
    env_id='MultiGrid-TaskList-8x8-Minigrid-v0',
    entry_point=module_path + ':TaskListEnv8x8Minigrid'
)
