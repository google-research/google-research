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

"""Implements a simple point navigation environment."""
from envs import multitask
import gin
import gym
import numpy as np
import tensorflow as tf

WALLS = {
    "Small":
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "Cross":
        np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
    "FourRooms":
        np.array([
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ]),
    "Spiral11x11":
        np.array(  # max_goal_dist = 45
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            ]),
    "Maze11x11":
        np.array(  # max_goal_dist = 49
            [
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]),
    "Tunnel":
        np.array(  # max_goal_dist = 70
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]),
    "FlyTrapBig":
        np.array(  # max_goal_dist = 38
            [
                [
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 1
                ],
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                    1, 1, 0
                ],
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0
                ],
                [
                    0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                    1, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0
                ],
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1
                ],
                [
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 1
                ],
            ]),
}


def resize_walls(walls, factor):
  """Increase the environment by rescaling.

  Args:
    walls: 0/1 array indicating obstacle locations.
    factor: (int) factor by which to rescale the environment.
  Returns:
    resized_walls: the resized array.
  """
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  col_indices = np.array([i for i in range(width) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls


@gin.configurable
class PointDynamics(multitask.Dynamics):
  """Abstract class for 2D navigation environments."""

  def __init__(
      self,
      walls=None,
      resize_factor=1,
      action_noise=1.0,
      random_initial_state=True,
  ):
    """Initialize the point environment.

    Args:
      walls: (str) name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
      action_noise: (float) Standard deviation of noise to add to actions. Use 0
        to add no noise.
      random_initial_state: (bool) Whether the initial state should be chosen
        uniformly across the state space, or fixed at (0, 0).
    """
    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._height = height
    self._width = width
    self._action_noise = action_noise
    self._random_initial_state = random_initial_state
    self._action_space = gym.spaces.Box(
        low=np.full(2, -1.0),
        high=np.full(2, 1.0),
        dtype=np.float32,
    )
    self._observation_space = gym.spaces.Box(
        low=np.array([0.0, 0.0]),
        high=np.array([self._height, self._width]),
        dtype=np.float32,
    )

  def _sample_empty_state(self):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state = np.array(
        [candidate_states[0][state_index], candidate_states[1][state_index]],
        dtype=np.float,
    )
    state += np.random.uniform(size=2)
    assert not self._is_blocked(state)
    return state

  def reset(self):
    if self._random_initial_state:
      self.state = self._sample_empty_state()
    else:
      self.state = np.zeros(2, dtype=np.float)
    assert not self._is_blocked(self.state)
    return self.state.copy().astype(np.float32)

  def _is_blocked(self, state):
    if not self.observation_space.contains(state):
      return True
    (i, j) = self._discretize_state(state)
    return self._walls[i, j] == 1

  def _discretize_state(self, state, resolution=1.0):
    (i, j) = np.floor(resolution * state).astype(np.int)
    # Round down to the nearest cell if at the boundary.
    if i == self._height:
      i -= 1
    if j == self._width:
      j -= 1
    return (i, j)

  def step(self, action):
    if self._action_noise > 0:
      action += np.random.normal(0, self._action_noise)
    action = np.clip(action, self.action_space.low, self.action_space.high)
    self.action_space.contains(action)
    action = action[:2]
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state

    return self.state.copy().astype(np.float32)


@gin.configurable
class PointGoalDistribution(multitask.TaskDistribution):
  """Defines the goal distribution for the point navigation environment."""

  def __init__(
      self,
      point_dynamics,
      random_goal=True,
      use_neg_rew=False,
      min_goal_distance=0.0,
      at_goal_dist=1.0,
  ):
    self._at_goal_dist = at_goal_dist
    self._point_dynamics = point_dynamics
    self._task_space = point_dynamics.observation_space
    self._random_goal = random_goal
    self._use_neg_rew = use_neg_rew
    self._min_goal_distance = min_goal_distance

  def sample(self):
    if self._random_goal:
      goal = None
      while (goal is None or np.linalg.norm(self._point_dynamics.state - goal) <
             self._min_goal_distance):
        goal = self._point_dynamics._sample_empty_state()  # pylint: disable=protected-access
      return goal.astype(np.float32)
    else:
      # The maximum observation is the top right corner. We subtract one such
      # that the full goal radius is within the environment.
      return self._point_dynamics.observation_space.high.astype(
          np.float32) - 1.0

  def _evaluate(self, states, actions, tasks):
    """Evaluates a goal-reaching task in the point navigation environment."""
    dist = tf.norm(states - tasks, axis=1)
    dones = dist < self._at_goal_dist
    if self._use_neg_rew:
      rewards = tf.cast(dones, tf.float32) - 1.0  # Rewards in {-1, 0}
    else:
      rewards = tf.cast(dones, tf.float32)  # Rewards in {0, 1}
    return rewards, dones

  def state_to_task(self, states):
    return states


@gin.configurable
class PointTaskDistribution(multitask.TaskDistribution):
  """Defines the goal distribution for the point navigation environment."""

  def __init__(self, point_dynamics, use_neg_rew=False, goals=None):
    self._use_neg_rew = use_neg_rew
    if goals is None:
      height = point_dynamics._height
      width = point_dynamics._width
      goals = [[1, 1], [height - 2, 1], [height - 2, width - 2], [1, width - 2]]
    self._point_dynamics = point_dynamics
    self._goals = tf.constant(goals, dtype=tf.float32)
    self._num_goals = len(goals)
    self._task_space = gym.spaces.Box(
        low=np.zeros(self._num_goals),
        high=np.ones(self._num_goals),
        dtype=np.float32,
    )
    self._tasks = np.eye(self._num_goals)

  def sample(self):
    task = np.zeros(self._num_goals)
    task[np.random.choice(self._num_goals)] = 1.0
    return task.astype(np.float32)

  def _evaluate(self, states, actions, tasks):
    """Evaluates a goal-reaching task in the point navigation environment."""
    goals = tf.matmul(tasks, self._goals)
    assert len(goals) == len(tasks)
    assert goals.shape[1] == 2
    dist = tf.norm(states - goals, axis=1)
    dones = dist < 1.0
    if self._use_neg_rew:
      rewards = tf.cast(dones, tf.float32) - 1.0  # rewards in {-1, 0}
    else:
      rewards = tf.cast(dones, tf.float32)  # Rewards in {0, 1}
    return rewards, dones

  @property
  def tasks(self):
    return np.eye(self._num_goals)
