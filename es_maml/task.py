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

# pylint: disable=g-doc-return-or-yield,unused-argument,missing-docstring,g-doc-args,line-too-long,invalid-name,pointless-string-statement
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import numpy as np


class Task(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def state_dimensionality(self):
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def action_dimensionality(self):
    raise NotImplementedError("Abstract method")


class ClassificationTask(Task):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def generate_samples(self):
    raise NotImplementedError("Abstract method")


class RLTask(Task):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def deterministic_start(self):
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def step(self, action):
    raise NotImplementedError("Abstract method")


class SinusodialTask(ClassificationTask):

  def __init__(self, task_id, sample_num=100, **kwargs):
    self.task_id = task_id
    np.random.seed(task_id)
    self.amp = np.random.uniform(0.1, 5.0)
    self.phase = np.random.uniform(0.0, np.pi)
    self.sample_num = sample_num

  def generate_samples(self):
    xs = np.random.uniform(-5.0, 5.0, self.sample_num)
    ys = np.array([self.amp * np.sin(x - self.phase) for x in xs])
    return xs, ys

  def state_dimensionality(self):
    return 1

  def action_dimensionality(self):
    return 1


class NavigationTask2d(RLTask):

  def __init__(self, task_id, **kwargs):
    self.task_id = task_id
    np.random.seed(task_id)
    self.goal_pos = np.random.uniform(low=-1.0, high=1.0, size=2)
    self.t = 0

  def random_start(self):
    self.agent_pos = np.array([0.0, 0.0])
    return self.agent_pos

  def deterministic_start(self):
    self.agent_pos = np.array([0.0, 0.0])
    return self.agent_pos

  def step(self, action):
    clipped_action = np.clip(action, a_min=-0.1, a_max=0.1)
    self.agent_pos += clipped_action
    self.agent_pos = np.clip(self.agent_pos, a_min=-1.0, a_max=1.0)

    reward = -1.0 * np.square(np.linalg.norm(self.agent_pos - self.goal_pos))
    done = False
    if reward >= -0.01:
      done = True

    return self.agent_pos, reward, done, None

  def reset(self):
    self.t = 0
    return self.deterministic_start()

  def restart(self):
    return self.reset()

  def state_dimensionality(self):
    return 2

  def action_dimensionality(self):
    return 2


class NavigationTask4corner(RLTask):

  def __init__(self, task_id, **kwargs):
    self.task_id = task_id
    corner_int = task_id % 4
    corner_id_to_pos = {
        0: np.array([2., 2.]),
        1: np.array([-2., 2.]),
        2: np.array([-2., -2.]),
        3: np.array([2., -2.])
    }
    self.goal_pos = corner_id_to_pos[corner_int]
    self.t = 0

  def random_start(self):
    self.agent_pos = np.array([0.0, 0.0])
    return self.agent_pos

  def deterministic_start(self):
    self.agent_pos = np.array([0.0, 0.0])
    return self.agent_pos

  def step(self, action):
    clipped_action = np.clip(action, a_min=-0.1, a_max=0.1)
    self.agent_pos += clipped_action
    self.agent_pos = np.clip(self.agent_pos, a_min=-5.0, a_max=5.0)

    sq_dist = np.square(np.linalg.norm(self.agent_pos - self.goal_pos))
    alive_penalty = -4.0
    # reward is only shown if near the corner
    reward = alive_penalty + max(0.0, 4.0 - sq_dist)
    return self.agent_pos, reward, False, None

  def reset(self):
    self.t = 0
    return self.deterministic_start()

  def restart(self):
    return self.reset()

  def state_dimensionality(self):
    return 2

  def action_dimensionality(self):
    return 2


class NavigationTaskCombo(RLTask):

  def __init__(self, task_id, num_subset_goals=2, num_goals=6, **kwargs):
    self.task_id = task_id
    self.id_to_goal = {}
    for i in range(num_goals):
      temp_goal = np.sqrt(8.0) * np.array([
          np.cos(2 * np.pi * i / float(num_goals)),
          np.sin(2 * np.pi * i / float(num_goals))
      ])
      self.id_to_goal[i] = np.copy(temp_goal)

    np.random.seed(task_id)
    self.goal_ids = np.random.choice(num_goals, num_subset_goals, replace=False)
    self.t = 0.0
    self.num_subset_goals = num_subset_goals
    self.num_goals = num_goals
    self.boundary = 4.0
    self.visited_goals = []

  def random_start(self):
    self.t = 0.0
    self.visited_goals = []
    self.agent_pos = np.array([0.0, 0.0])
    self.final_obs = np.concatenate((self.agent_pos, np.array([self.t])))
    # return self.final_obs
    return self.agent_pos

  def deterministic_start(self):
    self.t = 0.0
    self.visited_goals = []
    self.agent_pos = np.array([0.0, 0.0])
    self.final_obs = np.concatenate((self.agent_pos, np.array([self.t])))
    # return self.final_obs
    return self.agent_pos

  def step(self, action):
    self.t += 1.0
    clipped_action = np.clip(action, a_min=-0.1, a_max=0.1)
    self.agent_pos += clipped_action
    self.agent_pos = np.clip(self.agent_pos, a_min=-5.0, a_max=5.0)

    total_reward = 0.0
    for g in range(self.num_goals):
      if g not in self.goal_ids:
        temp_dist = np.square(
            np.linalg.norm(self.agent_pos - self.id_to_goal[g]))
        # higher penalties
        wrong_goal_penalty = 10000.0 * min(0.0, temp_dist - self.boundary)
        total_reward += wrong_goal_penalty
      else:  # g is a correct goal
        if g not in self.visited_goals:  # if it hasn't been turned off yet
          sq_dist = np.square(
              np.linalg.norm(self.agent_pos - self.id_to_goal[g]))
          alive_penalty = -1.0 * self.boundary
          # reward is only shown if near the corner
          total_reward += (alive_penalty + max(0.0, self.boundary - sq_dist))

          if sq_dist < 0.01:
            self.visited_goals.append(g)
        # g is a correct goal and was visited, and this goal is turned off
        else:
          total_reward += 0.0
    self.final_obs = np.concatenate((self.agent_pos, np.array([self.t])))
    # return self.final_obs, total_reward, False, None
    return self.agent_pos, total_reward, False, None

  def reset(self):
    self.t = 0.0
    self.visited_goals = []
    return self.deterministic_start()

  def restart(self):
    return self.reset()

  def state_dimensionality(self):
    return 2

  def action_dimensionality(self):
    return 2
