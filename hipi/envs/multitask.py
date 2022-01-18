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

"""Defines classes for handling multi-task environments."""
import gym
import numpy as np
import tensorflow as tf


class TaskDistribution(object):
  """Defines a distribution over tasks.

  Tasks can be parametrized by goals or other representations. The evaluate,
  combine, and split methods may be called on either tensorflow Tensors or
  Numpy arrays, so we must support both.
  """

  @property
  def task_space(self):
    return self._task_space

  def sample(self):
    """Samples a task."""
    pass

  def _assert_is_batched(self, *arrays):
    """Checks that all arrays are batched.

    Args:
      *arrays: any number of arrays.
    """
    shape_list = []
    for array in arrays:
      if isinstance(array, tf.Tensor):
        shape_list.append(array.shape.as_list())
      else:
        shape_list.append(np.shape(array))
    # All arrays should have at least two dimensions.
    assert all([len(shape) >= 2 for shape in shape_list])
    # All arrays should have the same batch size.
    assert len(set([shape[0] for shape in shape_list])) == 1

  def _tf_call(self, fn, *inputs):
    any_tf_inputs = any([isinstance(array, tf.Tensor) for array in inputs])
    tf_inputs = [
        tf.constant(array) if not isinstance(array, tf.Tensor) else array
        for array in inputs
    ]
    output = fn(*tf_inputs)
    if not any_tf_inputs:
      output = [array.numpy() for array in output]
    return output

  def evaluate(self, states, actions, tasks):
    """Evaluates states and actions under the provided tasks.

    Args:
      states: a batch of states.
      actions: a batch of actions.
      tasks: a batch of tasks

    Returns:
      rewards: a batch of rewards
      dones: a batch of boolean done flags that are True when the episode
        has terminated. Note that this is the logical negation of the discount.
    """
    self._assert_is_batched(states, actions, tasks)
    return self._tf_call(self._evaluate, states, actions, tasks)

  def _evaluate(self, states, actions, tasks):
    raise NotImplementedError

  def combine(self, states, tasks):
    """Combines the states and task into a single representation.

    Args:
      states: a batch of states.
      tasks: a batch of tasks.

    Returns:
      states_and_tasks: a batch of states concatenated with tasks
    """
    self._assert_is_batched(states, tasks)
    return self._tf_call(self._combine, states, tasks)

  def _combine(self, states, tasks):
    tasks = tf.cast(tasks, states.dtype)
    return tf.concat([states, tasks], axis=-1)

  def split(self, states_and_tasks):
    """Splits a concatenated state+task into a state and a task.

    Args:
      states_and_tasks: a batch of states concatenated with tasks.

    Returns:
      states: a batch of states.
      tasks: a batch of tasks.
    """
    self._assert_is_batched(states_and_tasks)
    return self._tf_call(self._split, states_and_tasks)

  def _split(self, states_and_tasks):
    task_last_dim = self.task_space.low.shape[-1]
    states = states_and_tasks[Ellipsis, :-task_last_dim]
    tasks = states_and_tasks[Ellipsis, -task_last_dim:]
    return states, tasks

  def state_to_task(self, states):
    """Extracts the coordinates of the state that correspond to the task.

    For example, if a manipulation task, this function might extract the current
    position of the block. If this method is not overwritten, it defaults to
    using the entire state.

    Args:
      states: the states to convert to tasks.

    Returns:
      tasks: the tasks extracted from the states.
    """
    tasks = states
    return tasks

  @property
  def tasks(self):
    return None


class Dynamics(object):
  """Implements the task-agnostic dynamics.

  The motivationg for decoupling the task distribution from the dynamics is
  that we can define multiple task distributions for the same dynamics, and
  we can pass the task distribution to the replay buffer to perform relabelling.
  """

  @property
  def action_space(self):
    return self._action_space

  @property
  def observation_space(self):
    return self._observation_space

  def reset(self):
    """Resets the dynamics.

    Returns:
      state - a state from the initial state distribution.
    """
    pass

  def step(self, action):
    """Executes the action in the dynamics.

    Args:
      action: an action to take.

    Returns:
      next_state: the state of the environment after taking the action.
    """
    pass


class Environment(gym.Env):
  """An environment defined in terms of a dynamics and a task distribution.

  Internally, this environment samples tasks from the task distribution and
  concatenates the tasks to the observations, and computes the rewards.

  While decoupling the dynamics from the task distribution is convenient for
  off-policy relabelling, it is still helpful to have a Gym environment for
  data collection and interfacing with the underlying MaxEnt RL algorithm.
  """

  def __init__(self, dynamics, task_distribution, constant_task=None):
    """Initialize the environment.

    Args:
      dynamics: an instance of Dynamics, which defines the task transitions.
      task_distribution: an instance of TaskDistribution, which defines the
        rewards and termination conditions.
      constant_task: specifies a fixed task to use for all episodes. Set to None
        to use tasks sampled from the task distribution.
    """
    self._t = 0
    self._dynamics = dynamics
    self._task_distribution = task_distribution
    assert isinstance(dynamics.observation_space, gym.spaces.Box)
    assert isinstance(task_distribution.task_space, gym.spaces.Box)
    if constant_task is None:
      self._hide_task = False
      low = task_distribution.combine([dynamics.observation_space.low],
                                      [task_distribution.task_space.low])[0]
      high = task_distribution.combine([dynamics.observation_space.high],
                                       [task_distribution.task_space.high])[0]
    else:
      self._hide_task = True
      low = dynamics.observation_space.low
      high = dynamics.observation_space.high
      constant_task = np.array(constant_task, dtype=np.float32)
    self._constant_task = constant_task

    # Needed to get TF Agents to work.
    high[Ellipsis] = np.max(high)
    low[Ellipsis] = np.min(low)
    self.observation_space = gym.spaces.Box(low=low, high=high)
    self.action_space = dynamics.action_space

    self._state = None

  def set_constant_task(self, task):
    if task is not None:
      assert self._task_distribution.task_space.contains(task)
      task = np.array(task, dtype=self._task_distribution.task_space.dtype)
    self._constant_task = task

  def reset(self):
    """Resets the environment.

    Returns:
      state_and_task: an observation, which contains the state and task ID.
    """
    self._t = 0
    self._state = self._dynamics.reset()
    if self._constant_task is None:
      self._task = self._task_distribution.sample()
    else:
      self._task = self._constant_task
    if self._hide_task:
      state_and_task = self._state
    else:
      state_and_task = self._task_distribution.combine([self._state],
                                                       [self._task])[0]
    return state_and_task

  def step(self, action):
    self._t += 1
    # print('Step:', self._t)
    rewards, dones = self._task_distribution.evaluate([self._state], [action],
                                                      [self._task])
    reward = rewards[0]
    done = dones[0]
    self._state = self._dynamics.step(action)
    if self._hide_task:
      state_and_task = self._state
    else:
      state_and_task = self._task_distribution.combine([self._state],
                                                       [self._task])[0]
    return state_and_task, reward, done, {}
