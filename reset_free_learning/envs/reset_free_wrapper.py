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

"""Gym wrapper environments for reset-free learning.

The environment retains its old state, but goals/tasks can be changed during the
episode.
"""

from gym import Wrapper
import tensorflow as tf


class ResetFreeWrapper(Wrapper):

  def __init__(self,
               env,
               reset_goal_frequency,
               variable_horizon_for_reset=False,
               num_success_states=1,
               full_reset_frequency=None,
               reset_goal_fn=None):

    super(ResetFreeWrapper, self).__init__(env)
    self._rfw_step_count = 0
    self._virtual_episode_steps = 0
    self._goal_switch = False
    self._reset_goal_frequency = reset_goal_frequency
    self._variable_horizon_for_reset = variable_horizon_for_reset
    self._num_success_states = num_success_states
    self._full_reset_frequency = full_reset_frequency
    self._reset_goal_fn = reset_goal_fn
    self._reset_state_candidates = None
    self._forward_or_reset_goal = True  # True -> forward, False -> reset goal
    self._goal_queue = []

  def reset(self):
    # variable resets
    if self._variable_horizon_for_reset:
      self._success_state_seq_length = 0
    self._virtual_episode_steps = 0

    if self._goal_switch:
      if self._reset_goal_fn is None or self._reset_state_candidates is None:
        self.env.reset_goal()
        self._forward_or_reset_goal = True
      else:
        if not self._goal_queue:
          self._forward_or_reset_goal = False
          next_task_goal = self.env.get_next_goal()
          self._goal_queue = [next_task_goal] + self._goal_queue
          reset_goal = self._reset_goal_fn(
              self._reset_state_candidates,
              tf.constant(self.env._get_obs(), dtype=tf.float32),
              tf.constant(next_task_goal, dtype=tf.float32))

          if isinstance(reset_goal, tf.Tensor):
            reset_goal = reset_goal.numpy()

          self.env.reset_goal(goal=reset_goal)
        else:
          self._forward_or_reset_goal = True
          next_task_goal = self._goal_queue.pop()
          self.env.reset_goal(goal=next_task_goal)

      self._goal_switch = False
      return self.env._get_obs()
    else:
      self._forward_or_reset_goal = True
      return self.env.reset()

  def step(self, action):
    obs, reward, done, info = self.env.step(
        action)  # always check if the underneath env is done

    self._rfw_step_count += 1
    self._virtual_episode_steps += 1

    # a full reset including robot arm and objects
    if not done and self._full_reset_frequency is not None and self._rfw_step_count % self._full_reset_frequency == 0:
      done = True

    # finishes the episode and reset the goal, but no change in env state
    if not done and self._virtual_episode_steps % self._reset_goal_frequency == 0:
      done = True
      self._goal_switch = True

    # only have variable horizon for reset goals
    # switch to forward goal when reset state has been achieved for a bit
    if self._variable_horizon_for_reset and not done and not self._forward_or_reset_goal:
      if self.env.is_successful():
        self._success_state_seq_length += 1
      else:
        self._success_state_seq_length = 0
      if self._success_state_seq_length >= self._num_success_states:
        done = True
        self._goal_switch = True

    return obs, reward, done, info

  def set_reset_goal_fn(self, reset_goal_fn):
    self._reset_goal_fn = reset_goal_fn

  # it is not possible to sample the replay buffer while the data is being
  # collected, therefore we have to lazily update the reset candidates.
  def set_reset_candidates(self, reset_candidates):
    self._reset_state_candidates = reset_candidates


class CustomOracleResetWrapper(Wrapper):

  def __init__(self,
               env,
               partial_reset_frequency,
               episodes_before_full_reset,
               reset_goal_fn=None):

    super(CustomOracleResetWrapper, self).__init__(env)
    self._rfw_step_count = 0
    self._partial_reset_frequency = partial_reset_frequency
    self._episodes_before_full_reset = episodes_before_full_reset
    self._episode_count = -1
    self._reset_goal_fn = reset_goal_fn
    self._reset_state_candidates = None

  def reset(self):
    self._episode_count += 1
    if self._episode_count % self._episodes_before_full_reset == 0:
      return self.env.reset()
    elif self._reset_state_candidates is not None:
      next_task_goal = self.env.get_next_goal()
      obs_before_teleporting = self.env._get_obs()
      reset_goal = self._reset_goal_fn(
          self._reset_state_candidates,
          tf.constant(obs_before_teleporting, dtype=tf.float32),
          tf.constant(next_task_goal, dtype=tf.float32))
      if isinstance(reset_goal, tf.Tensor):
        reset_goal = reset_goal.numpy()
      self.env.do_custom_reset(pos=reset_goal)
      self.env.reset()
      self.env.reset_goal(
          goal=next_task_goal
      )  # otherwise the reset state would be for the wrong goal
      return self.env._get_obs()
    else:
      return self.env.reset()

  def step(self, action):
    obs, reward, done, info = self.env.step(
        action)  # always check if the underneath env is done

    self._rfw_step_count += 1

    # a full reset including robot arm and objects
    if not done and self._rfw_step_count % self._partial_reset_frequency == 0:
      done = True

    return obs, reward, done, info

  def set_reset_goal_fn(self, reset_goal_fn):
    self._reset_goal_fn = reset_goal_fn

  # it is not possible to sample the replay buffer while the data is being
  # collected, therefore we have to lazily update the reset candidates.
  def set_reset_candidates(self, reset_candidates):
    self._reset_state_candidates = reset_candidates


class GoalTerminalResetFreeWrapper(Wrapper):

  def __init__(self,
               env,
               reset_goal_frequency,
               num_success_states=1,
               full_reset_frequency=None,
               reset_goal_fn=None):

    super(GoalTerminalResetFreeWrapper, self).__init__(env)
    self._rfw_step_count = 0
    self._virtual_episode_steps = 0
    self._goal_switch = False
    self._reset_goal_frequency = reset_goal_frequency
    self._num_success_states = num_success_states
    self._full_reset_frequency = full_reset_frequency
    self._reset_goal_fn = reset_goal_fn
    self._reset_state_candidates = None
    self._forward_or_reset_goal = True  # True -> forward, False -> reset goal
    self._goal_queue = []

  def reset(self):
    self._success_state_seq_length = 0
    self._virtual_episode_steps = 0

    if self._goal_switch:
      if self._reset_goal_fn is None:
        self.env.reset_goal()
        self._forward_or_reset_goal = True
      else:
        if not self._goal_queue:
          self._forward_or_reset_goal = False
          next_task_goal = self.env.get_next_goal()
          self._goal_queue = [next_task_goal] + self._goal_queue
          reset_goal = self._reset_goal_fn(
              self._reset_state_candidates,
              tf.constant(self.env._get_obs(), dtype=tf.float32),
              tf.constant(next_task_goal, dtype=tf.float32))

          if isinstance(reset_goal, tf.Tensor):
            reset_goal = reset_goal.numpy()

          self.env.reset_goal(goal=reset_goal)
        else:
          self._forward_or_reset_goal = True
          next_task_goal = self._goal_queue.pop()
          self.env.reset_goal(goal=next_task_goal)

      self._goal_switch = False
      return self.env._get_obs()
    else:
      self._forward_or_reset_goal = True
      return self.env.reset()

  def step(self, action):
    obs, reward, done, info = self.env.step(
        action)  # always check if the underneath env is done

    self._rfw_step_count += 1
    self._virtual_episode_steps += 1

    # a full reset including robot arm and objects
    if not done and self._full_reset_frequency is not None and self._rfw_step_count % self._full_reset_frequency == 0:
      done = True

    # finishes the episode and reset the goal, but no change in env state
    if not done and self._virtual_episode_steps % self._reset_goal_frequency == 0:
      done = True
      self._goal_switch = True

    # reset goal whenever the goal state is achieved
    if not done:
      if self.env.is_successful():
        self._success_state_seq_length += 1
      else:
        self._success_state_seq_length = 0
      if self._success_state_seq_length >= self._num_success_states:
        done = True
        self._goal_switch = True

    return obs, reward, done, info

  def set_reset_goal_fn(self, reset_goal_fn):
    self._reset_goal_fn = reset_goal_fn

  # it is not possible to sample the replay buffer while the data is being
  # collected, therefore we have to lazily update the reset candidates.
  def set_reset_candidates(self, reset_candidates):
    self._reset_state_candidates = reset_candidates


class GoalTerminalResetWrapper(Wrapper):

  def __init__(self, env, num_success_states=1, full_reset_frequency=None):

    super(GoalTerminalResetWrapper, self).__init__(env)
    self._virtual_episode_steps = 0
    self._num_success_states = num_success_states
    self._full_reset_frequency = full_reset_frequency
    self._goal_queue = []

  def reset(self):
    self._success_state_seq_length = 0
    self._virtual_episode_steps = 0

    return self.env.reset()

  def step(self, action):
    obs, reward, done, info = self.env.step(
        action)  # always check if the underneath env is done

    self._virtual_episode_steps += 1

    # finishes the episode and resets the environment
    if not done and self._virtual_episode_steps % self._full_reset_frequency == 0:
      done = True

    # reset whenever the goal state is achieved
    if not done and hasattr(self.env, 'is_successful'):
      if self.env.is_successful():
        self._success_state_seq_length += 1
      else:
        self._success_state_seq_length = 0
      if self._success_state_seq_length >= self._num_success_states:
        done = True

    return obs, reward, done, info


class CustomOracleResetGoalTerminalWrapper(Wrapper):

  def __init__(self,
               env,
               partial_reset_frequency,
               episodes_before_full_reset,
               reset_goal_fn=None):

    super(CustomOracleResetGoalTerminalWrapper, self).__init__(env)
    self._rfw_step_count = 0
    self._virtual_episode_steps = 0
    self._partial_reset_frequency = partial_reset_frequency
    self._episodes_before_full_reset = episodes_before_full_reset
    self._episode_count = -1
    self._reset_goal_fn = reset_goal_fn
    self._reset_state_candidates = None

  def reset(self):
    self._virtual_episode_steps = 0
    self._episode_count += 1
    if self._episode_count % self._episodes_before_full_reset == 0:
      return self.env.reset()

    elif self._reset_state_candidates is not None:
      next_task_goal = self.env.get_next_goal()
      obs_before_teleporting = self.env._get_obs()
      reset_goal = self._reset_goal_fn(
          self._reset_state_candidates,
          tf.constant(obs_before_teleporting, dtype=tf.float32),
          tf.constant(next_task_goal, dtype=tf.float32))
      if isinstance(reset_goal, tf.Tensor):
        reset_goal = reset_goal.numpy()
      self.env.do_custom_reset(pos=reset_goal)
      self.env.reset()
      self.env.reset_goal(
          goal=next_task_goal
      )  # otherwise the reset state would be for the wrong goal
      return self.env._get_obs()
    else:
      return self.env.reset()

  def step(self, action):
    obs, reward, done, info = self.env.step(
        action)  # always check if the underneath env is done

    self._rfw_step_count += 1
    self._virtual_episode_steps += 1

    # a full reset including robot arm and objects
    if not done and self._virtual_episode_steps % self._partial_reset_frequency == 0:
      done = True

    # reset whenever the goal state is achieved
    if not done and self.env.is_successful():
      done = True

    return obs, reward, done, info

  def set_reset_goal_fn(self, reset_goal_fn):
    self._reset_goal_fn = reset_goal_fn

  # it is not possible to sample the replay buffer while the data is being
  # collected, therefore we have to lazily update the reset candidates.
  def set_reset_candidates(self, reset_candidates):
    self._reset_state_candidates = reset_candidates


class VariableGoalTerminalResetWrapper(Wrapper):

  def __init__(self,
               env,
               num_success_states=1,
               full_reset_frequency=None,
               reset_goal_fn=None):

    super(VariableGoalTerminalResetWrapper, self).__init__(env)
    self._virtual_episode_steps = 0
    self._num_success_states = num_success_states
    self._full_reset_frequency = full_reset_frequency
    self._goal_queue = []
    self._reset_goal_fn = reset_goal_fn
    self._reset_state_candidates = None

  def reset(self):
    self._success_state_seq_length = 0
    self._virtual_episode_steps = 0

    # NOTE: assuming we are resetting at the goal
    if self._reset_goal_fn is not None:
      next_task_goal = self.env.get_next_goal()
      obs_before_teleporting = self.env._get_obs()
      reset_goal = self._reset_goal_fn(
          self._reset_state_candidates,
          tf.constant(obs_before_teleporting, dtype=tf.float32),
          tf.constant(next_task_goal, dtype=tf.float32))
      if isinstance(reset_goal, tf.Tensor):
        reset_goal = reset_goal.numpy()
      self.env.reset()
      self.env.reset_goal(
          goal=reset_goal
      )  # otherwise the reset state would be for the wrong goal
      return self.env._get_obs()
    else:
      reset_obs = self.env.reset()

    return reset_obs

  def step(self, action):
    obs, reward, done, info = self.env.step(
        action)  # always check if the underneath env is done

    self._virtual_episode_steps += 1

    # finishes the episode and resets the environment
    if not done and self._virtual_episode_steps % self._full_reset_frequency == 0:
      done = True

    # reset whenever the goal state is achieved
    if not done:
      if self.env.is_successful():
        self._success_state_seq_length += 1
      else:
        self._success_state_seq_length = 0
      if self._success_state_seq_length >= self._num_success_states:
        done = True

    return obs, reward, done, info

  def set_reset_goal_fn(self, reset_goal_fn):
    self._reset_goal_fn = reset_goal_fn

  # it is not possible to sample the replay buffer while the data is being
  # collected, therefore we have to lazily update the reset candidates.
  def set_reset_candidates(self, reset_candidates):
    self._reset_state_candidates = reset_candidates
