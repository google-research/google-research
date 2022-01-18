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

import abc
import logging
from strategic_exploration.hrl.env_wrapper import AtariWrapper


class CustomActionWrapper(AtariWrapper):
  """Supports custom actions: Teleport, EndEpisode.

  info["steps"] contains
    the number of steps that the action took
  """

  def __init__(self, env, clone_full_state=True):
    """
        Args: env (Environment | Wrapper)
            clone_full_state (bool): if True, teleport actions copy all
            pseudorandomness, otherwise they don't
    """
    super(CustomActionWrapper, self).__init__(env)
    self._clone_full_state = clone_full_state
    self._steps = 0
    self._reward = 0
    self._already_teleported = False

  def _reset(self):
    self._steps = 0
    self._reward = 0
    self._already_teleported = False
    return self.env.reset()

  def _step(self, action):
    # Support normal actions, too
    if isinstance(action, int):
      action = DefaultAction(action)

    if isinstance(action, DefaultAction):
      clone = self._get_clone()
      next_state, reward, done, info = self.env.step(action.action_num)
      self._steps += 1
      self._reward += reward
      info["steps"] = 1
      teleport = Teleport(clone, action.action_num, self._steps, self._reward)
      next_state.set_teleport(teleport)
      return next_state, reward, done, info
    elif isinstance(action, EndEpisode):
      state = self.env.reset()
      return state, 0., True, {"action": "EndEpisode", "steps": 0}
    elif isinstance(action, Teleport):
      self.restore_full_state(action.state_clone)
      clone = self._get_clone()
      next_state, reward, done, info = self.env.step(action.action)
      if done:
        logging.error("Done right after teleporting: {}".format(action))

      if self._already_teleported:
        logging.warning(("Already teleported, frame count and reward likely "
                         "incorrect: old reward: {}, old steps: {}, "
                         "teleport reward: {}, teleport steps: {}").format(
                             self._reward, self._steps, action.reward,
                             action.steps))

      # Overestimates the number of steps it took if already teleported
      info["steps"] = action.steps
      self._steps = action.steps
      reward = action.reward - self._reward
      self._reward = action.reward
      self._already_teleported = True
      teleport = Teleport(clone, action.action, self._steps, self._reward)
      next_state.set_teleport(teleport)
      return next_state, reward, done, info
    else:
      raise ValueError("Unsupported action: {}".format(action))

  def _get_clone(self):
    if self._clone_full_state:
      clone = self.env.clone_full_state()
    else:
      clone = self.env.clone_state()
    return clone

  def clone_full_state(self):
    logging.error("This most likely shouldn't be called. Use teleport instead")
    return super(CustomActionWrapper, self).clone_full_state()

  def clone_full_state(self):
    logging.error("This most likely shouldn't be called. Use teleport instead")
    return super(CustomActionWrapper, self).clone_state()


class CustomAction(object):
  """Marker class for custom actions supported by CustomActionWrapper"""
  pass


class EndEpisode(CustomAction):
  """The done bool of step returns True no matter what.

  The resulting state
    will be the first state of the next episode. The reward is 0.
  """

  def __str__(self):
    return "EndEpisode"

  __repr__ = __str__


class DefaultAction(CustomAction):
  """Wrapper around int action"""

  def __init__(self, action):
    """
        Args:
            action (int): normal gym.env action
    """
    self._action = action

  @property
  def action_num(self):
    return self._action

  def __str__(self):
    return "DefaultAction({})".format(self.action_num)

  __repr__ = __str__


class Teleport(CustomAction):
  """Step reloads the provided ALEState"""

  def __init__(self, state_clone, action, steps, reward):
    """
        Args:
            state_clone (np.array): output of clone_state from AtariEnv
            action (int): the one step to take to get to the desired copied
              state
            steps (int): number of steps to get to state_clone, including the
              last one (action)
            reward (int): reward when last at state_clone
    """
    self.state_clone = state_clone
    self.action = action
    self.steps = steps
    self.reward = reward

  def __str__(self):
    return "Teleport({}, {}, {})".format(self.action, self.steps, self.reward)

  __repr__ = __str__
