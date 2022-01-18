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

# Adapted from OpenAI Gym baselines
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import abc
import copy
import cv2
import gym
import logging
import numpy as np
import sys
import torch
from collections import deque
from PIL import Image
from strategic_exploration.hrl import abstract_state as AS
from strategic_exploration.hrl import reward_rule as R
from strategic_exploration.hrl.state import State
from gym import spaces
from gym.wrappers.time_limit import TimeLimit


class AtariWrapper(gym.Wrapper):
  """Additionally supports cloning and reloading Atari state

    NOTE:
        Any state that any AtariWrapper holds MUST be properly stored and
        restored inside of clone_state and restore_state. Otherwise,
        clone_state may not reproduce the same state.
    """

  def clone_state(self):
    if isinstance(self.env, TimeLimit):
      return self.env.unwrapped.clone_state()
    else:
      return self.env.clone_state()

  def restore_state(self, state):
    if isinstance(self.env, TimeLimit):
      self.env.unwrapped.restore_state(state)
    else:
      self.env.restore_state(state)

  def clone_full_state(self):
    if isinstance(self.env, TimeLimit):
      return self.env.unwrapped.clone_full_state()
    else:
      return self.env.clone_full_state()

  def restore_full_state(self, state):
    if isinstance(self.env, TimeLimit):
      self.env.unwrapped.restore_full_state(state)
    else:
      self.env.restore_full_state(state)


class FrameStack(AtariWrapper):

  def __init__(self, env, k):
    """Stack k last frames.

        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
    super(FrameStack, self).__init__(env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(
        low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]))

  def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.k
    return LazyFrames(list(self.frames))

  # Makes sure that past frames are saved / restored properly
  def clone_state(self):
    clone = super(FrameStack, self).clone_state()
    return (clone, copy.copy(self.frames))

  def restore_state(self, state):
    clone, self.frames = state
    self.frames = copy.deepcopy(self.frames)
    super(FrameStack, self).restore_state(clone)

  def clone_full_state(self):
    clone = super(FrameStack, self).clone_full_state()
    return (clone, copy.copy(self.frames))

  def restore_full_state(self, state):
    clone, self.frames = state
    self.frames = copy.deepcopy(self.frames)
    super(FrameStack, self).restore_full_state(clone)


class LazyFrames(object):

  def __init__(self, frames):
    """This object ensures that common frames between the observations

        are only stored once.  It exists purely to optimize memory usage
        which can be huge for DQN's 1M frames replay buffers.  This object
        should only be converted to numpy array before being passed to the
        model.  You'd not believe how complex the previous solution was.
        """
    self._frames = frames

  def __array__(self, dtype=None):
    out = np.concatenate(self._frames, axis=0)
    if dtype is not None:
      out = out.astype(dtype)
    return out


class NoopStarts(AtariWrapper):
  """Starts every episode with a random number of no-ops [0, max_noops]

    inclusive. Assumes that action 0 is a no-op.
    """

  def __init__(self, env, max_noops=30):
    super(NoopStarts, self).__init__(env)
    self._max_noops = max_noops
    assert env.unwrapped.get_action_meanings()[0] == "NOOP"

  def reset(self):
    state = self.env.reset()
    noops = np.random.randint(self._max_noops + 1)
    for _ in range(noops):
      state, _, done, _ = self.env.step(0)
      assert not done
    return state

  def step(self, action):
    return self.env.step(action)


class MaxAndSkipEnv(AtariWrapper):

  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame"""
    super(MaxAndSkipEnv, self).__init__(env)
    # most recent raw observations (for max pooling across time steps)
    # Does not need to be restored on clone
    self._obs_buffer = np.zeros(
        (2,) + env.observation_space.shape, dtype=np.uint8)
    self._skip = skip

  def reset(self):
    return self.env.reset()

  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2:
        self._obs_buffer[0] = obs
      if i == self._skip - 1:
        self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame
    # doesn't matter
    max_frame = self._obs_buffer.max(axis=0)

    return max_frame, total_reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


class StateWrapper(AtariWrapper):
  """Returns State objects.

  Expects that env returns pixel observations.
    Constructs State objects that also include the RAM if ram is True.

    Args: env (Environment)
        ram (bool): when True, step and reset return State objects with RAM and
        pixels
  """

  def __init__(self, env, ram=False):
    super(StateWrapper, self).__init__(env)
    self._step_num = 0
    self._ram = ram

  def _step(self, action):
    pixels, reward, done, info = self.env.step(action)
    ram = None
    if self._ram:
      ram = self.env.unwrapped._get_ram()
    state = State(ram, pixels, self._step_num)
    self._step_num += 1
    return state, reward, done, info

  def _reset(self):
    self._step_num = 0
    pixels = self.env.reset()
    ram = None
    if self._ram:
      ram = self.env.unwrapped._get_ram()
    return State(ram, pixels, self._step_num)

  def clone_full_state(self):
    clone = super(StateWrapper, self).clone_full_state()
    return (clone, self._step_num)

  def restore_full_state(self, state):
    clone, self._step_num = state
    super(StateWrapper, self).restore_full_state(clone)

  def clone_state(self):
    clone = super(StateWrapper, self).clone_state()
    return (clone, self._step_num)

  def restore_state(self, state):
    clone, self._step_num = state
    super(StateWrapper, self).restore_state(clone)


class OriginalPixelsWrapper(AtariWrapper):
  """Adds the original unmodified pixel arrays to the State."""

  def _step(self, action):
    state, reward, done, info = self.env.step(action)
    state.set_unmodified_pixels(self.env.unwrapped._get_image())
    return state, reward, done, info

  def _reset(self):
    state = self.env.reset()
    state.set_unmodified_pixels(self.env.unwrapped._get_image())
    return state


class LifeLostWrapper(AtariWrapper):
  """Returns whether the agent lost a life in the info dict under

    the key info['lift_lost'] = True
  """

  def _step(self, action):
    old_lives = self.env.unwrapped.ale.lives()
    state, reward, done, info = self.env.step(action)
    new_lives = self.env.unwrapped.ale.lives()
    info["life_lost"] = new_lives < old_lives
    return state, reward, done or new_lives < old_lives, info


class SubdomainWrapper(AtariWrapper):
  __metaclass__ = abc.ABCMeta

  def __init__(self, env, goal=True):
    super(SubdomainWrapper, self).__init__(env)
    self._steps = 0
    self._start = None
    self._goal = goal

  def _reset(self):
    self._steps = 0
    state = self.env.reset()
    if self._start is None:
      for action in self.action_prefix[:-1]:
        state, reward, done, info = self.env.step(action)
      self._start = self.clone_full_state()

    self.env.restore_full_state(self._start)
    state, reward, done, info = self.env.step(self.action_prefix[-1])

    if self._goal:
      return self._make_goal(state)[0]
    return state

  def _step(self, action):
    self._steps += 1
    state, reward, done, info = self.env.step(action)
    if self._goal:
      state, reward = self._make_goal(state)
      if self._steps == self.max_steps:
        done = True
    return state, reward, done, info

  def _make_goal(self, state):
    raise NotImplementedError("Deprecated! set_goal was updated")
    goal = np.zeros(AS.AbstractState.DIM + 2)
    goal[:AS.AbstractState.DIM] = \
        self.goal_abstract_state.numpy - AS.AbstractState(state).unbucketed
    goal[AS.AbstractState.DIM] = float(self._steps) / self.max_steps
    if AS.AbstractState(state) == self.goal_abstract_state:
      goal[AS.AbstractState.DIM + 1] = 1.
    difference = \
        self.goal_abstract_state.numpy - self.start_abstract_state.numpy
    normalization = np.linalg.norm(difference)
    goal[0] /= (normalization * 3)
    goal[1] /= (normalization * 3)

    state_copy = copy.copy(state)
    state_copy.set_goal(goal)
    reward = 0.
    if self.goal_abstract_state == AS.AbstractState(state):
      reward = 1.
    return state_copy, reward

  @abc.abstractproperty
  def max_steps(self):
    raise NotImplementedError()

  @abc.abstractproperty
  def goal_abstract_state(self):
    raise NotImplementedError()

  @abc.abstractproperty
  def start_abstract_state(self):
    raise NotImplementedError()

  @abc.abstractproperty
  def action_prefix(self):
    raise NotImplementedError


class PitfallVineWrapper(SubdomainWrapper):

  def _step(self, action):
    next_state, reward, done, info = super(PitfallVineWrapper,
                                           self)._step(action)
    if AS.AbstractState(next_state).room_number != 18:
      done = True
    return next_state, reward, done, info

  @property
  def max_steps(self):
    return 250

  @property
  def action_prefix(self):
    return [
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 11, 11, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 11, 11, 3, 3, 3, 3, 3, 3, 3,
        3, 11, 11, 3, 3, 3, 3, 3, 3, 11, 11, 3, 3, 3, 3, 11, 11, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 3,
        3, 3, 3, 3, 3, 3, 3, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
    ]

  @property
  def goal_abstract_state(self):
    ram = np.zeros(128)
    ram[[97, 105, 1, 113]] = [117, 30, 18, 31]
    goal_state = State(ram, None)
    return AS.AbstractState(goal_state)

  @property
  def start_abstract_state(self):
    ram = np.zeros(128)
    ram[[97, 105, 1, 113]] = [39, 30, 18, 31]
    state = State(ram, None)
    return AS.AbstractState(state)


class JumpMonsterWrapper(SubdomainWrapper):

  @property
  def max_steps(self):
    return 24

  @property
  def goal_abstract_state(self):
    goal_state = np.zeros(128).astype(np.uint8)
    goal_state[42] = 60
    goal_state[43] = 148
    goal_state[3] = 1
    goal_state[65] = 0
    goal_state[66] = 15
    goal_state = State(goal_state, None)
    goal_state = AS.AbstractState(goal_state)
    return goal_state

  @property
  def start_abstract_state(self):
    start_state = np.zeros(128).astype(np.uint8)
    start_state[42] = 133
    start_state[43] = 148
    start_state[3] = 1
    start_state[65] = 0
    start_state[66] = 15
    start_state = State(start_state, None)
    start_state = AS.AbstractState(start_state)
    return start_state

  @property
  def action_prefix(self):
    return [
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11, 11, 3,
        3, 3, 11, 11, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4
    ]


class BeamWrapper(SubdomainWrapper):

  def _step(self, action):
    next_state, reward, done, info = super(BeamWrapper, self)._step(action)
    if AS.AbstractState(next_state).room_number != 7:
      done = True
    return next_state, reward, done, info

  @property
  def max_steps(self):
    return 32

  @property
  def goal_abstract_state(self):
    goal_state = np.zeros(128).astype(np.uint8)
    # one beam
    goal_state[42] = 27
    goal_state[43] = 235
    goal_state[3] = 7
    goal_state[65] = 0
    goal_state[66] = 1
    goal_state = State(goal_state, None)
    goal_state = AS.AbstractState(goal_state)
    return goal_state

  @property
  def start_abstract_state(self):
    start_state = np.zeros(128).astype(np.uint8)
    start_state[42] = 3
    start_state[43] = 235
    start_state[3] = 7
    start_state[65] = 0
    start_state[66] = 1
    start_state = State(start_state, None)
    start_state = AS.AbstractState(start_state)
    return start_state

  @property
  def action_prefix(self):
    actions = [
        0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 8, 3, 3, 3, 3, 3, 3, 3, 11, 11, 11, 3,
        3, 11, 11, 11, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 7, 4, 1, 1, 3, 3, 3, 3, 3, 3,
        0, 0, 0, 0, 0, 0, 0, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 3, 3, 3, 3,
        3, 3, 3, 3, 11, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 4, 12, 12, 4, 4, 4, 4, 12, 12, 4,
        4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 11,
        11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3,
        3
    ]
    return actions


class DisappearingLadderWrapper(SubdomainWrapper):

  def _step(self, action):
    next_state, reward, done, info = super(DisappearingLadderWrapper,
                                           self)._step(action)
    return next_state, reward, done, info

  @property
  def max_steps(self):
    return 75

  @property
  def goal_abstract_state(self):
    goal_state = np.zeros(128).astype(np.uint8)
    indices = np.array([42, 43, 3, 65, 66])
    values = [149, 235, 8, 2, 0]
    goal_state[indices] = values
    goal_state = State(goal_state, None)
    goal_state.set_object_changes(3)
    goal_state = AS.AbstractState(goal_state)
    return goal_state

  @property
  def start_abstract_state(self):
    start_state = np.zeros(128).astype(np.uint8)
    indices = np.array([42, 43, 3, 65, 66])
    values = [149, 155, 8, 2, 0]
    start_state[indices] = values
    start_state = State(start_state, None)
    start_state.set_object_changes(3)
    start_state = AS.AbstractState(start_state)
    return start_state

  @property
  def action_prefix(self):
    actions = [
        0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 11, 11,
        11, 3, 3, 3, 11, 11, 3, 3, 3, 3, 3, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 4, 12, 1, 0, 3, 3, 3, 3, 3,
        3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 3, 3, 3, 3, 3, 0, 0, 0, 0,
        0, 0, 0, 0, 3, 3, 11, 11, 11, 11, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 4, 4, 12, 12, 4, 4, 4, 4, 4, 4,
        12, 12, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 12,
        12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 0, 0, 0, 0, 4, 4,
        4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 5, 5, 0, 0, 4, 4, 4, 2, 2, 2, 0, 0, 4, 4, 4, 4, 0, 0, 4,
        4, 4, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 12, 12, 12, 12, 4,
        4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 4, 4, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0,
        0, 3, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 0, 0, 4, 4, 4,
        4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 4, 12, 12, 12, 4, 4, 4,
        4, 4, 0, 0, 0, 0, 0, 0, 12, 12, 12, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0,
        0, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7,
        7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 1, 0, 0, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 8, 8, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
    ]
    return actions


class GifRecorder(AtariWrapper):
  """Records gifs accessible through get_images method."""

  def __init__(self, env):
    # Wrap in OriginalPixels to get the unmodified_pixels
    super(GifRecorder, self).__init__(OriginalPixelsWrapper(env))
    self._images = []

  def _step(self, action):
    state, reward, done, info = self.env.step(action)
    self._images.append(Image.fromarray(state.drop_unmodified_pixels()))
    return state, reward, done, info

  def _reset(self):
    state = self.env.reset()
    self._images.append(Image.fromarray(state.drop_unmodified_pixels()))
    return state

  @property
  def images(self):
    """Returns list[Image] of all of the states visited."""
    return self._images


class FixRamState(AtariWrapper):

  def __init__(self, env):
    super(FixRamState, self).__init__(env)
    self._prev_state = None

  def _step(self, action):
    state, reward, done, info = self.env.step(action)
    if self._prev_state is not None:
      ram_state = state.ram_state
      prev_ram_state = self._prev_state.ram_state
      dist_x = abs(int(ram_state[42]) - int(prev_ram_state[42]))
      dist_y = abs(int(ram_state[43]) - int(prev_ram_state[43]))
      if (dist_x >= 100 or dist_y >= 100) and \
              ram_state[3] == prev_ram_state[3]:
        logging.debug("Correcting coordinates: {} to {}".format(
            (ram_state[42], ram_state[43]),
            (prev_ram_state[42], prev_ram_state[43])))
        logging.debug("Observed ({}, {}, {})".format(ram_state[42],
                                                     ram_state[43],
                                                     ram_state[3]))
        logging.debug("Prev ({}, {}, {})".format(prev_ram_state[42],
                                                 prev_ram_state[43],
                                                 prev_ram_state[3]))
        state._ram_state[42] = prev_ram_state[42]
        state._ram_state[43] = prev_ram_state[43]
      elif dist_x <= 20 and dist_y <= 20 and \
              ram_state[3] != prev_ram_state[3]:
        logging.debug("Correcting room number: {} to {}".format(
            ram_state[3], prev_ram_state[3]))
        logging.debug("Observed ({}, {}, {})".format(ram_state[42],
                                                     ram_state[43],
                                                     ram_state[3]))
        logging.debug("Prev ({}, {}, {})".format(prev_ram_state[42],
                                                 prev_ram_state[43],
                                                 prev_ram_state[3]))
        state._ram_state[3] = prev_ram_state[3]

    # Needs to be copied! Otherwise, setting teleport will create large
    # linked list
    self._prev_state = copy.copy(state)
    return state, reward, done, info

  def _reset(self):
    state = self.env.reset()
    self._prev_state = copy.copy(state)
    return state

  def clone_full_state(self):
    clone = super(FixRamState, self).clone_full_state()
    return (clone, self._prev_state)

  def restore_full_state(self, state):
    clone, self._prev_state = state
    super(FixRamState, self).restore_full_state(clone)

  def clone_state(self):
    clone = super(FixRamState, self).clone_state()
    return (clone, self._prev_state)

  def restore_state(self, state):
    clone, self._prev_state = state
    super(FixRamState, self).restore_state(clone)


class ObjectChangesWrapper(AtariWrapper):

  def __init__(self, env):
    super(ObjectChangesWrapper, self).__init__(env)
    self._prev_state = None

  def _reset(self):
    state = self.env.reset()
    self._prev_state = copy.copy(state)
    return self._prev_state

  def _step(self, action):
    old_lives = self.env.unwrapped.ale.lives()
    next_state, reward, done, info = self.env.step(action)
    new_lives = self.env.unwrapped.ale.lives()
    next_state.set_object_changes(self._prev_state.object_changes)

    room_changed = \
        next_state.ram_state[3] != self._prev_state.ram_state[3]
    objects_changed = \
        next_state.ram_state[66] != self._prev_state.ram_state[66]
    life_lost = old_lives > new_lives
    # Check for change in reward and life lost because of Atari RAM
    # rendering lag
    if not room_changed and objects_changed and (reward > 0 or life_lost):
      object_changes = self._prev_state.object_changes + 1
      next_state.set_object_changes(object_changes)

    # Needs to be copied! Otherwise, setting teleport will create large
    # linked list
    self._prev_state = copy.copy(next_state)
    return next_state, reward, done, info

  def clone_full_state(self):
    clone = super(ObjectChangesWrapper, self).clone_full_state()
    return (clone, self._prev_state)

  def restore_full_state(self, state):
    clone_state, self._prev_state = state
    super(ObjectChangesWrapper, self).restore_full_state(clone_state)

  def clone_state(self):
    clone = super(ObjectChangesWrapper, self).clone_state()
    return (clone, self._prev_state)

  def restore_state(self, state):
    clone_state, self._prev_state = state
    super(ObjectChangesWrapper, self).restore_state(clone_state)


class WarpFrame(AtariWrapper):

  def __init__(self, env):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    super(WarpFrame, self).__init__(env)
    self.width = 84
    self.height = 84
    self.observation_space = spaces.Box(
        low=0, high=255, shape=(1, self.height, self.width))

  def _step(self, action):
    next_state, reward, done, info = self.env.step(action)
    return self._observation(next_state), reward, done, info

  def _reset(self):
    next_state = self.env.reset()
    return self._observation(next_state)

  def _observation(self, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(
        frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
    return frame[None, :, :]


class RewardRuleWrapper(AtariWrapper):
  """Environment reward is completely ignored.

  Reward
    is set to be sum of reward rules.
  """

  def __init__(self, env, reward_rules):
    """
        Args: reward_rules (list[RewardRule])
    """
    super(RewardRuleWrapper, self).__init__(env)
    self._reward_rules = reward_rules
    self._prev_abstract_state = None

  def _reset(self):
    state = self.env.reset()
    self._prev_abstract_state = AS.AbstractState(state)
    return state

  def _step(self, action):
    state, env_reward, done, info = self.env.step(action)
    info["env reward"] = env_reward
    abstract_state = AS.AbstractState(state)
    new_reward = sum(
        rule(self._prev_abstract_state, abstract_state)
        for rule in self._reward_rules)
    self._prev_abstract_state = abstract_state
    return state, new_reward, done, info

  def clone_full_state(self):
    clone = super(RewardRuleWrapper, self).clone_full_state()
    return (clone, self._prev_abstract_state)

  def restore_full_state(self, state):
    clone_state, self._prev_abstract_state = state
    super(RewardRuleWrapper, self).restore_full_state(clone_state)

  def clone_state(self):
    clone = super(RewardRuleWrapper, self).clone_state()
    return (clone, self._prev_abstract_state)

  def restore_state(self, state):
    clone_state, self._prev_abstract_state = state
    super(RewardRuleWrapper, self).restore_state(clone_state)


class DoneRuleWrapper(AtariWrapper):
  """Done is set to be environment done or done from rules."""

  def __init__(self, env, done_rules):
    """
        Args: done_rules (list[Rule])
    """
    super(DoneRuleWrapper, self).__init__(env)
    self._done_rules = done_rules
    self._prev_abstract_state = None

  def _reset(self):
    state = self.env.reset()
    self._prev_abstract_state = AS.AbstractState(state)
    return state

  def _step(self, action):
    state, reward, done, info = self.env.step(action)
    info["done"] = done
    abstract_state = AS.AbstractState(state)
    new_done = sum(
        rule(self._prev_abstract_state, abstract_state)
        for rule in self._done_rules)
    done = new_done or done
    self._prev_abstract_state = abstract_state
    return state, reward, done, info

  def clone_full_state(self):
    clone = super(DoneRuleWrapper, self).clone_full_state()
    return (clone, self._prev_abstract_state)

  def restore_full_state(self, state):
    clone_state, self._prev_abstract_state = state
    super(DoneRuleWrapper, self).restore_full_state(clone_state)

  def clone_state(self):
    clone = super(DoneRuleWrapper, self).clone_state()
    return (clone, self._prev_abstract_state)

  def restore_state(self, state):
    clone_state, self._prev_abstract_state = state
    super(DoneRuleWrapper, self).restore_state(clone_state)


def get_env(config):
  if config.stochastic:
    suffix = "-v0"
  else:
    suffix = "-v4"
  domain = config.domain + suffix

  env = gym.make(domain).env  # Manually use different TimeLimit wrapper
  env = TimeLimit(env, max_episode_steps=config.max_episode_len)
  env = NoopStarts(env, config.max_noops)
  if config.skip > 1:
    assert "NoFrameskip" in domain
    assert "-ram-" not in domain
    env = MaxAndSkipEnv(env, config.skip)
  env = WarpFrame(env)
  env = FrameStack(env, 4)
  env = LifeLostWrapper(env)
  env = StateWrapper(env, config.ram)
  if "MontezumaRevenge" in config.domain:
    env = FixRamState(env)
    env = ObjectChangesWrapper(env)
    reward_rules_config = config.get("reward_rules")
    if reward_rules_config is not None:
      rules = R.get_reward_rules(reward_rules_config)
      env = RewardRuleWrapper(env, rules)

    done_rules_config = config.get("done_rules")
    if done_rules_config is not None:
      rules = R.get_done_rules(done_rules_config)
      env = DoneRuleWrapper(env, rules)
  subdomain = config.get("subdomain", None)
  if subdomain == "JumpMonster":
    assert domain == "MontezumaRevengeNoFrameskip-v4"
    env = JumpMonsterWrapper(env, not config.teleport_only)
  elif subdomain == "Beam":
    assert domain == "MontezumaRevengeNoFrameskip-v4"
    env = BeamWrapper(env, not config.teleport_only)
  elif subdomain == "DisappearingLadder":
    assert domain == "MontezumaRevengeNoFrameskip-v4"
    env = DisappearingLadderWrapper(env, not config.teleport_only)
  elif subdomain == "Vine":
    assert domain == "PitfallNoFrameskip-v4", domain
    env = PitfallVineWrapper(env, not config.teleport_only)
  elif subdomain is not None:
    raise ValueError("{} is not a supported subdomain".format(subdomain))
  # This MUST be the last AtariWrapper because it sets Teleport
  from strategic_exploration.hrl.action import CustomActionWrapper
  env = CustomActionWrapper(env)
  return env
