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

"""Wrappers for OpenAI Gym environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import functools
import multiprocessing
import sys
import traceback

import gym
import gym.spaces
import numpy as np
import tensorflow.compat.v1 as tf


class AttributeModifier(object):
  """Provides getter and setter functions to access wrapped environments."""

  def __getattr__(self, name):
    return getattr(self._env, name)

  def set_attribute(self, name, value):
    """Set an attribute in the wrapped environment.

    Args:
      name: Attribute to access.
      value: New attribute value.
    """
    set_attr = getattr(self._env, 'set_attribute', None)
    if callable(set_attr):
      self._env.set_attribute(name, value)
    else:
      setattr(self._env, name, value)


class RangeNormalize(AttributeModifier):
  """Normalize the specialized observation and action ranges to [-1, 1]."""

  def __init__(self, env):
    self._env = env
    self._should_normalize_observ = self._is_finite(self._env.observation_space)
    if not self._should_normalize_observ:
      tf.logging.info('Not normalizing infinite observation range.')
    self._should_normalize_action = self._is_finite(self._env.action_space)
    if not self._should_normalize_action:
      tf.logging.info('Not normalizing infinite action range.')

  @property
  def observation_space(self):
    space = self._env.observation_space
    if not self._should_normalize_observ:
      return space
    return gym.spaces.Box(
        -np.ones(space.shape), np.ones(space.shape), dtype=np.float32)

  @property
  def action_space(self):
    space = self._env.action_space
    if not self._should_normalize_action:
      return space
    return gym.spaces.Box(
        -np.ones(space.shape), np.ones(space.shape), dtype=np.float32)

  def step(self, action):
    if self._should_normalize_action:
      action = self._denormalize_action(action)
    observ, reward, done, info = self._env.step(action)
    if self._should_normalize_observ:
      observ = self._normalize_observ(observ)
    return observ, reward, done, info

  def reset(self):
    observ = self._env.reset()
    if self._should_normalize_observ:
      observ = self._normalize_observ(observ)
    return observ

  def _denormalize_action(self, action):
    min_ = self._env.action_space.low
    max_ = self._env.action_space.high
    action = (action + 1) / 2 * (max_ - min_) + min_
    return action

  def _normalize_observ(self, observ):
    min_ = self._env.observation_space.low
    max_ = self._env.observation_space.high
    observ = 2 * (observ - min_) / (max_ - min_) - 1
    return observ

  def _is_finite(self, space):
    return np.isfinite(space.low).all() and np.isfinite(space.high).all()


class ClipAction(AttributeModifier):
  """Clip out of range actions to the action space of the environment."""

  def __init__(self, env):
    self._env = env

  @property
  def action_space(self):
    shape = self._env.action_space.shape
    return gym.spaces.Box(
        -np.inf * np.ones(shape), np.inf * np.ones(shape), dtype=np.float32)

  def step(self, action):
    action_space = self._env.action_space
    action = np.clip(action, action_space.low, action_space.high)
    return self._env.step(action)


class LimitDuration(AttributeModifier):
  """End episodes after specified number of steps."""

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def step(self, action):
    if self._step is None:
      raise RuntimeError('Must reset environment.')
    observ, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      self._step = None
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ExternalProcess(object):
  """Step environment in a separate process for lock free paralellism."""

  # Message types for communication via the pipe.
  _ACTION = 1
  _RESET = 2
  _CLOSE = 3
  _GETATTRIBUTE = 4
  _SETATTRIBUTE = 5
  _TRANSITION = 6
  _OBSERV = 7
  _EXCEPTION = 8
  _VALUE = 9

  def __init__(self, constructor):
    """Step environment in a separate process for lock free paralellism.

    The environment will be created in the external process by calling the
    specified callable. This can be an environment class, or a function
    creating the environment and potentially wrapping it. The returned
    environment should not access global variables.

    Args:
      constructor: Callable that creates and returns an OpenAI gym environment.
    Attributes:
      observation_space: The cached observation space of the environment.
      action_space: The cached action space of the environment.
    """
    self._conn, conn = multiprocessing.Pipe()
    self._process = multiprocessing.Process(
        target=self._worker, args=(constructor, conn))
    atexit.register(self.close)
    self._process.start()
    self._observ_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._observ_space:
      self._observ_space = self.__getattr__('observation_space')
    return self._observ_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    """Request an attribute from the environment.

    Note that this involves communication with the external process, so it can
    be slow.

    Args:
      name: Attribute to access.

    Returns:
      Value of the attribute.
    """
    self._conn.send((self._GETATTRIBUTE, name))
    return self._receive(self._VALUE)

  def set_attribute(self, name, value):
    """Set an attribute in the environment.

    Note that this involves communication with the external process, so it can
    be slow.

    Args:
      name: Attribute to access.
      value: New attribute value.
    """
    self._conn.send((self._SETATTRIBUTE, (name, value)))

  def step(self, action, blocking=True):
    """Step the environment.

    Args:
      action: The action to apply to the environment.
      blocking: Whether to wait for the result.

    Returns:
      Transition tuple when blocking, otherwise callable that returns the
      transition tuple.
    """
    self._conn.send((self._ACTION, action))
    if blocking:
      return self._receive(self._TRANSITION)
    else:
      return functools.partial(self._receive, self._TRANSITION)

  def reset(self, blocking=True):
    """Reset the environment.

    Args:
      blocking: Whether to wait for the result.

    Returns:
      New observation when blocking, otherwise callable that returns the new
      observation.
    """
    self._conn.send((self._RESET, None))
    if blocking:
      return self._receive(self._OBSERV)
    else:
      return functools.partial(self._receive, self._OBSERV)

  def close(self):
    """Send a close message to the external process and join it."""
    if self._process:
      try:
        self._conn.send((self._CLOSE, None))
        self._conn.close()
      except IOError:
        # The connection was already closed.
        pass
      self._process.join()
      # Python leaks file descriptors without the line below
      del self._process
      del self._conn
      self._conn = None
      self._process = None
    else:
      pass  # Don't close a connection twice

  def _receive(self, expected_message):
    """Wait for a message from the worker process and return its payload.

    Args:
      expected_message: Type of the expected message.

    Raises:
      Exception: An exception was raised inside the worker process.
      KeyError: The reveived message is not of the expected type.

    Returns:
      Payload object of the message.
    """
    message, payload = self._conn.recv()
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == expected_message:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, constructor, conn):
    """The process waits for actions and sends back environment results.

    Args:
      constructor: Constructor for the OpenAI Gym environment.
      conn: Connection for communication to the main process.
    """
    try:
      env = constructor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACTION:
          action = payload
          conn.send((self._TRANSITION, env.step(action)))
          continue
        if message == self._RESET:
          assert payload is None
          conn.send((self._OBSERV, env.reset()))
          continue
        if message == self._GETATTRIBUTE:
          name = payload
          conn.send((self._VALUE, getattr(env, name)))
          continue
        if message == self._SETATTRIBUTE:
          name, value = payload
          set_attr = getattr(env, 'set_attribute', None)
          if callable(set_attr):
            env.set_attribute(name, value)
          else:
            setattr(env, name, value)
          continue
        if message == self._CLOSE:
          assert payload is None
          if hasattr(env, 'close'):
            env.close()
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:  # pylint: disable=broad-except
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))  # pylint: disable=no-value-for-parameter
      conn.send((self._EXCEPTION, stacktrace))
      tf.logging.error('Error in environment process: {}'.format(stacktrace))
    conn.close()
