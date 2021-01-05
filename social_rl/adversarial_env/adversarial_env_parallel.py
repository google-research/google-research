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

"""Adapt TF-agents parallel process environments to adversarial env setting.

Note that the environments had to be copied and modified rather than overridden
because of the way parent processes are called for multiprocessing.

Adds two new functions: reset_agent, and step_adversary in addition to usual
RL env functions. Therefore we have the following environment functions:
  env.reset(): completely resets the environment and removes anything the
    adversary has built.
  env.reset_agent(): resets the position of the agent, but does not
    remove the obstacles the adversary has created when building the env.
  env.step(): steps the agent as before in the environment. i.e. if the agent
    passes action 'left' it will move left.
  env.step_adversary(): processes an adversary action, which involves choosing
    the location of the agent, goal, or an obstacle.

Adds additional functions for logging metrics related to the generated
environments, like the shortest path length to the goal.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import sys
import traceback

from absl import logging

import cloudpickle
import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.environments import py_environment
from tf_agents.system import system_multiprocessing
from tf_agents.utils import nest_utils

# Import needed to trigger env registration, so pylint: disable=unused-import
from social_rl import gym_multigrid


# Worker polling period in seconds.
_POLLING_PERIOD = 0.1


@gin.configurable
class AdversarialParallelPyEnvironment(py_environment.PyEnvironment):
  """Batch together environments and simulate them in external processes.

  The environments are created in external processes by calling the provided
  callables. This can be an environment class, or a function creating the
  environment and potentially wrapping it. The returned environment should not
  access global variables.
  """

  def __init__(self, env_constructors, start_serially=True, blocking=False,
               flatten=False):
    """Batch together environments and simulate them in external processes.

    The environments can be different but must use the same action and
    observation specs.

    Args:
      env_constructors: List of callables that create environments.
      start_serially: Whether to start environments serially or in parallel.
      blocking: Whether to step environments one after another.
      flatten: Boolean, whether to use flatten action and time_steps during
        communication to reduce overhead.

    Raises:
      ValueError: If the action or observation specs don't match.
    """
    super(AdversarialParallelPyEnvironment, self).__init__()
    self._envs = [AdversarialProcessPyEnvironment(ctor, flatten=flatten)
                  for ctor in env_constructors]
    self._num_envs = len(env_constructors)
    self._blocking = blocking
    self._start_serially = start_serially
    self.start()
    self._action_spec = self._envs[0].action_spec()
    self._observation_spec = self._envs[0].observation_spec()
    self._time_step_spec = self._envs[0].time_step_spec()
    self._parallel_execution = True
    if any(env.action_spec() != self._action_spec for env in self._envs):
      raise ValueError('All environments must have the same action spec.')
    if any(env.time_step_spec() != self._time_step_spec for env in self._envs):
      raise ValueError('All environments must have the same time_step_spec.')
    self._flatten = flatten

    self.adversary_action_spec = self._envs[0].adversary_action_spec
    self.adversary_observation_spec = self._envs[0].adversary_observation_spec
    self.adversary_time_step_spec = self._envs[0].adversary_time_step_spec

  def start(self):
    logging.info('Spawning all processes.')
    for env in self._envs:
      env.start(wait_to_start=self._start_serially)
    if not self._start_serially:
      logging.info('Waiting for all processes to start.')
      for env in self._envs:
        env.wait_start()
    logging.info('All processes started.')

  @property
  def batched(self):
    return True

  @property
  def batch_size(self):
    return self._num_envs

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

  def time_step_spec(self):
    return self._time_step_spec

  def _reset(self):
    """Reset all environments and combine the resulting observation.

    Returns:
      Time step with batch dimension.
    """
    time_steps = [env.reset(self._blocking) for env in self._envs]
    if not self._blocking:
      time_steps = [promise() for promise in time_steps]
    return self._stack_time_steps(time_steps)

  def _step(self, actions):
    """Forward a batch of actions to the wrapped environments.

    Args:
      actions: Batched action, possibly nested, to apply to the environment.

    Raises:
      ValueError: Invalid actions.

    Returns:
      Batch of observations, rewards, and done flags.
    """
    time_steps = [
        env.step(action, self._blocking)
        for env, action in zip(self._envs, self._unstack_actions(actions))]
    # When blocking is False we get promises that need to be called.
    if not self._blocking:
      time_steps = [promise() for promise in time_steps]
    return self._stack_time_steps(time_steps)

  def reset_agent(self):
    """Reset all environments and combine the resulting observation.

    Returns:
      Time step with batch dimension.
    """
    time_steps = [env.reset_agent(self._blocking) for env in self._envs]
    if not self._blocking:
      time_steps = [promise() for promise in time_steps]
    self._current_time_step = self._stack_time_steps(time_steps)
    return self._current_time_step

  def reset_random(self):
    """Reset all environments randomly and combine the resulting observation.

    Returns:
      Time step with batch dimension.
    """
    time_steps = [env.reset_random(self._blocking) for env in self._envs]
    if not self._blocking:
      time_steps = [promise() for promise in time_steps]
    self._current_time_step = self._stack_time_steps(time_steps)
    return self._current_time_step

  def step_adversary(self, actions):
    """Forward a batch of actions to the wrapped environments.

    Args:
      actions: Batched action, possibly nested, to apply to the environment.

    Raises:
      ValueError: Invalid actions.

    Returns:
      Batch of observations, rewards, and done flags.
    """
    time_steps = [
        env.step_adversary(action, self._blocking)
        for env, action in zip(self._envs, self._unstack_actions(actions))]
    # When blocking is False we get promises that need to be called.
    if not self._blocking:
      time_steps = [promise() for promise in time_steps]
    return self._stack_time_steps(time_steps)

  def get_num_blocks(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(
          tf.cast(self._envs[0].n_clutter_placed, tf.float32))
    else:
      return tf.stack(
          [tf.cast(env.n_clutter_placed, tf.float32) for env in self._envs])

  def get_distance_to_goal(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(
          tf.cast(self._envs[0].distance_to_goal, tf.float32))
    else:
      return tf.stack(
          [tf.cast(env.distance_to_goal, tf.float32) for env in self._envs])

  def get_deliberate_placement(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(
          tf.cast(self._envs[0].deliberate_agent_placement, tf.float32))
    else:
      return tf.stack(
          [tf.cast(env.deliberate_agent_placement,
                   tf.float32) for env in self._envs])

  def get_goal_x(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(
          tf.cast(self._envs[0].get_goal_x(), tf.float32))
    else:
      return tf.stack(
          [tf.cast(env.get_goal_x(), tf.float32) for env in self._envs])

  def get_goal_y(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(
          tf.cast(self._envs[0].get_goal_y(), tf.float32))
    else:
      return tf.stack(
          [tf.cast(env.get_goal_y(), tf.float32) for env in self._envs])

  def get_passable(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(
          tf.cast(self._envs[0].passable, tf.float32))
    else:
      return tf.stack(
          [tf.cast(env.passable, tf.float32) for env in self._envs])

  def get_shortest_path_length(self):
    if self._num_envs == 1:
      return nest_utils.batch_nested_array(
          tf.cast(self._envs[0].shortest_path_length, tf.float32))
    else:
      return tf.stack(
          [tf.cast(env.shortest_path_length, tf.float32) for env in self._envs])

  def close(self):
    """Close all external process."""
    logging.info('Closing all processes.')
    for env in self._envs:
      env.close()
    logging.info('All processes closed.')

  def _stack_time_steps(self, time_steps):
    """Given a list of TimeStep, combine to one with a batch dimension."""
    if self._flatten:
      return nest_utils.fast_map_structure_flatten(
          lambda *arrays: np.stack(arrays), self._time_step_spec, *time_steps)
    else:
      return nest_utils.fast_map_structure(
          lambda *arrays: np.stack(arrays), *time_steps)

  def _unstack_actions(self, batched_actions):
    """Returns a list of actions from potentially nested batch of actions."""
    flattened_actions = tf.nest.flatten(batched_actions)
    if self._flatten:
      unstacked_actions = zip(*flattened_actions)
    else:
      unstacked_actions = [
          tf.nest.pack_sequence_as(batched_actions, actions)
          for actions in zip(*flattened_actions)
      ]
    return unstacked_actions

  def seed(self, seeds):
    """Seeds the parallel environments."""
    if len(seeds) != len(self._envs):
      raise ValueError(
          'Number of seeds should match the number of parallel_envs.')

    promises = [env.call('seed', seed) for seed, env in zip(seeds, self._envs)]
    # Block until all envs are seeded.
    return [promise() for promise in promises]


class AdversarialProcessPyEnvironment(object):
  """Step a single env in a separate process for lock free paralellism."""

  # Message types for communication via the pipe.
  _READY = 1
  _ACCESS = 2
  _CALL = 3
  _RESULT = 4
  _EXCEPTION = 5
  _CLOSE = 6

  def __init__(self, env_constructor, flatten=False):
    """Step environment in a separate process for lock free paralellism.

    The environment is created in an external process by calling the provided
    callable. This can be an environment class, or a function creating the
    environment and potentially wrapping it. The returned environment should
    not access global variables.

    Args:
      env_constructor: Callable that creates and returns a Python environment.
      flatten: Boolean, whether to assume flattened actions and time_steps
        during communication to avoid overhead.

    Attributes:
      observation_spec: The cached observation spec of the environment.
      action_spec: The cached action spec of the environment.
      time_step_spec: The cached time step spec of the environment.
    """
    # NOTE(ebrevdo): multiprocessing uses the standard py3 pickler which does
    # not support anonymous lambdas.  Folks usually pass anonymous lambdas as
    # env constructors.  Here we work around this by manually pickling
    # the constructor using cloudpickle; which supports these.  In the
    # new process, we'll unpickle this constructor and run it.
    self._pickled_env_constructor = cloudpickle.dumps(env_constructor)
    self._flatten = flatten
    self._observation_spec = None
    self._action_spec = None
    self._time_step_spec = None

  def start(self, wait_to_start=True):
    """Start the process.

    Args:
      wait_to_start: Whether the call should wait for an env initialization.
    """
    mp_context = system_multiprocessing.get_context()
    self._conn, conn = mp_context.Pipe()
    self._process = mp_context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    if wait_to_start:
      self.wait_start()

  def wait_start(self):
    """Wait for the started process to finish initialization."""
    result = self._conn.recv()
    if isinstance(result, Exception):
      self._conn.close()
      self._process.join(5)
      raise result
    assert result == self._READY, result

  def observation_spec(self):
    if not self._observation_spec:
      self._observation_spec = self.call('observation_spec')()
    return self._observation_spec

  def action_spec(self):
    if not self._action_spec:
      self._action_spec = self.call('action_spec')()
    return self._action_spec

  def time_step_spec(self):
    if not self._time_step_spec:
      self._time_step_spec = self.call('time_step_spec')()
    return self._time_step_spec

  def __getattr__(self, name):
    """Request an attribute from the environment.

    Note that this involves communication with the external process, so it can
    be slow.

    This method is only called if the attribute is not found in the dictionary
    of `ParallelPyEnvironment`'s definition.

    Args:
      name: Attribute to access.

    Returns:
      Value of the attribute.
    """
    # Accessed by multiprocessing Pickler or this function.
    if name.startswith('_'):
      return super(AdversarialProcessPyEnvironment, self).__getattribute__(name)

    # All other requests get sent to the worker.
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    """Asynchronously call a method of the external environment.

    Args:
      name: Name of the method to call.
      *args: Positional arguments to forward to the method.
      **kwargs: Keyword arguments to forward to the method.

    Returns:
      Promise object that blocks and provides the return value when called.
    """
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    """Send a close message to the external process and join it."""
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    if self._process.is_alive():
      self._process.join(5)

  def step(self, action, blocking=True):
    """Step the environment.

    Args:
      action: The action to apply to the environment.
      blocking: Whether to wait for the result.

    Returns:
      time step when blocking, otherwise callable that returns the time step.
    """
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=True):
    """Reset the environment.

    Args:
      blocking: Whether to wait for the result.

    Returns:
      New observation when blocking, otherwise callable that returns the new
      observation.
    """
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def step_adversary(self, action, blocking=True):
    promise = self.call('step_adversary', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset_agent(self, blocking=True):
    promise = self.call('reset_agent')
    if blocking:
      return promise()
    else:
      return promise

  def reset_random(self, blocking=True):
    promise = self.call('reset_random')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    """Wait for a message from the worker process and return its payload.

    Raises:
      Exception: An exception was raised inside the worker process.
      KeyError: The reveived message is of an unknown type.

    Returns:
      Payload object of the message.
    """
    message, payload = self._conn.recv()
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    self.close()
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, conn):
    """The process waits for actions and sends back environment results.

    Args:
      conn: Connection for communication to the main process.

    Raises:
      KeyError: When receiving a message of unknown type.
    """
    try:
      env = cloudpickle.loads(self._pickled_env_constructor)()
      action_spec = env.action_spec()
      conn.send(self._READY)  # Ready.
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          if self._flatten and name == 'step':
            args = [tf.nest.pack_sequence_as(action_spec, args[0])]
          elif self._flatten and name == 'step_adversary':
            args = [tf.nest.pack_sequence_as(
                env.adversary_action_spec, args[0])]
          result = getattr(env, name)(*args, **kwargs)
          if self._flatten and name in [
              'step', 'reset', 'step_advesary', 'reset_agent', 'reset_random']:
            result = tf.nest.flatten(result)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          env.close()
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:  # pylint: disable=broad-except
      etype, evalue, tb = sys.exc_info()
      stacktrace = ''.join(traceback.format_exception(etype, evalue, tb))
      message = 'Error in environment process: {}'.format(stacktrace)
      logging.error(message)
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      conn.close()
