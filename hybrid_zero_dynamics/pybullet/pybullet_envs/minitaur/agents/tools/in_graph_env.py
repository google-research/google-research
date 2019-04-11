# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Put an OpenAI Gym environment into the TensorFlow graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import tensorflow as tf


class InGraphEnv(object):
  """Put an OpenAI Gym environment into the TensorFlow graph.

  The environment will be stepped and reset inside of the graph using
  tf.py_func(). The current observation, action, reward, and done flag are held
  in according variables.
  """

  def __init__(self, env):
    """Put an OpenAI Gym environment into the TensorFlow graph.

    Args:
      env: OpenAI Gym environment.
    """
    self._env = env
    observ_shape = self._parse_shape(self._env.observation_space)
    observ_dtype = self._parse_dtype(self._env.observation_space)
    action_shape = self._parse_shape(self._env.action_space)
    action_dtype = self._parse_dtype(self._env.action_space)
    with tf.name_scope('environment'):
      self._observ = tf.Variable(
          tf.zeros(observ_shape, observ_dtype), name='observ', trainable=False)
      self._action = tf.Variable(
          tf.zeros(action_shape, action_dtype), name='action', trainable=False)
      self._reward = tf.Variable(
          0.0, dtype=tf.float32, name='reward', trainable=False)
      self._done = tf.Variable(
          True, dtype=tf.bool, name='done', trainable=False)
      self._step = tf.Variable(
          0, dtype=tf.int32, name='step', trainable=False)

  def __getattr__(self, name):
    """Forward unimplemented attributes to the original environment.

    Args:
      name: Attribute that was accessed.

    Returns:
      Value behind the attribute name in the wrapped environment.
    """
    return getattr(self._env, name)

  def simulate(self, action):
    """Step the environment.

    The result of the step can be accessed from the variables defined below.

    Args:
      action: Tensor holding the action to apply.

    Returns:
      Operation.
    """
    with tf.name_scope('environment/simulate'):
      if action.dtype in (tf.float16, tf.float32, tf.float64):
        action = tf.check_numerics(action, 'action')
      observ_dtype = self._parse_dtype(self._env.observation_space)
      observ, reward, done = tf.py_func(
          lambda a: self._env.step(a)[:3], [action],
          [observ_dtype, tf.float32, tf.bool], name='step')
      observ = tf.check_numerics(observ, 'observ')
      reward = tf.check_numerics(reward, 'reward')
      return tf.group(
          self._observ.assign(observ),
          self._action.assign(action),
          self._reward.assign(reward),
          self._done.assign(done),
          self._step.assign_add(1))

  def reset(self):
    """Reset the environment.

    Returns:
      Tensor of the current observation.
    """
    observ_dtype = self._parse_dtype(self._env.observation_space)
    observ = tf.py_func(self._env.reset, [], observ_dtype, name='reset')
    observ = tf.check_numerics(observ, 'observ')
    with tf.control_dependencies([
        self._observ.assign(observ),
        self._reward.assign(0),
        self._done.assign(False)]):
      return tf.identity(observ)

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ

  @property
  def action(self):
    """Access the variable holding the last recieved action."""
    return self._action

  @property
  def reward(self):
    """Access the variable holding the current reward."""
    return self._reward

  @property
  def done(self):
    """Access the variable indicating whether the episode is done."""
    return self._done

  @property
  def step(self):
    """Access the variable containg total steps of this environment."""
    return self._step

  def _parse_shape(self, space):
    """Get a tensor shape from a OpenAI Gym space.

    Args:
      space: Gym space.

    Returns:
      Shape tuple.
    """
    if isinstance(space, gym.spaces.Discrete):
      return ()
    if isinstance(space, gym.spaces.Box):
      return space.shape
    raise NotImplementedError()

  def _parse_dtype(self, space):
    """Get a tensor dtype from a OpenAI Gym space.

    Args:
      space: Gym space.

    Returns:
      TensorFlow data type.
    """
    if isinstance(space, gym.spaces.Discrete):
      return tf.int32
    if isinstance(space, gym.spaces.Box):
      return tf.float32
    raise NotImplementedError()
