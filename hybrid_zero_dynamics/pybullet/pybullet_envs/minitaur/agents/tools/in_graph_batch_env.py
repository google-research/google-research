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

"""Batch of environments inside the TensorFlow graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pdb
import gym
import tensorflow as tf


class InGraphBatchEnv(object):
  """Batch of environments inside the TensorFlow graph.

  The batch of environments will be stepped and reset inside of the graph using
  a tf.py_func(). The current batch of observations, actions, rewards, and done
  flags are held in according variables.
  """

  def __init__(self, batch_env):
    """Batch of environments inside the TensorFlow graph.

    Args:
      batch_env: Batch environment.
    """
    self._batch_env = batch_env
    observ_shape = self._parse_shape(self._batch_env.observation_space)
    observ_dtype = self._parse_dtype(self._batch_env.observation_space)
    action_shape = self._parse_shape(self._batch_env.action_space)
    action_dtype = self._parse_dtype(self._batch_env.action_space)
    with tf.variable_scope('env_temporary'):
      self._observ = tf.Variable(
          tf.zeros((len(self._batch_env),) + observ_shape, observ_dtype),
          name='observ', trainable=False)
      self._action = tf.Variable(
          tf.zeros((len(self._batch_env),) + action_shape, action_dtype),
          name='action', trainable=False)
      self._reward = tf.Variable(
          tf.zeros((len(self._batch_env),), tf.float32),
          name='reward', trainable=False)
      self._done = tf.Variable(
          tf.cast(tf.ones((len(self._batch_env),)), tf.bool),
          name='done', trainable=False)

  def __getattr__(self, name):
    """Forward unimplemented attributes to one of the original environments.

    Args:
      name: Attribute that was accessed.

    Returns:
      Value behind the attribute name in one of the original environments.
    """
    return getattr(self._batch_env, name)

  def __len__(self):
    """Number of combined environments."""
    return len(self._batch_env)

  def __getitem__(self, index):
    """Access an underlying environment by index."""
    return self._batch_env[index]

  def simulate(self, action):
    """Step the batch of environments.

    The results of the step can be accessed from the variables defined below.

    Args:
      action: Tensor holding the batch of actions to apply.

    Returns:
      Operation.
    """
    with tf.name_scope('environment/simulate'):
      if action.dtype in (tf.float16, tf.float32, tf.float64):
        action = tf.check_numerics(action, 'action')
      observ_dtype = self._parse_dtype(self._batch_env.observation_space)
      observ, reward, done = tf.py_func(
          lambda a: self._batch_env.step(a)[:3], [action],
          [observ_dtype, tf.float32, tf.bool], name='step')
      observ = tf.check_numerics(observ, 'observ')
      reward = tf.check_numerics(reward, 'reward')
      return tf.group(
          self._observ.assign(observ),
          self._action.assign(action),
          self._reward.assign(reward),
          self._done.assign(done))

  def reset(self, indices=None):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset; defaults to all.

    Returns:
      Batch tensor of the new observations.
    """
    if indices is None:
      indices = tf.range(len(self._batch_env))
    observ_dtype = self._parse_dtype(self._batch_env.observation_space)
    observ = tf.py_func(
        self._batch_env.reset, [indices], observ_dtype, name='reset')
    observ = tf.check_numerics(observ, 'observ')
    reward = tf.zeros_like(indices, tf.float32)
    done = tf.zeros_like(indices, tf.bool)
    with tf.control_dependencies([
        tf.scatter_update(self._observ, indices, observ),
        tf.scatter_update(self._reward, indices, reward),
        tf.scatter_update(self._done, indices, done)]):
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

  def close(self):
    """Send close messages to the external process and join them."""
    self._batch_env.close()

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
