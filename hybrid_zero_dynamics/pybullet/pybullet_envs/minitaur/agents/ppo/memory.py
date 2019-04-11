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

"""Memory that stores episodes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class EpisodeMemory(object):
  """Memory that stores episodes."""

  def __init__(self, template, capacity, max_length, scope):
    """Create a memory that stores episodes.

    Each transition tuple consists of quantities specified by the template.
    These quantities would typically be be observartions, actions, rewards, and
    done indicators.

    Args:
      template: List of tensors to derive shapes and dtypes of each transition.
      capacity: Number of episodes, or rows, hold by the memory.
      max_length: Allocated sequence length for the episodes.
      scope: Variable scope to use for internal variables.
    """
    self._capacity = capacity
    self._max_length = max_length
    with tf.variable_scope(scope) as scope:
      self._scope = scope
      self._length = tf.Variable(tf.zeros(capacity, tf.int32), False)
      self._buffers = [
          tf.Variable(tf.zeros(
              [capacity, max_length] + elem.shape.as_list(),
              elem.dtype), False)
          for elem in template]

  def length(self, rows=None):
    """Tensor holding the current length of episodes.

    Args:
      rows: Episodes to select length from, defaults to all.

    Returns:
      Batch tensor of sequence lengths.
    """
    rows = tf.range(self._capacity) if rows is None else rows
    return tf.gather(self._length, rows)

  def append(self, transitions, rows=None):
    """Append a batch of transitions to rows of the memory.

    Args:
      transitions: Tuple of transition quantities with batch dimension.
      rows: Episodes to append to, defaults to all.

    Returns:
      Operation.
    """
    rows = tf.range(self._capacity) if rows is None else rows
    assert rows.shape.ndims == 1
    assert_capacity = tf.assert_less(
        rows, self._capacity,
        message='capacity exceeded')
    with tf.control_dependencies([assert_capacity]):
      assert_max_length = tf.assert_less(
          tf.gather(self._length, rows), self._max_length,
          message='max length exceeded')
    append_ops = []
    with tf.control_dependencies([assert_max_length]):
      for buffer_, elements in zip(self._buffers, transitions):
        timestep = tf.gather(self._length, rows)
        indices = tf.stack([rows, timestep], 1)
        append_ops.append(tf.scatter_nd_update(buffer_, indices, elements))
    with tf.control_dependencies(append_ops):
      episode_mask = tf.reduce_sum(tf.one_hot(
          rows, self._capacity, dtype=tf.int32), 0)
      return self._length.assign_add(episode_mask)

  def replace(self, episodes, length, rows=None):
    """Replace full episodes.

    Args:
      episodes: Tuple of transition quantities with batch and time dimensions.
      length: Batch of sequence lengths.
      rows: Episodes to replace, defaults to all.

    Returns:
      Operation.
    """
    rows = tf.range(self._capacity) if rows is None else rows
    assert rows.shape.ndims == 1
    assert_capacity = tf.assert_less(
        rows, self._capacity, message='capacity exceeded')
    with tf.control_dependencies([assert_capacity]):
      assert_max_length = tf.assert_less_equal(
          length, self._max_length, message='max length exceeded')
    replace_ops = []
    with tf.control_dependencies([assert_max_length]):
      for buffer_, elements in zip(self._buffers, episodes):
        replace_op = tf.scatter_update(buffer_, rows, elements)
        replace_ops.append(replace_op)
    with tf.control_dependencies(replace_ops):
      return tf.scatter_update(self._length, rows, length)

  def data(self, rows=None):
    """Access a batch of episodes from the memory.

    Padding elements after the length of each episode are unspecified and might
    contain old data.

    Args:
      rows: Episodes to select, defaults to all.

    Returns:
      Tuple containing a tuple of transition quantiries with batch and time
      dimensions, and a batch of sequence lengths.
    """
    rows = tf.range(self._capacity) if rows is None else rows
    assert rows.shape.ndims == 1
    episode = [tf.gather(buffer_, rows) for buffer_ in self._buffers]
    length = tf.gather(self._length, rows)
    return episode, length

  def clear(self, rows=None):
    """Reset episodes in the memory.

    Internally, this only sets their lengths to zero. The memory entries will
    be overridden by future calls to append() or replace().

    Args:
      rows: Episodes to clear, defaults to all.

    Returns:
      Operation.
    """
    rows = tf.range(self._capacity) if rows is None else rows
    assert rows.shape.ndims == 1
    return tf.scatter_update(self._length, rows, tf.zeros_like(rows))
