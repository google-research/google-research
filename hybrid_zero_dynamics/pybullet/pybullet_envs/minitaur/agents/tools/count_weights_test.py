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

"""Tests for the weight counting utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from agents.tools import count_weights


class CountWeightsTest(tf.test.TestCase):

  def test_count_trainable(self):
    tf.Variable(tf.zeros((5, 3)), trainable=True)
    tf.Variable(tf.zeros((1, 1)), trainable=True)
    tf.Variable(tf.zeros((5,)), trainable=True)
    self.assertEqual(15 + 1 + 5, count_weights())

  def test_ignore_non_trainable(self):
    tf.Variable(tf.zeros((5, 3)), trainable=False)
    tf.Variable(tf.zeros((1, 1)), trainable=False)
    tf.Variable(tf.zeros((5,)), trainable=False)
    self.assertEqual(0, count_weights())

  def test_trainable_and_non_trainable(self):
    tf.Variable(tf.zeros((5, 3)), trainable=True)
    tf.Variable(tf.zeros((8, 2)), trainable=False)
    tf.Variable(tf.zeros((1, 1)), trainable=True)
    tf.Variable(tf.zeros((5,)), trainable=True)
    tf.Variable(tf.zeros((3, 1)), trainable=False)
    self.assertEqual(15 + 1 + 5, count_weights())

  def test_include_scopes(self):
    tf.Variable(tf.zeros((3, 2)), trainable=True)
    with tf.variable_scope('foo'):
      tf.Variable(tf.zeros((5, 2)), trainable=True)
    self.assertEqual(6 + 10, count_weights())

  def test_restrict_scope(self):
    tf.Variable(tf.zeros((3, 2)), trainable=True)
    with tf.variable_scope('foo'):
      tf.Variable(tf.zeros((5, 2)), trainable=True)
      with tf.variable_scope('bar'):
        tf.Variable(tf.zeros((1, 2)), trainable=True)
    self.assertEqual(10 + 2, count_weights('foo'))

  def test_restrict_nested_scope(self):
    tf.Variable(tf.zeros((3, 2)), trainable=True)
    with tf.variable_scope('foo'):
      tf.Variable(tf.zeros((5, 2)), trainable=True)
      with tf.variable_scope('bar'):
        tf.Variable(tf.zeros((1, 2)), trainable=True)
    self.assertEqual(2, count_weights('foo/bar'))

  def test_restrict_invalid_scope(self):
    tf.Variable(tf.zeros((3, 2)), trainable=True)
    with tf.variable_scope('foo'):
      tf.Variable(tf.zeros((5, 2)), trainable=True)
      with tf.variable_scope('bar'):
        tf.Variable(tf.zeros((1, 2)), trainable=True)
    self.assertEqual(0, count_weights('bar'))

  def test_exclude_by_regex(self):
    tf.Variable(tf.zeros((3, 2)), trainable=True)
    with tf.variable_scope('foo'):
      tf.Variable(tf.zeros((5, 2)), trainable=True)
      with tf.variable_scope('bar'):
        tf.Variable(tf.zeros((1, 2)), trainable=True)
    self.assertEqual(0, count_weights(exclude=r'.*'))
    self.assertEqual(6, count_weights(exclude=r'(^|/)foo/.*'))
    self.assertEqual(16, count_weights(exclude=r'.*/bar/.*'))

  def test_non_default_graph(self):
    graph = tf.Graph()
    with graph.as_default():
      tf.Variable(tf.zeros((5, 3)), trainable=True)
      tf.Variable(tf.zeros((8, 2)), trainable=False)
    self.assertNotEqual(graph, tf.get_default_graph)
    self.assertEqual(15, count_weights(graph=graph))


if __name__ == '__main__':
  tf.test.main()
