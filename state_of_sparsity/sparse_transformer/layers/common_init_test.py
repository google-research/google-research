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

"""Tests for weight initializers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from state_of_sparsity.sparse_transformer.layers import common_init


class SparseGlorotUniformTest(tf.test.TestCase):

  def testSparseGlorotUniform_OutputShape(self):
    initializer = common_init.SparseGlorotUniform(.5)
    x = tf.get_variable(
        "x",
        shape=[512, 1024],
        initializer=initializer,
        dtype=tf.float32)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      res = sess.run(x)
    self.assertEqual(res.shape, (512, 1024))

  def testSparseGlorotUniform_NoSparsity(self):
    initializer = common_init.SparseGlorotUniform(0, seed=5)
    initializer_base = tf.glorot_uniform_initializer(seed=5)

    x = tf.get_variable(
        "x",
        shape=[512, 1024],
        initializer=initializer,
        dtype=tf.float32)
    y = tf.get_variable(
        "y",
        shape=[512, 1024],
        initializer=initializer_base,
        dtype=tf.float32)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      res_x = sess.run(x)
      res_y = sess.run(y)
    self.assertEqual(res_x.shape, (512, 1024))
    self.assertEqual(res_y.shape, (512, 1024))
    self.assertAllEqual(res_x, res_y)

if __name__ == "__main__":
  tf.test.main()
