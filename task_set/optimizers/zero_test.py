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

"""Tests for task_set.optimizers.zero."""
import numpy as np
from task_set.optimizers import zero
import tensorflow.compat.v1 as tf


class ZeroTest(tf.test.TestCase):

  def test_zero(self):
    opt = zero.ZeroOptimizer()
    v = tf.get_variable(
        name='weights',
        shape=[10],
        dtype=tf.float32,
        initializer=tf.initializers.random_normal())
    loss = tf.reduce_mean(v**2)
    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, var_list=[v], global_step=global_step)
    with self.cached_session() as sess:
      sess.run(tf.initializers.global_variables())
      sess.run(train_op)
      output = sess.run(v)
      self.assertEqual(output, np.zeros(shape=[10], dtype=np.float32))
      self.assertEqual(global_step, 1.0)
