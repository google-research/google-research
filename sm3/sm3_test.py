# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Tests for SM3 optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from sm3 import sm3


class SM3Test(tf.test.TestCase):

  def testDenseLayer(self):
    """SM3 update with gbar, and epsilon."""

    with self.cached_session() as sess:
      var = tf.Variable(0.5)
      grad = tf.Variable(0.1)
      opt = sm3.SM3Optimizer(learning_rate=0.1, momentum=0.9)

      step = opt.apply_gradients([(grad, var)])
      tf.global_variables_initializer().run()

      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose(0.5, pre_var)
      self.assertAllClose(0.0, pre_gbar)
      step.run()
      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose(0.49, pre_var)
      self.assertAllClose(0.1, pre_gbar)
      step.run()
      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose(0.4739, pre_var, atol=1e-4)
      self.assertAllClose(0.16, pre_gbar, atol=1e-2)

  def testDenseLayerVector(self):
    """SM3 update with gbar, and epsilon."""

    with self.cached_session() as sess:
      var = tf.Variable([0.5, 0.5])
      grad = tf.Variable([0.1, 0.1])
      opt = sm3.SM3Optimizer(learning_rate=0.1, momentum=0.9)

      step = opt.apply_gradients([(grad, var)])
      tf.global_variables_initializer().run()

      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose([0.5, 0.5], pre_var)
      self.assertAllClose([0.0, 0.0], pre_gbar)
      step.run()
      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose([0.49, 0.49], pre_var)
      self.assertAllClose([0.1, 0.1], pre_gbar)
      step.run()
      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose([0.4739, 0.4739], pre_var, atol=1e-4)
      self.assertAllClose([0.16, 0.16], pre_gbar, atol=1e-2)

  def testDenseLayerMatrix(self):
    """SM3 update with gbar, and epsilon."""

    with self.cached_session() as sess:
      var = tf.Variable([[0.5, 0.5], [0.5, 0.5]])
      grad = tf.Variable([[0.1, 0.1], [0.01, 0.01]])
      opt = sm3.SM3Optimizer(learning_rate=0.1, momentum=0.9)

      step = opt.apply_gradients([(grad, var)])
      tf.global_variables_initializer().run()

      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose([[0.5, 0.5], [0.5, 0.5]], pre_var)
      self.assertAllClose([[0.0, 0.0], [0.0, 0.0]], pre_gbar)
      step.run()
      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose([[0.49, 0.49], [0.49, 0.49]], pre_var)
      self.assertAllClose([[0.1, 0.1], [0.1, 0.1]], pre_gbar)
      step.run()
      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose([[0.4739, 0.4739], [0.4739, 0.4739]],
                          pre_var,
                          atol=1e-4)
      self.assertAllClose([[0.16, 0.16], [0.16, 0.16]], pre_gbar, atol=1e-2)

  def testNoEpsilon(self):
    """SM3 update without epsilon."""

    with self.cached_session() as sess:
      var = tf.Variable(0.5)
      grad = tf.Variable(0.0)
      opt = sm3.SM3Optimizer(learning_rate=0.1, momentum=0.9)

      step = opt.apply_gradients([(grad, var)])
      tf.global_variables_initializer().run()

      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose(0.5, pre_var)
      self.assertAllClose(0.0, pre_gbar)
      step.run()
      pre_var = sess.run(var)
      pre_gbar = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose(0.5, pre_var)
      self.assertAllClose(0.0, pre_gbar)

  def testSparseUpdates(self):
    """SM3 sparse updates."""

    with self.cached_session() as sess:
      var = tf.Variable([[0.5], [0.5], [0.5], [0.5]])
      grad = tf.IndexedSlices(
          tf.constant([0.1, 0.1], shape=[2, 1]), tf.constant([1, 3]),
          tf.constant([2, 1]))
      opt = sm3.SM3Optimizer(learning_rate=0.1, momentum=0.9)
      step = opt.apply_gradients([(grad, var)])
      tf.global_variables_initializer().run()

      pre_var = sess.run(var)
      self.assertAllClose([[0.5], [0.5], [0.5], [0.5]], pre_var)
      step.run()
      pre_var = sess.run(var)
      self.assertAllClose([[0.5], [0.4], [0.5], [0.4]], pre_var)


if __name__ == '__main__':
  tf.test.main()
