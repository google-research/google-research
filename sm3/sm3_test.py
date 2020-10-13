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

"""Tests for SM3 optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import tensorflow.compat.v1 as tf

from sm3 import sm3


class SM3Test(tf.test.TestCase):

  def setUp(self):
    super(SM3Test, self).setUp()
    self._learning_rate = 0.1
    self._momentum = 0.9

  def testDenseScalarLayer(self):
    """Test a single dense scalar layer."""

    with self.cached_session() as sess:
      var = tf.Variable(0.5)
      grad_np = 0.1
      grad = tf.Variable(grad_np)
      opt = sm3.SM3Optimizer(
          learning_rate=self._learning_rate, momentum=self._momentum)

      step = opt.apply_gradients([(grad, var)])
      sess.run(tf.global_variables_initializer())

      # Check that variable and momentum are as expected before starting
      # training.
      var_np = sess.run(var)
      gbar_np = sess.run(opt.get_slot(var, 'momentum'))

      self.assertAllClose(0.5, var_np)
      self.assertAllClose(0.0, gbar_np)

      accumulator = numpy.zeros_like(gbar_np)
      for _ in range(2):
        # Run a step of training.
        step.run()

        # Expected preconditioned gradient, momentum, and parameter.
        accumulator += numpy.square(grad_np)
        exp_p_grad = grad_np / numpy.sqrt(accumulator)
        exp_gbar_np = (
            self._momentum * gbar_np + (1 - self._momentum) * exp_p_grad)
        exp_var = var_np - self._learning_rate * exp_gbar_np
        # Check that variable and momentum are as expected after one step of
        # training.
        var_np = sess.run(var)
        gbar_np = sess.run(opt.get_slot(var, 'momentum'))

        self.assertAllClose(exp_var, var_np)
        self.assertAllClose(exp_gbar_np, gbar_np)

  def testDenseVectorLayer(self):
    """Test a single dense vector layer."""

    with self.cached_session() as sess:
      var = tf.Variable([0.5, 0.3])
      grad_np = [0.1, 0.1]
      grad = tf.Variable(grad_np)
      opt = sm3.SM3Optimizer(
          learning_rate=self._learning_rate, momentum=self._momentum)

      step = opt.apply_gradients([(grad, var)])
      sess.run(tf.global_variables_initializer())

      # Check that variable and momentum are as expected before starting
      # training.
      var_np = sess.run(var)
      gbar_np = sess.run(opt.get_slot(var, 'momentum'))

      self.assertAllClose([0.5, 0.3], var_np)
      self.assertAllClose([0.0, 0.0], gbar_np)

      accumulator = numpy.zeros_like(gbar_np)
      for _ in range(2):
        # Run a step of training.
        step.run()

        # Expected preconditioned gradient, momentum, and parameter.
        accumulator += numpy.square(grad_np)
        exp_p_grad = grad_np / numpy.sqrt(accumulator)
        exp_gbar_np = (
            self._momentum * gbar_np + (1 - self._momentum) * exp_p_grad)
        exp_var = var_np - self._learning_rate * exp_gbar_np
        # Check that variable and momentum are as expected after one step of
        # training.
        var_np = sess.run(var)
        gbar_np = sess.run(opt.get_slot(var, 'momentum'))

        self.assertAllClose(exp_var, var_np)
        self.assertAllClose(exp_gbar_np, gbar_np)

  def testDenseLayerMatrix(self):
    """Test a single dense matrix layer."""

    with self.cached_session() as sess:
      var = tf.Variable([[0.5, 0.5], [0.5, 0.5]])
      grad_np = [[0.1, 0.05], [0.03, 0.02]]
      grad = tf.Variable(grad_np)
      opt = sm3.SM3Optimizer(
          learning_rate=self._learning_rate, momentum=self._momentum)

      step = opt.apply_gradients([(grad, var)])
      sess.run(tf.global_variables_initializer())

      # Check that variable and momentum are as expected before starting
      # training.
      var_np = sess.run(var)
      gbar_np = sess.run(opt.get_slot(var, 'momentum'))

      self.assertAllClose(var_np, [[0.5, 0.5], [0.5, 0.5]])
      self.assertAllClose([[0.0, 0.0], [0.0, 0.0]], gbar_np)

      row_accumulator = numpy.zeros([2, 1])
      col_accumulator = numpy.zeros([1, 2])
      accumulator = numpy.zeros_like(gbar_np)
      for _ in range(2):
        # Run a step of training.
        step.run()

        accumulator = numpy.minimum(row_accumulator, col_accumulator)
        # Expected preconditioned gradient, momentum, and parameter.
        accumulator += numpy.square(grad_np)
        # Update SM3 accumulators.
        row_accumulator = numpy.amax(accumulator, axis=1, keepdims=True)
        col_accumulator = numpy.amax(accumulator, axis=0, keepdims=True)
        exp_p_grad = grad_np / numpy.sqrt(accumulator)
        exp_gbar_np = (
            self._momentum * gbar_np + (1 - self._momentum) * exp_p_grad)
        exp_var = var_np - self._learning_rate * exp_gbar_np
        # Check that variable and momentum are as expected after one step of
        # training.
        var_np = sess.run(var)
        gbar_np = sess.run(opt.get_slot(var, 'momentum'))

        self.assertAllClose(exp_var, var_np)
        self.assertAllClose(exp_gbar_np, gbar_np)

  def testZeroGradientNoOpAtFirstStep(self):
    """Test that checks that epsilon handling is unncessary."""

    with self.cached_session() as sess:
      var = tf.Variable(0.5)
      grad = tf.Variable(0.0)
      opt = sm3.SM3Optimizer(
          learning_rate=self._learning_rate, momentum=self._momentum)

      step = opt.apply_gradients([(grad, var)])
      sess.run(tf.global_variables_initializer())

      # Check that variable and momentum are as expected before starting
      # training.
      var_np = sess.run(var)
      gbar_np = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose(0.5, var_np)
      self.assertAllClose(0.0, gbar_np)

      # Run one step of training.
      step.run()
      var_np = sess.run(var)
      gbar_np = sess.run(opt.get_slot(var, 'momentum'))
      self.assertAllClose(0.5, var_np)
      self.assertAllClose(0.0, gbar_np)

  def testSparseUpdates(self):
    """Test that checks sparse updates."""

    with self.cached_session() as sess:
      var = tf.Variable([[0.5, 0.05], [0.05, 1.0], [0.15, 3.0], [0.35, 2.0]])
      # A sparse gradient that updates index 1, and 3.
      grad_np = [[0.1, 0.05], [0.01, 1.5]]
      indices_np = [1, 3]
      shape = [2, 2]
      grad = tf.IndexedSlices(
          tf.constant(grad_np, shape=shape),
          tf.constant(indices_np),  # indices
          tf.constant(shape))  # shape
      opt = sm3.SM3Optimizer(
          learning_rate=self._learning_rate, momentum=self._momentum)
      step = opt.apply_gradients([(grad, var)])
      sess.run(tf.global_variables_initializer())
      # Check that variable and momentum are as expected before starting
      # training.
      var_np = sess.run(var)
      self.assertAllClose([[0.5, 0.05], [0.05, 1.0], [0.15, 3.0], [0.35, 2.0]],
                          var_np)
      # Run one step of training.
      step.run()
      accumulator = numpy.zeros_like(var_np)
      accumulator[indices_np, :] += numpy.square(grad_np)
      row_accumulator = numpy.amax(accumulator, axis=1, keepdims=True)
      # Update SM3 accumulators.
      exp_p_grad = grad_np / numpy.sqrt(accumulator[indices_np, :])
      exp_var_np = var_np
      exp_var_np[indices_np, :] = var_np[
          indices_np, :] - self._learning_rate * exp_p_grad
      var_np = sess.run(var)
      self.assertAllClose(exp_var_np, var_np)
      row_accumulator_var = numpy.reshape(
          sess.run(opt.get_slot(var, 'accumulator_0')), [4, 1])
      self.assertAllClose(row_accumulator_var, row_accumulator)


if __name__ == '__main__':
  tf.test.main()
