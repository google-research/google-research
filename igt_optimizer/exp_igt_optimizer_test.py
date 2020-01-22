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

# Lint as: python2, python3
"""Tests for igt_optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from igt_optimizer import exp_igt_optimizer
# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import test
# pylint:enable=g-direct-tensorflow-import

LEARNING_RATE = 2.0


class IgtValidator(object):
  """A reference python implementation of the IGT optimizer."""

  def __init__(self,
               w_init,
               learning_rate,
               reset_steps=None,
               reset_shift=False):
    self.w = w_init
    self.learning_rate = learning_rate
    if reset_steps is None:
      reset_steps = []
    self.reset_steps = reset_steps
    self.reset_shift = reset_shift

    self.step = 0
    self.v = np.zeros(self.w.shape)
    self.w_hat = self.w.copy()

  def update(self, grad):
    momentum = self.step / (self.step + 1.)
    self.v = momentum * self.v + (1. - momentum) * grad
    update = -self.learning_rate * self.v
    self.w += update

    self.step += 1

    momentum_next = self.step / (self.step + 1.)
    self.w_hat = self.w + momentum_next / (1. - momentum_next) * update

    if self.step in self.reset_steps:
      if self.reset_shift:
        self.w = self.w_hat
      else:
        self.w_hat = self.w
      self.step = 0


class ExpIgtOptimizerTest(test.TestCase):

  def doTestApplyGradients(self, use_resource=False):
    """Validate the IGT update (i.e. apply_gradients) against a python impl."""
    # TODO(manzagop): try dtypes.half and dtypes.float64:
    for dtype in [dtypes.float32]:
      print('running for dtype {}'.format(dtype))

      with self.test_session():
        # Set up 2 variables and constants for their gradients.
        var0_value = np.array([1.0, 2.0])
        var1_value = np.array([3.0, 4.0])
        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_value, dtype=dtype)
          var1 = resource_variable_ops.ResourceVariable(var1_value, dtype=dtype)
        else:
          var0 = tf_variables.Variable(var0_value, dtype=dtype)
          var1 = tf_variables.Variable(var1_value, dtype=dtype)
        grads0 = tf.placeholder(dtype, shape=var0.get_shape())
        grads1 = tf.placeholder(dtype, shape=var1.get_shape())

        # TODO(manzagop): use a different tail fraction once validator support.
        igt_opt = exp_igt_optimizer.ExpIgtOptimizer(
            learning_rate=LEARNING_RATE, tail_fraction=1.)
        igt_update = igt_opt.apply_gradients(
            list(zip([grads0, grads1], [var0, var1])),
            global_step=tf.train.get_global_step())
        tf_variables.global_variables_initializer().run()

        # Validate we have slots.
        expected_slot_names = set(['estimate', 'true_param', 'update'])
        self.assertEqual(expected_slot_names, set(igt_opt.get_slot_names()))

        for slot_name in expected_slot_names:
          for var in [var0, var1]:
            slot = igt_opt.get_slot(var, slot_name)
            self.assertEqual(slot.get_shape(), var.get_shape())
            self.assertNotIn(slot, tf_variables.trainable_variables())

        # Validate initial values.
        validators = [
            IgtValidator(var0_value, LEARNING_RATE),
            IgtValidator(var1_value, LEARNING_RATE)
        ]
        self._validate(igt_opt, [var0, var1], validators)

        # Run first update and validate.
        g0_first = np.array([0.1, 0.1])
        g1_first = np.array([0.01, 0.01])
        igt_update.run({grads0: g0_first, grads1: g1_first})

        validators[0].update(g0_first)
        validators[1].update(g1_first)
        self._validate(igt_opt, [var0, var1], validators)

        # Run second update and validate.
        g0_second = np.array([0.1, 0.1])
        g1_second = np.array([0.01, 0.01])
        igt_update.run({grads0: g0_second, grads1: g1_second})

        validators[0].update(g0_second)
        validators[1].update(g1_second)
        self._validate(igt_opt, [var0, var1], validators)

  def _validate(self, opt, variables, validators):
    for var, validator in zip(variables, validators):
      slot = opt.get_slot(var, 'estimate')
      self.assertAllCloseAccordingToType(validator.v, slot.eval())

      slot = opt.get_slot(var, 'true_param')
      self.assertAllCloseAccordingToType(validator.w, slot.eval())

      self.assertAllCloseAccordingToType(validator.w_hat, var.eval())

  def testApplyGradients(self):
    self.doTestApplyGradients(use_resource=False)

  def testResourceApplyGradients(self):
    self.doTestApplyGradients(use_resource=True)

  def testMinimize(self):
    """Ensure that minimize actually lowers the loss."""
    with self.test_session():
      w_init = np.random.randn(10)
      w = tf.Variable(w_init, dtype=dtypes.float32)
      loss = tf.reduce_sum(w * w)

      igt_opt = exp_igt_optimizer.ExpIgtOptimizer(
          learning_rate=0.01, tail_fraction=2.)
      igt_update = igt_opt.minimize(loss)

      tf_variables.global_variables_initializer().run()

      loss_pre = loss.eval()
      igt_update.run()
      loss_post = loss.eval()
      self.assertLess(loss_post, loss_pre)

  def testSwap(self):
    with self.cached_session() as sess:
      v_init = np.random.randn(10)
      v = tf.Variable(v_init, dtype=dtypes.float32)
      loss = tf.reduce_sum(v * v)

      opt = exp_igt_optimizer.ExpIgtOptimizer(
          learning_rate=0.01, tail_fraction=2.)
      unused_igt_update = opt.minimize(loss)
      slot = opt.get_slot(v, 'true_param')

      tf_variables.global_variables_initializer().run()
      self.assertAllCloseAccordingToType(v_init, v.eval())
      self.assertAllCloseAccordingToType(v_init, slot.eval())

      zeros = np.zeros(10)
      sess.run(v.assign(zeros))
      self.assertAllCloseAccordingToType(zeros, v.eval())
      self.assertAllCloseAccordingToType(v_init, slot.eval())

      swap_op = opt.swap_true_and_shifted()
      swap_op.run()
      self.assertAllCloseAccordingToType(v_init, v.eval())
      self.assertAllCloseAccordingToType(zeros, slot.eval())


if __name__ == '__main__':
  test.main()
