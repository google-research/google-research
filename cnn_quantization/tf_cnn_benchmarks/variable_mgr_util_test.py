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

"""Tests for variable_mgr_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from cnn_quantization.tf_cnn_benchmarks import variable_mgr_util


class VariableMgrUtilTest(tf.test.TestCase):

  def testGetLossScaleUpdateOpTruePath(self):
    loss_scale = tf.Variable(4)
    # loss_scale_normal_steps >= inc_loss_scale_every_n
    loss_scale_normal_steps = tf.Variable(10)
    inc_loss_scale_every_n = 10
    update_op = variable_mgr_util.get_loss_scale_update_op(
        loss_scale, loss_scale_normal_steps, inc_loss_scale_every_n)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(update_op)

      self.assertEqual(sess.run(loss_scale), 8)
      self.assertEqual(sess.run(loss_scale_normal_steps), 0)

  def testGetLossScaleUpdateOpFalsePath(self):
    loss_scale = tf.Variable(4)
    # loss_scale_normal_steps < inc_loss_scale_every_n
    loss_scale_normal_steps = tf.Variable(9)
    inc_loss_scale_every_n = 10
    update_op = variable_mgr_util.get_loss_scale_update_op(
        loss_scale, loss_scale_normal_steps, inc_loss_scale_every_n)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(update_op)

      self.assertEqual(sess.run(loss_scale), 4)
      self.assertEqual(sess.run(loss_scale_normal_steps), 10)

  def testAppendGradientsWithLossScaleWithAutoScaleDisabled(self):
    v = tf.Variable(0)
    training_ops = []
    get_apply_gradients_ops_func = lambda: [tf.assign(v, v + 1)]
    loss_scale_params = variable_mgr_util.AutoLossScaleParams(
        enable_auto_loss_scale=False,  # no auto loss scale.
        loss_scale=tf.Variable(4),
        loss_scale_normal_steps=tf.Variable(10),
        inc_loss_scale_every_n=10,
        is_chief=True)
    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops,
        get_apply_gradients_ops_func,
        loss_scale_params,
        grad_has_inf_nan=True)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(training_ops)
      self.assertEqual(sess.run(v), 1)
      self.assertEqual(sess.run(loss_scale_params.loss_scale), 4)
      self.assertEqual(sess.run(loss_scale_params.loss_scale_normal_steps), 10)

  def testAppendGradientsWithLossScaleForNonChiefWorker(self):
    v = tf.Variable(0)
    training_ops = []
    get_apply_gradients_ops_func = lambda: [tf.assign(v, v + 1)]
    loss_scale_params = variable_mgr_util.AutoLossScaleParams(
        enable_auto_loss_scale=True,
        loss_scale=tf.Variable(4),
        loss_scale_normal_steps=tf.Variable(10),
        inc_loss_scale_every_n=10,
        is_chief=False)  # Non-chief
    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops,
        get_apply_gradients_ops_func,
        loss_scale_params,
        grad_has_inf_nan=False)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(training_ops)
      self.assertEqual(sess.run(v), 1)
      self.assertEqual(sess.run(loss_scale_params.loss_scale), 4)
      self.assertEqual(sess.run(loss_scale_params.loss_scale_normal_steps), 10)

  def testAppendGradientsWithLossScaleWithoutNan(self):
    v = tf.Variable(0)
    training_ops = []
    get_apply_gradients_ops_func = lambda: [tf.assign(v, v + 1)]
    loss_scale_params = variable_mgr_util.AutoLossScaleParams(
        enable_auto_loss_scale=True,
        loss_scale=tf.Variable(4, dtype=tf.float32),
        loss_scale_normal_steps=tf.Variable(10),
        inc_loss_scale_every_n=10,
        is_chief=True)
    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops,
        get_apply_gradients_ops_func,
        loss_scale_params,
        grad_has_inf_nan=tf.constant(False))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(training_ops)
      self.assertEqual(sess.run(v), 1)
      self.assertEqual(sess.run(loss_scale_params.loss_scale), 8)
      self.assertEqual(sess.run(loss_scale_params.loss_scale_normal_steps), 0)

  def testAppendGradientsWithLossScaleWithtNan(self):
    v = tf.Variable(0)
    training_ops = []
    get_apply_gradients_ops_func = lambda: [tf.assign(v, v + 1)]
    loss_scale_params = variable_mgr_util.AutoLossScaleParams(
        enable_auto_loss_scale=True,
        loss_scale=tf.Variable(4, dtype=tf.float32),
        loss_scale_normal_steps=tf.Variable(10),
        inc_loss_scale_every_n=10,
        is_chief=True)
    variable_mgr_util.append_gradients_with_loss_scale(
        training_ops,
        get_apply_gradients_ops_func,
        loss_scale_params,
        grad_has_inf_nan=tf.constant(True))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(training_ops)
      self.assertEqual(sess.run(v), 0)  # Skip updating for v.
      # halve loss_scale and reset local_scale_normal_steps.
      self.assertEqual(sess.run(loss_scale_params.loss_scale), 2)
      self.assertEqual(sess.run(loss_scale_params.loss_scale_normal_steps), 0)


if __name__ == '__main__':
  tf.test.main()
