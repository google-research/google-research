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

"""Tests for variational dropout convolutional neural network cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl.testing.parameterized as parameterized
import tensorflow.compat.v1 as tf
import state_of_sparsity.layers.l0_regularization as l0
import state_of_sparsity.layers.variational_dropout as vd
from state_of_sparsity.sparse_rn50 import pruning_layers
from state_of_sparsity.sparse_rn50 import utils
from tensorflow.contrib.model_pruning.python.layers import core_layers as core

PRUNING_METHODS = [{
    'pruning_method': 'threshold'
}, {
    'pruning_method': 'variational_dropout'
}, {
    'pruning_method': 'l0_regularization'
}]


class ConvLayerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ConvLayerTest, self).setUp()
    self.height, self.width = 7, 9

  @parameterized.parameters(PRUNING_METHODS)
  def testInvalidRank3(self, pruning_method):

    input_tensor = tf.ones((self.height, self.width, 3))
    with self.assertRaisesRegexp(ValueError, 'Rank'):
      pruning_layers.sparse_conv2d(
          x=input_tensor,
          units=32,
          kernel_size=3,
          sparsity_technique=pruning_method)

  @parameterized.parameters(PRUNING_METHODS)
  def testInvalidRank5(self, pruning_method):
    input_tensor = tf.ones((8, 8, self.height, self.width, 3))
    with self.assertRaisesRegexp(ValueError, 'Rank'):
      pruning_layers.sparse_conv2d(
          x=input_tensor,
          units=32,
          kernel_size=3,
          sparsity_technique=pruning_method)

  @parameterized.parameters(PRUNING_METHODS)
  def testSingleConvMaskAdded(self, pruning_method):
    kernel_size = [3, 3]
    input_depth, output_depth = 8, 32
    input_tensor = tf.ones((8, self.height, self.width, input_depth))
    pruning_layers.sparse_conv2d(
        x=input_tensor,
        units=32,
        kernel_size=kernel_size,
        sparsity_technique=pruning_method)

    if pruning_method == 'variational_dropout':
      theta_logsigma2 = tf.get_collection(
          vd.layers.THETA_LOGSIGMA2_COLLECTION)
      self.assertLen(theta_logsigma2, 1)
      self.assertListEqual(
          theta_logsigma2[0][0].get_shape().as_list(),
          [kernel_size[0], kernel_size[1], input_depth, output_depth])
    elif pruning_method == 'l0_regularization':
      theta_logalpha = tf.get_collection(
          l0.layers.THETA_LOGALPHA_COLLECTION)
      self.assertLen(theta_logalpha, 1)
      self.assertListEqual(
          theta_logalpha[0][0].get_shape().as_list(),
          [kernel_size[0], kernel_size[1], input_depth, output_depth])
    else:
      mask = tf.get_collection(core.MASK_COLLECTION)
      self.assertLen(mask, 1)
      self.assertListEqual(
          mask[0].get_shape().as_list(),
          [kernel_size[0], kernel_size[1], input_depth, output_depth])

  @parameterized.parameters(PRUNING_METHODS)
  def testMultipleConvMaskAdded(self, pruning_method):

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
      number_of_layers = 5

      kernel_size = [3, 3]
      base_depth = 4
      depth_step = 7

      input_tensor = tf.ones((8, self.height, self.width, base_depth))

      top_layer = input_tensor

      for ix in range(number_of_layers):
        units = base_depth + (ix + 1) * depth_step
        top_layer = pruning_layers.sparse_conv2d(
            x=top_layer,
            units=units,
            kernel_size=kernel_size,
            is_training=False,
            sparsity_technique=pruning_method)

      if pruning_method == 'variational_dropout':
        theta_logsigma2 = tf.get_collection(
            vd.layers.THETA_LOGSIGMA2_COLLECTION)
        self.assertLen(theta_logsigma2, number_of_layers)

        utils.add_vd_pruning_summaries(theta_logsigma2, threshold=3.0)

        dkl_loss_1 = utils.variational_dropout_dkl_loss(
            reg_scalar=1,
            start_reg_ramp_up=0,
            end_reg_ramp_up=1000,
            warm_up=False,
            use_tpu=False)
        dkl_loss_1 = tf.reshape(dkl_loss_1, [1])

        dkl_loss_2 = utils.variational_dropout_dkl_loss(
            reg_scalar=5,
            start_reg_ramp_up=0,
            end_reg_ramp_up=1000,
            warm_up=False,
            use_tpu=False)
        dkl_loss_2 = tf.reshape(dkl_loss_2, [1])

        for ix in range(number_of_layers):
          self.assertListEqual(theta_logsigma2[ix][0].get_shape().as_list(), [
              kernel_size[0], kernel_size[1], base_depth + ix * depth_step,
              base_depth + (ix + 1) * depth_step
          ])

        init_op = tf.global_variables_initializer()

        with self.test_session() as sess:
          sess.run(init_op)
          if pruning_method == 'variational_dropout':
            loss_1, loss_2 = sess.run([dkl_loss_1, dkl_loss_2])

            self.assertGreater(loss_2, loss_1)
      elif pruning_method == 'l0_regularization':
        theta_logalpha = tf.get_collection(
            l0.layers.THETA_LOGALPHA_COLLECTION)
        self.assertLen(theta_logalpha, number_of_layers)

        utils.add_l0_summaries(theta_logalpha)

        l0_norm_loss_1 = utils.l0_regularization_loss(
            reg_scalar=1,
            start_reg_ramp_up=0,
            end_reg_ramp_up=1000,
            warm_up=False,
            use_tpu=False)
        l0_norm_loss_1 = tf.reshape(l0_norm_loss_1, [1])

        l0_norm_loss_2 = utils.l0_regularization_loss(
            reg_scalar=5,
            start_reg_ramp_up=0,
            end_reg_ramp_up=1000,
            warm_up=False,
            use_tpu=False)
        l0_norm_loss_2 = tf.reshape(l0_norm_loss_2, [1])

        for ix in range(number_of_layers):
          self.assertListEqual(theta_logalpha[ix][0].get_shape().as_list(), [
              kernel_size[0], kernel_size[1], base_depth + ix * depth_step,
              base_depth + (ix + 1) * depth_step
          ])

        init_op = tf.global_variables_initializer()

        with self.test_session() as sess:
          sess.run(init_op)
          loss_1, loss_2 = sess.run([l0_norm_loss_1, l0_norm_loss_2])
          self.assertGreater(loss_2, loss_1)
      else:
        mask = tf.get_collection(core.MASK_COLLECTION)
        for ix in range(number_of_layers):
          self.assertListEqual(mask[ix].get_shape().as_list(), [
              kernel_size[0], kernel_size[1], base_depth + ix * depth_step,
              base_depth + (ix + 1) * depth_step
          ])


if __name__ == '__main__':
  tf.test.main()
