# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Tests loss utility functions."""

import tensorflow as tf

from poem.cv_mim import losses


class LossesTest(tf.test.TestCase):

  def test_compute_positive_expectation_shape(self):
    input_features = tf.ones([2, 3, 4], tf.float32)
    outputs = losses.compute_positive_expectation(
        input_features, losses.TYPE_MEASURE_W1, reduce_mean=False)
    self.assertAllEqual(outputs.shape.as_list(), [2, 3, 4])

    outputs = losses.compute_positive_expectation(
        input_features, losses.TYPE_MEASURE_W1, reduce_mean=True)
    self.assertAllEqual(outputs, 1.)

  def test_compute_negative_expectation_shape(self):
    input_features = tf.ones([2, 3, 4], tf.float32)
    outputs = losses.compute_negative_expectation(
        input_features, losses.TYPE_MEASURE_W1, reduce_mean=False)
    self.assertAllEqual(outputs.shape.as_list(), [2, 3, 4])

    outputs = losses.compute_negative_expectation(
        input_features, losses.TYPE_MEASURE_W1, reduce_mean=True)
    self.assertAllEqual(outputs, 1.)

  def test_compute_fenchel_dual_loss(self):
    input_features = tf.ones([4, 3, 5], tf.float32)
    loss = losses.compute_fenchel_dual_loss(
        input_features, input_features, losses.TYPE_MEASURE_W1)
    self.assertAllEqual(loss, 0.)

    input_features = tf.constant([[[1., 1.], [0., 1.]], [[1., 0.], [1., 1.]],
                                  [[1., 0.], [1., 1.]], [[1., 1.], [1., 1.]]])
    loss = losses.compute_fenchel_dual_loss(input_features, input_features,
                                            losses.TYPE_MEASURE_W1)
    self.assertAllClose(loss, 15.5 / 12 - 5.75 / 4)

    loss = losses.compute_fenchel_dual_loss(
        input_features, input_features, losses.TYPE_MEASURE_W1,
        tf.eye(4, dtype=tf.dtypes.float32))
    self.assertAllClose(loss, 15.5 / 12 - 5.75 / 4)

    loss = losses.compute_fenchel_dual_loss(
        input_features, input_features, losses.TYPE_MEASURE_W1,
        tf.ones((4, 4), dtype=tf.dtypes.float32))
    self.assertAllClose(loss, -(15.5 + 5.75) / 16)

  def test_compute_info_nce_loss_shape(self):
    input_features = tf.ones([4, 3, 5], tf.float32)
    loss = losses.compute_info_nce_loss(input_features, input_features)
    self.assertAllEqual(loss.shape.as_list(), ())

    input_features = tf.constant([[[0., 1.]], [[1., 0.]]])
    loss = losses.compute_info_nce_loss(input_features, input_features)
    self.assertAllClose(loss, -tf.nn.log_softmax([1.0, 0.0, -1e12])[0])

  def test_compute_log_likelihood_shape(self):
    x_mean = tf.ones([4, 8], tf.float32)
    x_logvar = tf.ones([4, 8], tf.float32)
    y = tf.ones([4, 8], tf.float32)

    outputs = losses.compute_log_likelihood(x_mean, x_logvar, y)
    self.assertAllEqual(outputs.shape.as_list(), ())

  def test_compute_contrastive_log_ratio_shape(self):
    x_mean = tf.ones([4, 8], tf.float32)
    x_logvar = tf.ones([4, 8], tf.float32)
    y = tf.ones([4, 8], tf.float32)

    outputs = losses.compute_contrastive_log_ratio(x_mean, x_logvar, y)
    self.assertAllEqual(outputs.shape.as_list(), ())

  def test_compute_gradient_penalty(self):
    network = tf.keras.layers.Dense(
        1, kernel_initializer='ones', bias_initializer='ones')
    inputs = tf.ones([4, 8], tf.float32)
    penalty, outputs = losses.compute_gradient_penalty(network, inputs, 1.0)
    self.assertAllClose(penalty, 8.0)
    self.assertAllClose(outputs, tf.ones([4, 1], tf.float32) * 9.0)

  def test_compute_discriminator_loss(self):
    network = tf.keras.layers.Dense(
        1, kernel_initializer='ones', bias_initializer='ones')
    real_inputs = tf.ones([4, 8], tf.float32)
    fake_inputs = tf.zeros([4, 8], tf.float32)
    loss, real_outputs, fake_outputs = losses.compute_discriminator_loss(
        network, real_inputs, fake_inputs)
    ones = tf.ones([4, 1], tf.float32)
    self.assertAllClose(
        loss, -(tf.reduce_mean(-tf.math.softplus(-ones * 9.0)) -
                tf.reduce_mean(tf.math.softplus(-ones) + ones)) * 0.5 + 8.0)
    self.assertAllClose(real_outputs, ones * 9.0)
    self.assertAllClose(fake_outputs, ones)

  def test_compute_generator_loss(self):
    fake_inputs = tf.ones([4, 8], tf.float32)
    loss = losses.compute_generator_loss(fake_inputs,
                                         losses.TYPE_GENERATOR_LOSS_MM)
    self.assertAllClose(loss, tf.math.softplus(-1.0) + 1.0)

    loss = losses.compute_generator_loss(fake_inputs,
                                         losses.TYPE_GENERATOR_LOSS_NS)
    self.assertAllClose(loss, tf.math.softplus(-1.0))


if __name__ == '__main__':
  tf.test.main()
