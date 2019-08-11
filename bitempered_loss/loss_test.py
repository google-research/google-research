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

"""Tests for bi-tempered loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
import tensorflow as tf

from bitempered_loss import loss

python_version = "PY3"


class LossTest(tf.test.TestCase):

  def test_normalization(self):
    """Test the normalization constant."""
    activations = tf.random.normal(shape=[100, 50000])
    with self.cached_session():
      normalization_constants = loss.compute_normalization(
          activations, 1.01, num_iters=5)
      self.assertEqual(normalization_constants.shape, [100, 1])
      probabilities = tf.reduce_sum(
          loss.exp_t(activations - normalization_constants, 1.01), -1)
      self.assertAllClose(probabilities.eval(), [1.0] * 100, atol=1e-5)
      normalization_constants = loss.compute_normalization(
          activations, 2.0, num_iters=5)
      probabilities = tf.reduce_sum(
          loss.exp_t(activations - normalization_constants, 2.0), -1)
      self.assertAllClose(probabilities.eval(), [1.0] * 100, atol=1e-5)

  def test_limit_case_logistic_loss(self):
    """Test for checking if t1 = t2 = 1.0 yields the logistic loss."""
    labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    activations = tf.random.normal(shape=[3, 3])
    with self.cached_session() as sess:
      actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 1.0,
                                                   1.0)
      logistic_loss = tf.nn.softmax_cross_entropy_with_logits(
          logits=activations, labels=labels)
      actual_loss_out, logistic_loss_out = sess.run(
          [actual_loss, logistic_loss])
      self.assertAllClose(actual_loss_out, logistic_loss_out)

  def test_loss_value(self):
    """Test the loss based on precomputed values."""
    labels = tf.constant([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.2, 0.8, 0.0]])
    activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]]
    with self.cached_session():
      actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 0.5,
                                                   1.5)
      self.assertAllClose(actual_loss.eval(),
                          [0.02301914, 0.18972909, 0.93874922])

  def test_constant_shift(self):
    """Test if adding a constant to all activations is vacuous."""
    labels = tf.constant([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.7, 0.2, 0.1]])
    activations = tf.random.normal(shape=[3, 3])
    bias = tf.random.normal(shape=[3, 1])
    with self.cached_session() as sess:
      actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 0.5,
                                                   1.2)
      shifted_loss = loss.bi_tempered_logistic_loss(activations + bias, labels,
                                                    0.5, 1.2)
      self.assertEqual(actual_loss.shape, [3])
      actual_loss_out, shifted_loss_out = sess.run([actual_loss, shifted_loss])
      self.assertAllClose(actual_loss_out, shifted_loss_out)

  def test_gradient_error(self):
    """Compare custom gradient with tf.gradient."""
    labels = tf.constant([[0.4, 0.3, 0.3], [0.8, 0.1, 0.1], [0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0]])
    activations = tf.random.normal(shape=[4, 3])
    internal_loss = loss._internal_bi_tempered_logistic_loss(
        activations, labels, 0.5, 1.5)
    numerical_gradient = tf.gradients(internal_loss, activations)
    actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 0.5, 1.5)
    actual_gradient = tf.gradients(actual_loss, activations)
    with self.cached_session() as sess:
      internal_loss_out, actual_loss_out = sess.run(
          [internal_loss, actual_loss])
      numerical_gradient_out, actual_gradient_out = sess.run(
          [numerical_gradient[0], actual_gradient[0]])
      self.assertEqual(actual_gradient_out.shape, (4, 3))
      self.assertAllClose(actual_loss_out, internal_loss_out)
      self.assertAllClose(
          actual_gradient_out, numerical_gradient_out, atol=1e-5)

  def test_label_smoothing(self):
    """Test label smoothing."""
    labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]]
    actual_loss = loss.bi_tempered_logistic_loss(
        activations, labels, 0.5, 1.5, label_smoothing=0.1)
    with self.cached_session() as sess:
      actual_loss_out = sess.run(actual_loss)
      self.assertAllClose(
          actual_loss_out, [0.76652711, 0.08627685, 1.35443510], atol=1e-5)

  def test_binary_logistic_loss(self):
    """Test binary logistic loss."""
    labels = tf.constant([1.0, 0.0])
    activations = [0.0, 0.0]
    actual_loss = loss.bi_tempered_binary_logistic_loss(activations, labels,
                                                        1.0, 1.0)
    with self.cached_session() as sess:
      actual_loss_out = sess.run(actual_loss)
      self.assertAllClose(actual_loss_out, [0.69314718, 0.69314718], atol=1e-5)


if __name__ == "__main__":
  absltest.main()
