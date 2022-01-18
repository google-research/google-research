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

"""Tests for bi-tempered loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing import absltest
import tensorflow as tf

from bitempered_loss import loss

python_version = "PY3"


class LossTest(tf.test.TestCase):

  def test_normalization(self):
    """Test the normalization constant."""
    activations = tf.random.normal(shape=[100, 50000])
    for t in [0.99, 1.01]:
      normalization_constants = loss.compute_normalization(
          activations, t, num_iters=20)
      self.assertEqual(normalization_constants.shape, [100, 1])
      probabilities = tf.reduce_sum(
          loss.exp_t(activations - normalization_constants, t), -1)
      self.assertAllClose(probabilities.numpy(), [1.0] * 100, atol=1e-5)
    for t in [0.1, 2.0]:
      normalization_constants = loss.compute_normalization(
          activations, t, num_iters=20)
      probabilities = tf.reduce_sum(
          loss.exp_t(activations - normalization_constants, t), -1)
      self.assertAllClose(probabilities.numpy(), [1.0] * 100, atol=1e-5)

  def test_limit_case_logistic_loss(self):
    """Test for checking if t1 = t2 = 1.0 yields the logistic loss."""
    labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    activations = tf.random.normal(shape=[3, 3])
    actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 1.0,
                                                 1.0)
    logistic_loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=activations, labels=labels)
    actual_loss_out, logistic_loss_out = (
        actual_loss.numpy(), logistic_loss.numpy())
    self.assertAllClose(actual_loss_out, logistic_loss_out)

  def test_loss_value(self):
    """Test the loss based on precomputed values."""
    labels = tf.constant([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.2, 0.8, 0.0]])
    activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]]
    actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 0.5,
                                                 1.5)
    self.assertAllClose(actual_loss.numpy(),
                        [0.02301914, 0.18972909, 0.93874922])
    actual_loss = loss.bi_tempered_logistic_loss(activations, labels, 0.5,
                                                 0.8, num_iters=20)
    self.assertAllClose(actual_loss.numpy(),
                        [0.21646356, 0.41836615, 1.33997854])

  def test_constant_shift(self):
    """Test if adding a constant to all activations is vacuous."""
    labels = tf.constant([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.7, 0.2, 0.1]])
    activations = tf.random.normal(shape=[3, 3])
    bias = tf.random.normal(shape=[3, 1])
    for t2 in [0.8, 1.2]:
      actual_loss = loss.bi_tempered_logistic_loss(
          activations, labels, 0.5, t2)
      shifted_loss = loss.bi_tempered_logistic_loss(
          activations + bias, labels, 0.5, t2)
      actual_loss_out, shifted_loss_out = (
          actual_loss.numpy(), shifted_loss.numpy())
      self.assertAllClose(actual_loss_out, shifted_loss_out)

  def test_gradient_error(self):
    """Compare custom gradient with tf.GradientTape."""
    labels = tf.constant([[0.4, 0.3, 0.3], [0.8, 0.1, 0.1], [0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0]])
    activations = tf.random.normal(shape=[4, 3])
    for t1, t2 in [[0.5, 1.0], [1.0, 1.5], [0.5, 1.5]]:
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(activations)
        internal_loss = loss._internal_bi_tempered_logistic_loss(
            activations, labels, t1, t2)
        actual_loss = loss.bi_tempered_logistic_loss(
            activations, labels, t1, t2)
      numerical_gradient = tape.gradient(internal_loss, activations)
      actual_gradient = tape.gradient(actual_loss, activations)
      internal_loss_out, actual_loss_out = (
          internal_loss.numpy(), actual_loss.numpy())
      numerical_gradient_out, actual_gradient_out = (
          numerical_gradient.numpy(), actual_gradient.numpy())
      self.assertEqual(actual_gradient_out.shape, (4, 3))
      self.assertAllClose(actual_loss_out, internal_loss_out, atol=1e-5)
      self.assertAllClose(
          actual_gradient_out, numerical_gradient_out, atol=1e-4)

  def test_label_smoothing(self):
    """Test label smoothing."""
    labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]]
    actual_loss = loss.bi_tempered_logistic_loss(
        activations, labels, 0.5, 1.5, label_smoothing=0.1)
    actual_loss_out = actual_loss.numpy()
    self.assertAllClose(
        actual_loss_out, [0.76652711, 0.08627685, 1.35443510], atol=1e-5)

  def test_binary_logistic_loss(self):
    """Test binary logistic loss."""
    labels = tf.constant([1.0, 0.0])
    activations = [0.0, 0.0]
    actual_loss = loss.bi_tempered_binary_logistic_loss(activations, labels,
                                                        1.0, 1.0)
    actual_loss_out = actual_loss.numpy()
    self.assertAllClose(actual_loss_out, [0.69314718, 0.69314718], atol=1e-5)

  def test_dynamic_temperatures(self):
    """Test changing temperatures dynamically."""
    labels = tf.constant([[0.2, 0.5, 0.3]])
    activations = [[-0.5, 0.1, 2.0]]
    actual_loss = functools.partial(
        loss.bi_tempered_logistic_loss,
        activations=activations,
        labels=labels,
        num_iters=5)
    t1_values = [1.0, 0.9, 0.8, 0.7]
    t2_values = [1.0, 1.1, 1.2, 1.3]
    loss_values = [[1.6583576], [0.45677936], [0.34298314], [0.26295574]]
    loss_out = []
    for t1_value, t2_value in zip(t1_values, t2_values):
      loss_out.append(actual_loss(t1=t1_value, t2=t2_value).numpy())
    self.assertAllClose(loss_values, loss_out, atol=1e-5)

  def test_sparse_loss(self):
    """Test int labels."""
    labels = tf.constant([0, 2, 1, 0])
    activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0],
                   [-1.5, 0.7, 5.2]]
    actual_loss = loss.bi_tempered_logistic_loss(activations,
                                                 tf.one_hot(labels, 3), 0.5,
                                                 1.5)
    sparse_loss = loss.sparse_bi_tempered_logistic_loss(activations, labels,
                                                        0.5, 1.5)
    actual_loss_out = actual_loss.numpy()
    sparse_loss_out = sparse_loss.numpy()
    self.assertAllClose(actual_loss_out, sparse_loss_out)
    labels = tf.constant([[0, 2], [1, 0]])
    activations = [[[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0]],
                   [[4.0, -3.0, -6.0], [-1.5, 0.7, 5.2]]]
    actual_loss = loss.bi_tempered_logistic_loss(activations,
                                                 tf.one_hot(labels, 3), 0.5,
                                                 1.5)
    sparse_loss = loss.sparse_bi_tempered_logistic_loss(activations, labels,
                                                        0.5, 1.5)
    actual_loss_out = actual_loss.numpy()
    sparse_loss_out = sparse_loss.numpy()
    self.assertAllClose(actual_loss_out, sparse_loss_out)

  def test_tempered_softmax(self):
    # Test softmax function with different temperatures.
    activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]]
    # Test with temperature = 1.0, which should recover regular
    # softmax probabilities.
    softmax_probabilities_t_1 = loss.tempered_softmax(
        activations, t=1.0).numpy()
    vanilla_softmax_probabilties = tf.nn.softmax(activations).numpy()
    self.assertAllClose(vanilla_softmax_probabilties,
                        softmax_probabilities_t_1)
    softmax_probabilities_t_4 = loss.tempered_softmax(
        activations, t=4.0).numpy()
    expected_softmax_probabilities_t_4 = ([[
        0.3205458, 0.32714278, 0.3523402
    ], [0.3430056, 0.36491093,
        0.29220778], [0.41369352, 0.30534995, 0.28299212]])
    self.assertAllClose(expected_softmax_probabilities_t_4,
                        softmax_probabilities_t_4)

  def test_tempered_sigmoid(self):
    # Test sigmoid function with different temperatures.
    activations = [0.0, 3.0, 6.0]
    # Test with temperature = 1.0, which should recover regular
    # sigmoid probabilities.
    sigmoid_probabilities_t_1 = loss.tempered_sigmoid(
        activations, t=1.0).numpy()
    vanilla_softmax_probabilties = tf.nn.sigmoid(activations).numpy()
    self.assertAllClose(vanilla_softmax_probabilties,
                        sigmoid_probabilities_t_1)
    sigmoid_probabilities_t_4 = loss.tempered_sigmoid(
        activations, t=4.0).numpy()
    expected_sigmoid_probabilities_t_4 = [0.5, 0.58516014, 0.6421035]
    self.assertAllClose(expected_sigmoid_probabilities_t_4,
                        sigmoid_probabilities_t_4)


if __name__ == "__main__":
  absltest.main()
