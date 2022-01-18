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

"""Tests for recomputing_dropout."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from etcmodel.layers import recompute_grad
from etcmodel.layers import recomputing_dropout


class RecomputingDropoutTest(tf.test.TestCase, parameterized.TestCase):

  def test_recompute_grad(self):
    """Tests that the gradient is computed correctly with recompute_grad."""
    dense = tf.keras.layers.Dense(10, input_shape=(8,))
    dropout = recomputing_dropout.RecomputingDropout(
        0.4, force_recomputation=True)

    @recompute_grad.recompute_grad
    def recompute_dense_dropout(x):
      return dropout(dense(x), training=True)

    # Define the model using dropout.
    def f(x):
      with tf.GradientTape() as tape:
        h1 = recompute_dense_dropout(x)
        h2 = recompute_dense_dropout(x)
        y = tf.math.reduce_sum(h1 + h2)
      return (tf.cast(tf.math.not_equal(h1, 0), tf.float32),
              tf.cast(tf.math.not_equal(h2, 0),
                      tf.float32), tape.gradient(y, dense.trainable_variables))

    x = tf.convert_to_tensor(np.random.normal(size=(4, 8)), tf.float32)
    mask1, mask2, gradients = f(x)
    self.evaluate(tf.compat.v1.initializers.global_variables())

    mask1, mask2, gradients = self.evaluate([mask1, mask2, gradients])
    # Make sure entries were masked and there is randomness.
    self.assertGreaterEqual(np.sum(mask1 == 0), 2)
    self.assertGreaterEqual(np.sum(mask2 == 0), 2)
    self.assertNotAllEqual(mask1, mask2)

    # Use the masks to compute exact gradients.
    def g(x):
      with tf.GradientTape() as tape:
        # Rescale proportional to dropout rate.
        h1 = (dense(x) * mask1) / 0.6
        h2 = (dense(x) * mask2) / 0.6
        y = tf.math.reduce_sum(h1 + h2)
      return tape.gradient(y, dense.trainable_variables)

    expected_gradients = self.evaluate(g(x))
    self.assertAllClose(gradients, expected_gradients)

  def test_nested_recompute_grad(self):
    """Tests nested usage of recompute_grad."""
    dense = tf.keras.layers.Dense(
        5, input_shape=(8,), bias_initializer='glorot_normal')
    dropout = recomputing_dropout.RecomputingDropout(
        0.4, force_recomputation=True)

    @recompute_grad.recompute_grad
    def recompute_dense_dropout_tower(x):
      return dropout(dense(x), training=True)

    def make_head():
      inputs = tf.keras.Input(shape=(5,))
      x = tf.keras.layers.Dense(
          3, activation='tanh', name='dense', bias_initializer='glorot_normal')(
              inputs)
      x = recomputing_dropout.RecomputingDropout(0.45)(x)
      outputs = {
          'head_mask': tf.cast(tf.math.not_equal(x, 0), tf.float32),
          'y': tf.reduce_sum(x),
      }
      return tf.keras.Model(inputs, outputs, name='head')

    head = make_head()

    # Nest recompute_grad inside another recompute_grad function.
    @recompute_grad.recompute_grad
    def recompute_model(x):
      y1 = recompute_dense_dropout_tower(x)
      y2 = recompute_dense_dropout_tower(x)
      outputs = head(y1 + y2, training=True)
      outputs.update({
          'tower1_mask': tf.cast(tf.math.not_equal(y1, 0), tf.float32),
          'tower2_mask': tf.cast(tf.math.not_equal(y2, 0), tf.float32),
      })
      return outputs

    def f(x):
      with tf.GradientTape() as tape:
        outputs = recompute_model(x)
      outputs['gradients'] = tape.gradient(
          outputs.pop('y'),
          dense.trainable_variables + head.trainable_variables)
      return outputs

    x = tf.convert_to_tensor(np.random.normal(size=(4, 8)), tf.float32)
    outputs = f(x)
    self.evaluate(tf.compat.v1.initializers.global_variables())
    outputs = self.evaluate(outputs)

    # Verify gradients are correct.
    def g(x):
      with tf.GradientTape() as tape:
        y1 = dense(x) * outputs['tower1_mask'] / 0.6
        y2 = dense(x) * outputs['tower2_mask'] / 0.6
        y = tf.reduce_sum(
            head.get_layer('dense')(y1 + y2) * outputs['head_mask'] / 0.55)
      return tape.gradient(y,
                           dense.trainable_variables + head.trainable_variables)
    # Increase tolerance from default of 1e-6 to reduce flakiness.
    self.assertAllClose(
        outputs['gradients'], self.evaluate(g(x)), rtol=2e-5, atol=2e-5)

  def test_force_recomputation(self):
    """Tests that an error is thrown when there is no recompute context."""
    dropout = recomputing_dropout.RecomputingDropout(
        0.4, force_recomputation=True)
    with self.assertRaises(ValueError) as assert_raises_context:
      dropout(np.random.normal(size=(2, 8)), training=True)
    self.assertContainsExactSubsequence(
        str(assert_raises_context.exception), 'RecomputeContext is required')


if __name__ == '__main__':
  np.set_printoptions(suppress=True)
  tf.test.main()
