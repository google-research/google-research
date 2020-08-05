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

"""Tests for non_semantic_speech_benchmark.eval_embedding.keras.google.autopool."""

from absl.testing import absltest
from absl.testing import parameterized

import tensorflow.compat.v2 as tf

from non_semantic_speech_benchmark.eval_embedding.keras import autopool


def _softmax_pool(x, axis, keepdims):
  softmax = tf.keras.backend.softmax(x, axis=axis)
  return tf.keras.backend.sum(x * softmax, axis=axis, keepdims=keepdims)


class AutopoolTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'alpha_init': 0.0, 'agg_func': tf.keras.backend.mean},
      {'alpha_init': 1.0, 'agg_func': _softmax_pool},
      {'alpha_init': 100.0, 'agg_func': tf.keras.backend.max},
      {'alpha_init': -130.0, 'agg_func': tf.keras.backend.min},
  )
  def test_limiting_behavior(self, alpha_init, agg_func):
    """Test semantics of alpha.

    When `alpha` = 0, it reduces to an unweighted mean; when `alpha` = 1,
    it simplifies to soft-max pooling; and when `alpha` -> inf, it approaches
    the max operator, and `alpha` -> -inf, it approachs the min operator.

    Args:
      alpha_init: The alpha parameters.
      agg_func: The expected aggregation function.
    """
    a = tf.random.uniform(shape=(2, 3, 1), seed=12)
    actual = autopool.AutoPool(axis=1, alpha_init=alpha_init)(a, keepdims=True)
    expected = agg_func(a, axis=1, keepdims=True)
    self.assertAllClose(actual, expected)

  def test_average_alpha(self):
    a = tf.random.uniform(shape=(3, 4, 5), seed=123)
    layer = autopool.AutoPool(axis=1)
    layer(a)  # Triggers layer build.
    self.assertIsNotNone(layer.average_alpha)
    self.assertEqual(layer.average_alpha.shape, [])

  def test_keras_model_has_no_trainable_vars(self):
    """Test that pooling variable is properly trainable."""
    m = tf.keras.models.Sequential(
        [tf.keras.Input(shape=(4, 5)),
         autopool.AutoPool(axis=1, trainable=False)]
    )
    m.build()
    self.assertEmpty(m.trainable_variables)

  def test_keras_model_has_trainable_vars(self):
    """Test that pooling variable is properly trainable."""
    m = tf.keras.models.Sequential(
        [tf.keras.Input(shape=(4, 5)),
         autopool.AutoPool(axis=1)]
    )
    m.build()
    self.assertIsNotNone(m.trainable_variables)
    assert len(m.trainable_variables) == 1

    # Run a fake training step and check that values after the step are
    # different. Access through `average_alpha` to check that it works.
    original_var_val = m.get_layer(index=0).average_alpha.numpy()
    with tf.GradientTape() as tape:
      o = m(tf.random.uniform(shape=(3, 4, 5)), training=True)
      o.shape.assert_is_compatible_with((3, 5))
      loss_value = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
          y_true=tf.ones_like(o), y_pred=o)
    grads = tape.gradient(loss_value, m.trainable_variables)
    tf.keras.optimizers.Adam(learning_rate=1.0).apply_gradients(
        zip(grads, m.trainable_variables))
    post_training_var_val = m.get_layer(index=0).average_alpha.numpy()

    self.assertNotAllClose(original_var_val, post_training_var_val)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  assert tf.executing_eagerly()
  absltest.main()
