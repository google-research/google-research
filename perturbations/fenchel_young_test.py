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

# Lint as: python3
"""Tests for the fenchel_young module."""

import tensorflow.compat.v2 as tf

from perturbations import fenchel_young as fy


def ranks(inputs, axis=-1):
  """Returns the ranks of the input values among the given axis."""
  return 1 + tf.cast(
      tf.argsort(tf.argsort(inputs, axis=axis), axis=axis), dtype=inputs.dtype)


class FenchelYoungTest(tf.test.TestCase):
  """Testing the gradients obtained by the FenchelYoungLoss class."""

  def test_gradients(self):
    loss_fn = fy.FenchelYoungLoss(
        ranks, num_samples=10000, sigma=0.1, batched=False)

    theta = tf.constant([1, 20, 7.3, 7.35])
    y_true = tf.constant([1, 4, 3, 2], dtype=theta.dtype)
    y_hard_minimum = tf.constant([1, 4, 2, 3], dtype=theta.dtype)
    y_perturbed_minimum = tf.constant(loss_fn.perturbed(theta))

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(theta)
      g_true = tape.gradient(loss_fn(y_true, theta), theta)
      g_hard_minimum = tape.gradient(loss_fn(y_hard_minimum, theta), theta)
      g_perturbed_minimum = tape.gradient(
          loss_fn(y_perturbed_minimum, theta), theta)

    # The gradient should be close to zero for the two first values.
    self.assertAllClose(g_true[:2], [0.0, 0.0])
    self.assertLess(tf.norm(g_perturbed_minimum), tf.norm(g_hard_minimum))
    self.assertLess(tf.norm(g_hard_minimum), tf.norm(g_true))
    for g in [g_true, g_hard_minimum, g_perturbed_minimum]:
      self.assertAllClose(tf.math.reduce_sum(g), 0.0)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
