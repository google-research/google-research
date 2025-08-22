# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Test for training_utils.py."""

import functools

import unittest

import jax
import jax.numpy as jnp
from study_recommend.utils import training_utils as utils


class TrainUtilsTest(unittest.TestCase):

  def test_create_learning_rate_schedule(self):
    """Assert learning rate can be created and returns valid learning rates."""
    schedule = utils.create_learning_rate_schedule(
        learning_rate=1.0, warmup_steps=50
    )
    epsilon = 1e-7
    for step in range(100):
      learning_rate = schedule(step)
      self.assertGreater(learning_rate, 0.0 - epsilon)

  def test_compute_weight_matrix(self):
    """Assert that the weight matrix masks out padding and separators."""
    titles = jnp.array([[1, 2, 10, 3, 0, 0], [1, 10, 2, 10, 0, 0]])

    weights = utils.compute_weight_matrix(titles, separator_token=10)
    self.assertTrue(
        (
            weights
            == jnp.array(
                [[1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0]], dtype=jnp.bool_
            )
        ).all()
    )

  def test_compute_weighted_cross_entropy(self):
    targets = jnp.array([[2, 1, 2], [2, 2, 0], [1, 0, 0]], dtype=jnp.int32)
    weights = jnp.array([[[1, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=jnp.bool_)

    logits = jnp.array(
        [
            [[3.0, -0.5, 4.0], [1.0, 12.0, 4.0], [3.0, 3.0, 5.0]],
            [[3.0, -0.5, 4.0], [1.0, 2.0, 4.0], [3.0, 3.0, 5.0]],
            [[3.0, -0.5, 4.0], [1.0, 2.0, 4.0], [3.0, 3.0, 5.0]],
        ],
        dtype=jnp.float32,
    )

    loss, normalizing_factor = utils.compute_weighted_cross_entropy(
        logits, targets, weights
    )

    self.assertAlmostEqual(loss.item(), 5.7039475440979)
    self.assertAlmostEqual(normalizing_factor.item(), 5)

  def test_compute_weighted_accuracy(self):
    targets = jnp.array([[2, 1, 2], [2, 2, 0], [1, 0, 0]], dtype=jnp.int32)
    weights = jnp.array([[[1, 1, 1], [1, 0, 0], [1, 0, 0]]], dtype=jnp.bool_)

    logits = jnp.array(
        [
            [[3.0, -0.5, 4.0], [1.0, 12.0, 4.0], [3.0, 3.0, 5.0]],
            [[3.0, -0.5, 4.0], [1.0, 2.0, 4.0], [3.0, 3.0, 5.0]],
            [[3.0, -0.5, 4.0], [1.0, 2.0, 4.0], [3.0, 3.0, 5.0]],
        ],
        dtype=jnp.float32,
    )

    n_oov_corrects, n_corrects, normalizing_factor = (
        utils.compute_weighted_accuracy(
            logits, targets, oov_value=1, weights=weights
        )
    )

    self.assertAlmostEqual(n_corrects.item(), 4)
    self.assertAlmostEqual(n_oov_corrects.item(), 3)
    self.assertAlmostEqual(normalizing_factor.item(), 5)

  def test_compute_metrics(self):
    """Test the correction of the utility function to return all metrics."""
    # Test data arrays in this test have a leading axis with size 1.
    # This is the axis that represents n_devices for jax.pmap
    targets = jnp.array([[[2, 1, 2], [2, 2, 0], [1, 0, 0]]], dtype=jnp.int32)
    weights = jnp.array([[[[1, 1, 1], [1, 0, 0], [1, 0, 0]]]], dtype=jnp.bool_)

    logits = jnp.array(
        [[
            [[3.0, -0.5, 4.0], [1.0, 12.0, 4.0], [3.0, 3.0, 5.0]],
            [[3.0, -0.5, 4.0], [1.0, 2.0, 4.0], [3.0, 3.0, 5.0]],
            [[3.0, -0.5, 4.0], [1.0, 2.0, 4.0], [3.0, 3.0, 5.0]],
        ]],
        dtype=jnp.float32,
    )

    @functools.partial(jax.pmap, axis_name='batch')
    def f(logits, targets, weights):
      return utils.compute_metrics(logits, targets, weights, oov_value=1)

    metrics = f(logits, targets, weights)

    metrics = {key: value.item() for key, value in metrics.items()}
    reference_values = {
        'loss': 5.7039475440979,
        'accuracy': 4,
        'oov_corrected_accuracy': 3,
        'denominator': 5,
    }

    for key in reference_values:
      self.assertAlmostEqual(metrics[key], reference_values[key])

if __name__ == '__main__':
  unittest.main()
