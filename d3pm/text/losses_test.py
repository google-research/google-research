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

"""Tests for losses.py."""

from absl.testing import absltest
import jax.numpy as jnp

from d3pm.text import losses


class NormalMetricsTest(absltest.TestCase):
  """Test normal unmasked/unweighted metrics."""

  def setUp(self):
    super(NormalMetricsTest, self).setUp()

    self.targets = jnp.array([1, 0, 0])
    self.probs = jnp.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1]])
    self.logits = jnp.log(self.probs + 1e-9)

  def test_cross_entropy(self):
    """Test that cross entropy works properly with no masking."""
    loss = losses.cross_entropy_with_logits(
        logits=self.logits, targets=self.targets)
    kl = loss.mean()

    expected_loss = -(jnp.log(0.9) + jnp.log(0.1) + jnp.log(0.9)) / 3

    self.assertAlmostEqual(kl, expected_loss)

  def test_accuracy(self):
    """Test that accuracy works correctly."""
    total_accuracy, weights = losses.weighted_accuracy(
        logits=self.logits, targets=self.targets)

    expected_accuracy = 2 / 3

    self.assertEqual(weights, 3)
    self.assertAlmostEqual(total_accuracy / weights, expected_accuracy)


class WeightedMetricsTest(absltest.TestCase):
  """Test weighted metrics with masking."""

  def setUp(self):
    super(WeightedMetricsTest, self).setUp()

    self.targets = jnp.array([1, 0, 0, -1])
    self.probs = jnp.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [10, 0]])
    self.logits = jnp.log(self.probs + 1e-9)

  def test_weighted_cross_entropy(self):
    """Test that cross entropy works with masking an invalid entry."""
    kl_loss = losses.cross_entropy_with_logits(
        logits=self.logits, targets=self.targets)
    loss, weights = losses.weighted_mean(kl_loss, self.targets >= 0)

    expected_loss = -(jnp.log(0.9) + jnp.log(0.1) + jnp.log(0.9)) / 3

    self.assertEqual(weights, 3)
    self.assertAlmostEqual(loss / weights, expected_loss)

  def test_weighted_accuracy(self):
    """Test that accuracy works correctly."""
    total_accuracy, weights = losses.weighted_accuracy(
        logits=self.logits, targets=self.targets, weights=self.targets >= 0)

    expected_accuracy = 2 / 3

    self.assertEqual(weights, 3)
    self.assertAlmostEqual(total_accuracy / weights, expected_accuracy)


class SequenceMetricsTest(absltest.TestCase):
  """Test weighted metrics for sequences with 2D masking."""

  def setUp(self):
    super(SequenceMetricsTest, self).setUp()

    self.targets = jnp.array([[1, -1], [0, 0]])
    self.probs = jnp.array([[
        [0.1, 0.9],
        [0.5, 0.5],
    ], [
        [0.9, 0.1],
        [0.2, 0.8],
    ]])

    self.logits = jnp.log(self.probs + 1e-9)

  def test_weighted_sequence_cross_entropy(self):
    """Test that cross entropy works with masking an invalid entry."""
    kl_loss = losses.cross_entropy_with_logits(
        logits=self.logits, targets=self.targets)
    loss, weights = losses.weighted_mean(kl_loss, weights=self.targets >= 0)

    expected_loss = -(jnp.log(0.9) + jnp.log(0.9) + jnp.log(0.2)) / 3

    self.assertEqual(weights, 3)
    self.assertAlmostEqual(loss / weights, expected_loss)


class KLDivergenceTest(absltest.TestCase):
  """Test the KL divergence loss."""

  def test_kl_divergence(self):
    p = jnp.array([[0.1, 0.3, 0.3, 0.3]])
    q = jnp.array([[0.2, 0.5, 0.1, 0.2]])

    eps = 1e-8

    kl = losses.kl_divergence_with_logits(jnp.log(p + eps), jnp.log(q + eps))
    self.assertAlmostEqual(kl, 0.22866088, places=4)

  def test_kl_divergence_probs(self):
    p = jnp.array([[0.1, 0.3, 0.3, 0.3]])
    q = jnp.array([[0.2, 0.5, 0.1, 0.2]])

    kl = losses.kl_divergence_with_probs(p, q)
    self.assertAlmostEqual(kl, 0.22866088, places=4)


if __name__ == '__main__':
  absltest.main()
