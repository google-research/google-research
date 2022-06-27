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

"""Tests for losses."""
import numpy as np
import tensorflow as tf

from simulation_research.next_day_wildfire_spread.models import losses


class LossesTest(tf.test.TestCase):

  def testMaskedInput(self):
    """Checks that the loss function ignores the masked input."""
    labels = -1 * np.ones((2, 2), dtype=np.float32)
    logits = np.random.uniform(0., 1., (2, 2)).astype(np.float32)
    loss = losses.weighted_cross_entropy_with_logits_with_masked_class(
        pos_weight=1.)(labels, logits)
    self.assertEqual(loss.shape, ())
    self.assertEqual(loss.numpy(), 0.0)

  def testUnmaskedInput(self):
    """Checks that the loss function processes the unmasked input correctly."""
    labels = np.array([[0., 1.], [0., 1.]], dtype=np.float32)
    logits = np.array([[100., 100.], [-100., -100.]], dtype=np.float32)
    loss = losses.weighted_cross_entropy_with_logits_with_masked_class(
        pos_weight=1.)(labels, logits)
    self.assertEqual(loss.shape, ())
    self.assertAllClose(loss.numpy(), 50.0)

  def testUnmaskedInputWithWeights(self):
    """Checks that the loss function processes the weighted unmasked input correctly."""
    labels = np.array([[0., 1.], [0., 1.]], dtype=np.float32)
    logits = np.array([[100., 100.], [-100., -100.]], dtype=np.float32)
    loss = losses.weighted_cross_entropy_with_logits_with_masked_class(
        pos_weight=2.)(labels, logits)
    self.assertEqual(loss.shape, ())
    self.assertAllClose(loss.numpy(), 75.0)

  def testPartiallyMaskedInput(self):
    """Checks that the loss function processes the partially masked input correctly."""
    labels = np.array([[-1., -1.], [0., 1.]], dtype=np.float32)
    logits = np.array([[100., 100.], [-100., -100.]], dtype=np.float32)
    loss = losses.weighted_cross_entropy_with_logits_with_masked_class(
        pos_weight=1.)(labels, logits)
    self.assertEqual(loss.shape, ())
    self.assertAllClose(loss.numpy(), 25.0)

  def testPartiallyMaskedInputWithWeights(self):
    """Checks that the loss function processes the weighted partially masked input correctly."""
    labels = np.array([[-1., -1.], [0., 1.]], dtype=np.float32)
    logits = np.array([[100., 100.], [-100., -100.]], dtype=np.float32)
    loss = losses.weighted_cross_entropy_with_logits_with_masked_class(
        pos_weight=2.)(labels, logits)
    self.assertEqual(loss.shape, ())
    self.assertAllClose(loss.numpy(), 50.0)


if __name__ == '__main__':
  tf.test.main()
