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

"""Tests for the softsort and softranks operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from soft_sort import ops


class OpsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(OpsTest, self).setUp()
    tf.random.set_seed(0)
    np.random.seed(seed=0)

  def test_preprocess(self):
    """Tests that _preprocess prepares the tensor as expected."""
    # Test preprocessing with input of dimension 1.
    n = 10
    x = tf.random.uniform((n,), dtype=tf.float64)
    z = ops._preprocess(x, axis=-1)
    self.assertEqual(z.shape.rank, 2)
    self.assertEqual(z.shape, (1, n))
    self.assertAllEqual(z[0], x)

    # Test preprocessing with input of dimension 2.
    x = tf.random.uniform((3, n), dtype=tf.float64)
    z = ops._preprocess(x, axis=-1)
    self.assertEqual(z.shape.rank, 2)
    self.assertEqual(z.shape, x.shape)
    self.assertAllEqual(z, x)

    # Test preprocessing with input of dimension 2, preparing for axis 0
    x = tf.random.uniform((3, n), dtype=tf.float64)
    z = ops._preprocess(x, axis=0)
    self.assertEqual(z.shape.rank, 2)
    self.assertEqual(z.shape, (x.shape[1], x.shape[0]))
    batch = 1
    self.assertAllEqual(z[batch], x[:, batch])

    # Test preprocessing with input of dimension > 2
    shape = [4, 21, 7, 10]
    x = tf.random.uniform(shape, dtype=tf.float64)
    axis = 2
    n = shape.pop(axis)
    z = ops._preprocess(x, axis=axis)
    self.assertEqual(z.shape.rank, 2)
    self.assertEqual(z.shape, (np.prod(shape), n))

  def test_postprocess(self):
    """Tests that _postprocess is the inverse of _preprocess."""
    shape = (4, 21, 7, 10)
    for i in range(1, len(shape)):
      x = tf.random.uniform(shape[:i])
      for axis in range(x.shape.rank):
        z = ops._postprocess(ops._preprocess(x, axis), x.shape, axis)
        self.assertAllEqual(x, z)

  def test_softsort(self):
    # Tests that the values are sorted (epsilon being small enough)
    x = tf.constant([3, 4, 1, 5, 2], dtype=tf.float32)
    eps = 1e-3
    sinkhorn_threshold = 1e-3
    values = ops.softsort(x, direction='ASCENDING',
                          epsilon=eps, sinkhorn_threshold=sinkhorn_threshold)
    self.assertEqual(values.shape, x.shape)
    self.assertAllGreater(np.diff(values), 0.0)

    # Since epsilon is not very small, we cannot expect to retrieve the sorted
    # values with high precision.
    tolerance = 1e-1
    self.assertAllClose(tf.sort(x), values, tolerance, tolerance)

    # Test descending sort.
    direction = 'DESCENDING'
    values = ops.softsort(x, direction=direction,
                          epsilon=eps, sinkhorn_threshold=sinkhorn_threshold)
    self.assertEqual(values.shape, x.shape)
    self.assertAllLess(np.diff(values), 0.0)
    self.assertAllClose(
        tf.sort(x, direction=direction), values, tolerance, tolerance)

  @parameterized.named_parameters(
      ('axis', 0, 'direction', 'ASCENDING'),
      ('axis', 1, 'direction', 'ASCENDING'),
      ('axis', 2, 'direction', 'ASCENDING'),
      ('axis', 0, 'direction', 'DESCENDING'),
      ('axis', 1, 'direction', 'DESCENDING'),
      ('axis', 2, 'direction', 'DESCENDING'))
  def softranks(self, axis, direction):
    """Test ops.softranks for a given shape, axis and direction."""
    shape = tf.TensorShape((3, 8, 6))
    n = shape[axis]
    p = int(np.prod(shape) / shape[axis])

    # Build a target tensor of ranks, of rank 2.
    # Those targets are zero based.
    target = tf.constant(
        [np.random.permutation(n) for _ in range(p)], dtype=tf.float32)

    # Turn it into a tensor of desired shape.
    target = ops._postprocess(target, shape, axis)

    # Apply a monotonic transformation to turn ranks into values
    sign = 2 * float(direction == 'ASCENDING') - 1
    x = sign * (1.2 * target - 0.4)

    # The softranks of x along the axis should be close to the target.
    eps = 1e-3
    sinkhorn_threshold = 1e-3
    tolerance = 0.5
    for zero_based in [False, True]:
      ranks = ops.softranks(
          x, direction=direction, axis=axis, zero_based=zero_based,
          epsilon=eps, sinkhorn_threshold=sinkhorn_threshold)
      targets = target + 1 if not zero_based else target
      self.assertAllClose(ranks, targets, tolerance, tolerance)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
