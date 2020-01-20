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
    z, _, _ = ops._preprocess(x, axis=-1)
    self.assertEqual(z.shape.rank, 2)
    self.assertEqual(z.shape, (1, n))
    self.assertAllEqual(z[0], x)

    # Test preprocessing with input of dimension 2.
    x = tf.random.uniform((3, n), dtype=tf.float64)
    z, _, _ = ops._preprocess(x, axis=-1)
    self.assertEqual(z.shape.rank, 2)
    self.assertEqual(z.shape, x.shape)
    self.assertAllEqual(z, x)

    # Test preprocessing with input of dimension 2, preparing for axis 0
    x = tf.random.uniform((3, n), dtype=tf.float64)
    z, _, _ = ops._preprocess(x, axis=0)
    self.assertEqual(z.shape.rank, 2)
    self.assertEqual(z.shape, (x.shape[1], x.shape[0]))
    batch = 1
    self.assertAllEqual(z[batch], x[:, batch])

    # Test preprocessing with input of dimension > 2
    shape = [4, 21, 7, 10]
    x = tf.random.uniform(shape, dtype=tf.float64)
    axis = 2
    n = shape.pop(axis)
    z, _, _ = ops._preprocess(x, axis=axis)
    self.assertEqual(z.shape.rank, 2)
    self.assertEqual(z.shape, (np.prod(shape), n))

  def test_postprocess(self):
    """Tests that _postprocess is the inverse of _preprocess."""
    shape = (4, 21, 7, 10)
    for i in range(1, len(shape)):
      x = tf.random.uniform(shape[:i])
      for axis in range(x.shape.rank):
        y, transp, s = ops._preprocess(x, axis)
        z = ops._postprocess(y, transp, s)
        self.assertAllEqual(x, z)

  @parameterized.named_parameters(
      ('all', None),
      ('top3', 3),
      ('top5', 5))
  def test_softsort(self, topk):
    # Tests that the values are sorted (epsilon being small enough)
    x = tf.constant([3, 4, 1, 5, 2, 9, 12, 11, 8, 15], dtype=tf.float32)
    eps = 1e-3
    sinkhorn_threshold = 1e-3
    values = ops.softsort(x, direction='ASCENDING', topk=topk,
                          epsilon=eps, threshold=sinkhorn_threshold)
    expect_shape = x.shape if topk is None else (topk,)
    self.assertEqual(values.shape, expect_shape)
    self.assertAllGreater(np.diff(values), 0.0)

    # Since epsilon is not very small, we cannot expect to retrieve the sorted
    # values with high precision.
    tolerance = 1e-1
    expected_values = tf.sort(x)
    if topk is not None:
      expected_values = expected_values[:topk]
    self.assertAllClose(expected_values, values, tolerance, tolerance)

    # Test descending sort.
    direction = 'DESCENDING'
    values = ops.softsort(x, direction=direction, topk=topk,
                          epsilon=eps, threshold=sinkhorn_threshold)
    expected_values = tf.sort(x, direction=direction)
    if topk is not None:
      expected_values = expected_values[:topk]
    self.assertEqual(values.shape, expect_shape)
    self.assertAllLess(np.diff(values), 0.0)
    self.assertAllClose(expected_values, values, tolerance, tolerance)

  @parameterized.named_parameters(
      ('ascending_0', 0, 'ASCENDING'),
      ('ascending_1', 1, 'ASCENDING'),
      ('ascending_2', 2, 'ASCENDING'),
      ('descending_0', 0, 'DESCENDING'),
      ('descending_1', 1, 'DESCENDING'),
      ('descending_2', 2, 'DESCENDING'))
  def test_softranks(self, axis, direction):
    """Test ops.softranks for a given shape, axis and direction."""
    shape = tf.TensorShape((3, 8, 6))
    n = shape[axis]
    p = int(np.prod(shape) / shape[axis])

    # Build a target tensor of ranks, of rank 2.
    # Those targets are zero based.
    target = tf.constant(
        [np.random.permutation(n) for _ in range(p)], dtype=tf.float32)

    # Turn it into a tensor of desired shape.
    dims = np.arange(shape.rank)
    dims[axis], dims[-1] = dims[-1], dims[axis]
    fake = tf.zeros(shape)
    transposition = tf.transpose(fake, dims).shape
    target = ops._postprocess(target, dims, transposition)

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
          epsilon=eps, threshold=sinkhorn_threshold)
      targets = target + 1 if not zero_based else target
      self.assertAllClose(ranks, targets, tolerance, tolerance)

  @parameterized.parameters([0.1, 0.2, 0.5])
  def test_softquantile(self, quantile):
    # Builds the input vector so that the desired quantile always corresponds to
    # an exact integer index.
    num_points_before_quantile = 10
    step = quantile / num_points_before_quantile
    num_points = int(1.0 / step + 1.0)
    quantile_width = step

    axis = 1
    x = tf.random.uniform((3, num_points, 4), dtype=tf.float32)
    soft_q = ops.softquantiles(
        x, quantile, quantile_width, axis=axis, epsilon=1e-3)

    # Compare to getting the exact quantile.
    hard_q = tf.gather(
        tf.sort(x, direction='ASCENDING', axis=axis),
        int(quantile * num_points), axis=1)

    self.assertAllClose(soft_q, hard_q, 0.2, 0.2)

  def test_softquantiles(self):
    num_points = 19
    sorted_values = tf.range(0, num_points, dtype=tf.float32)
    x = tf.random.shuffle(sorted_values)

    target_quantiles = [0.25, 0.50, 0.75]
    target_indices = [4, 9, 14]
    soft_q = ops.softquantiles(x, target_quantiles, epsilon=1e-3)
    hard_q = tf.gather(sorted_values, target_indices)

    self.assertAllClose(soft_q, hard_q, 0.2, 0.2)

  @parameterized.named_parameters(
      ('small_gap', 0.5, 0.6, 0.4),
      ('small_first_gap', 0.09, 0.6, 0.2),
      ('wrong_order', 0.7, 0.3, 0.2))
  def test_softquantile_errors(self, q1, q2, width):
    x = tf.random.uniform((3, 10))
    with self.assertRaises(tf.errors.InvalidArgumentError):
      ops.softquantiles(x, [q1, q2], quantile_width=width, axis=-1)

  def test_soft_quantile_normalization(self):
    x = tf.constant([1.2, 1.3, 1.5, -4.0, 1.8, 2.4, -1.0])
    target = tf.cumsum(tf.ones(x.shape[0]))
    xn = ops.soft_quantile_normalization(x, target)
    # Make sure that the order of x and xn are identical
    self.assertAllEqual(tf.argsort(x), tf.argsort(xn))
    # Make sure that the values of xn and target are close.
    self.assertAllClose(tf.sort(target), tf.sort(xn), atol=1e-1)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
