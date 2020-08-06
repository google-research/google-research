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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from single_view_mpi.libs import utils

# Because there are tables of numbers that line up to make them readable.
# pylint: disable=bad-whitespace


class UtilsTest(tf.test.TestCase):

  def test_broadcast_to_match(self):
    data = [
        # Each pair of lines is a test example. First line is the two inputs,
        # second line is the two outputs (or None if it should fail).
        #
        # Equal
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]], 0,
         [[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        # B broadcast up to A
        ([10, 11, 12], 1, 0,
         [10, 11, 12], [1, 1, 1]),
        ([[[10, 20], [30, 40]]], [1, 2], 0,
         [[[10, 20], [30, 40]]], [[[1, 2], [1, 2]]]),
        # A broadcast up to B
        ([[1],    [2]],    [[10, 11], [20, 21]], 0,
         [[1, 1], [2, 2]], [[10, 11], [20, 21]]),
        # Both broadcast: shapes [1, 2] and [2, 1]
        ([[1, 2]],         [[3], [4]], 0,
         [[1, 2], [1, 2]], [[3,3], [4,4]]),
        # Dimension added to A
        ([1, 2],           [[5], [6]], 0,
         [[1, 2], [1, 2]], [[5, 5], [6, 6]]),
        # Dimensions added to B
        ([[[1.0]]], 5.0, 0,
         [[[1.0]]], [[[5.0]]]),
        # Error case: non-matching shapes
        ([[1,2,3], [4,5,6]], [[10,20], [30,40]], 0,
         None, None),
        # Ignoring last two axes (e.g. for matrix multiply of shapes
        # [2,1,1,3] * [1,1,3,1]
        ([[[[1, 2, 3]]], [[[4, 5, 6]]]], [[[[10], [11], [12]]]], 2,
         [[[[1, 2, 3]]], [[[4, 5, 6]]]],
         [[[[10], [11], [12]]], [[[10], [11], [12]]]])
    ]
    for (a, b, ignore_axes, expected_a, expected_b) in data:
      if expected_a is None:
        with self.assertRaises(Exception):
          (aa, bb) = utils.broadcast_to_match(
              tf.constant(a), tf.constant(b), ignore_axes=ignore_axes)
      else:
        (aa, bb) = utils.broadcast_to_match(
            tf.constant(a), tf.constant(b), ignore_axes=ignore_axes)
        self.assertAllEqual(expected_a, aa)
        self.assertAllEqual(expected_b, bb)

  def test_collapse_dim(self):
    # A tensor of shape [2, 3, 4].
    tensor = tf.constant(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
         [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]],
        dtype=tf.uint8)
    # Collapse using positive axis number
    self.assertAllEqual(
        tf.reshape(tensor, [6, 4]), utils.collapse_dim(tensor, 1))
    self.assertAllEqual(
        tf.reshape(tensor, [2, 12]), utils.collapse_dim(tensor, 2))
    # Collapse using negative axis number
    self.assertAllEqual(
        tf.reshape(tensor, [2, 12]), utils.collapse_dim(tensor, -1))
    self.assertAllEqual(
        tf.reshape(tensor, [6, 4]), utils.collapse_dim(tensor, -2))

  def test_split_dim(self):
    # A tensor with 24 values which we'll reshape and split various ways.
    tensor = tf.constant(list(range(0, 24)), dtype=tf.uint8)
    # Split first axis
    self.assertAllEqual(
        tf.reshape(tensor, [2, 3, 4]),
        utils.split_dim(tf.reshape(tensor, [6, 4]), 0, 2))
    self.assertAllEqual(
        tf.reshape(tensor, [3, 2, 4]),
        utils.split_dim(tf.reshape(tensor, [6, 4]), 0, 3))
    # ... also using negative axis number
    self.assertAllEqual(
        tf.reshape(tensor, [2, 3, 4]),
        utils.split_dim(tf.reshape(tensor, [6, 4]), -2, 2))
    self.assertAllEqual(
        tf.reshape(tensor, [3, 2, 4]),
        utils.split_dim(tf.reshape(tensor, [6, 4]), -2, 3))
    # Split last axis
    self.assertAllEqual(
        tf.reshape(tensor, [2, 6, 2]),
        utils.split_dim(tf.reshape(tensor, [2, 12]), 1, 6))
    self.assertAllEqual(
        tf.reshape(tensor, [2, 2, 6]),
        utils.split_dim(tf.reshape(tensor, [2, 12]), 1, 2))
    # ... also using negative axis number
    self.assertAllEqual(
        tf.reshape(tensor, [2, 6, 2]),
        utils.split_dim(tf.reshape(tensor, [2, 12]), -1, 6))
    self.assertAllEqual(
        tf.reshape(tensor, [2, 2, 6]),
        utils.split_dim(tf.reshape(tensor, [2, 12]), -1, 2))
    # Split fails if the size given isn't a factor of the dimension
    with self.assertRaises(Exception):
      _ = utils.split_dim(tf.reshape(tensor, [4, 6]), 0, 3)

  def test_flatten_batch(self):
    # Example data of shape 3 x 2 x 2 x 1.
    data = [[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]],
            [[[9], [10]], [[11], [12]]]]
    # Flatten the initial 1 (no-op) 2 or even all 4 dimensions.
    flattened_1, unflatten_1 = utils.flatten_batch(tf.constant(data), 1)
    flattened_2, unflatten_2 = utils.flatten_batch(tf.constant(data), 2)
    flattened_3, unflatten_3 = utils.flatten_batch(tf.constant(data), 3)
    flattened_4, unflatten_4 = utils.flatten_batch(tf.constant(data), 4)
    unflattened_1 = unflatten_1(flattened_1)
    unflattened_2 = unflatten_2(flattened_2)
    unflattened_3 = unflatten_3(flattened_3)
    unflattened_4 = unflatten_4(flattened_4)

    # Check flattened shapes
    self.assertAllEqual([3, 2, 2, 1], flattened_1.shape.as_list())
    self.assertAllEqual([6, 2, 1], flattened_2.shape.as_list())
    self.assertAllEqual([12, 1], flattened_3.shape.as_list())
    self.assertAllEqual([12], flattened_4.shape.as_list())

    # Check flattened values
    # pylint: disable=g-complex-comprehension
    self.assertAllEqual(data, flattened_1)
    self.assertAllEqual([e for row in data for e in row], flattened_2)

    self.assertAllEqual([e for row in data for col in row
                         for e in col], flattened_3)
    self.assertAllEqual(
        [e for row in data for col in row for item in col
         for e in item], flattened_4)
    # pylint: enable=g-complex-comprehension

    # Unflattening restores initial shape and values
    self.assertAllEqual(data, unflattened_1)
    self.assertAllEqual(data, unflattened_2)
    self.assertAllEqual(data, unflattened_3)
    self.assertAllEqual(data, unflattened_4)

    # Unflattening works also on tensors with different shape after the batch
    # dimension, and with different type. Here are two results with shape
    # [3, 2 ...].
    result_a = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    result_b = [[[[1, 1, 1, 1]], [[2, 2, 2, 2]]],
                [[[3, 3, 3, 3]], [[4, 4, 4, 4]]],
                [[[5, 5, 5, 5]], [[6, 6, 6, 6]]]]
    flattened_result_a = [x for row in result_a for x in row]
    flattened_result_b = [x for row in result_b for x in row]

    self.assertAllEqual(result_a, unflatten_2(tf.constant(flattened_result_a)))
    self.assertAllEqual(result_b, unflatten_2(tf.constant(flattened_result_b)))


# pylint: enable=bad-whitespace

if __name__ == '__main__':
  tf.test.main()
