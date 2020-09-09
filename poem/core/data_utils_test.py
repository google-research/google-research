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

"""Tests for data utility functions."""

import numpy as np
import tensorflow as tf

from poem.core import data_utils


class DataUtilsTest(tf.test.TestCase):

  def test_flatten_last_dims(self):
    # Shape = [2, 3, 4].
    x = tf.constant([[[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]],
                     [[31, 32, 33, 34], [41, 42, 43, 44], [51, 52, 53, 54]]])
    flattened_x = data_utils.flatten_last_dims(x, num_last_dims=2)
    self.assertAllEqual(flattened_x,
                        [[1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24],
                         [31, 32, 33, 34, 41, 42, 43, 44, 51, 52, 53, 54]])

  def test_flatten_first_dims(self):
    # Shape = [1, 2, 3, 4, 1].
    x = tf.constant([[[[[1], [2], [3], [4]], [[11], [12], [13], [14]],
                       [[21], [22], [23], [24]]],
                      [[[31], [32], [33], [34]], [[41], [42], [43], [44]],
                       [[51], [52], [53], [54]]]]])
    flattened_x = data_utils.flatten_first_dims(x, num_last_dims_to_keep=2)
    self.assertAllEqual(flattened_x,
                        [[[1], [2], [3], [4]], [[11], [12], [13], [14]],
                         [[21], [22], [23], [24]], [[31], [32], [33], [34]],
                         [[41], [42], [43], [44]], [[51], [52], [53], [54]]])

  def test_tile_first_dims(self):
    # Shape = [1, 2, 1].
    x = tf.constant([[[1], [2]]])
    tiled_x = data_utils.tile_first_dims(x, first_dim_multiples=[2, 2])
    self.assertAllEqual(tiled_x, [[[1], [2], [1], [2]], [[1], [2], [1], [2]]])

  def test_tile_last_dims(self):
    # Shape = [2, 1, 2, 1].
    x = tf.constant([[[[1], [2]]], [[[3], [4]]]])
    tiled_x = data_utils.tile_last_dims(x, last_dim_multiples=[2, 2])
    self.assertAllEqual(tiled_x, [[[[1, 1], [2, 2], [1, 1], [2, 2]]],
                                  [[[3, 3], [4, 4], [3, 3], [4, 4]]]])

  def test_recursively_expand_dims(self):
    # Shape = [2, 3].
    x = tf.constant([[1, 2, 3], [4, 5, 6]])
    # Shape = [2, 1, 3, 1]
    expanded_x = data_utils.recursively_expand_dims(x, axes=[-1, 1])
    self.assertAllEqual(expanded_x, [[[[1], [2], [3]]], [[[4], [5], [6]]]])

  def test_reshape_by_last_dims(self):
    # Shape = [2, 4, 1].
    x = tf.constant([[[1], [2], [3], [4]], [[5], [6], [7], [8]]])
    # Shape = [2, 2, 2]
    reshaped_x = data_utils.reshape_by_last_dims(x, last_dim_shape=[2, 2])
    self.assertAllEqual(reshaped_x, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

  def test_reduce_mean(self):
    # Shape = [2, 3, 2].
    tensor = tf.constant([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                          [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    # Shape = [2, 3, 1].
    weights = tf.constant([[[1.0], [0.0], [1.0]], [[0.0], [1.0], [0.0]]])
    # Shape = [2, 1, 2].
    means = data_utils.reduce_weighted_mean(
        tensor, weights, axis=-2, keepdims=True)

    self.assertAllClose(means, [[[3.0, 4.0]], [[9.0, 10.0]]])

  def test_sample_gaussians(self):
    means = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stddevs = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    samples = data_utils.sample_gaussians(
        means, stddevs, num_samples=10, seed=1)
    self.assertAllClose(
        samples,
        [[[0.9188682, 2.2969198, 3.0195987], [0.75572956, 2.0198498, 3.1773672],
          [1.0592823, 1.5754141, 2.783131], [0.99437296, 2.1287088, 2.9207027],
          [1.1856633, 2.1135683, 2.8851492], [0.85146564, 2.2523541, 2.9924083],
          [0.973537, 2.3065627, 2.4771068], [0.95621073, 1.886798, 3.0962007],
          [1.1132832, 1.5443486, 3.1448436], [0.8687291, 2.0713701, 2.480915]],
         [[3.983933, 5.449831, 5.1716466], [4.592585, 4.8772526, 5.5604115],
          [3.9216413, 5.035854, 6.3797884], [3.3715236, 5.6646905, 5.2959795],
          [4.012618, 5.2385263, 6.262165], [3.8732765, 4.774625, 4.9163604],
          [4.0499597, 4.6146727, 5.552255], [3.8872187, 4.020592, 5.7974334],
          [4.4120793, 5.756701, 6.1350946], [3.8857353, 5.134413, 7.0477266]]])

  def test_compute_lower_percentile_means(self):
    # Shape = [2, 3, 3].
    x = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                     [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0],
                      [17.0, 18.0, 19.0]]])
    lower_half = data_utils.compute_lower_percentile_means(x, axis=[-2, -1])
    self.assertAllClose(lower_half, [3.0, 13.0])

  def test_mix_pair_batch_evenly(self):
    lhs_pairs = tf.constant([
        [[1.0], [2.0]],
        [[1.1], [2.1]],
        [[3.0], [4.0]],
        [[3.1], [4.1]],
        [[5.0], [6.0]],
        [[5.1], [6.1]],
        [[7.0], [8.0]],
        [[7.1], [8.1]],
    ])
    rhs_pairs = tf.constant([
        [[11.0], [12.0]],
        [[11.1], [12.1]],
        [[13.0], [14.0]],
        [[13.1], [14.1]],
        [[15.0], [16.0]],
        [[15.1], [16.1]],
        [[17.0], [18.0]],
        [[17.1], [18.1]],
    ])
    mixed_batch = data_utils.mix_pair_batch(lhs_pairs, rhs_pairs, axis=1)
    self.assertAllEqual(
        mixed_batch,
        np.array([
            [[1.0], [2.0]],
            [[1.1], [2.1]],
            [[13.0], [14.0]],
            [[13.1], [14.1]],
            [[5.0], [16.0]],
            [[5.1], [16.1]],
            [[17.0], [8.0]],
            [[17.1], [8.1]],
        ],
                 dtype=np.float32))

  def test_mix_pair_batch_porportionally(self):
    lhs_pairs = tf.constant([
        [[1.0], [2.0]],
        [[1.1], [2.1]],
        [[3.0], [4.0]],
        [[3.1], [4.1]],
        [[5.0], [6.0]],
        [[5.1], [6.1]],
        [[7.0], [8.0]],
        [[7.1], [8.1]],
    ])
    rhs_pairs = tf.constant([
        [[11.0], [12.0]],
        [[11.1], [12.1]],
        [[13.0], [14.0]],
        [[13.1], [14.1]],
        [[15.0], [16.0]],
        [[15.1], [16.1]],
        [[17.0], [18.0]],
        [[17.1], [18.1]],
    ])
    mixed_batch = data_utils.mix_pair_batch(
        lhs_pairs, rhs_pairs, axis=1, sub_batch_ratios=(1.0, 4.0, 1.0, 2.0))
    self.assertAllEqual(
        mixed_batch,
        np.array([
            [[1.0], [2.0]],
            [[11.1], [12.1]],
            [[13.0], [14.0]],
            [[13.1], [14.1]],
            [[15.0], [16.0]],
            [[5.1], [16.1]],
            [[17.0], [8.0]],
            [[17.1], [8.1]],
        ],
                 dtype=np.float32))

  def test_shuffle_batches(self):
    # Shape = [3, 2].
    tensor_1 = tf.constant([[1, 2], [3, 4], [5, 6]])
    tensor_2 = tf.constant([[11, 12], [13, 14], [15, 16]])
    tensor_3 = tf.constant([[21, 22], [23, 24], [25, 26]])
    shuffled_tensor_1, shuffled_tensor_2, shuffled_tensor_3 = (
        data_utils.shuffle_batches([tensor_1, tensor_2, tensor_3]))
    tensor_diff_21 = shuffled_tensor_2 - shuffled_tensor_1
    tensor_diff_31 = shuffled_tensor_3 - shuffled_tensor_1
    self.assertAllEqual(tensor_diff_21, [[10, 10], [10, 10], [10, 10]])
    self.assertAllEqual(tensor_diff_31, [[20, 20], [20, 20], [20, 20]])

  def test_update_sub_tensor(self):
    # Shape = [3, 5, 2].
    x = tf.constant([
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
        [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0], [18.0, 19.0]],
        [[20.0, 21.0], [22.0, 23.0], [24.0, 25.0], [26.0, 27.0], [28.0, 29.0]],
    ])

    def update_func(sub_tensor):
      # Shape = [3, 3, 2].
      delta = tf.constant([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                           [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
                           [[1.3, 1.4], [1.5, 1.6], [1.7, 1.8]]])
      return sub_tensor + delta

    updated_x = data_utils.update_sub_tensor(
        x, indices=[0, 2, 4], axis=-2, update_func=update_func)

    self.assertAllClose(updated_x, [
        [[0.1, 1.2], [2.0, 3.0], [4.3, 5.4], [6.0, 7.0], [8.5, 9.6]],
        [[10.7, 11.8], [12.0, 13.0], [14.9, 16.0], [16.0, 17.0], [19.1, 20.2]],
        [[21.3, 22.4], [22.0, 23.0], [25.5, 26.6], [26.0, 27.0], [29.7, 30.8]],
    ])


if __name__ == '__main__':
  tf.test.main()
