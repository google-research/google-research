# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

  def test_unflatten_first_dim(self):
    # Shape = [6, 2].
    x = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    unflattened_x = data_utils.unflatten_first_dim(
        x, shape_to_unflatten=tf.constant([2, 3]))
    self.assertAllEqual(unflattened_x,
                        [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])

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

  def test_get_shape_by_last_dims(self):
    # Shape = [1, 2, 3, 4, 5].
    x = tf.zeros([1, 2, 3, 4, 5])
    shape = data_utils.get_shape_by_last_dims(x, num_last_dims=2)
    self.assertAllEqual(shape, [4, 5])

  def test_get_shape_by_first_dims(self):
    # Shape = [1, 2, 3, 4, 5].
    x = tf.zeros([1, 2, 3, 4, 5])
    shape = data_utils.get_shape_by_first_dims(x, num_last_dims=2)
    self.assertAllEqual(shape, [1, 2, 3])

  def test_reshape_by_last_dims(self):
    # Shape = [2, 4, 1].
    x = tf.constant([[[1], [2], [3], [4]], [[5], [6], [7], [8]]])
    # Shape = [2, 2, 2]
    reshaped_x = data_utils.reshape_by_last_dims(x, last_dim_shape=[2, 2])
    self.assertAllEqual(reshaped_x, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

  def test_swap_axes(self):
    # Shape = [2, 4, 1, 3].
    x = tf.constant([[[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]],
                     [[[13, 14, 15]], [[16, 17, 18]], [[19, 20, 21]],
                      [[22, 23, 24]]]])
    # Shape = [2, 3, 1, 4].
    permuted_x = data_utils.swap_axes(x, lhs_axis=-3, rhs_axis=-1)
    self.assertAllEqual(
        permuted_x,
        [[[[1, 4, 7, 10]], [[2, 5, 8, 11]], [[3, 6, 9, 12]]],
         [[[13, 16, 19, 22]], [[14, 17, 20, 23]], [[15, 18, 21, 24]]]])

  def test_move_axis(self):
    # Shape = [1, 2, 3, 4, 1].
    x = tf.constant([[[[[1], [2], [3], [4]], [[5], [6], [7], [8]],
                       [[9], [10], [11], [12]]],
                      [[[13], [14], [15], [16]], [[17], [18], [19], [20]],
                       [[21], [22], [23], [24]]]]])
    # Shape = [1, 3, 4, 2, 1].
    permuted_x = data_utils.move_axis(x, input_axis=1, output_axis=-2)
    self.assertAllEqual(
        permuted_x, [[[[[1], [13]], [[2], [14]], [[3], [15]], [[4], [16]]],
                      [[[5], [17]], [[6], [18]], [[7], [19]], [[8], [20]]],
                      [[[9], [21]], [[10], [22]], [[11], [23]], [[12], [24]]]]])

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

  def test_mix_batch_singletons(self):
    # Shape = [8, 1, 2].
    lhs_batch = tf.constant([
        [[[1.0, 1.01]]],
        [[[1.1, 1.11]]],
        [[[3.0, 3.01]]],
        [[[3.1, 3.11]]],
        [[[5.0, 5.01]]],
        [[[5.1, 5.11]]],
        [[[7.0, 7.01]]],
        [[[7.1, 7.11]]],
    ])
    rhs_batch = tf.constant([
        [[[11.0, 11.01]]],
        [[[11.1, 11.11]]],
        [[[13.0, 13.01]]],
        [[[13.1, 13.11]]],
        [[[15.0, 15.01]]],
        [[[15.1, 15.11]]],
        [[[17.0, 17.01]]],
        [[[17.1, 17.11]]],
    ])
    # Shape = [8, 1].
    assignment = tf.constant([[True], [True], [False], [False], [True], [True],
                              [False], [False]])
    mixed_batch = data_utils.mix_batch([lhs_batch], [rhs_batch],
                                       axis=1,
                                       assignment=assignment)[0]
    self.assertAllEqual(
        mixed_batch,
        np.array([
            [[[1.0, 1.01]]],
            [[[1.1, 1.11]]],
            [[[13.0, 13.01]]],
            [[[13.1, 13.11]]],
            [[[5.0, 5.01]]],
            [[[5.1, 5.11]]],
            [[[17.0, 17.01]]],
            [[[17.1, 17.11]]],
        ],
                 dtype=np.float32))

  def test_mix_batch_pairs(self):
    # Shape = [8, 2, 2].
    lhs_batch = tf.constant([
        [[[1.0, 1.01]], [[2.0, 2.01]]],
        [[[1.1, 1.11]], [[2.1, 2.11]]],
        [[[3.0, 3.01]], [[4.0, 4.01]]],
        [[[3.1, 3.11]], [[4.1, 4.11]]],
        [[[5.0, 5.01]], [[6.0, 6.01]]],
        [[[5.1, 5.11]], [[6.1, 6.11]]],
        [[[7.0, 7.01]], [[8.0, 8.01]]],
        [[[7.1, 7.11]], [[8.1, 8.11]]],
    ])
    rhs_batch = tf.constant([
        [[[11.0, 11.01]], [[12.0, 12.01]]],
        [[[11.1, 11.11]], [[12.1, 12.11]]],
        [[[13.0, 13.01]], [[14.0, 14.01]]],
        [[[13.1, 13.11]], [[14.1, 14.11]]],
        [[[15.0, 15.01]], [[16.0, 16.01]]],
        [[[15.1, 15.11]], [[16.1, 16.11]]],
        [[[17.0, 17.01]], [[18.0, 18.01]]],
        [[[17.1, 17.11]], [[18.1, 18.11]]],
    ])
    # Shape = [8, 2].
    assignment = tf.constant([[True, True], [True, True], [False, False],
                              [False, False], [True, False], [True, False],
                              [False, True], [False, True]])
    mixed_batch = data_utils.mix_batch([lhs_batch], [rhs_batch],
                                       axis=1,
                                       assignment=assignment)[0]
    self.assertAllEqual(
        mixed_batch,
        np.array([
            [[[1.0, 1.01]], [[2.0, 2.01]]],
            [[[1.1, 1.11]], [[2.1, 2.11]]],
            [[[13.0, 13.01]], [[14.0, 14.01]]],
            [[[13.1, 13.11]], [[14.1, 14.11]]],
            [[[5.0, 5.01]], [[16.0, 16.01]]],
            [[[5.1, 5.11]], [[16.1, 16.11]]],
            [[[17.0, 17.01]], [[8.0, 8.01]]],
            [[[17.1, 17.11]], [[8.1, 8.11]]],
        ],
                 dtype=np.float32))

  def test_mix_batch_pair_lists(self):
    lhs_batches, rhs_batches = [None, None], [None, None]

    # Shape = [4, 3, 2, 1].
    lhs_batches[0] = tf.constant([
        [[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
        [[[2.0], [2.1]], [[2.2], [2.3]], [[2.4], [2.5]]],
        [[[3.0], [3.1]], [[3.2], [3.3]], [[3.4], [3.5]]],
        [[[4.0], [4.1]], [[4.2], [4.3]], [[4.4], [4.5]]],
    ])
    rhs_batches[0] = tf.constant([
        [[[11.0], [11.1]], [[11.2], [11.3]], [[11.4], [11.5]]],
        [[[12.0], [12.1]], [[12.2], [12.3]], [[12.4], [12.5]]],
        [[[13.0], [13.1]], [[13.2], [13.3]], [[13.4], [13.5]]],
        [[[14.0], [14.1]], [[14.2], [14.3]], [[14.4], [14.5]]],
    ])

    # Shape = [4, 3, 2, 2, 1].
    lhs_batches[1] = tf.constant([[[[[1.0], [10.0]], [[1.1], [10.1]]],
                                   [[[1.2], [10.2]], [[1.3], [10.3]]],
                                   [[[1.4], [10.4]], [[1.5], [10.5]]]],
                                  [[[[2.0], [20.0]], [[2.1], [20.1]]],
                                   [[[2.2], [20.2]], [[2.3], [20.3]]],
                                   [[[2.4], [20.4]], [[2.5], [20.5]]]],
                                  [[[[3.0], [30.0]], [[3.1], [30.1]]],
                                   [[[3.2], [30.2]], [[3.3], [30.3]]],
                                   [[[3.4], [30.4]], [[3.5], [30.5]]]],
                                  [[[[4.0], [40.0]], [[4.1], [40.1]]],
                                   [[[4.2], [40.2]], [[4.3], [40.3]]],
                                   [[[4.4], [40.4]], [[4.5], [40.5]]]]])
    rhs_batches[1] = tf.constant([[[[[11.0], [110.0]], [[11.1], [110.1]]],
                                   [[[11.2], [110.2]], [[11.3], [110.3]]],
                                   [[[11.4], [110.4]], [[11.5], [110.5]]]],
                                  [[[[12.0], [120.0]], [[12.1], [120.1]]],
                                   [[[12.2], [120.2]], [[12.3], [120.3]]],
                                   [[[12.4], [120.4]], [[12.5], [120.5]]]],
                                  [[[[13.0], [130.0]], [[13.1], [130.1]]],
                                   [[[13.2], [130.2]], [[13.3], [130.3]]],
                                   [[[13.4], [130.4]], [[13.5], [130.5]]]],
                                  [[[[14.0], [140.0]], [[14.1], [140.1]]],
                                   [[[14.2], [140.2]], [[14.3], [140.3]]],
                                   [[[14.4], [140.4]], [[14.5], [140.5]]]]])

    # Shape = [4, 1, 2].
    assignment = tf.constant([[[True, True]], [[True, False]], [[False, True]],
                              [[False, False]]])

    mixed_batches = data_utils.mix_batch(
        lhs_batches, rhs_batches, axis=2, assignment=assignment)
    self.assertLen(mixed_batches, 2)
    self.assertAllEqual(
        mixed_batches[0],
        # Shape = [4, 3, 2, 1].
        np.array([[[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
                  [[[2.0], [12.1]], [[2.2], [12.3]], [[2.4], [12.5]]],
                  [[[13.0], [3.1]], [[13.2], [3.3]], [[13.4], [3.5]]],
                  [[[14.0], [14.1]], [[14.2], [14.3]], [[14.4], [14.5]]]],
                 dtype=np.float32))
    self.assertAllEqual(
        mixed_batches[1],
        # Shape = [4, 3, 2, 2, 1].
        np.array([[[[[1.0], [10.0]], [[1.1], [10.1]]],
                   [[[1.2], [10.2]], [[1.3], [10.3]]],
                   [[[1.4], [10.4]], [[1.5], [10.5]]]],
                  [[[[2.0], [20.0]], [[12.1], [120.1]]],
                   [[[2.2], [20.2]], [[12.3], [120.3]]],
                   [[[2.4], [20.4]], [[12.5], [120.5]]]],
                  [[[[13.0], [130.0]], [[3.1], [30.1]]],
                   [[[13.2], [130.2]], [[3.3], [30.3]]],
                   [[[13.4], [130.4]], [[3.5], [30.5]]]],
                  [[[[14.0], [140.0]], [[14.1], [140.1]]],
                   [[[14.2], [140.2]], [[14.3], [140.3]]],
                   [[[14.4], [140.4]], [[14.5], [140.5]]]]],
                 dtype=np.float32))

  def test_mix_batch_pairs_with_idle_dim(self):
    # Shape = [4, 3, 2, 1].
    lhs_batch = tf.constant([[[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
                             [[[2.0], [2.1]], [[2.2], [2.3]], [[2.4], [2.5]]],
                             [[[3.0], [3.1]], [[3.2], [3.3]], [[3.4], [3.5]]],
                             [[[4.0], [4.1]], [[4.2], [4.3]], [[4.4], [4.5]]]])
    rhs_batch = tf.constant([[[[11.0], [11.1]], [[11.2], [11.3]],
                              [[11.4], [11.5]]],
                             [[[12.0], [12.1]], [[12.2], [12.3]],
                              [[12.4], [12.5]]],
                             [[[13.0], [13.1]], [[13.2], [13.3]],
                              [[13.4], [13.5]]],
                             [[[14.0], [14.1]], [[14.2], [14.3]],
                              [[14.4], [14.5]]]])
    # Shape = [4, 1, 2].
    assignment = tf.constant([[[True, True]], [[True, False]], [[False, True]],
                              [[False, False]]])
    mixed_batch = data_utils.mix_batch([lhs_batch], [rhs_batch],
                                       axis=2,
                                       assignment=assignment)[0]
    self.assertAllEqual(
        mixed_batch,
        np.array([[[[1.0], [1.1]], [[1.2], [1.3]], [[1.4], [1.5]]],
                  [[[2.0], [12.1]], [[2.2], [12.3]], [[2.4], [12.5]]],
                  [[[13.0], [3.1]], [[13.2], [3.3]], [[13.4], [3.5]]],
                  [[[14.0], [14.1]], [[14.2], [14.3]], [[14.4], [14.5]]]],
                 dtype=np.float32))

  def test_mix_batch_random(self):
    # Shape = [8, 7, 6, 5, 4, 3, 2].
    lhs_batch = tf.ones([8, 7, 6, 5, 4, 3, 2])
    rhs_batch = tf.zeros([8, 7, 6, 5, 4, 3, 2])
    mixed_batch = data_utils.mix_batch([lhs_batch], [rhs_batch], axis=3)[0]
    # We only trivially test the shape to make sure the code runs.
    self.assertAllEqual(mixed_batch.shape.as_list(), [8, 7, 6, 5, 4, 3, 2])

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

  def test_merge_dict(self):
    target_dict = {'a': 1, 'b': 2}
    source_dict = {'c': 4}
    data_utils.merge_dict(source_dict, target_dict)
    self.assertDictEqual(target_dict, {'a': 1, 'b': 2, 'c': 4})

    target_dict = {'a': 1, 'b': 2}
    source_dict = {'b': 3, 'c': 4}
    with self.assertRaisesRegexp(ValueError, 'Key conflict: `b`.'):
      data_utils.merge_dict(source_dict, target_dict)


if __name__ == '__main__':
  tf.test.main()
