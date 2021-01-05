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

"""Tests for ...tf3d.layers.sparse_conv_utils."""

import numpy as np
import tensorflow as tf

from tf3d.layers import sparse_voxel_net_utils


class SparseConvUtilsTest(tf.test.TestCase):

  def test_voxel_pooling(self):
    voxel_features = tf.constant([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [2.0, 0.5, 0.5],
                                  [3.0, 0.2, 0.7],
                                  [0.0, 0.0, 1.0],
                                  [0.0, 0.5, 0.7]], dtype=tf.float32)
    voxel_features = tf.stack([voxel_features, voxel_features], axis=0)
    voxel_indices = tf.constant([[0, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 0],
                                 [3, 0, 0],
                                 [3, 1, 0],
                                 [5, 4, 1]], dtype=tf.int32)
    voxel_indices = tf.stack([voxel_indices, voxel_indices], axis=0)
    num_valid_voxels = tf.constant([6, 4], dtype=tf.int32)
    (pooled_voxel_features, pooled_voxel_indices,
     num_valid_pooled_voxels, idx) = sparse_voxel_net_utils.voxel_pooling(
         voxel_features=voxel_features,
         voxel_xyz_indices=voxel_indices,
         num_valid_voxels=num_valid_voxels,
         pooling_size=(2, 2, 2))
    np_pooled_voxel_features = pooled_voxel_features.numpy()
    np_pooled_voxel_indices = pooled_voxel_indices.numpy()
    expected_features_0 = np.array([[2.0, 1.0, 0.5],
                                    [3.0, 0.2, 1.0],
                                    [0.0, 0.5, 0.7]], dtype=np.float32)
    expected_features_1 = np.array([[2.0, 1.0, 0.5],
                                    [3.0, 0.2, 0.7],
                                    [0.0, 0.0, 0.0]], dtype=np.float32)
    expected_indices_0 = np.array([[0, 0, 0],
                                   [1, 0, 0],
                                   [2, 2, 0]], dtype=np.int32)
    expected_indices_1 = np.array([[0, 0, 0],
                                   [1, 0, 0],
                                   [0, 0, 0]], dtype=np.int32)
    expected_num_valid_pooled_voxels = np.array([3, 2], dtype=np.int32)
    expected_idx = np.array([[0, 0, 0, 1, 1, 2],
                             [0, 0, 0, 1, 0, 0]])
    self.assertAllClose(np_pooled_voxel_features[0], expected_features_0)
    self.assertAllClose(np_pooled_voxel_features[1], expected_features_1)
    self.assertAllEqual(np_pooled_voxel_indices[0], expected_indices_0)
    self.assertAllEqual(np_pooled_voxel_indices[1], expected_indices_1)
    self.assertAllEqual(num_valid_pooled_voxels.numpy(),
                        expected_num_valid_pooled_voxels)
    self.assertAllEqual(idx.numpy(), expected_idx)

  def test_voxel_upsampling(self):
    pooled_voxel_features = tf.constant(
        [[[10.0, 0.0, 0.0],
          [0.0, 10.0, 0.0],
          [0.0, 0.0, 10.0]],
         [[5.0, 0.0, 0.0],
          [0.0, 5.0, 0.0],
          [0.0, 0.0, 5.0]]], dtype=tf.float32)
    index_mapping = tf.constant(
        [[0, 1, 2, 2, 1, 0],
         [1, 2, 0, 1, 1, 0]], dtype=tf.int32)
    voxel_features = sparse_voxel_net_utils.voxel_upsampling(
        pooled_voxel_features=pooled_voxel_features,
        index_mapping=index_mapping)
    expected_voxel_features = np.array(
        [[[10.0, 0.0, 0.0],
          [0.0, 10.0, 0.0],
          [0.0, 0.0, 10.0],
          [0.0, 0.0, 10.0],
          [0.0, 10.0, 0.0],
          [10.0, 0.0, 0.0]],
         [[0.0, 5.0, 0.0],
          [0.0, 0.0, 5.0],
          [5.0, 0.0, 0.0],
          [0.0, 5.0, 0.0],
          [0.0, 5.0, 0.0],
          [5.0, 0.0, 0.0]]], dtype=np.float32)
    self.assertAllClose(voxel_features.numpy(), expected_voxel_features)

  def test_pool_features_given_indices(self):
    features = tf.constant([[1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0],
                            [4.0, 0.0, 4.0],
                            [0.0, 5.0, 0.0],
                            [0.0, 0.0, 6.0],
                            [7.0, 7.0, 0.0],
                            [8.0, 0.0, 8.0],
                            [9.0, 9.0, 0.0],
                            [10.0, 10.0, 10.0]], dtype=tf.float32)
    indices = tf.constant([2, 2, 1, 3, 1, 1, 5, 3, 2, 1], dtype=tf.int32)
    pooled_features, segment_ids, num_segments = (
        sparse_voxel_net_utils.pool_features_given_indices(
            features=features,
            indices=indices,
            segment_func=tf.math.unsorted_segment_mean))
    use_external_impl = True

    print('segment ids')
    print(segment_ids.numpy())
    print('pooled features')
    print(pooled_features.numpy())
    if use_external_impl:
      self.assertAllEqual(segment_ids.numpy(),
                          np.array([0, 0, 1, 2, 1, 1, 3, 2, 0, 1]))
      self.assertAllClose(pooled_features.numpy(), np.array([[4.0, 4.0, 1.0],
                                                             [3.25, 4.5, 4.75],
                                                             [6.0, 0.0, 6.0],
                                                             [7.0, 7.0, 0.0]]))
    else:
      self.assertAllEqual(segment_ids.numpy(),
                          np.array([1, 1, 0, 2, 0, 0, 3, 2, 1, 0]))
      self.assertAllClose(pooled_features.numpy(), np.array([[3.25, 4.5, 4.75],
                                                             [4.0, 4.0, 1.0],
                                                             [6.0, 0.0, 6.0],
                                                             [7.0, 7.0, 0.0]]))
    self.assertAllEqual(num_segments.numpy(), np.array(4))

  def test_masked_batch_norm(self):
    batch_norm_fn = sparse_voxel_net_utils.MaskedBatchNorm()
    voxel_features = tf.random.uniform([4, 100, 8],
                                       minval=-2.0,
                                       maxval=2.0,
                                       dtype=tf.float32)
    num_valid_voxels = tf.constant([20, 30, 10, 70], dtype=tf.int32)
    voxel_features_bn = batch_norm_fn([voxel_features, num_valid_voxels],
                                      training=False)
    self.assertAllEqual(voxel_features_bn.shape, [4, 100, 8])

  def test_sparse_conv_block_3d(self):
    sparse_conv_fn = sparse_voxel_net_utils.SparseConvBlock3D(
        num_convolution_channels_list=[16, 8, 4, 2, 1],
        conv_filter_size=3,
        use_batch_norm=True,
        dropout_prob=0.1,
        apply_relu_to_last_conv=True,
        normalize_sparse_conv=True)
    voxel_features = tf.constant([[[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0],
                                   [1.0, 2.0, 3.0]]], dtype=tf.float32)
    voxel_xyz_indices = tf.constant([[[0, 0, 0],
                                      [0, 1, 0],
                                      [1, 0, 0],
                                      [0, 0, 1]]], dtype=tf.int32)
    num_valid_voxels = tf.constant([4], dtype=tf.int32)
    convolved_features = sparse_conv_fn(
        [voxel_features, voxel_xyz_indices, num_valid_voxels], training=False)
    self.assertAllEqual(convolved_features.shape, [1, 4, 1])


if __name__ == '__main__':
  tf.test.main()
