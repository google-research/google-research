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

"""Tests for ...tf3d.layers.sparse_voxel_unet."""

import tensorflow as tf

from tf3d.layers import sparse_voxel_unet


class SparseVoxelUnetTest(tf.test.TestCase):

  def test_sparse_voxel_unet(self):
    basenet = sparse_voxel_unet.SparseConvUNet(
        task_names_to_num_output_channels={'feature': 64})
    voxel_features = tf.constant(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]],
        dtype=tf.float32)
    voxel_xyz_indices = tf.constant(
        [[[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]], dtype=tf.int32)
    num_valid_voxels = tf.constant([4], dtype=tf.int32)
    outputs = basenet([voxel_features, voxel_xyz_indices, num_valid_voxels],
                      training=True)

    self.assertAllEqual(outputs['feature'].shape, [1, 4, 64])


if __name__ == '__main__':
  tf.test.main()
