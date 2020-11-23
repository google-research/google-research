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

"""Test sparse_conv ops."""
import numpy as np
import tensorflow as tf

try:
  from python import sparse_conv_ops  # pylint: disable=g-import-not-at-top
except ImportError:
  import sparse_conv_ops  # pylint: disable=g-import-not-at-top


class SparseConvOpTest(tf.test.TestCase):
  """Test sparse_conv ops."""

  def test_spar_conv_op(self):
    voxel_features = tf.constant(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 8.0]]],
        dtype=tf.float32)
    voxel_xyz_indices = tf.constant(
        [[[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]], dtype=tf.int32)
    num_valid_voxels = tf.constant([4], dtype=tf.int32)
    init_value = np.ones([3, 3, 3, 3, 5], np.float32)
    filters = tf.Variable(initial_value=init_value, trainable=True)
    with tf.GradientTape() as g:
      voxel_outputs = sparse_conv_ops.submanifold_sparse_conv3d(
          voxel_xyz_indices, num_valid_voxels, voxel_features, filters)
    print('voxel_outputs:', voxel_outputs)
    self.assertAllEqual(voxel_outputs.shape, [1, 4, 5])
    self.assertAllEqual(
        g.gradient(voxel_outputs, filters).shape, [3, 3, 3, 3, 5])


if __name__ == '__main__':
  tf.test.main()
