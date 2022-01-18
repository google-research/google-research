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

"""Tests algorithm functions."""

import tensorflow as tf

from poem.core import keypoint_utils
from poem.cv_mim import algorithms
from poem.cv_mim import models


class AlgorithmsTest(tf.test.TestCase):

  def test_compute_positive_indicator_matrix(self):
    anchors = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    matches = tf.constant([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
    indicator_matrix = algorithms.compute_positive_indicator_matrix(
        anchors, matches,
        distance_fn=tf.math.squared_difference,
        max_positive_distance=5.0)

    self.assertAllEqual(
        indicator_matrix,
        [[[0., 0.], [0., 1.], [1., 1.]], [[0., 1.], [1., 1.], [1., 0.]]])

  def test_compute_squared_positive_indicator_matrix(self):
    anchors = tf.random.uniform(shape=(16, 13, 3))
    indicator_matrix = algorithms.compute_positive_indicator_matrix(
        anchors, anchors,
        distance_fn=keypoint_utils.compute_procrustes_aligned_mpjpes,
        max_positive_distance=0.1)

    self.assertAllEqual(indicator_matrix, tf.transpose(indicator_matrix))
    self.assertAllEqual(tf.linalg.diag_part(indicator_matrix), tf.ones((16)))

  def test_get_encoder_shapes(self):
    input_features = tf.zeros([4, 6], tf.float32)
    model = algorithms.get_encoder(embedding_dim=128)

    outputs = model(input_features)
    self.assertAllEqual(outputs[0].shape, [4, 128])
    self.assertAllEqual(outputs[1]['flatten'].shape, [4, 6])
    self.assertAllEqual(outputs[1]['fc0'].shape, [4, 1024])
    self.assertAllEqual(outputs[1]['res_fcs1'].shape, [4, 1024])
    self.assertAllEqual(outputs[1]['res_fcs2'].shape, [4, 1024])
    self.assertAllEqual(outputs[1]['embedder'].shape, [4, 128])

  def test_autoencoder_shape(self):
    input_features = tf.zeros([4, 8, 2], tf.float32)
    model = algorithms.AutoEncoder(
        pose_embedding_dim=16,
        view_embedding_dim=16,
        embedder_type=models.TYPE_EMBEDDER_POINT)
    outputs = model(input_features)

    self.assertEqual(len(outputs), 2)
    self.assertAllEqual(outputs[0].shape, [4, 16])
    self.assertAllEqual(outputs[1].shape, [4, 16])

  def test_infomix_shape(self):
    input_features = tf.zeros([4, 6], tf.float32)
    model = algorithms.InfoMix(
        embedding_dim=128, embedder_type=models.TYPE_EMBEDDER_POINT)
    outputs = model(input_features)

    self.assertAllEqual(outputs[0].shape, [4, 1, 128])
    self.assertAllEqual(outputs[1].shape, [4, 1, 128])

  def test_infodisentangle_shape(self):
    input_features = tf.zeros([4, 6], tf.float32)

    model = algorithms.InfoDisentangle(
        pose_embedding_dim=16,
        view_embedding_dim=16,
        fusion_op_type=algorithms.TYPE_FUSION_OP_CAT,
        embedder_type=models.TYPE_EMBEDDER_POINT)
    outputs = model(input_features)
    self.assertAllEqual(outputs[0].shape, [4, 1, 32])
    self.assertAllEqual(outputs[1].shape, [4, 1, 32])

    model = algorithms.InfoDisentangle(
        pose_embedding_dim=16,
        view_embedding_dim=16,
        fusion_op_type=algorithms.TYPE_FUSION_OP_POE,
        embedder_type=models.TYPE_EMBEDDER_POINT)
    outputs = model(input_features)
    self.assertAllEqual(outputs[0].shape, [4, 1, 32])
    self.assertAllEqual(outputs[1].shape, [4, 1, 16])

    model = algorithms.InfoDisentangle(
        pose_embedding_dim=16,
        view_embedding_dim=16,
        fusion_op_type=algorithms.TYPE_FUSION_OP_MOE,
        embedder_type=models.TYPE_EMBEDDER_POINT)
    outputs = model(input_features)
    self.assertAllEqual(outputs[0].shape, [4, 1, 32])
    self.assertAllEqual(outputs[1].shape, [4, 1, 16])

  def test_infomax_shape(self):
    input_features = tf.zeros([4, 6], tf.float32)

    model = algorithms.InfoMax(
        pose_embedding_dim=16,
        view_embedding_dim=16,
        fusion_op_type=algorithms.TYPE_FUSION_OP_CAT,
        embedder_type=models.TYPE_EMBEDDER_POINT)
    outputs = model(input_features)
    self.assertEqual(len(outputs), 3)
    self.assertAllEqual(outputs[0].shape, [4, 1, 16])
    self.assertAllEqual(outputs[1].shape, [4, 1, 16])
    self.assertAllEqual(outputs[2].shape, [4, 1, 32])

    model = algorithms.InfoMax(
        pose_embedding_dim=16,
        view_embedding_dim=16,
        fusion_op_type=algorithms.TYPE_FUSION_OP_POE,
        embedder_type=models.TYPE_EMBEDDER_POINT)
    outputs = model(input_features)
    self.assertEqual(len(outputs), 3)
    self.assertAllEqual(outputs[0].shape, [4, 1, 16])
    self.assertAllEqual(outputs[1].shape, [4, 1, 16])
    self.assertAllEqual(outputs[2].shape, [4, 1, 16])

    model = algorithms.InfoMax(
        pose_embedding_dim=16,
        view_embedding_dim=16,
        fusion_op_type=algorithms.TYPE_FUSION_OP_MOE,
        embedder_type=models.TYPE_EMBEDDER_POINT)
    outputs = model(input_features)
    self.assertEqual(len(outputs), 3)
    self.assertAllEqual(outputs[0].shape, [4, 1, 16])
    self.assertAllEqual(outputs[1].shape, [4, 1, 16])
    self.assertAllEqual(outputs[2].shape, [4, 1, 16])


if __name__ == '__main__':
  tf.test.main()
