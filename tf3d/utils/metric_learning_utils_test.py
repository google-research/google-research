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

"""Tests for instance_segmentation.core.instance_segment_ops."""
import numpy as np
import tensorflow as tf
from tf3d.utils import metric_learning_utils


class EmbeddingUtilsTest(tf.test.TestCase):

  def get_embedding(self):
    embedding_dim0 = tf.constant([[0.0, 0.1, 0.0, 1.0, 1.0],
                                  [0.0, 0.05, 0.07, 1.2, 1.1],
                                  [-0.1, 0.03, -0.06, 0.4, 0.95],
                                  [0.01, 0.04, 0.08, 0.25, 1.05]],
                                 dtype=tf.float32)
    embedding_dim0 = tf.expand_dims(embedding_dim0, 2)
    embedding_dim1 = tf.constant([[-1.0, -1.1, -0.9, 0.04, 0.09],
                                  [-1.1, -1.05, -1.07, 0.2, -0.1],
                                  [-1.1, -1.03, -0.06, 0.4, 0.05],
                                  [0.01, 0.04, 0.08, 0.15, 0.05]],
                                 dtype=tf.float32)
    embedding_dim1 = tf.expand_dims(embedding_dim1, 2)
    embedding_dim2 = tf.constant([[0.0, 0.1, 0.0, 0.07, 0.1],
                                  [0.0, 0.05, 0.07, -0.2, -0.1],
                                  [0.1, 0.03, 1.06, 1.1, -0.05],
                                  [1.01, 1.04, 1.08, 1.15, 0.05]],
                                 dtype=tf.float32)
    embedding_dim2 = tf.expand_dims(embedding_dim2, 2)
    embedding = tf.concat([embedding_dim0, embedding_dim1, embedding_dim2],
                          axis=2)
    return embedding

  def test_embedding_to_cluster_centers_random(self):
    embedding = self.get_embedding()
    embedding = tf.reshape(embedding, [20, -1])
    centers, indices = (
        metric_learning_utils.embedding_to_cluster_centers_random(embedding, 3))
    expected_centers = tf.gather(embedding, indices)
    np_centers_shape_expected = np.array([6, 3], dtype=np.int32)
    self.assertAllEqual(centers.shape, np_centers_shape_expected)
    self.assertAllClose(centers.numpy(), expected_centers.numpy())

  def test_embedding_to_cluster_centers_kpp(self):
    embedding = self.get_embedding()
    embedding = tf.reshape(embedding, [20, -1])
    centers, indices = metric_learning_utils.embedding_to_cluster_centers_kpp(
        embedding, 3)
    expected_centers = tf.gather(embedding, indices)
    np_centers_shape_expected = np.array([6, 3], dtype=np.int32)
    self.assertAllEqual(centers.shape, np_centers_shape_expected)
    self.assertAllClose(centers.numpy(), expected_centers.numpy())

  def test_embedding_to_cluster_centers_given_scores(self):
    embedding = self.get_embedding()
    embedding = tf.reshape(embedding, [20, -1])
    scores = tf.random.uniform([20], maxval=1.0, dtype=tf.float32)
    centers, indices = metric_learning_utils.embedding_to_cluster_centers_given_scores(
        embedding, scores, 3)
    expected_centers = tf.gather(embedding, indices)
    np_centers_shape_expected = np.array([6, 3], dtype=np.int32)
    self.assertAllEqual(centers.shape, np_centers_shape_expected)
    self.assertAllClose(centers.numpy(), expected_centers.numpy())

  def test_embedding_centers_to_soft_masks_relative_dot_product(self):
    embedding = tf.constant([[1.0, 0.0],
                             [0.9, 0.01],
                             [0.01, 1.0],
                             [0.0, 1.0]], dtype=tf.float32)
    centers = tf.constant([[1.0, 0.0],
                           [0.0, 1.0]], dtype=tf.float32)
    soft_masks = metric_learning_utils.embedding_centers_to_soft_masks_by_relative_dot_product(
        embedding, centers)
    expected_soft_masks = tf.constant([[1.0, 0.90483743, 0.3715767, 0.36787945],
                                       [0.36787945, 0.3715767, 1.0, 1.0]],
                                      dtype=tf.float32)
    self.assertAllClose(soft_masks.numpy(), expected_soft_masks.numpy())

  def test_embedding_centers_to_soft_masks_dot_product(self):
    embedding = tf.constant([[1.0, 0.0],
                             [0.9, 0.01],
                             [0.01, 1.0],
                             [0.0, 1.0]], dtype=tf.float32)
    centers = tf.constant([[1.0, 0.0],
                           [0.0, 1.0]], dtype=tf.float32)
    soft_masks = metric_learning_utils.embedding_centers_to_soft_masks_by_dot_product(
        embedding, centers)
    expected_soft_masks = tf.constant([[0.731059, 0.710949, 0.5025, 0.5],
                                       [0.5, 0.5025, 0.731059, 0.731059]],
                                      dtype=tf.float32)
    self.assertAllClose(soft_masks.numpy(), expected_soft_masks.numpy())

  def test_embedding_centers_to_soft_masks_by_distance(self):
    embedding = tf.constant([[1.0, 0.0],
                             [0.9, 0.01],
                             [0.01, 1.0],
                             [0.0, 1.0]], dtype=tf.float32)
    centers = tf.constant([[1.0, 0.0],
                           [0.0, 1.0]], dtype=tf.float32)
    soft_masks = metric_learning_utils.embedding_centers_to_soft_masks_by_distance(
        embedding, centers)
    expected_soft_masks = tf.constant([[1.0, 0.99495, 0.242616, 0.238406],
                                       [0.238406, 0.286121, 0.99995, 1.0]],
                                      dtype=tf.float32)
    self.assertAllClose(soft_masks.numpy(), expected_soft_masks.numpy())

  def test_embedding_centers_to_soft_masks_by_distance2(self):
    embedding = tf.constant([[1.0, 0.0],
                             [0.9, 0.01],
                             [0.01, 1.0],
                             [0.0, 1.0]], dtype=tf.float32)
    centers = tf.constant([[1.0, 0.0],
                           [0.0, 1.0]], dtype=tf.float32)
    soft_masks = metric_learning_utils.embedding_centers_to_soft_masks_by_distance2(
        embedding, centers)
    expected_soft_masks = tf.constant([[1.0, 0.999899, 0.980199, 0.98],
                                       [0.98, 0.982099, 0.999999, 1.0]],
                                      dtype=tf.float32)
    self.assertAllClose(soft_masks.numpy(), expected_soft_masks.numpy())

  def test_embedding_centers_to_soft_masks_by_dot_product(self):
    embedding = tf.constant([[1.0, 0.0],
                             [0.9, 0.01],
                             [0.01, 1.0],
                             [0.0, 1.0]], dtype=tf.float32)
    centers = tf.constant([[1.0, 0.0],
                           [0.0, 1.0]], dtype=tf.float32)
    soft_masks = metric_learning_utils.embedding_centers_to_soft_masks_by_dot_product(
        embedding, centers)
    expected_soft_masks = tf.constant([[0.731059, 0.710949, 0.5025, 0.5],
                                       [0.5, 0.5025, 0.731059, 0.731059]],
                                      dtype=tf.float32)
    self.assertAllClose(soft_masks.numpy(), expected_soft_masks.numpy())

  def test_get_similarity_between_corresponding_embedding_vectors(self):
    embedding1 = tf.constant([[1.0, 0.0],
                              [0.9, 0.01],
                              [0.01, 1.0],
                              [0.0, -1.0]], dtype=tf.float32)
    embedding2 = tf.constant([[1.0, 0.4],
                              [0.9, 0.21],
                              [0.11, 0.8],
                              [0.0, -0.8]], dtype=tf.float32)
    similarity1 = metric_learning_utils.get_similarity_between_corresponding_embedding_vectors(
        embedding1, embedding2, 'dotproduct')
    expected_similarity1 = tf.constant([[1.0],
                                        [0.81209993],
                                        [0.80110002],
                                        [0.80000001]], dtype=tf.float32)
    similarity2 = metric_learning_utils.get_similarity_between_corresponding_embedding_vectors(
        embedding1, embedding2, 'distance')
    expected_similarity2 = tf.constant([[-0.16],
                                        [-0.04],
                                        [-0.05],
                                        [-0.04]], dtype=tf.float32)
    self.assertAllClose(similarity1.numpy(), expected_similarity1.numpy())
    self.assertAllClose(similarity2.numpy(), expected_similarity2.numpy())

  def test_embedding_centers_to_logits(self):
    embedding = tf.constant([[1.0, 0.0],
                             [0.9, 0.01],
                             [0.01, 1.0],
                             [0.0, -1.0]], dtype=tf.float32)
    centers = tf.constant([[1.0, 0.0],
                           [0.0, 1.0]], dtype=tf.float32)
    logits1 = metric_learning_utils.embedding_centers_to_logits(
        embedding, centers, 'dotproduct')
    expected_logits1 = tf.constant([[1.0, 0.89999998, 0.01, 0.0],
                                    [0.0, 0.01, 1.0, -1.0]], dtype=tf.float32)
    logits2 = metric_learning_utils.embedding_centers_to_logits(
        embedding, centers, 'distance')
    expected_logits2 = tf.constant(
        [[1.19209290e-07, -1.00998878e-02, -1.98009968, -1.99999988],
         [-1.99999988, -1.79009986, -9.96589661e-05, -4]], dtype=tf.float32)
    self.assertAllClose(logits1.numpy(), expected_logits1.numpy())
    self.assertAllClose(logits2.numpy(), expected_logits2.numpy())

  def test_embedding_centers_to_soft_masks_given_similarity_strategy(self):
    embedding = tf.constant([[1.0, 0.0],
                             [0.9, 0.01],
                             [0.01, 1.0],
                             [0.0, -1.0]], dtype=tf.float32)
    centers = tf.constant([[1.0, 0.0],
                           [0.0, 1.0]], dtype=tf.float32)
    masks1 = metric_learning_utils.embedding_centers_to_soft_masks(
        embedding, centers, 'dotproduct')
    expected_masks1 = tf.constant([[0.7310586, 0.71094948, 0.5025, 0.5],
                                   [0.5, 0.5025, 0.7310586, 0.26894143]],
                                  dtype=tf.float32)
    masks2 = metric_learning_utils.embedding_centers_to_soft_masks(
        embedding, centers, 'distance')
    expected_masks2 = tf.constant(
        [[1.0, 0.99495012, 0.24261642, 0.23840588],
         [0.23840588, 0.28612095, 0.99995017, 0.03597248]],
        dtype=tf.float32)
    self.assertAllClose(masks1.numpy(), expected_masks1.numpy())
    self.assertAllClose(masks2.numpy(), expected_masks2.numpy())


if __name__ == '__main__':
  tf.test.main()
