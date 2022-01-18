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

"""Tests for ...utils/metric_learning_losses."""
import numpy as np
import tensorflow as tf
from tf3d.utils import metric_learning_losses as mll


class MetricLearningLossesTest(tf.test.TestCase):

  def get_embedding2d_unit_length(self):
    embedding = tf.constant([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0],
                             [0.8, -0.6, 0.0]], dtype=tf.float32)
    return embedding

  def get_embedding2d_close_to_0(self):
    embedding = tf.constant([[0.01, 0.0, 0.0],
                             [0.0, 0.01, 0.0],
                             [-0.01, 0.0, 0.0],
                             [0.008, -0.006, 0.0]], dtype=tf.float32)
    return embedding

  def get_embedding1(self):
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

  def get_embedding2(self):
    embedding_dim0 = tf.constant([[0.0, 0.1, 0.0, 1.0, 1.0],
                                  [1.0, 0.05, 0.07, 1.2, 1.1],
                                  [-0.1, 0.03, -0.06, 1.4, 0.95],
                                  [0.01, 0.04, 0.08, 1.25, 1.05]],
                                 dtype=tf.float32)
    embedding_dim0 = tf.expand_dims(embedding_dim0, 2)
    embedding_dim1 = tf.constant([[-1.0, -1.1, -0.9, 0.04, 0.09],
                                  [-0.1, -1.05, -1.07, 0.2, -0.1],
                                  [-1.1, -1.03, -0.06, 0.4, 0.05],
                                  [0.01, 0.04, 0.08, 0.15, 0.05]],
                                 dtype=tf.float32)
    embedding_dim1 = tf.expand_dims(embedding_dim1, 2)
    embedding_dim2 = tf.constant([[0.0, 0.1, 0.0, 0.07, 0.1],
                                  [0.0, 0.05, 0.07, -0.2, -0.1],
                                  [0.1, 0.03, 1.06, 0.1, -0.05],
                                  [1.01, 1.04, 1.08, 0.15, 0.05]],
                                 dtype=tf.float32)
    embedding_dim2 = tf.expand_dims(embedding_dim2, 2)
    embedding = tf.concat([embedding_dim0, embedding_dim1, embedding_dim2],
                          axis=2)
    return embedding

  def get_embedding3(self):
    embedding_dim0 = tf.constant([[10.0, 0.1],
                                  [10.0, 0.05]],
                                 dtype=tf.float32)
    embedding_dim0 = tf.expand_dims(embedding_dim0, 2)
    embedding_dim1 = tf.constant([[-0.0, -0.1],
                                  [-0.1, -10.05]],
                                 dtype=tf.float32)
    embedding_dim1 = tf.expand_dims(embedding_dim1, 2)
    embedding_dim2 = tf.constant([[0.0, 10.1],
                                  [0.0, 0.05]],
                                 dtype=tf.float32)
    embedding_dim2 = tf.expand_dims(embedding_dim2, 2)
    embedding = tf.concat([embedding_dim0, embedding_dim1, embedding_dim2],
                          axis=2)
    return embedding

  def get_embedding4(self):
    embedding_dim0 = tf.constant([[0.0, 0.1],
                                  [10.0, 0.05]],
                                 dtype=tf.float32)
    embedding_dim0 = tf.expand_dims(embedding_dim0, 2)
    embedding_dim1 = tf.constant([[-10.0, -0.1],
                                  [-0.1, -10.05]],
                                 dtype=tf.float32)
    embedding_dim1 = tf.expand_dims(embedding_dim1, 2)
    embedding_dim2 = tf.constant([[0.0, 10.1],
                                  [0.0, 0.05]],
                                 dtype=tf.float32)
    embedding_dim2 = tf.expand_dims(embedding_dim2, 2)
    embedding = tf.concat([embedding_dim0, embedding_dim1, embedding_dim2],
                          axis=2)
    return embedding

  def get_labels1(self):
    labels = tf.constant([[0, 0, 0, -1, 1],
                          [0, 0, 0, 1, 1],
                          [0, 0, 2, 2, 1],
                          [2, 2, 2, 2, 1]], dtype=tf.int32)
    return labels

  def get_labels2(self):
    labels = tf.constant([[0, 1],
                          [0, 2]], dtype=tf.int32)
    return labels

  def test_mean_square_regularization_loss(self):
    embedding_unit_length = self.get_embedding2d_unit_length()
    loss1 = mll.regularization_loss(embedding_unit_length, 1.0, 'msq')
    embedding_0 = self.get_embedding2d_close_to_0()
    loss2 = mll.regularization_loss(embedding_0, 1.0, 'msq')
    expected_loss1 = np.array(0.5, dtype=np.float32)
    self.assertAllClose(loss1.numpy(), expected_loss1)
    self.assertLess(loss2.numpy(), loss1.numpy())

  def test_unit_length_regularization_loss(self):
    embedding_unit_length = self.get_embedding2d_unit_length()
    loss1 = mll.regularization_loss(embedding_unit_length, 1.0, 'unit_length')
    embedding_0 = self.get_embedding2d_close_to_0()
    loss2 = mll.regularization_loss(embedding_0, 1.0, 'unit_length')
    expected_loss1 = np.array(0.0, dtype=np.float32)
    self.assertAllClose(loss1.numpy(), expected_loss1)
    self.assertLess(loss1.numpy(), loss2.numpy())

  def test_npair_loss(self):
    embedding1 = tf.reshape(self.get_embedding1(), [20, 3])
    embedding2 = tf.reshape(self.get_embedding2(), [20, 3])
    labels_onehot = tf.one_hot(tf.reshape(self.get_labels1(), [20]), depth=3)
    valid_labels = tf.greater_equal(tf.reshape(self.get_labels1(), [20]), 0)
    embedding1 = tf.boolean_mask(embedding1, valid_labels)
    embedding2 = tf.boolean_mask(embedding2, valid_labels)
    labels_onehot = tf.boolean_mask(labels_onehot, valid_labels)
    loss11 = mll.npair_loss(embedding1, labels_onehot, 'dotproduct', 'softmax')
    loss12 = mll.npair_loss(embedding2, labels_onehot, 'dotproduct', 'softmax')
    loss21 = mll.npair_loss(embedding1, labels_onehot, 'dotproduct', 'sigmoid')
    loss22 = mll.npair_loss(embedding2, labels_onehot, 'dotproduct', 'sigmoid')
    loss31 = mll.npair_loss(embedding1, labels_onehot, 'distance', 'softmax')
    loss32 = mll.npair_loss(embedding2, labels_onehot, 'distance', 'softmax')
    loss41 = mll.npair_loss(embedding1, labels_onehot, 'distance', 'sigmoid')
    loss42 = mll.npair_loss(embedding2, labels_onehot, 'distance', 'sigmoid')
    self.assertLess(loss11.numpy(), loss12.numpy())
    self.assertLess(loss21.numpy(), loss22.numpy())
    self.assertLess(loss31.numpy(), loss32.numpy())
    self.assertLess(loss41.numpy(), loss42.numpy())

  def test_instance_embedding_npair_loss(self):
    embedding1 = self.get_embedding1()
    embedding2 = self.get_embedding2()
    labels = self.get_labels1()
    loss1 = mll.instance_embedding_npair_loss(embedding1, labels, 10, 100)
    loss2 = mll.instance_embedding_npair_loss(embedding2, labels, 10, 100)
    self.assertLess(loss1.numpy(), loss2.numpy())

  def test_instance_embedding_iou_loss(self):
    embedding1 = self.get_embedding1()
    embedding2 = self.get_embedding2()
    labels = self.get_labels1()
    loss1 = mll.instance_embedding_iou_loss(embedding1, labels, 10)
    loss2 = mll.instance_embedding_iou_loss(embedding2, labels, 10)
    self.assertLess(loss1.numpy(), loss2.numpy())

  def test_instance_embedding_npair_random_center_loss(self):
    embedding1 = self.get_embedding1()
    embedding2 = self.get_embedding2()
    labels = self.get_labels1()
    loss11 = mll.instance_embedding_npair_random_center_loss(
        embedding1, labels, 'dotproduct', 'softmax')
    loss12 = mll.instance_embedding_npair_random_center_loss(
        embedding2, labels, 'dotproduct', 'softmax')
    loss21 = mll.instance_embedding_npair_random_center_loss(
        embedding1, labels, 'dotproduct', 'sigmoid')
    loss22 = mll.instance_embedding_npair_random_center_loss(
        embedding2, labels, 'dotproduct', 'sigmoid')
    loss31 = mll.instance_embedding_npair_random_center_loss(
        embedding1, labels, 'distance', 'softmax')
    loss32 = mll.instance_embedding_npair_random_center_loss(
        embedding2, labels, 'distance', 'softmax')
    loss41 = mll.instance_embedding_npair_random_center_loss(
        embedding1, labels, 'distance', 'sigmoid')
    loss42 = mll.instance_embedding_npair_random_center_loss(
        embedding2, labels, 'distance', 'sigmoid')
    self.assertLess(loss11.numpy(), loss12.numpy())
    self.assertLess(loss21.numpy(), loss22.numpy())
    self.assertLess(loss31.numpy(), loss32.numpy())
    self.assertLess(loss41.numpy(), loss42.numpy())

  def test_instance_embedding_npair_random_center_random_sample_loss(self):
    embedding1 = self.get_embedding1()
    embedding2 = self.get_embedding2()
    labels = self.get_labels1()
    loss11 = mll.instance_embedding_npair_random_center_random_sample_loss(
        embedding1, labels, 20, 'dotproduct', 'softmax')
    loss12 = mll.instance_embedding_npair_random_center_random_sample_loss(
        embedding2, labels, 20, 'dotproduct', 'softmax')
    loss21 = mll.instance_embedding_npair_random_center_random_sample_loss(
        embedding1, labels, 20, 'dotproduct', 'sigmoid')
    loss22 = mll.instance_embedding_npair_random_center_random_sample_loss(
        embedding2, labels, 20, 'dotproduct', 'sigmoid')
    loss31 = mll.instance_embedding_npair_random_center_random_sample_loss(
        embedding1, labels, 20, 'distance', 'softmax')
    loss32 = mll.instance_embedding_npair_random_center_random_sample_loss(
        embedding2, labels, 20, 'distance', 'softmax')
    loss41 = mll.instance_embedding_npair_random_center_random_sample_loss(
        embedding1, labels, 20, 'distance', 'sigmoid')
    loss42 = mll.instance_embedding_npair_random_center_random_sample_loss(
        embedding2, labels, 20, 'distance', 'sigmoid')
    self.assertLess(loss11.numpy(), loss12.numpy())
    self.assertLess(loss21.numpy(), loss22.numpy())
    self.assertLess(loss31.numpy(), loss32.numpy())
    self.assertLess(loss41.numpy(), loss42.numpy())

  def test_instance_embedding_npair_random_sample_loss(self):
    embedding1 = self.get_embedding1()
    embedding2 = self.get_embedding2()
    labels = self.get_labels1()
    loss11 = mll.instance_embedding_npair_random_sample_loss(
        embedding1, labels, 20, 'dotproduct', 'softmax')
    loss12 = mll.instance_embedding_npair_random_sample_loss(
        embedding2, labels, 20, 'dotproduct', 'softmax')
    loss21 = mll.instance_embedding_npair_random_sample_loss(
        embedding1, labels, 20, 'dotproduct', 'sigmoid')
    loss22 = mll.instance_embedding_npair_random_sample_loss(
        embedding2, labels, 20, 'dotproduct', 'sigmoid')
    loss31 = mll.instance_embedding_npair_random_sample_loss(
        embedding1, labels, 20, 'distance', 'softmax')
    loss32 = mll.instance_embedding_npair_random_sample_loss(
        embedding2, labels, 20, 'distance', 'softmax')
    loss41 = mll.instance_embedding_npair_random_sample_loss(
        embedding1, labels, 20, 'distance', 'sigmoid')
    loss42 = mll.instance_embedding_npair_random_sample_loss(
        embedding2, labels, 20, 'distance', 'sigmoid')
    self.assertLess(loss11.numpy(), loss12.numpy())
    self.assertLess(loss21.numpy(), loss22.numpy())
    self.assertLess(loss31.numpy(), loss32.numpy())
    self.assertLess(loss41.numpy(), loss42.numpy())

  def test_get_instance_embedding_loss(self):
    embedding1 = self.get_embedding1()
    embedding2 = self.get_embedding2()
    labels = self.get_labels1()
    l11 = mll.get_instance_embedding_loss(embedding1, 'npair', labels, 100, 100,
                                          10)
    l12 = mll.get_instance_embedding_loss(embedding2, 'npair', labels, 100, 100,
                                          10)
    l21 = mll.get_instance_embedding_loss(embedding1, 'npair_r_c', labels, 10,
                                          100, 10)
    l22 = mll.get_instance_embedding_loss(embedding2, 'npair_r_c', labels, 10,
                                          100, 10)
    l31 = mll.get_instance_embedding_loss(embedding1, 'npair_r_c_r_s', labels,
                                          10, 100, 10)
    l32 = mll.get_instance_embedding_loss(embedding2, 'npair_r_c_r_s', labels,
                                          10, 100, 10)
    l41 = mll.get_instance_embedding_loss(embedding1, 'npair_r_s', labels, 10,
                                          100, 10)
    l42 = mll.get_instance_embedding_loss(embedding2, 'npair_r_s', labels, 10,
                                          100, 10)
    l51 = mll.get_instance_embedding_loss(embedding1, 'iou', labels, 10, 100,
                                          10)
    l52 = mll.get_instance_embedding_loss(embedding2, 'iou', labels, 10, 100,
                                          10)
    self.assertLess(l11.numpy(), l12.numpy())
    self.assertLess(l21.numpy(), l22.numpy())
    self.assertLess(l31.numpy(), l32.numpy())
    self.assertLess(l41.numpy(), l42.numpy())
    self.assertLess(l51.numpy(), l52.numpy())

  def testGetInstanceEmbeddingLossBatch(self):
    embedding1 = self.get_embedding1()
    embedding1 = tf.stack([embedding1, embedding1, embedding1])
    embedding2 = self.get_embedding2()
    embedding2 = tf.stack([embedding2, embedding2, embedding2])
    labels = self.get_labels1()
    labels = tf.stack([labels, labels, labels])
    l11 = mll.get_instance_embedding_loss(embedding1, 'npair', labels, 100, 100,
                                          10)
    l12 = mll.get_instance_embedding_loss(embedding2, 'npair', labels, 100, 100,
                                          10)
    l21 = mll.get_instance_embedding_loss(embedding1, 'npair_r_c', labels, 10,
                                          100, 10)
    l22 = mll.get_instance_embedding_loss(embedding2, 'npair_r_c', labels, 10,
                                          100, 10)
    l31 = mll.get_instance_embedding_loss(embedding1, 'npair_r_c_r_s', labels,
                                          10, 100, 10)
    l32 = mll.get_instance_embedding_loss(embedding2, 'npair_r_c_r_s', labels,
                                          10, 100, 10)
    l41 = mll.get_instance_embedding_loss(embedding1, 'npair_r_s', labels, 10,
                                          100, 10)
    l42 = mll.get_instance_embedding_loss(embedding2, 'npair_r_s', labels, 10,
                                          100, 10)
    l51 = mll.get_instance_embedding_loss(embedding1, 'iou', labels, 10, 100,
                                          10)
    l52 = mll.get_instance_embedding_loss(embedding2, 'iou', labels, 10, 100,
                                          10)
    self.assertLess(l11.numpy(), l12.numpy())
    self.assertLess(l21.numpy(), l22.numpy())
    self.assertLess(l31.numpy(), l32.numpy())
    self.assertLess(l41.numpy(), l42.numpy())
    self.assertLess(l51.numpy(), l52.numpy())


if __name__ == '__main__':
  tf.test.main()
