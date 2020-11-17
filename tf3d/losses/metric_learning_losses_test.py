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

"""Tests for research.vale.masternet.threed.instance_segmentation.losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf3d import standard_fields
from tf3d.losses import metric_learning_losses


class LossesTest(tf.test.TestCase):

  def _get_embeddings(self):
    embeddings1 = tf.constant([[[1.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0]]], dtype=tf.float32)
    embeddings2 = tf.constant([[[1.0, 1.0, 0.0],
                                [1.0, 1.0, 0.0],
                                [0.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0]]], dtype=tf.float32)
    embeddings3 = tf.constant([[[0.5, 0.0, 0.0],
                                [0.5, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.0, 0.5],
                                [0.0, 0.0, 0.5]]], dtype=tf.float32)
    instance_ids = tf.constant([[0, 0, 1, 1, 2, 2]])
    return embeddings1, embeddings2, embeddings3, instance_ids

  def test_npair_loss_func_dotproduct(self):
    embeddings1, embeddings2, embeddings3, instance_ids = self._get_embeddings()
    loss1 = metric_learning_losses.npair_loss_func(
        embeddings=embeddings1,
        instance_ids=instance_ids,
        num_samples=40,
        valid_mask=None,
        max_instance_id=5,
        similarity_strategy='dotproduct',
        loss_strategy='softmax')
    loss2 = metric_learning_losses.npair_loss_func(
        embeddings=embeddings2,
        instance_ids=instance_ids,
        num_samples=40,
        valid_mask=None,
        max_instance_id=5,
        similarity_strategy='dotproduct',
        loss_strategy='softmax')
    loss3 = metric_learning_losses.npair_loss_func(
        embeddings=embeddings3,
        instance_ids=instance_ids,
        num_samples=40,
        valid_mask=None,
        max_instance_id=5,
        similarity_strategy='dotproduct',
        loss_strategy='softmax')
    self.assertLessEqual(loss1.numpy(), loss2.numpy())
    self.assertLessEqual(loss1.numpy(), loss3.numpy())

  def test_npair_loss_distance(self):
    embeddings1, embeddings2, embeddings3, instance_ids = self._get_embeddings()
    loss1 = metric_learning_losses.npair_loss_func(
        embeddings=embeddings1,
        instance_ids=instance_ids,
        num_samples=40,
        valid_mask=None,
        max_instance_id=5,
        similarity_strategy='distance',
        loss_strategy='softmax')
    loss2 = metric_learning_losses.npair_loss_func(
        embeddings=embeddings2,
        instance_ids=instance_ids,
        num_samples=40,
        valid_mask=None,
        max_instance_id=5,
        similarity_strategy='distance',
        loss_strategy='softmax')
    loss3 = metric_learning_losses.npair_loss_func(
        embeddings=embeddings3,
        instance_ids=instance_ids,
        num_samples=40,
        valid_mask=None,
        max_instance_id=5,
        similarity_strategy='distance',
        loss_strategy='softmax')
    self.assertLessEqual(loss1.numpy(), loss2.numpy())
    self.assertLessEqual(loss1.numpy(), loss3.numpy())

  def test_npair_loss(self):
    embeddings1, embeddings2, embeddings3, instance_ids = self._get_embeddings()
    inputs = {
        standard_fields.InputDataFields.object_instance_id_voxels:
            instance_ids,
        standard_fields.InputDataFields.num_valid_voxels:
            tf.constant([5], dtype=tf.int32),
    }
    outputs1 = {
        standard_fields.DetectionResultFields.instance_embedding_voxels:
            embeddings1,
    }
    outputs2 = {
        standard_fields.DetectionResultFields.instance_embedding_voxels:
            embeddings2,
    }
    outputs3 = {
        standard_fields.DetectionResultFields.instance_embedding_voxels:
            embeddings3,
    }
    loss1 = metric_learning_losses.npair_loss(
        inputs=inputs,
        outputs=outputs1,
        num_samples=40,
        max_instance_id=5,
        similarity_strategy='distance',
        loss_strategy='softmax')
    loss2 = metric_learning_losses.npair_loss(
        inputs=inputs,
        outputs=outputs2,
        num_samples=40,
        max_instance_id=5,
        similarity_strategy='distance',
        loss_strategy='softmax')
    loss3 = metric_learning_losses.npair_loss(
        inputs=inputs,
        outputs=outputs3,
        num_samples=40,
        max_instance_id=5,
        similarity_strategy='distance',
        loss_strategy='softmax')
    self.assertLessEqual(loss1.numpy(), loss2.numpy())
    self.assertLessEqual(loss1.numpy(), loss3.numpy())

  def test_embedding_regularization_loss(self):
    embeddings = tf.constant([[[1.0, 0.0, 0.0],
                               [0.2, 0.2, 0.2],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.5, 0.5, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]]], dtype=tf.float32)
    instance_ids = tf.constant([[0, 1, 2, 3, 1, 0, 0, 0]], dtype=tf.int32)
    inputs = {
        standard_fields.InputDataFields.object_instance_id_voxels:
            instance_ids,
        standard_fields.InputDataFields.num_valid_voxels:
            tf.constant([5], dtype=tf.int32),
    }
    outputs = {
        standard_fields.DetectionResultFields.instance_embedding_voxels:
            embeddings,
    }
    loss = metric_learning_losses.embedding_regularization_loss(
        inputs=inputs, outputs=outputs)
    self.assertGreater(loss.numpy(), 0.0)


if __name__ == '__main__':
  tf.test.main()
