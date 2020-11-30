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

"""Tests for ...tf3d.losses.classification_losses."""
import tensorflow as tf
from tf3d import standard_fields
from tf3d.losses import classification_losses


class ClassificationLossesTest(tf.test.TestCase):

  def test_classification_loss_scalar_weight_scales_inputs(self):
    n = 12
    num_classes = 5
    weight = 4.0
    logits = tf.random.uniform((n, num_classes), maxval=1)
    labels = tf.cast(
        tf.math.round(tf.random.uniform(
            (n, 1), minval=1, maxval=num_classes)) - 1,
        dtype=tf.int32)
    outputs = {
        standard_fields.DetectionResultFields.object_semantic_voxels: logits
    }
    inputs1 = {
        standard_fields.InputDataFields.object_class_voxels: labels,
        standard_fields.InputDataFields.voxel_loss_weights: 1.0,
    }
    inputs2 = {
        standard_fields.InputDataFields.object_class_voxels: labels,
        standard_fields.InputDataFields.voxel_loss_weights: weight,
    }
    loss_unscaled = classification_losses.classification_loss(inputs1, outputs)
    loss_scaled = classification_losses.classification_loss(inputs2, outputs)
    self.assertEqual(loss_unscaled.numpy() * weight, loss_scaled.numpy())

  def test_classification_loss_scalar_weight_scales_inputs_batch(self):
    batch_size = 4
    n = 12
    num_classes = 5
    weight = 4.0
    logits = tf.random.uniform((batch_size, n, num_classes), maxval=1)
    labels = tf.cast(
        tf.math.round(
            tf.random.uniform(
                (batch_size, n, 1), minval=1, maxval=num_classes)) - 1,
        dtype=tf.int32)
    outputs = {
        standard_fields.DetectionResultFields.object_semantic_voxels: logits
    }
    inputs1 = {
        standard_fields.InputDataFields.object_class_voxels:
            labels,
        standard_fields.InputDataFields.voxel_loss_weights:
            1.0,
        standard_fields.InputDataFields.num_valid_voxels:
            tf.ones([batch_size], dtype=tf.int32) * n,
    }
    inputs2 = {
        standard_fields.InputDataFields.object_class_voxels:
            labels,
        standard_fields.InputDataFields.voxel_loss_weights:
            weight,
        standard_fields.InputDataFields.num_valid_voxels:
            tf.ones([batch_size], dtype=tf.int32) * n,
    }
    loss_unscaled = classification_losses.classification_loss(inputs1, outputs)
    loss_scaled = classification_losses.classification_loss(inputs2, outputs)
    self.assertEqual(loss_unscaled.numpy() * weight, loss_scaled.numpy())

  def test_box_classification_loss_relative(self):
    gt_classes = tf.reshape(tf.constant([1, 1], dtype=tf.int32), [2, 1])
    gt_length = tf.reshape(tf.constant([1, 1], dtype=tf.float32), [2, 1])
    gt_height = tf.reshape(tf.constant([1, 1], dtype=tf.float32), [2, 1])
    gt_width = tf.reshape(tf.constant([1, 1], dtype=tf.float32), [2, 1])
    gt_center = tf.reshape(
        tf.constant([1, 1, 1, 1, 1, 1], dtype=tf.float32), [2, 3])
    gt_rotation_matrix = tf.tile(tf.expand_dims(tf.eye(3), axis=0), [2, 1, 1])
    logits1 = tf.reshape(
        tf.constant([[-2.0, 2.0, -3.0, -2.0, 0.0],
                     [-2.0, 2.0, -3.0, -2.0, 0.0]], dtype=tf.float32),
        [2, 5])
    logits2 = tf.reshape(
        tf.constant([[-2.0, 0.0, -3.0, -2.0, 2.0],
                     [-2.0, 0.0, -3.0, -2.0, 2.0]], dtype=tf.float32),
        [2, 5])
    gt_instance_ids = tf.reshape(tf.constant([1, 1], dtype=tf.int32), [2, 1])
    inputs = {
        standard_fields.InputDataFields.num_valid_voxels:
            tf.constant([2, 2], dtype=tf.int32),
        standard_fields.InputDataFields.object_class_voxels:
            tf.stack([gt_classes, gt_classes], axis=0),
        standard_fields.InputDataFields.object_length_voxels:
            tf.stack([gt_length, gt_length], axis=0),
        standard_fields.InputDataFields.object_height_voxels:
            tf.stack([gt_height, gt_height], axis=0),
        standard_fields.InputDataFields.object_width_voxels:
            tf.stack([gt_width, gt_width], axis=0),
        standard_fields.InputDataFields.object_center_voxels:
            tf.stack([gt_center, gt_center], axis=0),
        standard_fields.InputDataFields.object_rotation_matrix_voxels:
            tf.stack([gt_rotation_matrix, gt_rotation_matrix], axis=0),
        standard_fields.InputDataFields.object_instance_id_voxels:
            tf.stack([gt_instance_ids, gt_instance_ids], axis=0),
    }
    outputs1 = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            tf.stack([logits1, logits1], axis=0),
        standard_fields.DetectionResultFields.object_length_voxels:
            tf.stack([gt_length, gt_length], axis=0),
        standard_fields.DetectionResultFields.object_height_voxels:
            tf.stack([gt_height, gt_height], axis=0),
        standard_fields.DetectionResultFields.object_width_voxels:
            tf.stack([gt_width, gt_width], axis=0),
        standard_fields.DetectionResultFields.object_center_voxels:
            tf.stack([gt_center, gt_center], axis=0),
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            tf.stack([gt_rotation_matrix, gt_rotation_matrix], axis=0),
    }
    outputs2 = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            tf.stack([logits2, logits2], axis=0),
        standard_fields.DetectionResultFields.object_length_voxels:
            tf.stack([gt_length, gt_length], axis=0),
        standard_fields.DetectionResultFields.object_height_voxels:
            tf.stack([gt_height, gt_height], axis=0),
        standard_fields.DetectionResultFields.object_width_voxels:
            tf.stack([gt_width, gt_width], axis=0),
        standard_fields.DetectionResultFields.object_center_voxels:
            tf.stack([gt_center, gt_center], axis=0),
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            tf.stack([gt_rotation_matrix, gt_rotation_matrix], axis=0),
    }
    loss1 = classification_losses.box_classification_loss(
        inputs=inputs, outputs=outputs1)
    loss2 = classification_losses.box_classification_loss(
        inputs=inputs, outputs=outputs2)
    self.assertGreater(loss2.numpy(), loss1.numpy())

  def test_box_classification_using_center_distance_loss_relative(self):
    gt_classes = tf.reshape(tf.constant([1, 1], dtype=tf.int32), [2, 1])
    gt_length = tf.reshape(tf.constant([1, 1], dtype=tf.float32), [2, 1])
    gt_height = tf.reshape(tf.constant([1, 1], dtype=tf.float32), [2, 1])
    gt_width = tf.reshape(tf.constant([1, 1], dtype=tf.float32), [2, 1])
    gt_center = tf.reshape(
        tf.constant([1, 1, 1, 1, 1, 1], dtype=tf.float32), [2, 3])
    gt_rotation_matrix = tf.tile(tf.expand_dims(tf.eye(3), axis=0), [2, 1, 1])
    logits1 = tf.reshape(
        tf.constant([[-2.0, 2.0, -3.0, -2.0, 0.0],
                     [-2.0, 2.0, -3.0, -2.0, 0.0]], dtype=tf.float32),
        [2, 5])
    logits2 = tf.reshape(
        tf.constant([[-2.0, 0.0, -3.0, -2.0, 2.0],
                     [-2.0, 0.0, -3.0, -2.0, 2.0]], dtype=tf.float32),
        [2, 5])
    gt_instance_ids = tf.reshape(tf.constant([1, 1], dtype=tf.int32), [2, 1])
    inputs = {
        standard_fields.InputDataFields.num_valid_voxels:
            tf.constant([2, 2], dtype=tf.int32),
        standard_fields.InputDataFields.object_class_voxels:
            tf.stack([gt_classes, gt_classes], axis=0),
        standard_fields.InputDataFields.object_length_voxels:
            tf.stack([gt_length, gt_length], axis=0),
        standard_fields.InputDataFields.object_height_voxels:
            tf.stack([gt_height, gt_height], axis=0),
        standard_fields.InputDataFields.object_width_voxels:
            tf.stack([gt_width, gt_width], axis=0),
        standard_fields.InputDataFields.object_center_voxels:
            tf.stack([gt_center, gt_center], axis=0),
        standard_fields.InputDataFields.object_rotation_matrix_voxels:
            tf.stack([gt_rotation_matrix, gt_rotation_matrix], axis=0),
        standard_fields.InputDataFields.object_instance_id_voxels:
            tf.stack([gt_instance_ids, gt_instance_ids], axis=0),
    }
    outputs1 = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            tf.stack([logits1, logits1], axis=0),
        standard_fields.DetectionResultFields.object_length_voxels:
            tf.stack([gt_length, gt_length], axis=0),
        standard_fields.DetectionResultFields.object_height_voxels:
            tf.stack([gt_height, gt_height], axis=0),
        standard_fields.DetectionResultFields.object_width_voxels:
            tf.stack([gt_width, gt_width], axis=0),
        standard_fields.DetectionResultFields.object_center_voxels:
            tf.stack([gt_center, gt_center], axis=0),
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            tf.stack([gt_rotation_matrix, gt_rotation_matrix], axis=0),
    }
    outputs2 = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            tf.stack([logits2, logits2], axis=0),
        standard_fields.DetectionResultFields.object_length_voxels:
            tf.stack([gt_length, gt_length], axis=0),
        standard_fields.DetectionResultFields.object_height_voxels:
            tf.stack([gt_height, gt_height], axis=0),
        standard_fields.DetectionResultFields.object_width_voxels:
            tf.stack([gt_width, gt_width], axis=0),
        standard_fields.DetectionResultFields.object_center_voxels:
            tf.stack([gt_center, gt_center], axis=0),
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            tf.stack([gt_rotation_matrix, gt_rotation_matrix], axis=0),
    }
    loss1 = classification_losses.box_classification_using_center_distance_loss(
        inputs=inputs, outputs=outputs1)
    loss2 = classification_losses.box_classification_using_center_distance_loss(
        inputs=inputs, outputs=outputs2)
    self.assertGreater(loss2.numpy(), loss1.numpy())

  def test_classification_loss_using_mask_iou_func_unbatched(self):
    embeddings = tf.constant([[1.0, 0.0, 0.0],
                              [0.2, 0.2, 0.2],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0],
                              [0.5, 0.5, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 1.0]], dtype=tf.float32)
    instance_ids = tf.constant([0, 1, 2, 3, 1, 0, 0, 0], dtype=tf.int32)
    class_labels = tf.constant([[0], [1], [1], [1], [1], [0], [2], [2]],
                               dtype=tf.int32)
    logits = tf.random.uniform([8, 5],
                               minval=-1.0,
                               maxval=1.0,
                               dtype=tf.float32)
    sampled_indices = tf.constant([0, 2, 3, 5], dtype=tf.int32)
    sampled_embeddings = tf.gather(embeddings, sampled_indices)
    sampled_instance_ids = tf.gather(instance_ids, sampled_indices)
    sampled_class_labels = tf.gather(class_labels, sampled_indices)
    sampled_logits = tf.gather(logits, sampled_indices)
    loss = classification_losses.classification_loss_using_mask_iou_func_unbatched(
        embeddings=embeddings,
        instance_ids=instance_ids,
        sampled_embeddings=sampled_embeddings,
        sampled_instance_ids=sampled_instance_ids,
        sampled_class_labels=sampled_class_labels,
        sampled_logits=sampled_logits,
        similarity_strategy='dotproduct',
        is_balanced=True)
    self.assertGreater(loss.numpy(), 0.0)

  def test_classification_loss_using_mask_iou_func(self):
    embeddings = tf.constant([[[1.0, 0.0, 0.0],
                               [0.2, 0.2, 0.2],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.5, 0.5, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]]], dtype=tf.float32)
    instance_ids = tf.constant([[0, 1, 2, 3, 1, 0, 0, 0]], dtype=tf.int32)
    class_labels = tf.constant([[[0], [1], [1], [1], [1], [0], [2], [2]]],
                               dtype=tf.int32)
    logits = tf.random.uniform([1, 8, 5],
                               minval=-1.0,
                               maxval=1.0,
                               dtype=tf.float32)
    valid_mask = tf.cast(
        tf.constant([[1, 1, 1, 0, 0, 1, 0, 0]], dtype=tf.int32), dtype=tf.bool)
    loss = classification_losses.classification_loss_using_mask_iou_func(
        embeddings=embeddings,
        logits=logits,
        instance_ids=instance_ids,
        class_labels=class_labels,
        num_samples=4,
        valid_mask=valid_mask,
        max_instance_id=5,
        similarity_strategy='dotproduct',
        is_balanced=True)
    self.assertGreater(loss.numpy(), 0.0)

  def test_classification_loss_using_mask_iou(self):
    outputs = {
        standard_fields.DetectionResultFields.instance_embedding_voxels:
            tf.constant([[[1.0, 0.0, 0.0], [0.2, 0.2, 0.2], [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
                        dtype=tf.float32),
        standard_fields.DetectionResultFields.object_semantic_voxels:
            tf.random.uniform([1, 8, 5],
                              minval=-1.0,
                              maxval=1.0,
                              dtype=tf.float32),
    }
    inputs = {
        standard_fields.InputDataFields.object_instance_id_voxels:
            tf.constant([[0, 1, 2, 3, 1, 0, 0, 0]], dtype=tf.int32),
        standard_fields.InputDataFields.object_class_voxels:
            tf.constant([[[0], [1], [1], [1], [1], [0], [2], [2]]],
                        dtype=tf.int32),
        standard_fields.InputDataFields.num_valid_voxels:
            tf.constant([5], dtype=tf.int32),
    }
    loss = classification_losses.classification_loss_using_mask_iou(
        inputs=inputs, outputs=outputs, num_samples=4, max_instance_id=5)
    self.assertGreater(loss.numpy(), 0.0)

  def test_hard_negative_classification_loss_relative(self):
    gt_classes = tf.reshape(tf.constant([1, 1], dtype=tf.int32), [2, 1])
    gt_length = tf.reshape(tf.constant([1, 1], dtype=tf.float32), [2, 1])
    gt_height = tf.reshape(tf.constant([1, 1], dtype=tf.float32), [2, 1])
    gt_width = tf.reshape(tf.constant([1, 1], dtype=tf.float32), [2, 1])
    gt_center = tf.reshape(
        tf.constant([1, 1, 1, 1, 1, 1], dtype=tf.float32), [2, 3])
    gt_rotation_matrix = tf.tile(tf.expand_dims(tf.eye(3), axis=0), [2, 1, 1])
    logits1 = tf.reshape(
        tf.constant([[-2.0, 2.0, -3.0, -2.0, 0.0],
                     [-2.0, 2.0, -3.0, -2.0, 0.0]], dtype=tf.float32),
        [2, 5])
    logits2 = tf.reshape(
        tf.constant([[-2.0, 0.0, -3.0, -2.0, 2.0],
                     [-2.0, 0.0, -3.0, -2.0, 2.0]], dtype=tf.float32),
        [2, 5])
    gt_instance_ids = tf.reshape(tf.constant([1, 1], dtype=tf.int32), [2, 1])
    inputs = {
        standard_fields.InputDataFields.num_valid_voxels:
            tf.constant([2, 2], dtype=tf.int32),
        standard_fields.InputDataFields.object_class_voxels:
            tf.stack([gt_classes, gt_classes], axis=0),
        standard_fields.InputDataFields.object_length_voxels:
            tf.stack([gt_length, gt_length], axis=0),
        standard_fields.InputDataFields.object_height_voxels:
            tf.stack([gt_height, gt_height], axis=0),
        standard_fields.InputDataFields.object_width_voxels:
            tf.stack([gt_width, gt_width], axis=0),
        standard_fields.InputDataFields.object_center_voxels:
            tf.stack([gt_center, gt_center], axis=0),
        standard_fields.InputDataFields.object_rotation_matrix_voxels:
            tf.stack([gt_rotation_matrix, gt_rotation_matrix], axis=0),
        standard_fields.InputDataFields.object_instance_id_voxels:
            tf.stack([gt_instance_ids, gt_instance_ids], axis=0),
    }
    outputs1 = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            tf.stack([logits1, logits1], axis=0),
        standard_fields.DetectionResultFields.object_length_voxels:
            tf.stack([gt_length, gt_length], axis=0),
        standard_fields.DetectionResultFields.object_height_voxels:
            tf.stack([gt_height, gt_height], axis=0),
        standard_fields.DetectionResultFields.object_width_voxels:
            tf.stack([gt_width, gt_width], axis=0),
        standard_fields.DetectionResultFields.object_center_voxels:
            tf.stack([gt_center, gt_center], axis=0),
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            tf.stack([gt_rotation_matrix, gt_rotation_matrix], axis=0),
    }
    outputs2 = {
        standard_fields.DetectionResultFields.object_semantic_voxels:
            tf.stack([logits2, logits2], axis=0),
        standard_fields.DetectionResultFields.object_length_voxels:
            tf.stack([gt_length, gt_length], axis=0),
        standard_fields.DetectionResultFields.object_height_voxels:
            tf.stack([gt_height, gt_height], axis=0),
        standard_fields.DetectionResultFields.object_width_voxels:
            tf.stack([gt_width, gt_width], axis=0),
        standard_fields.DetectionResultFields.object_center_voxels:
            tf.stack([gt_center, gt_center], axis=0),
        standard_fields.DetectionResultFields.object_rotation_matrix_voxels:
            tf.stack([gt_rotation_matrix, gt_rotation_matrix], axis=0),
    }
    loss1 = classification_losses.hard_negative_classification_loss(
        inputs=inputs, outputs=outputs1)
    loss2 = classification_losses.hard_negative_classification_loss(
        inputs=inputs, outputs=outputs2)
    self.assertGreaterEqual(loss2.numpy(), loss1.numpy())


if __name__ == '__main__':
  tf.test.main()
