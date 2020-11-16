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
import tensorflow as tf

from tf3d.utils import instance_segmentation_utils as isu


class InstanceSegmentUtilsTest(tf.test.TestCase):

  def get_instance_masks(self):
    mask0 = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0]],
                        dtype=tf.float32)
    mask1 = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]],
                        dtype=tf.float32)
    mask2 = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1]],
                        dtype=tf.float32)
    masks = tf.stack([mask0, mask1, mask2])
    return masks

  def test_map_labels_to_0_to_n1(self):
    labels = tf.constant([[-1, 2, 5],
                          [0, 9, 1]], dtype=tf.int32)
    labels_0_n = isu.map_labels_to_0_to_n(labels)
    expected_labels_0_n = tf.constant([[-1, 2, 3],
                                       [0, 4, 1]], dtype=tf.int32)
    self.assertAllEqual(labels_0_n.numpy(), expected_labels_0_n.numpy())

  def test_map_labels_to_0_to_n2(self):
    labels = tf.constant([[-1, 1, 2],
                          [1, 1, 2]], dtype=tf.int32)
    labels_0_n = isu.map_labels_to_0_to_n(labels)
    expected_labels_0_n = tf.constant([[-1, 0, 1],
                                       [0, 0, 1]], dtype=tf.int32)
    self.assertAllEqual(labels_0_n.numpy(), expected_labels_0_n.numpy())

  def test_randomly_select_one_point_per_segment(self):
    instance_labels = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 2, 2, 2, 2, 2, 2],
                                   [1, 2, 2, 2, 2, 2, 2, 2],
                                   [0, 0, 0, 0, 2, 2, 2, 2],
                                   [0, 0, 0, 0, 2, 2, 2, 2]],
                                  dtype=tf.int32)
    instance_labels = tf.reshape(instance_labels, [-1])
    (indices,
     masks_t) = isu.randomly_select_one_point_per_segment(instance_labels)
    masks = tf.transpose(masks_t)
    masks = tf.reshape(masks, [3, 5, 8])
    expected_masks = self.get_instance_masks()
    selected_instances = tf.gather(instance_labels, indices)
    expected_selected_instances = tf.constant([0, 1, 2], dtype=tf.int32)
    self.assertAllEqual(selected_instances.numpy(),
                        expected_selected_instances.numpy())
    self.assertAllClose(masks.numpy(), expected_masks.numpy())

  def test_inputs_Distances_to_centers(self):
    inputs = tf.random.uniform(
        [100, 8], minval=-10, maxval=10.0, dtype=tf.float32)
    centers = tf.random.uniform(
        [5, 8], minval=-10, maxval=10.0, dtype=tf.float32)
    distances1 = isu.inputs_distances_to_centers(inputs, centers)
    num_centers = tf.shape(centers)[0]
    inputs_reshaped = tf.tile(tf.expand_dims(inputs, axis=1),
                              tf.stack([1, num_centers, 1]))
    distances2 = tf.reduce_sum(tf.square(inputs_reshaped - centers), axis=2)
    self.assertAllClose(distances1.numpy(), distances2.numpy(), atol=0.001)

  def test_pairwise_iou_matrix(self):
    mask0 = tf.constant([[1, 0],
                         [0, 1]], dtype=tf.float32)
    mask1 = tf.constant([[1, 1],
                         [0, 1]], dtype=tf.float32)
    mask2 = tf.constant([[1, 0],
                         [1, 1]], dtype=tf.float32)
    mask3 = tf.constant([[1, 1],
                         [1, 1]], dtype=tf.float32)
    mask4 = tf.constant([[0, 0],
                         [0, 0]], dtype=tf.float32)
    mask5 = tf.constant([[1, 0],
                         [1, 0]], dtype=tf.float32)
    masks1 = tf.stack([mask0, mask1, mask2])
    masks2 = tf.stack([mask3, mask4, mask5])
    ious = isu.get_pairwise_iou_matrix(masks1, masks2)
    expected_ious = tf.constant([[0.5, 0.0, 1.0/3.0],
                                 [0.75, 0.0, 0.25],
                                 [0.75, 0.0, 2.0/3.0]],
                                dtype=tf.float32)
    self.assertAllClose(ious.numpy(), expected_ious.numpy())

  def test_instance_non_maximum_suppression_1d_scores(self):
    mask0 = tf.constant([[1, 0],
                         [0, 1]], dtype=tf.float32)
    mask1 = tf.constant([[1, 1],
                         [0, 1]], dtype=tf.float32)
    mask2 = tf.constant([[1, 0],
                         [1, 1]], dtype=tf.float32)
    mask3 = tf.constant([[1, 1],
                         [1, 1]], dtype=tf.float32)
    mask4 = tf.constant([[0, 0],
                         [0, 0]], dtype=tf.float32)
    mask5 = tf.constant([[1, 0],
                         [1, 0]], dtype=tf.float32)
    masks = tf.stack([mask0, mask1, mask2, mask3, mask4, mask5])
    classes = tf.constant([1, 2, 3, 1, 2, 3], dtype=tf.int32)
    scores = tf.constant([1.0, 0.9, 0.8, 0.95, 0.85, 0.6], dtype=tf.float32)
    (nms_masks1,
     nms_scores1,
     nms_classes1,
     _) = isu.instance_non_maximum_suppression_1d_scores(
         masks,
         scores,
         classes,
         min_score_thresh=0.65,
         min_iou_thresh=0.5,
         is_class_agnostic=True)
    nms_masks_expected1 = tf.stack([mask0, mask4])
    nms_scores_expected1 = tf.constant([1.0, 0.85], dtype=tf.float32)
    nms_classes_expected1 = tf.constant([1, 2], dtype=tf.int32)
    (nms_masks2,
     nms_scores2,
     nms_classes2,
     _) = isu.instance_non_maximum_suppression_1d_scores(
         masks,
         scores,
         classes,
         min_score_thresh=0.65,
         min_iou_thresh=0.5,
         is_class_agnostic=False)
    nms_masks_expected2 = tf.stack([mask0, mask1, mask4, mask2])
    nms_scores_expected2 = tf.constant([1.0, 0.9, 0.85, 0.8], dtype=tf.float32)
    nms_classes_expected2 = tf.constant([1, 2, 2, 3], dtype=tf.int32)
    self.assertAllEqual(nms_masks1.numpy(), nms_masks_expected1.numpy())
    self.assertAllClose(nms_scores1.numpy(), nms_scores_expected1.numpy())
    self.assertAllEqual(nms_classes1.numpy(), nms_classes_expected1.numpy())
    self.assertAllEqual(nms_masks2.numpy(), nms_masks_expected2.numpy())
    self.assertAllClose(nms_scores2.numpy(), nms_scores_expected2.numpy())
    self.assertAllEqual(nms_classes2.numpy(), nms_classes_expected2.numpy())

  def test_instance_non_maximum_suppression_1d_scores_empty_inputs(self):
    masks = tf.constant(1.0, shape=[0, 2, 2], dtype=tf.float32)
    scores = tf.constant([], dtype=tf.float32)
    classes = tf.constant([], dtype=tf.int32)
    (nms_masks1,
     nms_scores1,
     nms_classes1,
     _) = isu.instance_non_maximum_suppression_1d_scores(
         masks,
         scores,
         classes,
         min_score_thresh=0.65,
         min_iou_thresh=0.5,
         is_class_agnostic=True)
    nms_masks_expected1 = tf.constant(1.0, shape=[0, 2, 2], dtype=tf.float32)
    nms_scores_expected1 = tf.constant([], dtype=tf.float32)
    nms_classes_expected1 = tf.constant([], dtype=tf.int32)
    (nms_masks2,
     nms_scores2,
     nms_classes2,
     _) = isu.instance_non_maximum_suppression_1d_scores(
         masks,
         scores,
         classes,
         min_score_thresh=0.65,
         min_iou_thresh=0.5,
         is_class_agnostic=False)
    nms_masks_expected2 = tf.constant(1.0, shape=[0, 2, 2], dtype=tf.float32)
    nms_scores_expected2 = tf.constant([], dtype=tf.float32)
    nms_classes_expected2 = tf.constant([], dtype=tf.int32)
    self.assertAllEqual(nms_masks1.numpy(), nms_masks_expected1.numpy())
    self.assertAllClose(nms_scores1.numpy(), nms_scores_expected1.numpy())
    self.assertAllEqual(nms_classes1.numpy(), nms_classes_expected1.numpy())
    self.assertAllEqual(nms_masks2.numpy(), nms_masks_expected2.numpy())
    self.assertAllClose(nms_scores2.numpy(), nms_scores_expected2.numpy())
    self.assertAllEqual(nms_classes2.numpy(), nms_classes_expected2.numpy())

  def test_instance_non_maximum_suppression_2d_scores(self):
    mask0 = tf.constant([[1, 0],
                         [0, 1]], dtype=tf.float32)
    mask1 = tf.constant([[1, 1],
                         [0, 1]], dtype=tf.float32)
    mask2 = tf.constant([[1, 0],
                         [1, 1]], dtype=tf.float32)
    mask3 = tf.constant([[1, 1],
                         [1, 1]], dtype=tf.float32)
    mask4 = tf.constant([[0, 0],
                         [0, 0]], dtype=tf.float32)
    mask5 = tf.constant([[1, 0],
                         [1, 0]], dtype=tf.float32)
    masks = tf.stack([mask0, mask1, mask2, mask3, mask4, mask5])
    scores = tf.constant([[0.05, 1.0, 0.2],
                          [0.9, 0.1, 0.3],
                          [0.95, 0.92, 0.1],
                          [0.1, 0.05, 0.0],
                          [0.2, 0.3, 0.7],
                          [0.1, 0.2, 0.8]],
                         dtype=tf.float32)
    (nms_masks1,
     nms_scores1,
     nms_classes1) = isu.instance_non_maximum_suppression_2d_scores(
         masks,
         scores,
         3,
         min_score_thresh=0.65,
         min_iou_thresh=0.5,
         is_class_agnostic=True)
    nms_masks_expected1 = tf.stack([mask0, mask5, mask4])
    nms_scores_expected1 = tf.constant([1.0, 0.8, 0.7], dtype=tf.float32)
    nms_classes_expected1 = tf.constant([1, 2, 2], dtype=tf.int32)
    (nms_masks2,
     nms_scores2,
     nms_classes2) = isu.instance_non_maximum_suppression_2d_scores(
         masks,
         scores,
         3,
         min_score_thresh=0.65,
         min_iou_thresh=0.5,
         is_class_agnostic=False)
    nms_masks_expected2 = tf.stack([mask2, mask0, mask5, mask4])
    nms_scores_expected2 = tf.constant([0.95, 1.0, 0.8, 0.7], dtype=tf.float32)
    nms_classes_expected2 = tf.constant([0, 1, 2, 2], dtype=tf.int32)
    self.assertAllEqual(nms_masks1.numpy(), nms_masks_expected1.numpy())
    self.assertAllClose(nms_scores1.numpy(), nms_scores_expected1.numpy())
    self.assertAllEqual(nms_classes1.numpy(), nms_classes_expected1.numpy())
    self.assertAllEqual(nms_masks2.numpy(), nms_masks_expected2.numpy())
    self.assertAllClose(nms_scores2.numpy(), nms_scores_expected2.numpy())
    self.assertAllEqual(nms_classes2.numpy(), nms_classes_expected2.numpy())

  def test_points_mask_iou(self):
    masks1 = tf.constant([[0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 1],
                          [0, 1, 0, 1, 0]], dtype=tf.int32)
    masks2 = tf.constant([[0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 1]], dtype=tf.int32)
    iou = isu.points_mask_iou(masks1=masks1, masks2=masks2)
    expected_iou = tf.constant([[0, 0, 0],
                                [0, 1, 0.6],
                                [0, 0.6, 1.0],
                                [0, 0.4, 0]], dtype=tf.float32)
    self.assertAllClose(iou.numpy(), expected_iou.numpy())

  def test_points_mask_pairwise_iou(self):
    masks1 = tf.constant([[0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1],
                          [1, 0, 1, 0, 1],
                          [0, 1, 0, 1, 0]], dtype=tf.int32)
    masks2 = tf.constant([[0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 0],
                          [1, 0, 1, 1, 1]], dtype=tf.int32)
    pairwise_iou = isu.points_mask_pairwise_iou(masks1=masks1, masks2=masks2)
    expected_iou = tf.constant([0, 1, 0.4, 0.2], dtype=tf.float32)
    self.assertAllClose(pairwise_iou.numpy(), expected_iou.numpy())


if __name__ == '__main__':
  tf.test.main()
