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

"""Tests for ...box_utils.np_box_list_ops."""

import math
import numpy as np
import tensorflow as tf

from tf3d.object_detection.box_utils import np_box_list
from tf3d.object_detection.box_utils import np_box_list_ops


def _degree_to_radians(degree):
  return degree * math.pi / 180.0


def _decompose_box_tensor(boxes):
  rotation_z_radians = boxes[:, 0]
  length = boxes[:, 1]
  height = boxes[:, 2]
  width = boxes[:, 3]
  center_x = boxes[:, 4]
  center_y = boxes[:, 5]
  center_z = boxes[:, 6]
  return rotation_z_radians, length, height, width, center_x, center_y, center_z


class NpBoxListOpsTest(tf.test.TestCase):

  def setUp(self):
    boxes1 = np.array(
        [[_degree_to_radians(0.0), 1.0, 4.0, 3.0, 2.0, 5.0, 5.0],
         [_degree_to_radians(0.0), 2.0, 5.0, 6.0, 2.0, 7.5, 7.0]],
        dtype=float)
    (rotation_z_radians1, length1, height1, width1, center_x1, center_y1,
     center_z1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[_degree_to_radians(0.0), 1.0, 3.0, 4.0, 1.0, 4.5, 8.0],
         [_degree_to_radians(0.0), 14.0, 14.0, 14.0, 15.0, 8.0, 15.0],
         [_degree_to_radians(0.0), 1.0, 4.0, 3.0, 2.0, 5.0, 5.0],
         [_degree_to_radians(0.0), 2.0, 5.0, 6.0, 2.0, 7.5, 7.0]],
        dtype=float)
    (rotation_z_radians2, length2, height2, width2, center_x2, center_y2,
     center_z2) = _decompose_box_tensor(boxes2)
    rotation_matrix1 = np.tile(np.expand_dims(np.eye(3), axis=0), [2, 1, 1])
    rotation_matrix2 = np.tile(np.expand_dims(np.eye(3), axis=0), [4, 1, 1])
    self.boxlist1 = np_box_list.BoxList3d(
        length=length1,
        height=height1,
        width=width1,
        center_x=center_x1,
        center_y=center_y1,
        center_z=center_z1,
        rotation_z_radians=rotation_z_radians1)
    self.boxlist2 = np_box_list.BoxList3d(
        length=length2,
        height=height2,
        width=width2,
        center_x=center_x2,
        center_y=center_y2,
        center_z=center_z2,
        rotation_z_radians=rotation_z_radians2)
    self.boxlist1_matrix = np_box_list.BoxList3d(
        length=length1,
        height=height1,
        width=width1,
        center_x=center_x1,
        center_y=center_y1,
        center_z=center_z1,
        rotation_matrix=rotation_matrix1)
    self.boxlist2_matrix = np_box_list.BoxList3d(
        length=length2,
        height=height2,
        width=width2,
        center_x=center_x2,
        center_y=center_y2,
        center_z=center_z2,
        rotation_matrix=rotation_matrix2)

  def test_volume(self):
    volumes = np_box_list_ops.volume(self.boxlist1)
    volumes_matrix = np_box_list_ops.volume(self.boxlist1_matrix)
    expected_volumes = np.array([12.0, 60.0], dtype=float)
    self.assertAllClose(expected_volumes, volumes)
    self.assertAllClose(expected_volumes, volumes_matrix)

  def test_intersection3d(self):
    intersection = np_box_list_ops.intersection3d(self.boxlist1, self.boxlist2)
    intersection_matrix = np_box_list_ops.intersection3d(
        self.boxlist1_matrix, self.boxlist2_matrix)
    expected_intersection = np.array(
        [[0.0, 0.0, 12.0, 5.0],
         [3.0, 0.0, 5.0, 60.0]], dtype=float)
    self.assertAllClose(
        intersection, expected_intersection)
    self.assertAllClose(
        intersection_matrix, expected_intersection)

  def test_iou3d(self):
    iou = np_box_list_ops.iou3d(self.boxlist1, self.boxlist2)
    iou_matrix = np_box_list_ops.iou3d(self.boxlist1_matrix,
                                       self.boxlist2_matrix)
    expected_iou = np.array(
        [[0.0, 0.0, 1.0, 0.074627],
         [0.043478, 0.0, 0.074627, 1.0]],
        dtype=float)
    self.assertAllClose(iou, expected_iou)
    self.assertAllClose(iou_matrix, expected_iou)

  def test_iov3d(self):
    iov = np_box_list_ops.iov3d(self.boxlist1, self.boxlist2)
    iov_matrix = np_box_list_ops.iov3d(self.boxlist1_matrix,
                                       self.boxlist2_matrix)
    expected_iov = np.array(
        [[0.0, 0.0, 1.0, 0.083333],
         [0.25, 0.0, 0.416667, 1.0]],
        dtype=float)
    self.assertAllClose(iov, expected_iov)
    self.assertAllClose(iov_matrix, expected_iov)

  def test_nuscenes_center_distance_measure(self):
    center_distance_measure = np_box_list_ops.nuscenes_center_distance_measure(
        self.boxlist1, self.boxlist2)
    expected_center_distance = np.array(
        [[np.linalg.norm([1, 0.5]),
          np.linalg.norm([13, 3]),
          np.linalg.norm([0, 0]),
          np.linalg.norm([0, 2.5])],
         [np.linalg.norm([1, 3]),
          np.linalg.norm([13, 0.5]),
          np.linalg.norm([0, 2.5]),
          np.linalg.norm([0, 0])]], np.float32)
    expected_center_distance_measure = 1.0 / (1.0 +
                                              np.exp(expected_center_distance))
    self.assertAllClose(center_distance_measure,
                        expected_center_distance_measure)


class NpBoxListGatherTest(tf.test.TestCase):

  def setUp(self):
    boxes = np.array(
        [[_degree_to_radians(0.1), 1.0, 3.0, 4.0, 1.0, 6.0, 8.0],
         [_degree_to_radians(0.2), 14.0, 14.0, 14.0, 15.0, 15.0, 15.0],
         [_degree_to_radians(0.1), 1.0, 4.0, 3.0, 2.0, 7.0, 5.0],
         [_degree_to_radians(0.2), 2.0, 5.0, 6.0, 2.0, 10.0, 7.0]],
        dtype=float)
    (rotation_z_radians, length, height, width, center_x, center_y,
     center_z) = _decompose_box_tensor(boxes)
    self.boxlist = np_box_list.BoxList3d(
        length=length,
        height=height,
        width=width,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        rotation_z_radians=rotation_z_radians)
    self.boxlist.add_field('scores',
                           np.array([0.5, 0.7, 0.9, 0.4], dtype=float))
    self.boxlist.add_field(
        'labels',
        np.array([[0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 1, 0, 0]],
                 dtype=int))

  def _compare_boxlist_with_boxes(self, boxlist, expected_boxes):
    (expected_rotation_z_radians, expected_length, expected_height,
     expected_width, expected_center_x, expected_center_y,
     expected_center_z) = _decompose_box_tensor(expected_boxes)
    self.assertAllClose(expected_rotation_z_radians,
                        boxlist.get_rotation_z_radians())
    self.assertAllClose(expected_length, boxlist.get_length())
    self.assertAllClose(expected_height, boxlist.get_height())
    self.assertAllClose(expected_width, boxlist.get_width())
    self.assertAllClose(expected_center_x, boxlist.get_center_x())
    self.assertAllClose(expected_center_y, boxlist.get_center_y())
    self.assertAllClose(expected_center_z, boxlist.get_center_z())

  def test_gather3d_with_invalid_multidimensional_indices(self):
    indices = np.array([[0, 1, 4]], dtype=int)
    boxlist = self.boxlist
    with self.assertRaises(ValueError):
      np_box_list_ops.gather3d(boxlist, indices)

  def test_gather3d_without_fields_specified(self):
    indices = np.array([2, 0, 1], dtype=int)
    boxlist = self.boxlist
    subboxlist = np_box_list_ops.gather3d(boxlist, indices)
    expected_scores = np.array([0.9, 0.5, 0.7], dtype=float)
    self.assertAllClose(expected_scores, subboxlist.get_field('scores'))
    expected_boxes = np.array(
        [[_degree_to_radians(0.1), 1.0, 4.0, 3.0, 2.0, 7.0, 5.0],
         [_degree_to_radians(0.1), 1.0, 3.0, 4.0, 1.0, 6.0, 8.0],
         [_degree_to_radians(0.2), 14.0, 14.0, 14.0, 15.0, 15.0, 15.0]],
        dtype=float)
    self._compare_boxlist_with_boxes(subboxlist, expected_boxes)
    expected_labels = np.array(
        [[0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0]], dtype=int)
    self.assertAllClose(expected_labels, subboxlist.get_field('labels'))

  def test_gather3d_with_fields_specified(self):
    indices = np.array([2, 0, 1], dtype=int)
    boxlist = self.boxlist
    subboxlist = np_box_list_ops.gather3d(boxlist, indices, ['labels'])
    self.assertFalse(subboxlist.has_field('scores'))
    expected_boxes = np.array(
        [[_degree_to_radians(0.1), 1.0, 4.0, 3.0, 2.0, 7.0, 5.0],
         [_degree_to_radians(0.1), 1.0, 3.0, 4.0, 1.0, 6.0, 8.0],
         [_degree_to_radians(0.2), 14.0, 14.0, 14.0, 15.0, 15.0, 15.0]],
        dtype=float)
    self._compare_boxlist_with_boxes(subboxlist, expected_boxes)
    expected_labels = np.array(
        [[0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0],
         [0, 1, 0, 0, 0]], dtype=int)
    self.assertAllClose(expected_labels, subboxlist.get_field('labels'))

  def test_gather3d_with_invalid_field_specified(self):
    indices = np.array([2, 0, 1], dtype=int)
    boxlist = self.boxlist
    with self.assertRaises(ValueError):
      np_box_list_ops.gather3d(boxlist, indices, 'labels')
    with self.assertRaises(ValueError):
      np_box_list_ops.gather3d(boxlist, indices, ['objectness'])

  def test_filter_scores_greater_than(self):
    boxlist = np_box_list_ops.copy_boxlist(boxlist=self.boxlist)
    boxlist.add_field('scores', np.array([0.8, 0.2, 0.7, 0.4], np.float32))
    boxlist_greater = np_box_list_ops.filter_scores_greater_than3d(boxlist, 0.5)
    expected_boxes_greater = np.array(
        [[_degree_to_radians(0.1), 1.0, 3.0, 4.0, 1.0, 6.0, 8.0],
         [_degree_to_radians(0.1), 1.0, 4.0, 3.0, 2.0, 7.0, 5.0]],
        dtype=np.float32)
    self._compare_boxlist_with_boxes(boxlist_greater, expected_boxes_greater)


class NpBoxListSortTest(tf.test.TestCase):

  def setUp(self):
    boxes = np.array(
        [[_degree_to_radians(0.1), 1.0, 3.0, 4.0, 1.0, 6.0, 8.0],
         [_degree_to_radians(0.2), 14.0, 14.0, 14.0, 15.0, 15.0, 15.0],
         [_degree_to_radians(0.1), 1.0, 4.0, 3.0, 2.0, 7.0, 5.0],
         [_degree_to_radians(0.2), 2.0, 5.0, 6.0, 2.0, 10.0, 7.0]],
        dtype=float)
    (rotation_z_radians, length, height, width, center_x, center_y,
     center_z) = _decompose_box_tensor(boxes)
    self.boxlist = np_box_list.BoxList3d(
        length=length,
        height=height,
        width=width,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        rotation_z_radians=rotation_z_radians)
    self.boxlist.add_field('scores',
                           np.array([0.5, 0.7, 0.9, 0.4], dtype=float))
    self.boxlist.add_field(
        'labels',
        np.array([[0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 1, 0, 0]],
                 dtype=int))

  def _compare_boxlist_with_boxes(self, boxlist, expected_boxes):
    (expected_rotation_z_radians, expected_length, expected_height,
     expected_width, expected_center_x, expected_center_y,
     expected_center_z) = _decompose_box_tensor(expected_boxes)
    self.assertAllClose(expected_rotation_z_radians,
                        boxlist.get_rotation_z_radians())
    self.assertAllClose(expected_length, boxlist.get_length())
    self.assertAllClose(expected_height, boxlist.get_height())
    self.assertAllClose(expected_width, boxlist.get_width())
    self.assertAllClose(expected_center_x, boxlist.get_center_x())
    self.assertAllClose(expected_center_y, boxlist.get_center_y())
    self.assertAllClose(expected_center_z, boxlist.get_center_z())

  def test_sort_with_invalid_field(self):
    with self.assertRaises(ValueError):
      np_box_list_ops.sort_by_field3d(self.boxlist, 'objectness')
    with self.assertRaises(ValueError):
      np_box_list_ops.sort_by_field3d(self.boxlist, 'labels')

  def test_sort_with_invalid_sorting_order(self):
    with self.assertRaises(ValueError):
      np_box_list_ops.sort_by_field3d(self.boxlist, 'scores', 'Descending')

  def test_sort_descending(self):
    sorted_boxlist = np_box_list_ops.sort_by_field3d(self.boxlist, 'scores')
    expected_boxes = np.array(
        [[_degree_to_radians(0.1), 1.0, 4.0, 3.0, 2.0, 7.0, 5.0],
         [_degree_to_radians(0.2), 14.0, 14.0, 14.0, 15.0, 15.0, 15.0],
         [_degree_to_radians(0.1), 1.0, 3.0, 4.0, 1.0, 6.0, 8.0],
         [_degree_to_radians(0.2), 2.0, 5.0, 6.0, 2.0, 10.0, 7.0]],
        dtype=float)
    self._compare_boxlist_with_boxes(sorted_boxlist, expected_boxes)
    expected_scores = np.array([0.9, 0.7, 0.5, 0.4], dtype=float)
    self.assertAllClose(expected_scores, sorted_boxlist.get_field('scores'))

  def test_sort_ascending(self):
    sorted_boxlist = np_box_list_ops.sort_by_field3d(
        self.boxlist, 'scores', np_box_list_ops.SortOrder.ASCEND)
    expected_boxes = np.array(
        [[_degree_to_radians(0.2), 2.0, 5.0, 6.0, 2.0, 10.0, 7.0],
         [_degree_to_radians(0.1), 1.0, 3.0, 4.0, 1.0, 6.0, 8.0],
         [_degree_to_radians(0.2), 14.0, 14.0, 14.0, 15.0, 15.0, 15.0],
         [_degree_to_radians(0.1), 1.0, 4.0, 3.0, 2.0, 7.0, 5.0]],
        dtype=float)
    self._compare_boxlist_with_boxes(sorted_boxlist, expected_boxes)
    expected_scores = np.array([0.4, 0.5, 0.7, 0.9], dtype=float)
    self.assertAllClose(expected_scores, sorted_boxlist.get_field('scores'))


class NpBoxListNMSTest(tf.test.TestCase):

  def setUp(self):
    boxes = np.array(
        [[_degree_to_radians(0.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [_degree_to_radians(10.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [_degree_to_radians(20.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [_degree_to_radians(-10.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [_degree_to_radians(0.0), 1.0, 1.0, 1.0, 10.0, 0.0, 0.0],
         [_degree_to_radians(10.0), 1.0, 1.0, 1.0, 10.0, 0.0, 0.0],
         [_degree_to_radians(20.0), 1.0, 1.0, 1.0, 10.0, 0.0, 0.0]],
        dtype=float)
    (rotation_z_radians, length, height, width, center_x, center_y,
     center_z) = _decompose_box_tensor(boxes)
    self.boxlist = np_box_list.BoxList3d(
        length=length,
        height=height,
        width=width,
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        rotation_z_radians=rotation_z_radians)
    self.boxlist.add_field(
        'scores',
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                 dtype=float))

  def _compare_boxlist_with_boxes(self, boxlist, expected_boxes):
    (expected_rotation_z_radians, expected_length, expected_height,
     expected_width, expected_center_x, expected_center_y,
     expected_center_z) = _decompose_box_tensor(expected_boxes)
    self.assertAllClose(expected_rotation_z_radians,
                        boxlist.get_rotation_z_radians())
    self.assertAllClose(expected_length, boxlist.get_length())
    self.assertAllClose(expected_height, boxlist.get_height())
    self.assertAllClose(expected_width, boxlist.get_width())
    self.assertAllClose(expected_center_x, boxlist.get_center_x())
    self.assertAllClose(expected_center_y, boxlist.get_center_y())
    self.assertAllClose(expected_center_z, boxlist.get_center_z())

  def test_with_no_scores_field(self):
    boxlist = np_box_list_ops.copy_boxlist(boxlist=self.boxlist)
    max_output_size = 3
    iou_threshold = 0.5
    with self.assertRaises(ValueError):
      np_box_list_ops.non_max_suppression3d(boxlist, max_output_size,
                                            iou_threshold)

  def test_concatenate_boxes3d(self):
    boxes1 = np.array([[0.25, 0.25, 0.75, 0.75, 1.0, 2.0, 3.0],
                       [0.0, 0.1, 0.5, 0.75, 0.0, 0.0, 1.0]],
                      dtype=np.float32)
    (rotation_z_radians1, length1, height1, width1, center_x1, center_y1,
     center_z1) = _decompose_box_tensor(boxes1)
    boxlist1 = np_box_list.BoxList3d(
        length=length1,
        height=height1,
        width=width1,
        center_x=center_x1,
        center_y=center_y1,
        center_z=center_z1,
        rotation_z_radians=rotation_z_radians1)
    boxes2 = np.array([[0.5, 0.25, 1.0, 1.0, 2.0, 3.0, 4.0],
                       [0.0, 0.1, 1.0, 1.0, 10.0, 20.0, 30.0]],
                      dtype=np.float32)
    (rotation_z_radians2, length2, height2, width2, center_x2, center_y2,
     center_z2) = _decompose_box_tensor(boxes2)
    boxlist2 = np_box_list.BoxList3d(
        length=length2,
        height=height2,
        width=width2,
        center_x=center_x2,
        center_y=center_y2,
        center_z=center_z2,
        rotation_z_radians=rotation_z_radians2)
    boxlists = [boxlist1, boxlist2]
    boxlist_concatenated = np_box_list_ops.concatenate_boxes3d(boxlists)
    boxes_concatenated_expected = np.array(
        [[0.25, 0.25, 0.75, 0.75, 1.0, 2.0, 3.0],
         [0.0, 0.1, 0.5, 0.75, 0.0, 0.0, 1.0],
         [0.5, 0.25, 1.0, 1.0, 2.0, 3.0, 4.0],
         [0.0, 0.1, 1.0, 1.0, 10.0, 20.0, 30.0]],
        dtype=np.float32)
    self._compare_boxlist_with_boxes(boxlist_concatenated,
                                     boxes_concatenated_expected)

  def test_nms_disabled_max_output_size_equals_three(self):
    boxlist = self.boxlist
    max_output_size = 3
    iou_threshold = 0.99
    expected_boxes = np.array(
        [[_degree_to_radians(20.0), 1.0, 1.0, 1.0, 10.0, 0.0, 0.0],
         [_degree_to_radians(10.0), 1.0, 1.0, 1.0, 10.0, 0.0, 0.0],
         [_degree_to_radians(0.0), 1.0, 1.0, 1.0, 10.0, 0.0, 0.0]],
        dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression3d(
        boxlist, max_output_size, iou_threshold)
    self._compare_boxlist_with_boxes(nms_boxlist, expected_boxes)

  def test_nms(self):
    boxlist = self.boxlist
    max_output_size = 3
    iou_threshold = 0.1
    expected_boxes = np.array(
        [[_degree_to_radians(20.0), 1.0, 1.0, 1.0, 10.0, 0.0, 0.0],
         [_degree_to_radians(-10.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression3d(
        boxlist, max_output_size, iou_threshold)
    self._compare_boxlist_with_boxes(nms_boxlist, expected_boxes)

  def test_multiclass_nms(self):
    boxlist = np_box_list_ops.copy_boxlist(boxlist=self.boxlist)
    scores = np.array([[-0.2, 0.1, 0.5, -0.4, 0.3], [0.7, -0.7, 0.6, 0.2, -0.9],
                       [0.4, 0.34, -0.9, 0.2, 0.31], [0.1, 0.2, 0.3, 0.4, 0.5],
                       [-0.1, 0.1, -0.1, 0.1, -0.1], [0.3, 0.2, 0.1, 0.0, -0.1],
                       [0.0, 0.0, 0.0, 0.0, 0.0]],
                      dtype=np.float32)
    boxlist.add_field('scores', scores)
    boxlist_clean = np_box_list_ops.multi_class_non_max_suppression3d(
        boxlist, score_thresh=0.25, iou_thresh=0.1, max_output_size=3)

    scores_clean = boxlist_clean.get_field('scores')
    classes_clean = boxlist_clean.get_field('classes')
    expected_scores = np.array([0.7, 0.6, 0.5, 0.4, 0.34, 0.3])
    expected_classes = np.array([0, 2, 4, 3, 1, 0])
    expected_boxes = np.array(
        [[_degree_to_radians(10.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [_degree_to_radians(10.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [_degree_to_radians(-10.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [_degree_to_radians(-10.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [_degree_to_radians(20.0), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [_degree_to_radians(10.0), 1.0, 1.0, 1.0, 10.0, 0.0, 0.0]],
        dtype=np.float32)
    self.assertAllClose(scores_clean, expected_scores)
    self.assertAllClose(classes_clean, expected_classes)
    self._compare_boxlist_with_boxes(boxlist_clean, expected_boxes)


if __name__ == '__main__':
  tf.test.main()
