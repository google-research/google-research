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

"""Tests for ...box_utils.np_box_ops."""

import math
import numpy as np
import tensorflow as tf
from tf3d.object_detection.box_utils import np_box_ops


def _degree_to_radians(degree):
  return degree * math.pi / 180.0


def _decompose_box_tensor(boxes):
  rotation_z_radians = boxes[:, 0]
  length = boxes[:, 1]
  height = boxes[:, 2]
  width = boxes[:, 3]
  center = boxes[:, 4:]
  return rotation_z_radians, length, height, width, center


class NpBoxOpsTest(tf.test.TestCase):

  def test_diagonal_length(self):
    boxes = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, 1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, 1.0, 2.0, 3.0]],
        dtype=np.float32)
    (_, length, height, width, _) = _decompose_box_tensor(boxes)
    boxes_diagonal_length = np_box_ops.diagonal_length(
        length=length, height=height, width=width)
    self.assertAllClose(boxes_diagonal_length, [18.70829, 5.937171, 3.24037])

  def test_volume(self):
    boxes = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, 1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, 1.0, 2.0, 3.0]],
        dtype=np.float32)
    (_, length, height, width, _) = _decompose_box_tensor(boxes)
    boxes_volume = np_box_ops.volume(length=length, height=height, width=width)
    self.assertAllClose(boxes_volume, [6000, 200, 20])

  def test_center_distance(self):
    boxes1 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (_, _, _, _, center1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32)
    (_, _, _, _, center2) = _decompose_box_tensor(boxes2)
    center_distances = np_box_ops.center_distances(
        boxes1_center=center1, boxes2_center=center2)
    expected_distances = np.array([[0.0, 7.28, 10.0499, 8.775],
                                   [7.28, 0.0, 4.0, 3.741658],
                                   [10.0499, 4.0, 0.0, 3.741658]],
                                  dtype=np.float32)
    self.assertAllClose(
        center_distances, expected_distances, rtol=0.01, atol=0.01)

  def test_center_distance_pairwise(self):
    boxes1 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (_, _, _, _, center1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (_, _, _, _, center2) = _decompose_box_tensor(boxes2)
    center_distances = np_box_ops.center_distances_pairwise(
        boxes1_center=center1, boxes2_center=center2)
    expected_distances = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    self.assertAllClose(
        center_distances, expected_distances, rtol=0.01, atol=0.01)

  def test_intersection3d_9dof_box(self):
    boxes1_rotations = np.tile(np.expand_dims(np.eye(3), axis=0), [2, 1, 1])
    boxes1_length = np.array([1.0, 1.0])
    boxes1_height = np.array([1.0, 1.0])
    boxes1_width = np.array([1.0, 1.0])
    boxes1_center = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    boxes2_rotations = np.expand_dims(np.eye(3), axis=0)
    boxes2_length = np.array([1.0])
    boxes2_height = np.array([1.0])
    boxes2_width = np.array([1.0])
    boxes2_center = np.array([[0.0, 0.0, 0.0]])
    intersections = np_box_ops.intersection3d_9dof_box(
        boxes1_length=boxes1_length,
        boxes1_height=boxes1_height,
        boxes1_width=boxes1_width,
        boxes1_center=boxes1_center,
        boxes1_rotation_matrix=boxes1_rotations,
        boxes2_length=boxes2_length,
        boxes2_height=boxes2_height,
        boxes2_width=boxes2_width,
        boxes2_center=boxes2_center,
        boxes2_rotation_matrix=boxes2_rotations)
    expected_intersections = np.array([[1.0], [0.0]], dtype=np.float32)
    self.assertAllClose(
        intersections, expected_intersections, rtol=0.1, atol=1.0)

  def test_intersection3d_7dof_box(self):
    boxes1 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (rotation_z_radians1, length1, height1, width1,
     center1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, 100.0, 100.0, 100.0]],
        dtype=np.float32)
    (rotation_z_radians2, length2, height2, width2,
     center2) = _decompose_box_tensor(boxes2)
    intersections = np_box_ops.intersection3d_7dof_box(
        boxes1_length=length1,
        boxes1_height=height1,
        boxes1_width=width1,
        boxes1_center=center1,
        boxes1_rotation_z_radians=rotation_z_radians1,
        boxes2_length=length2,
        boxes2_height=height2,
        boxes2_width=width2,
        boxes2_center=center2,
        boxes2_rotation_z_radians=rotation_z_radians2)
    expected_intersections = np.array(
        [[6000.0, 49.91, 4.492, 0.0],
         [49.91, 200.0, 19.93, 0.0],
         [4.492, 19.93, 20.0, 0.0]],
        dtype=np.float32)
    self.assertAllClose(
        intersections, expected_intersections, rtol=0.1, atol=1.0)

  def test_intersection3d_7dof_box_pairwise(self):
    boxes1 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (rotation_z_radians1, length1, height1, width1,
     center1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (rotation_z_radians2, length2, height2, width2,
     center2) = _decompose_box_tensor(boxes2)
    intersections = np_box_ops.intersection3d_7dof_box_pairwise(
        boxes1_length=length1,
        boxes1_height=height1,
        boxes1_width=width1,
        boxes1_center=center1,
        boxes1_rotation_z_radians=rotation_z_radians1,
        boxes2_length=length2,
        boxes2_height=height2,
        boxes2_width=width2,
        boxes2_center=center2,
        boxes2_rotation_z_radians=rotation_z_radians2)
    expected_intersections = np.array([6000.0, 200.0, 20.0], dtype=np.float32)
    self.assertAllClose(
        intersections, expected_intersections, rtol=0.1, atol=1.0)

  def test_iou3d_7dof_box(self):
    boxes1 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (rotation_z_radians1, length1, height1, width1,
     center1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, 100.0, 100.0, 100.0]],
        dtype=np.float32)
    (rotation_z_radians2, length2, height2, width2,
     center2) = _decompose_box_tensor(boxes2)
    iou = np_box_ops.iou3d_7dof_box(
        boxes1_length=length1,
        boxes1_height=height1,
        boxes1_width=width1,
        boxes1_center=center1,
        boxes1_rotation_z_radians=rotation_z_radians1,
        boxes2_length=length2,
        boxes2_height=height2,
        boxes2_width=width2,
        boxes2_center=center2,
        boxes2_rotation_z_radians=rotation_z_radians2)
    expected_iou = np.array([[1.0, 0.008, 0.0008, 0.0],
                             [0.008, 1.0, 0.01845, 0.0],
                             [0.0008, 0.01845, 1.00, 0.0]],
                            dtype=np.float32)
    self.assertAllClose(iou, expected_iou, rtol=0.1, atol=1.0)

  def test_iou3d_7dof_box_pairwise(self):
    boxes1 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (rotation_z_radians1, length1, height1, width1,
     center1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (rotation_z_radians2, length2, height2, width2,
     center2) = _decompose_box_tensor(boxes2)
    iou = np_box_ops.iou3d_7dof_box_pairwise(
        boxes1_length=length1,
        boxes1_height=height1,
        boxes1_width=width1,
        boxes1_center=center1,
        boxes1_rotation_z_radians=rotation_z_radians1,
        boxes2_length=length2,
        boxes2_height=height2,
        boxes2_width=width2,
        boxes2_center=center2,
        boxes2_rotation_z_radians=rotation_z_radians2)
    expected_iou = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    self.assertAllClose(iou, expected_iou, rtol=0.1, atol=1.0)

  def test_iou3d_pairwise_simple(self):
    boxes1 = np.array([[0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                       [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                       [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                       [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                       [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                       [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                       [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                       [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]],
                      dtype=np.float32)
    (rotation_z_radians1, length1, height1, width1,
     center1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
         [0.0, 10.0, 10.0, 10.0, 10.0, 15.0, 10.0],
         [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 15.0],
         [0.0, 10.0, 10.0, 10.0, 15.0, 10.0, 10.0],
         [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0],
         [_degree_to_radians(90.0), 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
         [_degree_to_radians(180.0), 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
         [_degree_to_radians(90.0), 10.0, 10.0, 10.0, 10.0, 15.0, 10.0]],
        dtype=np.float32)
    (rotation_z_radians2, length2, height2, width2,
     center2) = _decompose_box_tensor(boxes2)
    iou = np_box_ops.iou3d_7dof_box_pairwise(
        boxes1_length=length1,
        boxes1_height=height1,
        boxes1_width=width1,
        boxes1_center=center1,
        boxes1_rotation_z_radians=rotation_z_radians1,
        boxes2_length=length2,
        boxes2_height=height2,
        boxes2_width=width2,
        boxes2_center=center2,
        boxes2_rotation_z_radians=rotation_z_radians2)
    expected_iou = np.array(
        [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0, 1.0, 1.0 / 3.0],
        dtype=np.float32)
    self.assertAllClose(iou, expected_iou, rtol=0.0001, atol=0.00001)

  def test_iov3d_7dof_box(self):
    boxes1 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (rotation_z_radians1, length1, height1, width1,
     center1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, 100.0, 100.0, 100.0]],
        dtype=np.float32)
    (rotation_z_radians2, length2, height2, width2,
     center2) = _decompose_box_tensor(boxes2)
    iov = np_box_ops.iov3d_7dof_box(
        boxes1_length=length1,
        boxes1_height=height1,
        boxes1_width=width1,
        boxes1_center=center1,
        boxes1_rotation_z_radians=rotation_z_radians1,
        boxes2_length=length2,
        boxes2_height=height2,
        boxes2_width=width2,
        boxes2_center=center2,
        boxes2_rotation_z_radians=rotation_z_radians2)
    expected_iov = np.array([[1.0, 0.25, 0.25, 0.0],
                             [0.008, 1.0, 0.2, 0.0],
                             [0.0008, 0.2, 1.0, 0.0]],
                            dtype=np.float32)
    self.assertAllClose(iov, expected_iov, rtol=0.1, atol=1.0)

  def test_iov3d_7dof_box_pairwise(self):
    boxes1 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (rotation_z_radians1, length1, height1, width1,
     center1) = _decompose_box_tensor(boxes1)
    boxes2 = np.array(
        [[_degree_to_radians(0.1), 10.0, 20.0, 30.0, 5.0, 6.0, 4.0],
         [_degree_to_radians(0.2), 4.0, 5.0, 10.0, -1.0, 2.0, 3.0],
         [_degree_to_radians(-0.2), 4.0, 5.0, 1.0, -1.0, -2.0, 3.0]],
        dtype=np.float32)
    (rotation_z_radians2, length2, height2, width2,
     center2) = _decompose_box_tensor(boxes2)
    iov = np_box_ops.iov3d_7dof_box_pairwise(
        boxes1_length=length1,
        boxes1_height=height1,
        boxes1_width=width1,
        boxes1_center=center1,
        boxes1_rotation_z_radians=rotation_z_radians1,
        boxes2_length=length2,
        boxes2_height=height2,
        boxes2_width=width2,
        boxes2_center=center2,
        boxes2_rotation_z_radians=rotation_z_radians2)
    expected_iov = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    self.assertAllClose(iov, expected_iov, rtol=0.1, atol=1.0)


if __name__ == '__main__':
  tf.test.main()
