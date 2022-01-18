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

"""Operations on 3D bounding boxes.

Example box operations that are supported:
  * volumes: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""

import logging
import numpy as np
import shapely.affinity
import shapely.geometry


def diagonal_length(length, height, width):
  """Computes volume of boxes.

  Args:
    length: Numpy array with shape [N].
    height: Numpy array with shape [N].
    width: Numpy array with shape [N].

  Returns:
    A Numpy array with shape [N] representing the length of the diagonal of
    each box.
  """
  return np.sqrt(np.square(length) + np.square(height) + np.square(width)) / 2


def volume(length, height, width):
  """Computes volume of boxes.

  Args:
    length: Numpy array with shape [N].
    height: Numpy array with shape [N].
    width: Numpy array with shape [N].

  Returns:
    A Numpy array with shape [N] representing the box volumes.
  """
  return length * height * width


def center_distances(boxes1_center, boxes2_center):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes2_center: A Numpy array with shape [M, 3] holding M box centers in the
      format of [cx, cy, cz].

  Returns:
    A Numpy array with shape [N, M] representing pairwise center distances.
  """
  n = boxes1_center.shape[0]
  m = boxes2_center.shape[0]
  boxes1_center = np.tile(np.expand_dims(boxes1_center, axis=1), [1, m, 1])
  boxes2_center = np.tile(np.expand_dims(boxes2_center, axis=0), [n, 1, 1])
  return np.sqrt(np.sum(np.square(boxes2_center - boxes1_center), axis=2))


def center_distances_pairwise(boxes1_center, boxes2_center):
  """Computes pairwise distance between corresponding box centers.

  Args:
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes2_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].

  Returns:
    A Numpy array with shape [N] representing pairwise center distances.
  """
  return np.sqrt(np.sum(np.square(boxes2_center - boxes1_center), axis=1))


def _rotation_matrix_to_rotation_z(boxes_rotation_matrix):
  cos = boxes_rotation_matrix[:, 0, 0]
  sin = boxes_rotation_matrix[:, 1, 0]
  return np.arctan2(sin, cos)


def intersection3d_9dof_box(boxes1_length, boxes1_height, boxes1_width,
                            boxes1_center, boxes1_rotation_matrix,
                            boxes2_length, boxes2_height, boxes2_width,
                            boxes2_center, boxes2_rotation_matrix):
  """Computes intersection between every pair of boxes in the box collections.

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_matrix: Numpy array with shape [N, 3, 3].
    boxes2_length: Numpy array with shape [M].
    boxes2_height: Numpy array with shape [M].
    boxes2_width: Numpy array with shape [M].
    boxes2_center: A Numpy array with shape [M, 3] holding M box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_matrix: Numpy array with shape [M, 3, 3].

  Returns:
    A Numpy array with shape [N, M] representing pairwise intersections.
  """
  boxes1_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes1_rotation_matrix)
  boxes2_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes2_rotation_matrix)
  return intersection3d_7dof_box(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)


def intersection3d_7dof_box(boxes1_length, boxes1_height, boxes1_width,
                            boxes1_center, boxes1_rotation_z_radians,
                            boxes2_length, boxes2_height, boxes2_width,
                            boxes2_center, boxes2_rotation_z_radians):
  """Computes intersection between every pair of boxes in the box collections.

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_z_radians: Numpy array with shape [N].
    boxes2_length: Numpy array with shape [M].
    boxes2_height: Numpy array with shape [M].
    boxes2_width: Numpy array with shape [M].
    boxes2_center: A Numpy array with shape [M, 3] holding M box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_z_radians: Numpy array with shape [M].

  Returns:
    A Numpy array with shape [N, M] representing pairwise intersections.
  """
  n = boxes1_center.shape[0]
  m = boxes2_center.shape[0]
  if n == 0 or m == 0:
    return np.zeros([n, m], dtype=np.float32)
  boxes1_diag = diagonal_length(
      length=boxes1_length, height=boxes1_height, width=boxes1_width)
  boxes2_diag = diagonal_length(
      length=boxes2_length, height=boxes2_height, width=boxes2_width)
  dists = center_distances(
      boxes1_center=boxes1_center, boxes2_center=boxes2_center)

  intersections = []
  for i in range(n):
    box_i_length = boxes1_length[i]
    box_i_height = boxes1_height[i]
    box_i_width = boxes1_width[i]
    box_i_center_x = boxes1_center[i, 0]
    box_i_center_y = boxes1_center[i, 1]
    box_i_center_z = boxes1_center[i, 2]
    box_i_rotation_z_radians = boxes1_rotation_z_radians[i]
    box_diag = boxes1_diag[i]
    dist = dists[i, :]
    non_empty_i = (box_diag + boxes2_diag) >= dist
    intersection_i = np.zeros(m, np.float32)
    if non_empty_i.any():
      boxes2_center_nonempty = boxes2_center[non_empty_i]
      height_int_i, _ = _height_metrics(
          box_center_z=box_i_center_z,
          box_height=box_i_height,
          boxes_center_z=boxes2_center_nonempty[:, 2],
          boxes_height=boxes2_height[non_empty_i])
      rect_int_i = _get_rectangular_metrics(
          box_length=box_i_length,
          box_width=box_i_width,
          box_center_x=box_i_center_x,
          box_center_y=box_i_center_y,
          box_rotation_z_radians=box_i_rotation_z_radians,
          boxes_length=boxes2_length[non_empty_i],
          boxes_width=boxes2_width[non_empty_i],
          boxes_center_x=boxes2_center_nonempty[:, 0],
          boxes_center_y=boxes2_center_nonempty[:, 1],
          boxes_rotation_z_radians=boxes2_rotation_z_radians[non_empty_i])
      intersection_i[non_empty_i] = height_int_i * rect_int_i
    intersections.append(intersection_i)
  return np.stack(intersections, axis=0)


def intersection3d_9dof_box_pairwise(boxes1_length, boxes1_height, boxes1_width,
                                     boxes1_center, boxes1_rotation_matrix,
                                     boxes2_length, boxes2_height, boxes2_width,
                                     boxes2_center, boxes2_rotation_matrix):
  """Computes intersection between corresponding pairs in the box collections.

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_matrix: Numpy array with shape [N, 3, 3].
    boxes2_length: Numpy array with shape [N].
    boxes2_height: Numpy array with shape [N].
    boxes2_width: Numpy array with shape [N].
    boxes2_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_matrix: Numpy array with shape [N, 3, 3].

  Returns:
    A Numpy array with shape [N] representing pairwise intersections.
  """
  boxes1_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes1_rotation_matrix)
  boxes2_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes2_rotation_matrix)
  return intersection3d_7dof_box_pairwise(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)


def intersection3d_7dof_box_pairwise(boxes1_length, boxes1_height, boxes1_width,
                                     boxes1_center, boxes1_rotation_z_radians,
                                     boxes2_length, boxes2_height, boxes2_width,
                                     boxes2_center, boxes2_rotation_z_radians):
  """Computes intersection between corresponding pairs in the box collections.

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_z_radians: Numpy array with shape [N].
    boxes2_length: Numpy array with shape [N].
    boxes2_height: Numpy array with shape [N].
    boxes2_width: Numpy array with shape [N].
    boxes2_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_z_radians: Numpy array with shape [N].

  Returns:
    A Numpy array with shape [N] representing pairwise intersections.
  """
  n = boxes1_center.shape[0]
  if n == 0:
    return np.zeros([n], dtype=np.float32)
  if boxes1_center.shape[0] != boxes2_center.shape[0]:
    raise ValueError('Unequal number of box centers.')
  if boxes1_length.shape[0] != boxes2_length.shape[0]:
    raise ValueError('Unequal number of box lengths.')
  if boxes1_height.shape[0] != boxes2_height.shape[0]:
    raise ValueError('Unequal number of box heights.')
  if boxes1_width.shape[0] != boxes2_width.shape[0]:
    raise ValueError('Unequal number of box widths.')
  if boxes1_rotation_z_radians.shape[0] != boxes2_rotation_z_radians.shape[0]:
    raise ValueError('Unequal number of box rotations.')
  boxes1_diag = diagonal_length(
      length=boxes1_length, height=boxes1_height, width=boxes1_width)
  boxes2_diag = diagonal_length(
      length=boxes2_length, height=boxes2_height, width=boxes2_width)
  dists = center_distances_pairwise(
      boxes1_center=boxes1_center, boxes2_center=boxes2_center)

  intersections = []
  for i in range(n):
    box_i_length = boxes1_length[i]
    box_i_height = boxes1_height[i]
    box_i_width = boxes1_width[i]
    box_i_center_x = boxes1_center[i, 0]
    box_i_center_y = boxes1_center[i, 1]
    box_i_center_z = boxes1_center[i, 2]
    box_i_rotation_z_radians = boxes1_rotation_z_radians[i]
    box_diag = boxes1_diag[i]
    dist = dists[i:i+1]
    non_empty_i = (box_diag + boxes2_diag[i:i+1]) >= dist
    intersection_i = np.zeros(1, np.float32)
    if non_empty_i.any():
      boxes2_center_nonempty = boxes2_center[i:i+1][non_empty_i]
      height_int_i, _ = _height_metrics(
          box_center_z=box_i_center_z,
          box_height=box_i_height,
          boxes_center_z=boxes2_center_nonempty[:, 2],
          boxes_height=boxes2_height[i:i+1][non_empty_i])
      rect_int_i = _get_rectangular_metrics(
          box_length=box_i_length,
          box_width=box_i_width,
          box_center_x=box_i_center_x,
          box_center_y=box_i_center_y,
          box_rotation_z_radians=box_i_rotation_z_radians,
          boxes_length=boxes2_length[i:i + 1][non_empty_i],
          boxes_width=boxes2_width[i:i + 1][non_empty_i],
          boxes_center_x=boxes2_center_nonempty[:, 0],
          boxes_center_y=boxes2_center_nonempty[:, 1],
          boxes_rotation_z_radians=boxes2_rotation_z_radians[i:i +
                                                             1][non_empty_i])
      intersection_i[non_empty_i] = height_int_i * rect_int_i
    intersections.append(intersection_i)
  return np.concatenate(intersections, axis=0)


def iou3d_9dof_box(boxes1_length, boxes1_height, boxes1_width, boxes1_center,
                   boxes1_rotation_matrix, boxes2_length, boxes2_height,
                   boxes2_width, boxes2_center, boxes2_rotation_matrix):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_matrix: Numpy array with shape [N, 3, 3].
    boxes2_length: Numpy array with shape [M].
    boxes2_height: Numpy array with shape [M].
    boxes2_width: Numpy array with shape [M].
    boxes2_center: A Numpy array with shape [M, 3] holding M box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_matrix: Numpy array with shape [M, 3, 3].

  Returns:
    A Numpy array with shape [N, M] representing pairwise iou scores.
  """
  boxes1_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes1_rotation_matrix)
  boxes2_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes2_rotation_matrix)
  return iou3d_7dof_box(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)


def iou3d_7dof_box(boxes1_length, boxes1_height, boxes1_width, boxes1_center,
                   boxes1_rotation_z_radians, boxes2_length, boxes2_height,
                   boxes2_width, boxes2_center, boxes2_rotation_z_radians):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_z_radians: Numpy array with shape [N].
    boxes2_length: Numpy array with shape [M].
    boxes2_height: Numpy array with shape [M].
    boxes2_width: Numpy array with shape [M].
    boxes2_center: A Numpy array with shape [M, 3] holding M box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_z_radians: Numpy array with shape [M].

  Returns:
    A Numpy array with shape [N, M] representing pairwise iou scores.
  """
  n = boxes1_center.shape[0]
  m = boxes2_center.shape[0]
  if n == 0 or m == 0:
    return np.zeros([n, m], dtype=np.float32)
  boxes1_volume = volume(
      length=boxes1_length, height=boxes1_height, width=boxes1_width)
  boxes1_volume = np.tile(np.expand_dims(boxes1_volume, axis=1), [1, m])
  boxes2_volume = volume(
      length=boxes2_length, height=boxes2_height, width=boxes2_width)
  boxes2_volume = np.tile(np.expand_dims(boxes2_volume, axis=0), [n, 1])
  intersection = intersection3d_7dof_box(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)
  union = boxes1_volume + boxes2_volume - intersection
  return intersection / union


def iou3d_9dof_box_pairwise(boxes1_length, boxes1_height, boxes1_width,
                            boxes1_center, boxes1_rotation_matrix,
                            boxes2_length, boxes2_height, boxes2_width,
                            boxes2_center, boxes2_rotation_matrix):
  """Computes pairwise intersection-over-union between pairs of boxes.

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_matrix: Numpy array with shape [N, 3, 3].
    boxes2_length: Numpy array with shape [N].
    boxes2_height: Numpy array with shape [N].
    boxes2_width: Numpy array with shape [n].
    boxes2_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_matrix: Numpy array with shape [N, 3, 3].

  Returns:
    A Numpy array with shape [N] representing pairwise iou scores.
  """
  boxes1_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes1_rotation_matrix)
  boxes2_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes2_rotation_matrix)
  return iou3d_7dof_box_pairwise(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)


def iou3d_7dof_box_pairwise(boxes1_length, boxes1_height, boxes1_width,
                            boxes1_center, boxes1_rotation_z_radians,
                            boxes2_length, boxes2_height, boxes2_width,
                            boxes2_center, boxes2_rotation_z_radians):
  """Computes pairwise intersection-over-union between pairs of boxes.

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_z_radians: Numpy array with shape [N].
    boxes2_length: Numpy array with shape [N].
    boxes2_height: Numpy array with shape [N].
    boxes2_width: Numpy array with shape [n].
    boxes2_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_z_radians: Numpy array with shape [N].

  Returns:
    A Numpy array with shape [N] representing pairwise iou scores.
  """
  n = boxes1_center.shape[0]
  if n == 0:
    return np.zeros([n], dtype=np.float32)
  if boxes1_center.shape[0] != boxes2_center.shape[0]:
    raise ValueError('Unequal number of box centers.')
  if boxes1_length.shape[0] != boxes2_length.shape[0]:
    raise ValueError('Unequal number of box lengths.')
  if boxes1_height.shape[0] != boxes2_height.shape[0]:
    raise ValueError('Unequal number of box heights.')
  if boxes1_width.shape[0] != boxes2_width.shape[0]:
    raise ValueError('Unequal number of box widths.')
  if boxes1_rotation_z_radians.shape[0] != boxes2_rotation_z_radians.shape[0]:
    raise ValueError('Unequal number of box rotations.')
  boxes1_volume = volume(
      length=boxes1_length, height=boxes1_height, width=boxes1_width)
  boxes2_volume = volume(
      length=boxes2_length, height=boxes2_height, width=boxes2_width)
  intersection = intersection3d_7dof_box_pairwise(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)
  union = boxes1_volume + boxes2_volume - intersection
  return intersection / union


def iov3d_9dof_box(boxes1_length, boxes1_height, boxes1_width, boxes1_center,
                   boxes1_rotation_matrix, boxes2_length, boxes2_height,
                   boxes2_width, boxes2_center, boxes2_rotation_matrix):
  """Computes pairwise intersection-over-volume between box collections.

  Intersection-over-volume (iov) between two boxes box1 and box2 is defined as
  their intersection volume over box2's volume. Note that iov is not symmetric,
  that is, IOV(box1, box2) != IOV(box2, box1).

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_matrix: Numpy array with shape [N, 3, 3].
    boxes2_length: Numpy array with shape [M].
    boxes2_height: Numpy array with shape [M].
    boxes2_width: Numpy array with shape [M].
    boxes2_center: A Numpy array with shape [M, 3] holding M box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_matrix: Numpy array with shape [M, 3, 3].

  Returns:
    A Numpy array with shape [N, M] representing pairwise iou scores.
  """
  boxes1_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes1_rotation_matrix)
  boxes2_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes2_rotation_matrix)
  return iov3d_7dof_box(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)


def iov3d_7dof_box(boxes1_length, boxes1_height, boxes1_width, boxes1_center,
                   boxes1_rotation_z_radians, boxes2_length, boxes2_height,
                   boxes2_width, boxes2_center, boxes2_rotation_z_radians):
  """Computes pairwise intersection-over-volume between box collections.

  Intersection-over-volume (iov) between two boxes box1 and box2 is defined as
  their intersection volume over box2's volume. Note that iov is not symmetric,
  that is, IOV(box1, box2) != IOV(box2, box1).

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_z_radians: Numpy array with shape [N].
    boxes2_length: Numpy array with shape [M].
    boxes2_height: Numpy array with shape [M].
    boxes2_width: Numpy array with shape [M].
    boxes2_center: A Numpy array with shape [M, 3] holding M box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_z_radians: Numpy array with shape [M].

  Returns:
    A Numpy array with shape [N, M] representing pairwise iou scores.
  """
  n = boxes1_center.shape[0]
  boxes2_volume = volume(
      length=boxes2_length, height=boxes2_height, width=boxes2_width)
  boxes2_volume = np.tile(np.expand_dims(boxes2_volume, axis=0), [n, 1])
  intersection = intersection3d_7dof_box(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)
  return intersection / boxes2_volume


def iov3d_9dof_box_pairwise(boxes1_length, boxes1_height, boxes1_width,
                            boxes1_center, boxes1_rotation_matrix,
                            boxes2_length, boxes2_height, boxes2_width,
                            boxes2_center, boxes2_rotation_matrix):
  """Computes pairwise intersection-over-volume between pairs of boxes.

  Intersection-over-volume (iov) between two boxes box1 and box2 is defined as
  their intersection volume over box2's volume. Note that iov is not symmetric,
  that is, IOV(box1, box2) != IOV(box2, box1).

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_matrix: Numpy array with shape [N, 3, 3].
    boxes2_length: Numpy array with shape [N].
    boxes2_height: Numpy array with shape [N].
    boxes2_width: Numpy array with shape [N].
    boxes2_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_matrix: Numpy array with shape [N, 3, 3].

  Returns:
    A Numpy array with shape [N] representing pairwise iou scores.
  """
  boxes1_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes1_rotation_matrix)
  boxes2_rotation_z_radians = _rotation_matrix_to_rotation_z(
      boxes_rotation_matrix=boxes2_rotation_matrix)
  return iov3d_7dof_box_pairwise(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)


def iov3d_7dof_box_pairwise(boxes1_length, boxes1_height, boxes1_width,
                            boxes1_center, boxes1_rotation_z_radians,
                            boxes2_length, boxes2_height, boxes2_width,
                            boxes2_center, boxes2_rotation_z_radians):
  """Computes pairwise intersection-over-volume between pairs of boxes.

  Intersection-over-volume (iov) between two boxes box1 and box2 is defined as
  their intersection volume over box2's volume. Note that iov is not symmetric,
  that is, IOV(box1, box2) != IOV(box2, box1).

  Args:
    boxes1_length: Numpy array with shape [N].
    boxes1_height: Numpy array with shape [N].
    boxes1_width: Numpy array with shape [N].
    boxes1_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes1_rotation_z_radians: Numpy array with shape [N].
    boxes2_length: Numpy array with shape [N].
    boxes2_height: Numpy array with shape [N].
    boxes2_width: Numpy array with shape [N].
    boxes2_center: A Numpy array with shape [N, 3] holding N box centers in the
      format of [cx, cy, cz].
    boxes2_rotation_z_radians: Numpy array with shape [N].

  Returns:
    A Numpy array with shape [N] representing pairwise iou scores.
  """
  n = boxes1_center.shape[0]
  if n == 0:
    return np.zeros([n], dtype=np.float32)
  if boxes1_center.shape[0] != boxes2_center.shape[0]:
    raise ValueError('Unequal number of box centers.')
  if boxes1_length.shape[0] != boxes2_length.shape[0]:
    raise ValueError('Unequal number of box lengths.')
  if boxes1_height.shape[0] != boxes2_height.shape[0]:
    raise ValueError('Unequal number of box heights.')
  if boxes1_width.shape[0] != boxes2_width.shape[0]:
    raise ValueError('Unequal number of box widths.')
  if boxes1_rotation_z_radians.shape[0] != boxes2_rotation_z_radians.shape[0]:
    raise ValueError('Unequal number of box rotations.')
  boxes2_volume = volume(
      length=boxes2_length, height=boxes2_height, width=boxes2_width)
  intersection = intersection3d_7dof_box_pairwise(
      boxes1_length=boxes1_length,
      boxes1_height=boxes1_height,
      boxes1_width=boxes1_width,
      boxes1_center=boxes1_center,
      boxes1_rotation_z_radians=boxes1_rotation_z_radians,
      boxes2_length=boxes2_length,
      boxes2_height=boxes2_height,
      boxes2_width=boxes2_width,
      boxes2_center=boxes2_center,
      boxes2_rotation_z_radians=boxes2_rotation_z_radians)
  return intersection / boxes2_volume


def _height_metrics(box_center_z, box_height, boxes_center_z, boxes_height):
  """Compute 3D height intersection and union between a box and a list of boxes.

  Args:
    box_center_z: A scalar.
    box_height: A scalar.
    boxes_center_z: A Numpy array of size [N].
    boxes_height: A Numpy array of size [N].

  Returns:
    height_intersection: A Numpy array containing the intersection along
      the gravity axis between the two bounding boxes.
    height_union: A Numpy array containing the union along the gravity
      axis between the two bounding boxes.
  """
  min_z_boxes = boxes_center_z - boxes_height / 2.0
  max_z_boxes = boxes_center_z + boxes_height / 2.0
  max_z_box = box_center_z + box_height / 2.0
  min_z_box = box_center_z - box_height / 2.0
  max_of_mins = np.maximum(min_z_box, min_z_boxes)
  min_of_maxs = np.minimum(max_z_box, max_z_boxes)
  offsets = min_of_maxs - max_of_mins
  height_intersection = np.maximum(0, offsets)
  height_union = (
      np.maximum(min_z_box, max_z_boxes) - np.minimum(min_z_box, min_z_boxes) -
      np.maximum(0, -offsets))
  return height_intersection, height_union


def _get_box_contour(length, width, center_x, center_y, rotation_z_radians):
  """Compute shapely contour."""
  c = shapely.geometry.box(-length / 2.0, -width / 2.0, length / 2.0,
                           width / 2.0)
  rc = shapely.affinity.rotate(c, rotation_z_radians, use_radians=True)
  return shapely.affinity.translate(rc, center_x, center_y)


def _get_boxes_contour(length, width, center_x, center_y, rotation_z_radians):
  """Compute shapely contour."""
  contours = []
  n = length.shape[0]
  for i in range(n):
    contour = _get_box_contour(
        length=length[i],
        width=width[i],
        center_x=center_x[i],
        center_y=center_y[i],
        rotation_z_radians=rotation_z_radians[i])
    contours.append(contour)
  return contours


def _get_rectangular_metrics(box_length, box_width, box_center_x, box_center_y,
                             box_rotation_z_radians, boxes_length, boxes_width,
                             boxes_center_x, boxes_center_y,
                             boxes_rotation_z_radians):
  """Computes the intersection of the bases of 3d boxes.

  Args:
    box_length: A float scalar.
    box_width: A float scalar.
    box_center_x: A float scalar.
    box_center_y: A float scalar.
    box_rotation_z_radians: A float scalar.
    boxes_length: A np.float32 Numpy array of size [N].
    boxes_width: A np.float32 Numpy array of size [N].
    boxes_center_x: A np.float32 Numpy array of size [N].
    boxes_center_y: A np.float32 Numpy array of size [N].
    boxes_rotation_z_radians: A np.float32 Numpy array of size [N].

  Returns:
    intersection: A Numpy array containing intersection between the
      base of box and all other boxes.
  """
  m = boxes_length.shape[0]
  intersections = np.zeros([m], dtype=np.float32)
  try:
    contour_box = _get_box_contour(
        length=box_length,
        width=box_width,
        center_x=box_center_x,
        center_y=box_center_y,
        rotation_z_radians=box_rotation_z_radians)
    contours2 = _get_boxes_contour(
        length=boxes_length,
        width=boxes_width,
        center_x=boxes_center_x,
        center_y=boxes_center_y,
        rotation_z_radians=boxes_rotation_z_radians)
    for j in range(m):
      intersections[j] = contour_box.intersection(contours2[j]).area
  except Exception as e:  # pylint: disable=broad-except
    error_message = ('Error calling shapely : {}'.format(e))
    logging.info(error_message)
  return intersections
