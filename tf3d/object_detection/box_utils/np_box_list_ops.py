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

"""Bounding Box List operations for Numpy BoxList3d's.

Example box operations that are supported:
  * Volumes: compute bounding box volume
  * IOU: pairwise intersection-over-union scores
"""

import numpy as np
from tf3d.object_detection.box_utils import np_box_list
from tf3d.object_detection.box_utils import np_box_ops
from tensorflow_models.object_detection.utils import np_box_list_ops

SortOrder = np_box_list_ops.SortOrder  # pylint: disable=invalid-name


def copy_boxlist(boxlist, indices=None):
  """Copy the boxes of a BoxList3d object into a new BoxList3d object.

  Args:
    boxlist: A np_box_list.BoxList3d object.
    indices: A indices of the boxes to be copied. It is not used if None.

  Returns:
    new_boxlist: A new np_box_list.BoxList3d object.
  """
  length = boxlist.get_length()
  height = boxlist.get_height()
  width = boxlist.get_width()
  center_x = boxlist.get_center_x()
  center_y = boxlist.get_center_y()
  center_z = boxlist.get_center_z()
  rotation_matrix = boxlist.get_rotation_matrix()
  rotation_z_radians = boxlist.get_rotation_z_radians()
  if indices is not None:
    length = length[indices]
    height = height[indices]
    width = width[indices]
    center_x = center_x[indices]
    center_y = center_y[indices]
    center_z = center_z[indices]
    if rotation_matrix is not None:
      rotation_matrix = rotation_matrix[indices, :, :]
    if rotation_z_radians is not None:
      rotation_z_radians = rotation_z_radians[indices]
  new_boxlist = np_box_list.BoxList3d(
      length=length,
      height=height,
      width=width,
      center_x=center_x,
      center_y=center_y,
      center_z=center_z,
      rotation_matrix=rotation_matrix,
      rotation_z_radians=rotation_z_radians)
  return new_boxlist


def volume(boxlist):
  """Computes area of boxes.

  Args:
    boxlist: BoxList3d holding N boxes.

  Returns:
    A numpy array with shape [N*1] representing box volumes.
  """
  return np_box_ops.volume(
      length=boxlist.get_length(),
      height=boxlist.get_height(),
      width=boxlist.get_width())


def intersection3d(boxlist1, boxlist2):
  """Computes pairwise intersection areas between boxes.

  Args:
    boxlist1: BoxList3d holding N boxes.
    boxlist2: BoxList3d holding M boxes.

  Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
  """
  boxlist1_rotation_matrix = boxlist1.get_rotation_matrix()
  boxlist2_rotation_matrix = boxlist2.get_rotation_matrix()
  if (boxlist1_rotation_matrix is not None) and (boxlist2_rotation_matrix is
                                                 not None):
    return np_box_ops.intersection3d_9dof_box(
        boxes1_length=boxlist1.get_length(),
        boxes1_height=boxlist1.get_height(),
        boxes1_width=boxlist1.get_width(),
        boxes1_center=boxlist1.get_center(),
        boxes1_rotation_matrix=boxlist1_rotation_matrix,
        boxes2_length=boxlist2.get_length(),
        boxes2_height=boxlist2.get_height(),
        boxes2_width=boxlist2.get_width(),
        boxes2_center=boxlist2.get_center(),
        boxes2_rotation_matrix=boxlist2_rotation_matrix)
  else:
    return np_box_ops.intersection3d_7dof_box(
        boxes1_length=boxlist1.get_length(),
        boxes1_height=boxlist1.get_height(),
        boxes1_width=boxlist1.get_width(),
        boxes1_center=boxlist1.get_center(),
        boxes1_rotation_z_radians=boxlist1.get_rotation_z_radians(),
        boxes2_length=boxlist2.get_length(),
        boxes2_height=boxlist2.get_height(),
        boxes2_width=boxlist2.get_width(),
        boxes2_center=boxlist2.get_center(),
        boxes2_rotation_z_radians=boxlist2.get_rotation_z_radians())


def iou3d(boxlist1, boxlist2):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxlist1: BoxList3d holding N boxes.
    boxlist2: BoxList3d holding M boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
  """
  boxlist1_rotation_matrix = boxlist1.get_rotation_matrix()
  boxlist2_rotation_matrix = boxlist2.get_rotation_matrix()
  if (boxlist1_rotation_matrix is not None) and (boxlist2_rotation_matrix is
                                                 not None):
    return np_box_ops.iou3d_9dof_box(
        boxes1_length=boxlist1.get_length(),
        boxes1_height=boxlist1.get_height(),
        boxes1_width=boxlist1.get_width(),
        boxes1_center=boxlist1.get_center(),
        boxes1_rotation_matrix=boxlist1.get_rotation_matrix(),
        boxes2_length=boxlist2.get_length(),
        boxes2_height=boxlist2.get_height(),
        boxes2_width=boxlist2.get_width(),
        boxes2_center=boxlist2.get_center(),
        boxes2_rotation_matrix=boxlist2.get_rotation_matrix())
  else:
    return np_box_ops.iou3d_7dof_box(
        boxes1_length=boxlist1.get_length(),
        boxes1_height=boxlist1.get_height(),
        boxes1_width=boxlist1.get_width(),
        boxes1_center=boxlist1.get_center(),
        boxes1_rotation_z_radians=boxlist1.get_rotation_z_radians(),
        boxes2_length=boxlist2.get_length(),
        boxes2_height=boxlist2.get_height(),
        boxes2_width=boxlist2.get_width(),
        boxes2_center=boxlist2.get_center(),
        boxes2_rotation_z_radians=boxlist2.get_rotation_z_radians())


def iov3d(boxlist1, boxlist2):
  """Computes pairwise intersection-over-volume between box collections.

  Args:
    boxlist1: BoxList3d holding N boxes.
    boxlist2: BoxList3d holding M boxes.

  Returns:
    a numpy array with shape [N, M] representing pairwise iov scores.
  """
  boxlist1_rotation_matrix = boxlist1.get_rotation_matrix()
  boxlist2_rotation_matrix = boxlist2.get_rotation_matrix()
  if (boxlist1_rotation_matrix is not None) and (boxlist2_rotation_matrix is
                                                 not None):
    return np_box_ops.iov3d_9dof_box(
        boxes1_length=boxlist1.get_length(),
        boxes1_height=boxlist1.get_height(),
        boxes1_width=boxlist1.get_width(),
        boxes1_center=boxlist1.get_center(),
        boxes1_rotation_matrix=boxlist1.get_rotation_matrix(),
        boxes2_length=boxlist2.get_length(),
        boxes2_height=boxlist2.get_height(),
        boxes2_width=boxlist2.get_width(),
        boxes2_center=boxlist2.get_center(),
        boxes2_rotation_matrix=boxlist2.get_rotation_matrix())
  else:
    return np_box_ops.iov3d_7dof_box(
        boxes1_length=boxlist1.get_length(),
        boxes1_height=boxlist1.get_height(),
        boxes1_width=boxlist1.get_width(),
        boxes1_center=boxlist1.get_center(),
        boxes1_rotation_z_radians=boxlist1.get_rotation_z_radians(),
        boxes2_length=boxlist2.get_length(),
        boxes2_height=boxlist2.get_height(),
        boxes2_width=boxlist2.get_width(),
        boxes2_center=boxlist2.get_center(),
        boxes2_rotation_z_radians=boxlist2.get_rotation_z_radians())


def nuscenes_center_distance_measure(boxlist1, boxlist2):
  """Computes pairwise intersection-over-volume between box collections.

  Args:
    boxlist1: BoxList3d holding N boxes.
    boxlist2: BoxList3d holding M boxes.

  Returns:
    A numpy array with shape [N, M] representing pairwise closeness scores
      based on center distance.
  """
  boxes1_center = boxlist1.get_center()
  boxes2_center = boxlist2.get_center()
  boxes1_center_xy = boxes1_center[:, 0:2]
  boxes2_center_xy = boxes2_center[:, 0:2]
  distances = np.linalg.norm(
      np.expand_dims(boxes1_center_xy, axis=1) -
      np.expand_dims(boxes2_center_xy, axis=0),
      axis=2)
  return 1.0 / (1.0 + np.exp(distances))


def gather3d(boxlist, indices, fields=None):
  """Gathers boxes from BoxList3d according to indices and return new BoxList3d.

  By default, gather returns boxes corresponding to the input index list, as
  well as all additional fields stored in the boxlist (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.

  Args:
    boxlist: BoxList3d holding N boxes.
    indices: A 1-d numpy array of type np.int32.
    fields: (optional) List of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.

  Returns:
    subboxlist: A BoxList3d corresponding to the subset of the input BoxList3d
        specified by indices.

  Raises:
    ValueError: If specified field is not contained in boxlist or if the
        indices are not of type np.int32.
  """
  if indices.size:
    if np.amax(indices) >= boxlist.num_boxes() or np.amin(indices) < 0:
      raise ValueError('Indices are out of valid range.')
  subboxlist = copy_boxlist(boxlist=boxlist, indices=indices)
  if fields is None:
    fields = boxlist.get_extra_fields()
  for field in fields:
    extra_field_data = boxlist.get_field(field)
    subboxlist.add_field(field, extra_field_data[indices, Ellipsis])
  return subboxlist


def sort_by_field3d(boxlist, field, order=SortOrder.DESCEND):
  """Sorts boxes and associated fields according to a scalar field.

  A common use case is reordering the boxes according to descending scores.

  Args:
    boxlist: BoxList3d holding N boxes.
    field: A BoxList3d field for sorting and reordering the BoxList3d.
    order: (Optional) 'descend' or 'ascend'. Default is descend.

  Returns:
    sorted_boxlist: A sorted BoxList3d with the field in the specified order.

  Raises:
    ValueError: If specified field does not exist or is not of single dimension.
    ValueError: If the order is not either descend or ascend.
  """
  if not boxlist.has_field(field):
    raise ValueError('Field ' + field + ' does not exist')
  if len(boxlist.get_field(field).shape) != 1:
    raise ValueError('Field ' + field + 'should be single dimension.')
  if order != SortOrder.DESCEND and order != SortOrder.ASCEND:
    raise ValueError('Invalid sort order')

  field_to_sort = boxlist.get_field(field)
  if order == SortOrder.DESCEND:
    sorted_indices = np.lexsort((-field_to_sort,))  # stable sort
  else:
    sorted_indices = np.lexsort((field_to_sort,))  # stable sort
  return gather3d(boxlist, sorted_indices)


def filter_scores_greater_than3d(boxlist, thresh):
  """Filters boxlist to keep only boxes with score exceeding a given threshold.

  This op keeps the collection of boxes whose corresponding scores are
  greater than the input threshold.

  Args:
    boxlist: BoxList3d holding N boxes. Must contain a 'scores' field
      representing detection scores.
    thresh: Scalar threshold.

  Returns:
    A BoxList3d holding M boxes where M <= N.

  Raises:
    ValueError: If boxlist not a BoxList3d object or if it does not
      have a scores field.
  """
  if not isinstance(boxlist, np_box_list.BoxList3d):
    raise ValueError('boxlist must be a BoxList3d')
  if not boxlist.has_field('scores'):
    raise ValueError('input boxlist must have \'scores\' field')
  scores = boxlist.get_field('scores')
  if len(scores.shape) > 2:
    raise ValueError('Scores should have rank 1 or 2')
  if len(scores.shape) == 2 and scores.shape[1] != 1:
    raise ValueError('Scores should have rank 1 or have shape '
                     'consistent with [None, 1]')
  high_score_indices = np.reshape(np.where(np.greater(scores, thresh)),
                                  [-1]).astype(np.int32)
  return gather3d(boxlist, high_score_indices)


def concatenate_boxes3d(boxlists, fields=None):
  """Concatenates list of BoxList3d-s.

  This op concatenates a list of input BoxList3d-s into a larger BoxList.  It
  also handles concatenation of BoxList fields as long as the field tensor
  shapes are equal except for the first dimension.

  Args:
    boxlists: list of BoxList3d objects.
    fields: optional list of fields to also concatenate.  By default, all
      fields from the first BoxList3d in the list are included in the
      concatenation.

  Returns:
    A BoxList3d with number of boxes equal to
      sum([boxlist.num_boxes() for boxlist in boxlists]).

  Raises:
    ValueError: If boxlists is invalid (i.e., is not a list, is empty, or
      contains non BoxList3d objects), or if requested fields are not contained
      in all boxlists.
  """
  if not isinstance(boxlists, list):
    raise ValueError('boxlists should be a list')
  if not boxlists:
    raise ValueError('boxlists should have nonzero length')
  for boxlist in boxlists:
    if not isinstance(boxlist, np_box_list.BoxList3d):
      raise ValueError('all elements of boxlists should be BoxList3d objects')
  length_list = []
  height_list = []
  width_list = []
  center_x_list = []
  center_y_list = []
  center_z_list = []
  rotation_matrix_list = []
  rotation_z_radians_list = []
  for boxlist in boxlists:
    length_list.append(boxlist.get_length())
    height_list.append(boxlist.get_height())
    width_list.append(boxlist.get_width())
    center_x_list.append(boxlist.get_center_x())
    center_y_list.append(boxlist.get_center_y())
    center_z_list.append(boxlist.get_center_z())
    rotation_matrix_list.append(boxlist.get_rotation_matrix())
    rotation_z_radians_list.append(boxlist.get_rotation_z_radians())
  length = np.concatenate(length_list)
  height = np.concatenate(height_list)
  width = np.concatenate(width_list)
  center_x = np.concatenate(center_x_list)
  center_y = np.concatenate(center_y_list)
  center_z = np.concatenate(center_z_list)
  if rotation_matrix_list[0] is None:
    rotation_matrix = None
  else:
    rotation_matrix = np.concatenate(rotation_matrix_list)
  if rotation_z_radians_list[0] is None:
    rotation_z_radians = None
  else:
    rotation_z_radians = np.concatenate(rotation_z_radians_list)
  concatenated = np_box_list.BoxList3d(
      length=length,
      height=height,
      width=width,
      center_x=center_x,
      center_y=center_y,
      center_z=center_z,
      rotation_matrix=rotation_matrix,
      rotation_z_radians=rotation_z_radians)

  if fields is None:
    fields = boxlists[0].get_extra_fields()
  for field in fields:
    first_field_shape = boxlists[0].get_field(field).shape
    first_field_shape = first_field_shape[1:]
    for boxlist in boxlists:
      if not boxlist.has_field(field):
        raise ValueError('boxlist must contain all requested fields')
      field_shape = boxlist.get_field(field).shape
      field_shape = field_shape[1:]
      if field_shape != first_field_shape:
        raise ValueError('field %s must have same shape for all boxlists '
                         'except for the 0th dimension.' % field)
    concatenated_field = np.concatenate(
        [boxlist.get_field(field) for boxlist in boxlists], axis=0)
    concatenated.add_field(field, concatenated_field)
  return concatenated


def non_max_suppression3d(boxlist,
                          max_output_size=10000,
                          iou_threshold=1.0,
                          score_threshold=-10.0):
  """Non maximum suppression.

  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes. In each iteration, the detected bounding box with
  highest score in the available pool is selected.

  Args:
    boxlist: BoxList3d holding N boxes.  Must contain a 'scores' field
      representing detection scores. All scores belong to the same class.
    max_output_size: Maximum number of retained boxes.
    iou_threshold: Intersection over union threshold.
    score_threshold: Minimum score threshold. Remove the boxes with scores
      less than this value. Default value is set to -10. A very low threshold to
      pass pretty much all the boxes, unless the user sets a different score
      threshold.

  Returns:
    A BoxList3d holding M boxes where M <= max_output_size.

  Raises:
    ValueError: If 'scores' field does not exist.
    ValueError: If threshold is not in [0, 1].
    ValueError: If max_output_size < 0.
  """
  if not boxlist.has_field('scores'):
    raise ValueError('Field scores does not exist')
  if iou_threshold < 0. or iou_threshold > 1.0:
    raise ValueError('IOU threshold must be in [0, 1]')
  if max_output_size < 0:
    raise ValueError('max_output_size must be bigger than 0.')

  boxlist = filter_scores_greater_than3d(boxlist, score_threshold)
  if boxlist.num_boxes() == 0:
    return boxlist

  boxlist = sort_by_field3d(boxlist, 'scores')

  # Prevent further computation if NMS is disabled.
  if iou_threshold == 1.0:
    if boxlist.num_boxes() > max_output_size:
      selected_indices = np.arange(max_output_size)
      return gather3d(boxlist, selected_indices)
    else:
      return boxlist

  num_boxes = boxlist.num_boxes()
  # is_index_valid is True only for all remaining valid boxes,
  is_index_valid = np.full(num_boxes, 1, dtype=bool)
  selected_indices = []
  num_output = 0
  for i in range(num_boxes):
    if num_output < max_output_size:
      if is_index_valid[i]:
        num_output += 1
        selected_indices.append(i)
        is_index_valid[i] = False
        valid_indices = np.where(is_index_valid)[0]
        if valid_indices.size == 0:
          break
        intersect_over_union = iou3d(
            boxlist1=copy_boxlist(boxlist=boxlist, indices=[i]),
            boxlist2=copy_boxlist(boxlist=boxlist, indices=valid_indices))
        intersect_over_union = np.squeeze(intersect_over_union, axis=0)
        is_index_valid[valid_indices] = np.logical_and(
            is_index_valid[valid_indices],
            intersect_over_union <= iou_threshold)
  return gather3d(boxlist, np.array(selected_indices))


def multi_class_non_max_suppression3d(boxlist, score_thresh, iou_thresh,
                                      max_output_size):
  """Multi-class version of non maximum suppression.

  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes.  It operates independently for each class for
  which scores are provided (via the scores field of the input box_list),
  pruning boxes with score less than a provided threshold prior to
  applying NMS.

  Args:
    boxlist: BoxList holding N boxes.  Must contain a 'scores' field
      representing detection scores.  This scores field is a tensor that can
      be 1 dimensional (in the case of a single class) or 2-dimensional, which
      which case we assume that it takes the shape [num_boxes, num_classes].
      We further assume that this rank is known statically and that
      scores.shape[1] is also known (i.e., the number of classes is fixed
      and known at graph construction time).
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (boxes that that high IOU overlap
      with previously selected boxes are removed).
    max_output_size: maximum number of retained boxes per class.

  Returns:
    a BoxList holding M boxes with a rank-1 scores field representing
      corresponding scores for each box with scores sorted in decreasing order
      and a rank-1 classes field representing a class label for each box.
  Raises:
    ValueError: if iou_thresh is not in [0, 1] or if input boxlist does not have
      a valid scores field.
  """
  if not 0 <= iou_thresh <= 1.0:
    raise ValueError('thresh must be between 0 and 1')
  if not isinstance(boxlist, np_box_list.BoxList3d):
    raise ValueError('boxlist must be a BoxList3d')
  if not boxlist.has_field('scores'):
    raise ValueError('input boxlist must have \'scores\' field')
  scores = boxlist.get_field('scores')
  if len(scores.shape) == 1:
    scores = np.reshape(scores, [-1, 1])
  elif len(scores.shape) == 2:
    if scores.shape[1] is None:
      raise ValueError('scores field must have statically defined second '
                       'dimension')
  else:
    raise ValueError('scores field must be of rank 1 or 2')
  num_boxes = boxlist.num_boxes()
  num_scores = scores.shape[0]
  num_classes = scores.shape[1]

  if num_boxes != num_scores:
    raise ValueError('Incorrect scores field length: actual vs expected.')

  selected_boxes_list = []
  for class_idx in range(num_classes):
    boxlist_and_class_scores = copy_boxlist(boxlist=boxlist)
    class_scores = np.reshape(scores[0:num_scores, class_idx], [-1])
    boxlist_and_class_scores.add_field('scores', class_scores)
    boxlist_filt = filter_scores_greater_than3d(
        boxlist=boxlist_and_class_scores, thresh=score_thresh)
    nms_result = non_max_suppression3d(boxlist=boxlist_filt,
                                       max_output_size=max_output_size,
                                       iou_threshold=iou_thresh,
                                       score_threshold=score_thresh)
    nms_result.add_field(
        'classes', np.zeros_like(nms_result.get_field('scores')) + class_idx)
    selected_boxes_list.append(nms_result)
  selected_boxes = concatenate_boxes3d(selected_boxes_list)
  sorted_boxes = sort_by_field3d(boxlist=selected_boxes, field='scores')
  return sorted_boxes
