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

"""tensorflow box ops."""

import functools
import numpy as np
import tensorflow as tf
from tf3d.object_detection.box_utils import np_box_list
from tf3d.object_detection.box_utils import np_box_list_ops
from tf3d.object_detection.box_utils import np_box_ops


def pairwise_iou3d(boxes1_length, boxes1_height, boxes1_width, boxes1_center,
                   boxes1_rotation_matrix, boxes2_length, boxes2_height,
                   boxes2_width, boxes2_center, boxes2_rotation_matrix):
  """Return the pairwise IOU between boxes1 and boxes2."""
  boxes1_length = tf.reshape(boxes1_length, [-1])
  boxes1_height = tf.reshape(boxes1_height, [-1])
  boxes1_width = tf.reshape(boxes1_width, [-1])
  boxes1_center = tf.reshape(boxes1_center, [-1, 3])
  boxes1_rotation_matrix = tf.reshape(boxes1_rotation_matrix, [-1, 3, 3])
  boxes2_length = tf.reshape(boxes2_length, [-1])
  boxes2_height = tf.reshape(boxes2_height, [-1])
  boxes2_width = tf.reshape(boxes2_width, [-1])
  boxes2_center = tf.reshape(boxes2_center, [-1, 3])
  boxes2_rotation_matrix = tf.reshape(boxes2_rotation_matrix, [-1, 3, 3])
  iou = tf.numpy_function(np_box_ops.iou3d_9dof_box_pairwise, [
      boxes1_length, boxes1_height, boxes1_width, boxes1_center,
      boxes1_rotation_matrix, boxes2_length, boxes2_height, boxes2_width,
      boxes2_center, boxes2_rotation_matrix
  ], tf.float32)
  return tf.reshape(iou, [-1])


def iou3d(boxes1_length, boxes1_height, boxes1_width, boxes1_center,
          boxes1_rotation_matrix, boxes2_length, boxes2_height,
          boxes2_width, boxes2_center, boxes2_rotation_matrix):
  """Return the IOU between boxes1 and boxes2."""
  boxes1_length = tf.reshape(boxes1_length, [-1])
  boxes1_height = tf.reshape(boxes1_height, [-1])
  boxes1_width = tf.reshape(boxes1_width, [-1])
  boxes1_center = tf.reshape(boxes1_center, [-1, 3])
  boxes1_rotation_matrix = tf.reshape(boxes1_rotation_matrix, [-1, 3, 3])
  boxes2_length = tf.reshape(boxes2_length, [-1])
  boxes2_height = tf.reshape(boxes2_height, [-1])
  boxes2_width = tf.reshape(boxes2_width, [-1])
  boxes2_center = tf.reshape(boxes2_center, [-1, 3])
  boxes2_rotation_matrix = tf.reshape(boxes2_rotation_matrix, [-1, 3, 3])
  n = tf.shape(boxes1_length)[0]
  m = tf.shape(boxes2_length)[0]
  iou = tf.numpy_function(np_box_ops.iou3d_9dof_box, [
      boxes1_length, boxes1_height, boxes1_width, boxes1_center,
      boxes1_rotation_matrix, boxes2_length, boxes2_height, boxes2_width,
      boxes2_center, boxes2_rotation_matrix
  ], tf.float32)
  return tf.reshape(iou, [n, m])


def np_nms(np_boxes_length, np_boxes_height, np_boxes_width, np_boxes_center,
           np_boxes_rotation_matrix, np_boxes_score, score_thresh, iou_thresh,
           max_output_size):
  """Non maximum suppression.

  Args:
    np_boxes_length: A numpy array of size [N, 1].
    np_boxes_height: A numpy array of size [N, 1].
    np_boxes_width: A numpy array of size [N, 1].
    np_boxes_center: A numpy array of size [N, 3].
    np_boxes_rotation_matrix: A numpy array of size [N, 3, 3].
    np_boxes_score: A numpy array of size [N, num_classes].
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (boxes that that high IOU overlap
      with previously selected boxes are removed).
    max_output_size: maximum number of retained boxes per class.

  Returns:
    np_nms_boxes_length: A numpy array of size [N', 1].
    np_nms_boxes_height: A numpy array of size [N', 1].
    np_nms_boxes_width: A numpy array of size [N', 1].
    np_nms_boxes_center: A numpy array of size [N', 3].
    np_nms_boxes_rotation_matrix: A numpy array of size [N', 3, 3].
    np_nms_boxes_class: A numpy array of size [N', 1].
    np_nms_boxes_score: A numpy array of size [N', 1].
  """
  boxlist = np_box_list.BoxList3d(
      length=np.squeeze(np_boxes_length, axis=1),
      height=np.squeeze(np_boxes_height, axis=1),
      width=np.squeeze(np_boxes_width, axis=1),
      center_x=np_boxes_center[:, 0],
      center_y=np_boxes_center[:, 1],
      center_z=np_boxes_center[:, 2],
      rotation_matrix=np.reshape(np_boxes_rotation_matrix, [-1, 3, 3]))
  boxlist.add_field('scores', np_boxes_score)
  boxlist_nms = np_box_list_ops.multi_class_non_max_suppression3d(
      boxlist=boxlist,
      score_thresh=score_thresh,
      iou_thresh=iou_thresh,
      max_output_size=max_output_size)
  return (np.expand_dims(boxlist_nms.get_length(), axis=1),
          np.expand_dims(boxlist_nms.get_height(), axis=1),
          np.expand_dims(boxlist_nms.get_width(), axis=1),
          boxlist_nms.get_center(),
          np.expand_dims(boxlist_nms.get_rotation_matrix(), axis=1),
          np.expand_dims(boxlist_nms.get_field('classes'),
                         axis=1).astype(np.int32),
          np.expand_dims(boxlist_nms.get_field('scores'), axis=1))


def nms(boxes_length, boxes_height, boxes_width, boxes_center,
        boxes_rotation_matrix, boxes_score, score_thresh, iou_thresh,
        max_output_size):
  """Non maximum suppression in tensorflow.

  Args:
    boxes_length: A tf.float32 tensor of size [N, 1].
    boxes_height: A tf.float32 tensor of size [N, 1].
    boxes_width: A tf.float32 tensor of size [N, 1].
    boxes_center: A tf.float32 tensor of size [N, 3].
    boxes_rotation_matrix: A tf.float32 tensor of size [N, 3, 3].
    boxes_score: A tf.float32 tensor of size [N, num_classes].
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (boxes that that high IOU overlap
      with previously selected boxes are removed).
    max_output_size: maximum number of retained boxes per class.

  Returns:
    boxes_nms_length: A tf.float32 tensor of size [N', 1].
    boxes_nms_height: A tf.float32 tensor of size [N', 1].
    boxes_nms_width: A tf.float32 tensor of size [N', 1].
    boxes_nms_center: A tf.float32 tensor of size [N', 3].
    boxes_nms_rotation_matrix: A tf.float32 tensor of size [N', 3, 3].
    boxes_nms_class: A tf.int32 tensor of size [N', 1].
    boxes_nms_score: A tf.float32 tensor of size [N', 1].
  """
  (boxes_nms_length, boxes_nms_height, boxes_nms_width, boxes_nms_center,
   boxes_nms_rotation_matrix, boxes_nms_class,
   boxes_nms_score) = tf.numpy_function(
       func=functools.partial(
           np_nms,
           score_thresh=score_thresh,
           iou_thresh=iou_thresh,
           max_output_size=max_output_size),
       inp=[
           boxes_length, boxes_height, boxes_width, boxes_center,
           boxes_rotation_matrix, boxes_score
       ],
       Tout=[
           tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32,
           tf.float32
       ])
  boxes_nms_length = tf.reshape(boxes_nms_length, [-1, 1])
  boxes_nms_height = tf.reshape(boxes_nms_height, [-1, 1])
  boxes_nms_width = tf.reshape(boxes_nms_width, [-1, 1])
  boxes_nms_center = tf.reshape(boxes_nms_center, [-1, 3])
  boxes_nms_rotation_matrix = tf.reshape(boxes_nms_rotation_matrix, [-1, 3, 3])
  boxes_nms_class = tf.reshape(boxes_nms_class, [-1, 1])
  boxes_nms_score = tf.reshape(boxes_nms_score, [-1, 1])
  return (boxes_nms_length, boxes_nms_height, boxes_nms_width, boxes_nms_center,
          boxes_nms_rotation_matrix, boxes_nms_class, boxes_nms_score)
