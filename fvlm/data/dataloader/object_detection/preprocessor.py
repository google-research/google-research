# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Preprocess images and bounding boxes for detection.

We perform two sets of operations in preprocessing stage:
(a) operations that are applied to both training and testing data,
(b) operations that are applied only to training data for the purpose of
    data augmentation.

A preprocessing function receives a set of inputs,
e.g. an image and bounding boxes,
performs an operation on them, and returns them.
Some examples are: randomly cropping the image, randomly mirroring the image,
                   randomly changing the brightness, contrast, hue and
                   randomly jittering the bounding boxes.

The image is a rank 4 tensor: [1, height, width, channels] with
dtype=tf.float32. The groundtruth_boxes is a rank 2 tensor: [N, 4] where
in each row there is a box with [ymin xmin ymax xmax].
Boxes are in normalized coordinates meaning
their coordinate values range in [0, 1]

Important Note: In tensor_dict, images is a rank 4 tensor, but preprocessing
functions receive a rank 3 tensor for processing the image. Thus, inside the
preprocess function we squeeze the image to become a rank 3 tensor and then
we pass it to the functions. At the end of the preprocess we expand the image
back to rank 4.
"""
import sys

from data.dataloader.object_detection import box_list
from data.dataloader.object_detection import box_list_ops
import numpy as np
import tensorflow.compat.v1 as tf


def _flip_boxes_left_right(boxes):
  """Left-right flip the boxes.

  Args:
    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].

  Returns:
    Flipped boxes.
  """
  ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
  flipped_xmin = tf.subtract(1.0, xmax)
  flipped_xmax = tf.subtract(1.0, xmin)
  flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
  return flipped_boxes


def _flip_masks_left_right(masks):
  """Left-right flip masks.

  Args:
    masks: rank 3 float32 tensor with shape
      [num_instances, height, width] representing instance masks.

  Returns:
    flipped masks: rank 3 float32 tensor with shape
      [num_instances, height, width] representing instance masks.
  """
  return tf.reverse(masks, axis=[2])


def keypoint_flip_horizontal(keypoints, flip_point, flip_permutation,
                             scope=None):
  """Flips the keypoints horizontally around the flip_point.

  This operation flips the x coordinate for each keypoint around the flip_point
  and also permutes the keypoints in a manner specified by flip_permutation.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    flip_point:  (float) scalar tensor representing the x coordinate to flip the
      keypoints around.
    flip_permutation: rank 1 int32 tensor containing the keypoint flip
      permutation. This specifies the mapping from original keypoint indices
      to the flipped keypoint indices. This is used primarily for keypoints
      that are not reflection invariant. E.g. Suppose there are 3 keypoints
      representing ['head', 'right_eye', 'left_eye'], then a logical choice for
      flip_permutation might be [0, 2, 1] since we want to swap the 'left_eye'
      and 'right_eye' after a horizontal flip.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'FlipHorizontal'):
    keypoints = tf.transpose(keypoints, [1, 0, 2])
    keypoints = tf.gather(keypoints, flip_permutation)
    v, u = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    u = flip_point * 2.0 - u
    new_keypoints = tf.concat([v, u], 2)
    new_keypoints = tf.transpose(new_keypoints, [1, 0, 2])
    return new_keypoints


def keypoint_change_coordinate_frame(keypoints, window, scope=None):
  """Changes coordinate frame of the keypoints to be relative to window's frame.

  Given a window of the form [y_min, x_min, y_max, x_max], changes keypoint
  coordinates from keypoints of shape [num_instances, num_keypoints, 2]
  to be relative to this window.

  An example use case is data augmentation: where we are given groundtruth
  keypoints and would like to randomly crop the image to some window. In this
  case we need to change the coordinate frame of each groundtruth keypoint to be
  relative to this new window.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window we should change the coordinate frame to.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'ChangeCoordinateFrame'):
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    new_keypoints = box_list_ops.scale(
        keypoints - [window[0], window[1]], 1.0 / win_height, 1.0 / win_width)
    return new_keypoints


def keypoint_prune_outside_window(keypoints, window, scope=None):
  """Prunes keypoints that fall outside a given window.

  This function replaces keypoints that fall outside the given window with nan.
  See also clip_to_window which clips any keypoints that fall outside the given
  window.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window outside of which the op should prune the keypoints.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'PruneOutsideWindow'):
    y, x = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)

    valid_indices = tf.logical_and(
        tf.logical_and(y >= win_y_min, y <= win_y_max),
        tf.logical_and(x >= win_x_min, x <= win_x_max))

    new_y = tf.where(valid_indices, y, np.nan * tf.ones_like(y))
    new_x = tf.where(valid_indices, x, np.nan * tf.ones_like(x))
    new_keypoints = tf.concat([new_y, new_x], 2)

    return new_keypoints


def random_horizontal_flip(image,
                           boxes=None,
                           masks=None,
                           roi_boxes=None,
                           keypoints=None,
                           keypoint_flip_permutation=None,
                           seed=None):
  """Randomly flips the image and detections horizontally.

  The probability of flipping the image is 50%.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    roi_boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    keypoints: (optional) rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]. The keypoints are in y-x
               normalized coordinates.
    keypoint_flip_permutation: rank 1 int32 tensor containing the keypoint flip
                               permutation.
    seed: random seed

  Returns:
    image: image which is the same shape as input image.

    If boxes, masks, keypoints, and keypoint_flip_permutation are not None,
    the function also returns the following tensors.

    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    roi_boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]

  Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  if keypoints is not None and keypoint_flip_permutation is None:
    raise ValueError(
        'keypoints are provided but keypoints_flip_permutation is not provided')

  with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
    result = []
    # random variable defining whether to do flip or not
    do_a_flip_random = tf.greater(tf.random_uniform([], seed=seed), 0.5)

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
    result.append(image)

    # flip boxes
    if boxes is not None:
      boxes = tf.cond(do_a_flip_random, lambda: _flip_boxes_left_right(boxes),
                      lambda: boxes)
      result.append(boxes)

    # flip masks
    if masks is not None:
      masks = tf.cond(do_a_flip_random, lambda: _flip_masks_left_right(masks),
                      lambda: masks)
      result.append(masks)

    # flip roi_boxes
    if roi_boxes is not None:
      roi_boxes = tf.cond(do_a_flip_random,
                          lambda: _flip_boxes_left_right(roi_boxes),
                          lambda: roi_boxes)
      result.append(roi_boxes)

    # flip keypoints
    if keypoints is not None and keypoint_flip_permutation is not None:
      permutation = keypoint_flip_permutation
      keypoints = tf.cond(
          do_a_flip_random,
          lambda: keypoint_flip_horizontal(keypoints, 0.5, permutation),
          lambda: keypoints)
      result.append(keypoints)

    return tuple(result)


def _compute_new_static_size(image, min_dimension, max_dimension):
  """Compute new static shape for resize_to_range method."""
  image_shape = image.get_shape().as_list()
  orig_height = image_shape[0]
  orig_width = image_shape[1]
  num_channels = image_shape[2]
  orig_min_dim = min(orig_height, orig_width)
  # Calculates the larger of the possible sizes
  large_scale_factor = min_dimension / float(orig_min_dim)
  # Scaling orig_(height|width) by large_scale_factor will make the smaller
  # dimension equal to min_dimension, save for floating point rounding errors.
  # For reasonably-sized images, taking the nearest integer will reliably
  # eliminate this error.
  large_height = int(round(orig_height * large_scale_factor))
  large_width = int(round(orig_width * large_scale_factor))
  large_size = [large_height, large_width]
  if max_dimension:
    # Calculates the smaller of the possible sizes, use that if the larger
    # is too big.
    orig_max_dim = max(orig_height, orig_width)
    small_scale_factor = max_dimension / float(orig_max_dim)
    # Scaling orig_(height|width) by small_scale_factor will make the larger
    # dimension equal to max_dimension, save for floating point rounding
    # errors. For reasonably-sized images, taking the nearest integer will
    # reliably eliminate this error.
    small_height = int(round(orig_height * small_scale_factor))
    small_width = int(round(orig_width * small_scale_factor))
    small_size = [small_height, small_width]
    new_size = large_size
    if max(large_size) > max_dimension:
      new_size = small_size
  else:
    new_size = large_size
  return tf.constant(new_size + [num_channels])


def _compute_new_dynamic_size(image, min_dimension, max_dimension):
  """Compute new dynamic shape for resize_to_range method."""
  image_shape = tf.shape(image)
  orig_height = tf.to_float(image_shape[0])
  orig_width = tf.to_float(image_shape[1])
  num_channels = image_shape[2]
  orig_min_dim = tf.minimum(orig_height, orig_width)
  # Calculates the larger of the possible sizes
  min_dimension = tf.constant(min_dimension, dtype=tf.float32)
  large_scale_factor = min_dimension / orig_min_dim
  # Scaling orig_(height|width) by large_scale_factor will make the smaller
  # dimension equal to min_dimension, save for floating point rounding errors.
  # For reasonably-sized images, taking the nearest integer will reliably
  # eliminate this error.
  large_height = tf.to_int32(tf.round(orig_height * large_scale_factor))
  large_width = tf.to_int32(tf.round(orig_width * large_scale_factor))
  large_size = tf.stack([large_height, large_width])
  if max_dimension:
    # Calculates the smaller of the possible sizes, use that if the larger
    # is too big.
    orig_max_dim = tf.maximum(orig_height, orig_width)
    max_dimension = tf.constant(max_dimension, dtype=tf.float32)
    small_scale_factor = max_dimension / orig_max_dim
    # Scaling orig_(height|width) by small_scale_factor will make the larger
    # dimension equal to max_dimension, save for floating point rounding
    # errors. For reasonably-sized images, taking the nearest integer will
    # reliably eliminate this error.
    small_height = tf.to_int32(tf.round(orig_height * small_scale_factor))
    small_width = tf.to_int32(tf.round(orig_width * small_scale_factor))
    small_size = tf.stack([small_height, small_width])
    new_size = tf.cond(
        tf.to_float(tf.reduce_max(large_size)) > max_dimension,
        lambda: small_size, lambda: large_size)
  else:
    new_size = large_size
  return tf.stack(tf.unstack(new_size) + [num_channels])


def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
                    pad_to_max_dimension=False):
  """Resizes an image so its dimensions are within the provided value.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum dimension is equal to the
     provided value without the other dimension exceeding max_dimension,
     then do so.
  2. Otherwise, resize so the largest dimension is equal to max_dimension.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks.
    min_dimension: (optional) (scalar) desired size of the smaller image
                   dimension.
    max_dimension: (optional) (scalar) maximum allowed size
                   of the larger image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
            BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
                   and output. Defaults to False.
    pad_to_max_dimension: Whether to resize the image and pad it with zeros
      so the resulting image is of the spatial size
      [max_dimension, max_dimension]. If masks are included they are padded
      similarly.

  Returns:
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      min(new_height, new_width) == min_dimension or
      max(new_height, new_width) == max_dimension.
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width].
    resized_image_shape: A 1D tensor of shape [3] containing shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
    if image.get_shape().is_fully_defined():
      new_size = _compute_new_static_size(image, min_dimension, max_dimension)
    else:
      new_size = _compute_new_dynamic_size(image, min_dimension, max_dimension)
    new_image = tf.image.resize_images(
        image, new_size[:-1], method=method, align_corners=align_corners)

    if pad_to_max_dimension:
      new_image = tf.image.pad_to_bounding_box(
          new_image, 0, 0, max_dimension, max_dimension)

    result = [new_image]
    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize_images(
          new_masks,
          new_size[:-1],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          align_corners=align_corners)
      new_masks = tf.squeeze(new_masks, 3)
      if pad_to_max_dimension:
        new_masks = tf.image.pad_to_bounding_box(
            new_masks, 0, 0, max_dimension, max_dimension)
      result.append(new_masks)

    result.append(new_size)
    return result


def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
  """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

  Args:
    boxlist_to_copy_to: BoxList to which extra fields are copied.
    boxlist_to_copy_from: BoxList from which fields are copied.

  Returns:
    boxlist_to_copy_to with extra fields.
  """
  for field in boxlist_to_copy_from.get_extra_fields():
    boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(field))
  return boxlist_to_copy_to


def box_list_scale(boxlist, y_scale, x_scale, scope=None):
  """scale box coordinates in x and y dimensions.

  Args:
    boxlist: BoxList holding N boxes
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    boxlist: BoxList holding N boxes
  """
  with tf.name_scope(scope, 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = box_list.BoxList(
        tf.concat([y_min, x_min, y_max, x_max], 1))
    return _copy_extra_fields(scaled_boxlist, boxlist)


def keypoint_scale(keypoints, y_scale, x_scale, scope=None):
  """Scales keypoint coordinates in x and y dimensions.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope, 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    new_keypoints = keypoints * [[[y_scale, x_scale]]]
    return new_keypoints


def scale_boxes_to_pixel_coordinates(image, boxes, keypoints=None):
  """Scales boxes from normalized to pixel coordinates.

  Args:
    image: A 3D float32 tensor of shape [height, width, channels].
    boxes: A 2D float32 tensor of shape [num_boxes, 4] containing the bounding
      boxes in normalized coordinates. Each row is of the form
      [ymin, xmin, ymax, xmax].
    keypoints: (optional) rank 3 float32 tensor with shape
      [num_instances, num_keypoints, 2]. The keypoints are in y-x normalized
      coordinates.

  Returns:
    image: unchanged input image.
    scaled_boxes: a 2D float32 tensor of shape [num_boxes, 4] containing the
      bounding boxes in pixel coordinates.
    scaled_keypoints: a 3D float32 tensor with shape
      [num_instances, num_keypoints, 2] containing the keypoints in pixel
      coordinates.
  """
  boxlist = box_list.BoxList(boxes)
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  scaled_boxes = box_list_scale(boxlist, image_height, image_width).get()
  result = [image, scaled_boxes]
  if keypoints is not None:
    scaled_keypoints = keypoint_scale(keypoints, image_height, image_width)
    result.append(scaled_keypoints)
  return tuple(result)


def _strict_random_crop_image(image,
                              boxes,
                              labels,
                              masks=None,
                              keypoints=None,
                              min_object_covered=1.0,
                              aspect_ratio_range=(0.75, 1.33),
                              area_range=(0.1, 1.0),
                              overlap_thresh=0.3,
                              clip_boxes=True):
  """Performs random crop.

  Note: Keypoint coordinates that are outside the crop will be set to NaN, which
  is consistent with the original keypoint encoding for non-existing keypoints.
  This function always crops the image and is supposed to be used by
  `random_crop_image` function which sometimes returns the image unchanged.

  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    boxes: rank 2 float32 tensor containing the bounding boxes with shape
           [num_instances, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    labels: rank 1 int32 tensor containing the object classes.
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]. The keypoints are in y-x
               normalized coordinates.
    min_object_covered: the cropped image must cover at least this fraction of
                        at least one of the input bounding boxes.
    aspect_ratio_range: allowed range for aspect ratio of cropped image.
    area_range: allowed range for area ratio between cropped image and the
                original image.
    overlap_thresh: minimum overlap thresh with new cropped
                    image to keep the box.
    clip_boxes: whether to clip the boxes to the cropped image.

  Returns:
    image: image which is the same rank as input image.
    boxes: boxes which is the same rank as input boxes.
           Boxes are in normalized form.
    labels: new labels.

    If label_weights, multiclass_scores, masks, or keypoints is not None, the
    function also returns:
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]
  """
  with tf.name_scope('RandomCropImage', values=[image, boxes]):
    image_shape = tf.shape(image)

    # boxes are [N, 4]. Lets first make them [N, 1, 4].
    boxes_expanded = tf.expand_dims(
        tf.clip_by_value(
            boxes, clip_value_min=0.0, clip_value_max=1.0), 1)

    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.sample_distorted_bounding_box,
        image_shape,
        bounding_boxes=boxes_expanded,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)

    im_box_begin, im_box_size, im_box = sample_distorted_bounding_box

    new_image = tf.slice(image, im_box_begin, im_box_size)
    new_image.set_shape([None, None, image.get_shape()[2]])

    # [1, 4]
    im_box_rank2 = tf.squeeze(im_box, axis=[0])
    # [4]
    im_box_rank1 = tf.squeeze(im_box)

    boxlist = box_list.BoxList(boxes)
    boxlist.add_field('labels', labels)

    im_boxlist = box_list.BoxList(im_box_rank2)

    # remove boxes that are outside cropped image
    boxlist, inside_window_ids = box_list_ops.prune_completely_outside_window(
        boxlist, im_box_rank1)

    # remove boxes that are outside image
    overlapping_boxlist, keep_ids = box_list_ops.prune_non_overlapping_boxes(
        boxlist, im_boxlist, overlap_thresh)

    # change the coordinate of the remaining boxes
    new_labels = overlapping_boxlist.get_field('labels')
    new_boxlist = box_list_ops.change_coordinate_frame(overlapping_boxlist,
                                                       im_box_rank1)
    new_boxes = new_boxlist.get()
    if clip_boxes:
      new_boxes = tf.clip_by_value(
          new_boxes, clip_value_min=0.0, clip_value_max=1.0)

    result = [new_image, new_boxes, new_labels]

    if masks is not None:
      masks_of_boxes_inside_window = tf.gather(masks, inside_window_ids)
      masks_of_boxes_completely_inside_window = tf.gather(
          masks_of_boxes_inside_window, keep_ids)
      masks_box_begin = [0, im_box_begin[0], im_box_begin[1]]
      masks_box_size = [-1, im_box_size[0], im_box_size[1]]
      new_masks = tf.slice(
          masks_of_boxes_completely_inside_window,
          masks_box_begin, masks_box_size)
      result.append(new_masks)

    if keypoints is not None:
      keypoints_of_boxes_inside_window = tf.gather(keypoints, inside_window_ids)
      keypoints_of_boxes_completely_inside_window = tf.gather(
          keypoints_of_boxes_inside_window, keep_ids)
      new_keypoints = keypoint_change_coordinate_frame(
          keypoints_of_boxes_completely_inside_window, im_box_rank1)
      if clip_boxes:
        new_keypoints = keypoint_prune_outside_window(
            new_keypoints, [0.0, 0.0, 1.0, 1.0])
      result.append(new_keypoints)

    return tuple(result)


def random_crop_image(image,
                      boxes,
                      labels,
                      masks=None,
                      keypoints=None,
                      min_object_covered=1.0,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.1, 1.0),
                      overlap_thresh=0.3,
                      clip_boxes=True,
                      random_coef=0.0,
                      seed=None):
  """Randomly crops the image.

  Given the input image and its bounding boxes, this op randomly
  crops a subimage.  Given a user-provided set of input constraints,
  the crop window is resampled until it satisfies these constraints.
  If within 100 trials it is unable to find a valid crop, the original
  image is returned. See the Args section for a description of the input
  constraints. Both input boxes and returned Boxes are in normalized
  form (e.g., lie in the unit square [0, 1]).
  This function will return the original image with probability random_coef.

  Note: Keypoint coordinates that are outside the crop will be set to NaN, which
  is consistent with the original keypoint encoding for non-existing keypoints.

  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    boxes: rank 2 float32 tensor containing the bounding boxes with shape
           [num_instances, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    labels: rank 1 int32 tensor containing the object classes.
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]. The keypoints are in y-x
               normalized coordinates.
    min_object_covered: the cropped image must cover at least this fraction of
                        at least one of the input bounding boxes.
    aspect_ratio_range: allowed range for aspect ratio of cropped image.
    area_range: allowed range for area ratio between cropped image and the
                original image.
    overlap_thresh: minimum overlap thresh with new cropped
                    image to keep the box.
    clip_boxes: whether to clip the boxes to the cropped image.
    random_coef: a random coefficient that defines the chance of getting the
                 original image. If random_coef is 0, we will always get the
                 cropped image, and if it is 1.0, we will always get the
                 original image.
    seed: random seed.

  Returns:
    image: Image shape will be [new_height, new_width, channels].
    boxes: boxes which is the same rank as input boxes. Boxes are in normalized
           form.
    labels: new labels.

    If label_weights, multiclass_scores, masks, or keypoints is not None, the
    function also returns:
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]
  """

  def strict_random_crop_image_fn():
    return _strict_random_crop_image(
        image,
        boxes,
        labels,
        masks=masks,
        keypoints=keypoints,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        overlap_thresh=overlap_thresh,
        clip_boxes=clip_boxes)

  # avoids tf.cond to make faster RCNN training on borg. See b/140057645.
  if random_coef < sys.float_info.min:
    result = strict_random_crop_image_fn()
  else:
    do_a_crop_random = tf.greater(tf.random_uniform([], seed=seed), random_coef)

    outputs = [image, boxes, labels]

    if masks is not None:
      outputs.append(masks)
    if keypoints is not None:
      outputs.append(keypoints)

    result = tf.cond(do_a_crop_random, strict_random_crop_image_fn,
                     lambda: tuple(outputs))
  return result
