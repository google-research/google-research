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

"""Input for the FindIt demo."""

import collections
import itertools

import math
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from flax import traverse_util
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import tensorflow as tf

Array = jnp.ndarray


class Anchor(object):
  """Anchor class for anchor-based object detectors."""

  def __init__(self,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               image_size):
    """Constructs multiscale anchors.

    Args:
      min_level: integer number of minimum level of the output feature pyramid.
      max_level: integer number of maximum level of the output feature pyramid.
      num_scales: integer number representing intermediate scales added
        on each level. For instances, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: list of float numbers representing the aspect ratio anchors
        added on each level. The number indicates the ratio of width to height.
        For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each
        scale level.
      anchor_size: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      image_size: a list of integer numbers or Tensors representing
        [height, width] of the input image size.The image_size should be divided
        by the largest feature stride 2^max_level.
    """
    self.min_level = min_level
    self.max_level = max_level
    self.num_scales = num_scales
    self.aspect_ratios = aspect_ratios
    self.anchor_size = anchor_size
    self.image_size = image_size
    self.boxes = self._generate_boxes()

  def _generate_boxes(self):
    """Generates multiscale anchor boxes.

    Returns:
      a Tensor of shape [N, 4], representing anchor boxes of all levels
      concatenated together.
    """
    boxes_all = []
    for level in range(self.min_level, self.max_level + 1):
      boxes_l = []
      for scale in range(self.num_scales):
        for aspect_ratio in self.aspect_ratios:
          stride = 2 ** level
          intermidate_scale = 2 ** (scale / float(self.num_scales))
          base_anchor_size = self.anchor_size * stride * intermidate_scale
          aspect_x = aspect_ratio ** 0.5
          aspect_y = aspect_ratio ** -0.5
          half_anchor_size_x = base_anchor_size * aspect_x / 2.0
          half_anchor_size_y = base_anchor_size * aspect_y / 2.0
          x = tf.range(stride / 2, self.image_size[1], stride)
          y = tf.range(stride / 2, self.image_size[0], stride)
          xv, yv = tf.meshgrid(x, y)
          xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
          yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
          # Tensor shape Nx4.
          boxes = tf.stack([yv - half_anchor_size_y, xv - half_anchor_size_x,
                            yv + half_anchor_size_y, xv + half_anchor_size_x],
                           axis=1)
          boxes_l.append(boxes)
      # Concat anchors on the same level to tensor shape NxAx4.
      boxes_l = tf.stack(boxes_l, axis=1)
      boxes_l = tf.reshape(boxes_l, [-1, 4])
      boxes_all.append(boxes_l)
    return tf.concat(boxes_all, axis=0)

  def unpack_labels(self, labels, is_box=False):
    """Unpacks an array of labels into multiscales labels.

    Args:
      labels: labels to unpack.
      is_box: to unpack anchor boxes or not. If it is true, will unpack to 2D,
        otherwise, will unpack to 3D.

    Returns:
      unpacked_labels: a dictionary contains unpack labels in different levels.
    """
    unpacked_labels = collections.OrderedDict()
    count = 0
    for level in range(self.min_level, self.max_level + 1):
      feat_size_y = tf.cast(self.image_size[0] / 2 ** level, tf.int32)
      feat_size_x = tf.cast(self.image_size[1] / 2 ** level, tf.int32)
      steps = feat_size_y * feat_size_x * self.anchors_per_location
      if is_box:
        unpacked_labels[level] = tf.reshape(labels[count:count + steps],
                                            [-1, 4])
      else:
        unpacked_labels[level] = tf.reshape(labels[count:count + steps],
                                            [feat_size_y, feat_size_x, -1])
      count += steps
    return unpacked_labels

  @property
  def anchors_per_location(self):
    return self.num_scales * len(self.aspect_ratios)

  @property
  def multilevel_boxes(self):
    return self.unpack_labels(self.boxes, is_box=True)


def normalize_and_resize_image_for_eval_fn(
    data,
    max_level,
    output_size,
    image_feature_name = 'image',
    box_feature_name = 'groundtruth_boxes',
    image_info_feature_name = 'image_info'):
  """Normalizes, resizes and crops the images.

  Args:
    data: the decoded tensor dictionary that contains the raw image, boxes and
      class labels.
    max_level: `int` number of maximum level of the output feature pyramid.
    output_size: `Tensor` or `list` for [height, width] of output image. The
      output_size should be divided by the largest feature stride 2^max_level.
    image_feature_name: feature name for image.
    box_feature_name: feature name for boxes.
    image_info_feature_name: feature name for image info which has the original
      image shape, new image shape and the image scale information.

  Returns:
    data: tensor dictionary with the normalized image and image_info.
  """
  image = data[image_feature_name]
  image_shape = tf.shape(image)[0:2]
  # Normalizes image with mean and std pixel values.
  image = normalize_image(image)
  # Resizes and crops image.
  image, image_info = resize_and_crop_image(
      image,
      output_size,
      padded_size=compute_padded_size(output_size, 2**max_level),
      aug_scale_min=1.0,
      aug_scale_max=1.0)
  data[image_feature_name] = image
  data[image_info_feature_name] = image_info
  # Converts boxes from normalized coordinates to pixel coordinates.
  # Now the coordinates of boxes are w.r.t. the original image.
  data[box_feature_name] = denormalize_boxes(
      data[box_feature_name], image_shape)
  return data


def denormalize_boxes(boxes, image_shape):
  """Converts boxes normalized by [height, width] to pixel coordinates.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    denormalized_boxes: a tensor whose shape is the same as `boxes` representing
      the denormalized boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  with tf.name_scope('denormalize_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.split(image_shape, 2, axis=-1)

    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
    ymin = ymin * height
    xmin = xmin * width
    ymax = ymax * height
    xmax = xmax * width

    denormalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return denormalized_boxes


def resize_and_crop_image(image,
                          desired_size,
                          padded_size,
                          aug_scale_min=1.0,
                          aug_scale_max=1.0,
                          seed=1,
                          method=tf.image.ResizeMethod.BILINEAR):
  """Resizes the input image to output size (RetinaNet style).

  Resize and pad images given the desired output size of the image and
  stride size.

  Here are the preprocessing steps.
  1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `desired_size`.
  2. Pad the rescaled image to the padded_size.

  Args:
    image: a `Tensor` of shape [height, width, 3] representing an image.
    desired_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the desired actual output image size.
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size. Padding will be applied
      after scaling the image to the desired_size.
    aug_scale_min: a `float` with range between [0, 1.0] representing minimum
      random scale applied to desired_size for training scale jittering.
    aug_scale_max: a `float` with range between [1.0, inf] representing maximum
      random scale applied to desired_size for training scale jittering.
    seed: seed for random scale jittering.
    method: function to resize input image to scaled image.

  Returns:
    output_image: `Tensor` of shape [height, width, 3] where [height, width]
      equals to `output_size`.
    image_info: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [desired_height, desired_width],
       [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desired_width] is the actual scaled image size, and [y_scale, x_scale] is
      the scaling factor, which is the ratio of
      scaled dimension / original dimension.
  """
  with tf.name_scope('resize_and_crop_image'):
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

    random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

    if random_jittering:
      random_scale = tf.random_uniform(
          [], aug_scale_min, aug_scale_max, seed=seed)
      scaled_size = tf.round(random_scale * desired_size)
    else:
      scaled_size = desired_size

    scale = tf.minimum(
        scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
    scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
      max_offset = scaled_size - desired_size
      max_offset = tf.where(tf.less(max_offset, 0),
                            tf.zeros_like(max_offset),
                            max_offset)
      offset = max_offset * tf.random_uniform([2,], 0, 1, seed=seed)
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
      scaled_image = scaled_image[
          offset[0]:offset[0] + desired_size[0],
          offset[1]:offset[1] + desired_size[1], :]

    output_image = tf.image.pad_to_bounding_box(
        scaled_image, 0, 0, padded_size[0], padded_size[1])

    image_info = tf.stack([
        image_size,
        tf.constant(desired_size, dtype=tf.float32),
        image_scale,
        tf.cast(offset, tf.float32)])
    return output_image, image_info


def compute_padded_size(desired_size, stride):
  """Compute the padded size given the desired size and the stride.

  The padded size will be the smallest rectangle, such that each dimension is
  the smallest multiple of the stride which is larger than the desired
  dimension. For example, if desired_size = (100, 200) and stride = 32,
  the output padded_size = (128, 224).

  Args:
    desired_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the target output image size.
    stride: an integer, the stride of the backbone network.

  Returns:
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size.
  """
  if isinstance(desired_size, list) or isinstance(desired_size, tuple):
    padded_size = [int(math.ceil(d * 1.0 / stride) * stride)
                   for d in desired_size]
  else:
    padded_size = tf.cast(
        tf.math.ceil(
            tf.cast(desired_size, dtype=tf.float32) / stride) * stride,
        tf.int32)
  return padded_size


def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
  """Normalizes the image to zero mean and unit variance."""
  with tf.name_scope('normalize_image'):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    image -= offset

    scale = tf.constant(scale)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    image /= scale
    return image


def cast_tensor_dict(dict_tensor, dtype):
  """Cast a nested dictionary of tensors to a data type.

  Args:
    dict_tensor: Input nested dictionary of tensors to cast type.
    dtype: The data type to cast to.

  Returns:
    A casted nesteddictionary of tensors.
  """
  casted_tensors = {}
  for k, v in traverse_util.flatten_dict(dict_tensor).items():
    casted_tensors[k] = tf.cast(v, dtype)
  return traverse_util.unflatten_dict(casted_tensors)


def tile_tensor_dict(dict_tensor,
                     repeats,
                     axis = 0):
  """Tile a nested dictionary of tensors along a given axis.

  Args:
    dict_tensor: Input dictionary of tensors to tile.
    repeats: The number of repetition.
    axis: The axis to tile on. Default to 0 the leading dimension.

  Returns:
    A tiled nested dictionary of tensors.
  """
  tiled_tensors = {}
  for k, v in traverse_util.flatten_dict(dict_tensor).items():
    tiled_tensors[k] = tf.repeat(tf.expand_dims(v, axis), repeats, axis=axis)
  return traverse_util.unflatten_dict(tiled_tensors)


def reshape_expr_boxes(
    expressions, boxes, max_exprs_per_image,
    shuffle_exprs):
  """Reshape and pad expressions and boxes to max_exprs_per_image.

  This function reshapes the expressions and boxes to 2D array for easier
  downstream processing in the FindIt models.

  Args:
    expressions: A `tf.Tensor` of expressions with shape:
      [max_boxes_per_image, max_exprs_per_box, max_expr_len].
    boxes: A `tf.Tensor` of boxes corresponding to expressions with shape:
      [max_boxes_per_image, 4].
    max_exprs_per_image: Maximum expressions per image.
    shuffle_exprs: Shuffle the expressions for training time.

  Returns:
    expressions: A `tf.Tensor` of shape [max_exprs_per_image, max_expr_len]
      containing token IDs of expressions for each box. Each expression is
      appended with EOS and cropped/padded to get the fixed shape.
    expr_boxes: A `tf.Tensor` of shape [max_exprs_per_image, 4]
      and contains normalized box coordinates of type
      [y_min, x_min, y_max, x_max]. It is cropped/padded with zeros since
      actual boxes can be less/more than max_boxes_per_image.
    expr_mask: A `tf.Tensor` of shape [max_exprs_per_image] with 1/0 indicating
      whether the expression/box is valid or not.
  """

  _, max_exprs_per_box, max_expr_len = expressions.shape.as_list()
  # [max_boxes_per_image * max_exprs_per_box, max_expr_len]
  expressions = tf.reshape(expressions, (-1, max_expr_len))
  # [max_boxes_per_image * max_exprs_per_box]
  valid_exprs = tf.reduce_any(tf.greater(expressions, 0), axis=-1)
  # [num_valid_exprs]
  valid_indices = tf.where(valid_exprs)[:, 0]
  if shuffle_exprs:
    valid_indices = tf.random.shuffle(valid_indices)
  box_dim = boxes.shape[-1]
  # [max_boxes_per_image, max_exprs_per_box, box_dim].
  expr_boxes = tf.tile(tf.expand_dims(boxes, 1), (1, max_exprs_per_box, 1))
  # [max_boxes_per_image * max_exprs_per_box, box_dim].
  expr_boxes = tf.reshape(expr_boxes, (-1, box_dim))
  # [num_valid_exprs, box_dim].
  expr_boxes = tf.gather(expr_boxes, valid_indices)
  # [num_valid_exprs, max_expr_len].
  expressions = tf.gather(expressions, valid_indices)

  # [max_exprs_per_image, box_dim].
  expr_boxes = clip_or_pad_to_fixed_size(
      expr_boxes, max_exprs_per_image)
  # [max_exprs_per_image, max_expr_len].
  expressions = clip_or_pad_to_fixed_size(
      expressions, max_exprs_per_image)
  # [max_exprs_per_image].
  expr_mask = tf.reduce_any(tf.greater(expressions, 0), axis=-1)
  expr_mask = tf.cast(expr_mask, tf.int32)

  return expressions, expr_boxes, expr_mask


def clip_or_pad_to_fixed_size(input_tensor, size, constant_values=0):
  """Pads data to a fixed length at the first dimension.

  Args:
    input_tensor: `Tensor` with any dimension.
    size: `int` number for the first dimension of output Tensor.
    constant_values: `int` value assigned to the paddings.

  Returns:
    `Tensor` with the first dimension padded to `size`.
  """
  input_shape = input_tensor.get_shape().as_list()
  padding_shape = []

  # Computes the padding length on the first dimension, clip input tensor if it
  # is longer than `size`.
  input_length = tf.shape(input_tensor)[0]
  input_length = tf.clip_by_value(input_length, 0, size)
  input_tensor = input_tensor[:input_length]

  padding_length = tf.maximum(0, size - input_length)
  padding_shape.append(padding_length)

  # Copies shapes of the rest of input shape dimensions.
  for i in range(1, len(input_shape)):
    padding_shape.append(tf.shape(input_tensor)[i])

  # Pads input tensor to the fixed first dimension.
  paddings = tf.cast(constant_values * tf.ones(padding_shape),
                     input_tensor.dtype)
  padded_tensor = tf.concat([input_tensor, paddings], axis=0)
  output_shape = input_shape
  output_shape[0] = size
  padded_tensor.set_shape(output_shape)
  return padded_tensor


def fake_tokenizer(text, output_shape):
  """Return fake tokens of output_shape."""
  del text
  return tf.ones(shape=output_shape, dtype=tf.int32)


def ref_expr_map_fn(
    features,
    is_training = False,
    image_feature_name = ('image',),
    box_feature_name = ('objects', 'bbox'),
    expr_feature_name = ('objects', 'refexp', 'raw'),
    source_id_feature_name = ('image/id',),
    box_output_name = 'gt_boxes',
    class_output_name = 'gt_classes',
    anchor_boxes_output_name = 'anchor_boxes',
    image_info_output_name = 'image_info',
    source_id_output_name = 'source_id',
    output_size = (128, 128),
    min_level = 2,
    max_level = 6,
    max_expr_len = 64,
    dtype = jnp.float32,
    image_output_name = 'image',
    expr_output_name = 'text',
    label_output_name = 'labels',
    class_feature_name = 'classes',
    max_exprs_per_image = 1,
    max_boxes_per_image = 100,
    max_exprs_per_box = 10,
):
  """Referring expression map function without anchors.

  Args:
    features: a nested dictionary of features.
    is_training: Training mode or not.
    image_feature_name: a string tuple representing the image feature name in
      the flattened features.
    box_feature_name: a string tuple representing the box feature name in the
      flattened features.
    expr_feature_name: a string tuple representing the expressions feature name
      in the flattened features.
    source_id_feature_name: A string tuple representing the source id feature
      name in the flattened features.
    box_output_name: the key for gt boxes in the output dictionary.
    class_output_name: the key for gt classes in the output dictionary.
    anchor_boxes_output_name: the key for anchor_boxes in output dictionary.
    image_info_output_name: the key for image_info in output dictionary.
    source_id_output_name: The key for source id in the output dictionary.
    output_size: the image output size.
    min_level: max level of feature pyramid to use.
    max_level: max level of feature pyramid to use.
    max_expr_len: maximum expression length.
    dtype: The output dtype.
    image_output_name: the key for image in the output dictionary.
    expr_output_name: the key for expressions in the output dictionary.
    label_output_name: the key for labels in output dictionary.
    class_feature_name: a string representing the class feature name to provide
      for RPN label generation.
    max_exprs_per_image: maximum expressions per image.
    max_boxes_per_image: maximum boxes per image.
    max_exprs_per_box: maximum expressions per box.

  Returns:
    a dictionary like {<image_output_name>: image, <box_output_name>: boxes,
    <expr_output_name>: exprs}.
    * image is of shape <output_size> that is scaled/padded so that the
    output_size is a bounding box of the original image.
    * boxes is of shape [max_exprs_per_image, 4] and contains normalized box
    coordinates of type [y_min, x_min, y_max, x_max]. It is cropped/padded with
    zeros since actual boxes can be less/more than max_boxes_per_image.
    * exprs is of shape [max_exprs_per_image, max_expr_len]
    containing token IDs of expressions for each box. Each expression is
    appended with EOS and cropped/padded to get the fixed shape. Type is int32.
    * valid_expr_mask is of shape [max_exprs_per_image] with 1/0 indicating
    whether the expression/box is valid or not. Type is int32.
    * image_info: a 2D `tf.Tensor` that encodes the information of the image and
      the applied preprocessing. It is in the format of
      [[original_height, original_width], [scaled_height, scaled_width].
    * anchor_boxes: ordered dictionary with keys
      [min_level, min_level+1, ..., max_level]. The values are tensor with
      shape [height_l, width_l, 4] representing anchor boxes at each level.
    Optional RPN label fields below:
    * rpn_score_targets: ordered dictionary with keys
      [min_level, min_level+1, ..., max_level]. The values are tensor with
      shape [height_l, width_l, anchors_per_location]. The height_l and
      width_l represent the dimension of class logits at l-th level.
    * rpn_box_targets: ordered dictionary with keys
      [min_level, min_level+1, ..., max_level]. The values are tensor with
      shape [height_l, width_l, anchors_per_location * 4]. The height_l and
      width_l represent the dimension of bounding box regression output at
      l-th level.
  """
  if is_training and max_exprs_per_image != 1:
    raise ValueError('Only support max_exprs_per_image=1 training for now!')

  features = traverse_util.flatten_dict(features)
  source_id = features[source_id_feature_name]
  # [height, width, channels]
  image = features[image_feature_name]
  # [boxes, 4]
  boxes = features[box_feature_name]
  # ragged tensor with shape [boxes, expr_per_box]
  expressions = features[expr_feature_name]
  # [max_boxes_per_image, 4]
  boxes = clip_or_pad_to_fixed_size(boxes, max_boxes_per_image)

  # [max_boxes_per_image, max_exprs_per_box, max_expr_len]
  expressions = fake_tokenizer(
      expressions,
      output_shape=[max_boxes_per_image, max_exprs_per_box, max_expr_len])

  expressions, boxes, expr_mask = reshape_expr_boxes(
      expressions, boxes, max_exprs_per_image, shuffle_exprs=is_training)

  data = {
      image_feature_name: image,
      box_feature_name: boxes,
      class_feature_name: expr_mask,
  }
  data = normalize_and_resize_image_for_eval_fn(
      data,
      image_feature_name=image_feature_name,
      box_feature_name=box_feature_name,
      output_size=output_size,
      max_level=max_level)

  image = data[image_feature_name]
  boxes = data[box_feature_name]
  classes = data[class_feature_name]
  num_scales = 1
  aspect_ratios = (1.0, 2.0, 0.5)
  anchor_size = 8
  image_info_feature_name = 'image_info'
  image_height, image_width, _ = image.get_shape().as_list()
  input_anchor = Anchor(min_level, max_level, num_scales, aspect_ratios,
                        anchor_size, (image_height, image_width))
  rpn_labels = {
      'anchor_boxes': input_anchor.multilevel_boxes,
      'image_info': data[image_info_feature_name],
  }
  # Tile the image by max_exprs_per_image for evaluation by first unbatch
  # and then batch.
  # For training, index the first element to remove the first dimension.
  # This assumes max_exprs_per_image = 1.
  image = data[image_feature_name]
  boxes = data[box_feature_name]
  classes = data[class_feature_name]

  # Pad one extra dimension for N > 1 gt boxes in detection down the road.
  boxes = boxes[:, None, :]
  image = tf.repeat(image[None, Ellipsis], max_exprs_per_image, axis=0)
  source_id = tf.repeat(source_id, max_exprs_per_image, axis=0)
  rpn_labels = tile_tensor_dict(
      rpn_labels, max_exprs_per_image, axis=0)
  rpn_labels = cast_tensor_dict(rpn_labels, dtype)

  labels = {
      box_output_name: tf.cast(boxes, dtype),
      class_output_name: tf.cast(classes, tf.int32),
      anchor_boxes_output_name: rpn_labels['anchor_boxes'],
      image_info_output_name: rpn_labels['image_info'],
  }

  output = {
      image_output_name: tf.cast(image, dtype),
      expr_output_name: tf.cast(expressions, tf.int32),
      source_id_output_name: tf.cast(source_id, tf.int64),
      label_output_name: {'refexp': labels},
  }
  return output


def get_input(loader_fn = None,
              map_fn = None,
              batch_size = 1,
              feature_names = ('image', 'text', 'labels'),
              ):
  """input function to obtain a data generator.

  Args:
    loader_fn: a function which returns a `tf.data.Dataset`.
    map_fn: a function that operates on individual items in the dataset. NOTE:
      this is called before batching hence it operates on a per example basis.
    batch_size: the global batch size.
    feature_names: a sequence of feature names to select from the dataset.

  Returns:
    A generator that yields the features as a list of values.
  """
  dataset = loader_fn()
  dataset = dataset.map(
      map_fn,
      num_parallel_calls=1,
      deterministic=True)
  dataset = dataset.unbatch()
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(
      lambda x: [x[name] for name in feature_names],
      num_parallel_calls=1,
      deterministic=True)
  return dataset.as_numpy_iterator()


def refexp_input_dict_from_image_text(
    image, text):
  """Get input dict for Ref/Loc/Det task from an image and a string.

  This generates the input dictionary in the format as expected by
  referring expression map function. Only the image and texts are provided by
  the user and the other fields are fake inputs.

  Args:
    image: A numpy array of shape (height, width, 3).
    text: A string.

  Returns:
    data: A dictionary in the format of the referring expression parser.
  """
  if len(image.shape) != 3:
    raise ValueError('Image must have 3 dimension!')
  if image.shape[-1] != 3:
    raise ValueError('Image must have 3 channels!')

  image = tf.constant(image, dtype=tf.uint8)
  image.set_shape([None, None, 3])
  text = tf.ragged.constant([[text]], dtype=tf.string)
  # fake boxes un-used by FI models in the predict mode.
  boxes = tf.zeros((1, 4), tf.float32)
  boxes.set_shape([1, 4])
  data = {
      'image': image, 'image/id': tf.constant(1, tf.int64),
      'objects': {'refexp': {'raw': text}, 'bbox': boxes}
  }
  return data


def tfds_from_tensor_dict(tensor_dict):
  """Generate tf.data.Dataset from a dictionary of tensors.

  Args:
    tensor_dict: A dictionary of tensors.

  Returns:
    A tf.data.Dataset converted from the dictionary of tensors.
  """
  def map_fn(x):
    if isinstance(x, tf.RaggedTensor):
      spec_fn = tf.RaggedTensorSpec
    else:
      spec_fn = tf.TensorSpec
    return spec_fn(shape=x.shape, dtype=x.dtype)

  generator = lambda: itertools.repeat(tensor_dict)
  # Define output signatures so that the shapes will not be None as a result of
  # tf.data.Dataset.from_generator. See tf.data.Dataset.from_generator
  # docstring for more details.
  output_signature = tree_map(map_fn, tensor_dict)
  return tf.data.Dataset.from_generator(generator,
                                        output_signature=output_signature)


def fake_image_text():
  """Generate fake image/text inputs."""
  image = np.zeros((128, 128, 3))
  text = 'Test string.'
  return image, text

