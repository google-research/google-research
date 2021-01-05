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

# Lint as: python3
r"""Preprocessing helper functions for resnet wide.

Code based upon resnet cloud pre-processing (tensorflow/tpu/blob/master/
models/official/resnet/resnet_preprocessing)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

IMAGE_SIZE = 224
CROP_PADDING = 32


def crop_image(x, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes."""
  original_shape = tf.shape(x)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(x), 3), ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  with tf.control_dependencies([size_assertion]):
    x = tf.slice(x, offsets, cropped_shape)
    x = tf.reshape(x, cropped_shape)
  return x


def bounding_box_crop(x,
                      bbox,
                      min_object_covered=0.1,
                      aspect_ratio_range=(0.75, 1.33),
                      area_range=(0.05, 1.0),
                      max_attempts=100):
  """Generates cropped_image using a one of the bboxes randomly distorted."""

  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      tf.shape(x),
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  cropped_image = tf.slice(x, bbox_begin, bbox_size)
  return cropped_image, distort_bbox


def random_crop_image(x, size):
  """Make a random crop of the image."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  random_image, bbox = bounding_box_crop(
      x,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=1)
  condition_ = compare_dims(tf.shape(x), tf.shape(random_image), 3)
  x = tf.cond(condition_, lambda: center_crop(rescale_image(x, size), size),
              lambda: tf.image.resize_bicubic([random_image], [size, size])[0])
  return x


def flip_image(image):
  """Random horizontal image flip."""
  image = tf.image.random_flip_left_right(image)
  return image


def compare_dims(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are true."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def rescale_image(x, size):
  """Rescale the image by scaling the smaller spatial dimension to `size`."""

  shape = tf.cast(tf.shape(x), tf.float32)
  w_greater = tf.greater(shape[0], shape[1])
  shape = tf.cond(
      w_greater, lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
      lambda: tf.cast([size, shape[1] / shape[0] * size], tf.int32))

  return tf.image.resize_bicubic([x], shape)[0]


def center_crop(image, size):
  """Crops to center of image with specified `size`."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  offset_height = ((image_height - size) + 1) / 2
  offset_width = ((image_width - size) + 1) / 2
  image = crop_image(image, offset_height, offset_width, size, size)
  return image


def rescale_input(x):
  """Rescales image input to be in range [0,1]."""

  current_min = tf.reduce_min(x)
  current_max = tf.reduce_max(x)

  # we add an epsilon value to prevent division by zero
  epsilon = 1e-5
  rescaled_x = tf.div(
      tf.subtract(x, current_min),
      tf.maximum(tf.subtract(current_max, current_min), epsilon))
  return rescaled_x


def preprocess_for_train(image, image_size):
  """Preprocess image for training."""

  image = random_crop_image(image, image_size)
  image = flip_image(image)
  image = tf.reshape(image, [image_size, image_size, 3])
  return image


def preprocess_for_eval(image, image_size):
  """Preprocesses the given image for eval."""

  image = rescale_image(image, image_size + 32)
  image = center_crop(image, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  return image


def preprocess_image(image, image_size, is_training=False):
  """Preprocesses the given image."""
  if is_training:
    return preprocess_for_train(image, image_size)
  else:
    return preprocess_for_eval(image, image_size)
