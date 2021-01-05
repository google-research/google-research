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
r"""Eval preprocessing functions."""

import tensorflow as tf

CROP_PADDING = 32


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(3. / 4, 4. / 3.),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
  """Generates cropped_image using one of the bboxes randomly distorted."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  return image


def _at_least_x_are_equal(a, b, x):
  """At least x of a and b Tensors are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_center_crop(image_bytes, image_size):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]
  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
  ])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  image = tf.image.resize([image], [image_size, image_size],
                          method='bicubic')[0]

  return image


def _decode_and_random_crop(image_bytes, image_size):
  """Make a random crop of image_size."""

  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(bad, lambda: _decode_and_center_crop(image_bytes, image_size),
                  lambda: tf.image.resize([image], [image_size, image_size],  # pylint: disable=g-long-lambda
                                          method='bicubic')[0])  # pylint: disable=g-long-lambda

  return image


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


def rescale_image(x, size):
  """Rescale the image by scaling the smaller spatial dimension to `size`."""

  shape = tf.cast(tf.shape(x), tf.float32)
  w_greater = tf.greater(shape[0], shape[1])
  shape = tf.cond(w_greater,
                  lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
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


def preprocess_for_eval(image_bytes, image_size):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)
  return image


def preprocess_for_train(image_bytes, image_size):
  """Preprocesses the given image for training.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image size.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image_bytes, image_size)
  image = tf.image.convert_image_dtype(image, tf.float32)
  return image


def preprocess_image(image_bytes, image_size, is_training=False):
  """Preprocesses the given image."""
  if is_training:
    return preprocess_for_train(image_bytes, image_size)
  else:
    return preprocess_for_eval(image_bytes, image_size)


def preprocess_for_eval_non_bytes(image, image_size):
  """Preprocesses the given image."""

  image = rescale_image(image, image_size + 32)
  image = center_crop(image, image_size)
  image = tf.reshape(image, [image_size, image_size, 3])
  return image
