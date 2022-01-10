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

"""Utility functions for preprocessing and augmenting image datasets."""

import functools

import math
import tensorflow as tf
import tensorflow_addons as tfa


def degree2radian(angel_in_degree):
  """Converts degrees to radians.

  Args:
    angel_in_degree: float; Angle value in degrees.

  Returns:
    Converted angle value.
  """

  return (angel_in_degree / 180.0) * math.pi


def rotate_image(img, angle):
  """Rotates the input image.

  Args:
    img: float array; Input image with shape `[w, h, c]`.
    angle: float; The rotation angle in degrees.

  Returns:
    Rotated image.
  """
  new_img = tfa.image.rotate(
      images=[img], angles=[degree2radian(angle)], interpolation='nearest')[0]

  return new_img


def shift_image(img, shift, bg_value=-.5):
  """Shifts the input image.

  Args:
    img: Input image.
    shift: tuple(float); How much the input should be shifted in each dimension.
    bg_value: float; Value to fill past edges of input if None mode is set to
      nearest which means the input is extended by replicating the last pixel.

  Returns:
    Shifted image.
  """

  # TODO(samiraabnar): Implement this.
  raise NotImplementedError


def resize_image(image, image_size):
  """Resize image to [image_size, image_size] and pad if necessary.

  Args:
    image: tensor; Image tensor with shape [height, width, channels].
    image_size: int; Target height and width of the image.

  Returns:
    resized image of shape [image_size, image_size, channels],
    with the same type as the input.
  """
  original_type = image.dtype
  if original_type in ['uint8', 'int32', 'int64']:
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0

  image = tf.image.resize_with_pad([image],
                                   image_size,
                                   image_size,
                                   method=tf.image.ResizeMethod.BICUBIC)[0]

  if original_type in ['uint8', 'int32', 'int64']:
    image = tf.cast(image * 255.0, dtype=original_type)

  return image


def color_grayscale_image(image, color=(.0, .0, 255.)):
  """Converts grayscale image to a colored image.

  Args:
    image: tensor; Input image of shape `[bs, ..., 1]`.
    color: tuple; Color value which is a tuple pf three float scalars (r, g, b).

  Returns:
    Recolored image of shape `[bs, ..., 3]`, with three channels for RGB.
  """
  shape = tf.shape(image)
  assert_op1 = tf.Assert(tf.equal(shape[-1], 1), [shape])
  assert_op2 = tf.Assert(tf.less_equal(tf.reduce_max(color), 255), [color])
  assert_op3 = tf.Assert(tf.greater_equal(tf.reduce_min(color), 0), [color])

  with tf.control_dependencies([assert_op1, assert_op2, assert_op3]):
    # Normalize the color values because we are multiplying this the intesities
    # of the pixels in the original image.
    color = tf.cast(color, dtype=tf.float32)
    color = color / 255

    # Reshape the image to have three channels with the same values.
    tile_shape = tf.concat([tf.ones((len(shape) - 1,), dtype=tf.int32), [3]],
                           axis=0)
    image = tf.tile(image, tile_shape)

    # Normalize the image channel values and apply the color.
    image = tf.cast(image, dtype=tf.float32) * color

    return image


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                channels=3):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: TF tensor; Binary image data.
    bbox: `ensor; Bounding boxes arranged `[1, num_boxes, coords]` where each
      coordinate is [0, 1) and the coordinates are arranged as `[ymin, xmin,
      ymax, xmax]`. If num_boxes is 0 then use the whole image.
    min_object_covered: float; Defaults to `0.1`. The cropped area of the image
      must contain at least this fraction of any bounding box supplied.
    aspect_ratio_range: list[float]; The cropped area of the image must have an
      aspect ratio = width / height within this range.
    area_range: list[float]; The cropped area of the image must contain a
      fraction of the supplied image within in this range.
    max_attempts: int; Number of attempts at generating a cropped region of the
      image of the specified constraints. After `max_attempts` failures, return
      the entire image.
    channels: int; number of channels.

  Returns:
    Cropped image TF Tensor.
  """
  shape = tf.image.extract_jpeg_shape(image_bytes)
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

  image = tf.image.decode_and_crop_jpeg(
      image_bytes, crop_window, channels=channels)

  return image


def decode_and_random_crop(image_bytes, image_size, crop_padding, channels):
  """Make a random crop of `image_size`."""
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      channels=channels)

  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)
  image = tf.cond(
      bad,
      functools.partial(
          decode_and_center_crop,
          image_bytes=image_bytes,
          image_size=image_size,
          crop_padding=crop_padding,
          channels=channels), functools.partial(resize_image, image,
                                                image_size))

  return image


def random_crop(image, image_size):
  """Make a random crop of `image_size`."""

  original_shape = tf.shape(image)
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      original_shape,
      bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  image = tf.image.crop_to_bounding_box(image, offset_y, offset_x,
                                        target_height, target_width)

  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: tf.image.resize_with_crop_or_pad(image, image_size, image_size),
      lambda: resize_image(image, image_size))

  return image


def random_reflect_crop(image, image_size, crop_padding, channels):
  """Make a random crop of `image_size`."""

  image = tf.pad(image, [[crop_padding, crop_padding],
                         [crop_padding, crop_padding], [0, 0]], 'REFLECT')
  image = tf.image.random_crop(image, [image_size, image_size, channels])

  return image


def decode_and_center_crop(image_bytes, image_size, crop_padding, channels):
  """Crops to center of image with padding then scales `image_size`.

  Args:
    image_bytes: bytes; Image input in bytes.
    image_size: int; Target size of the image.
    crop_padding: int; Amount of padding before cropping.
    channels: int; Number of channels of the input image.

  Returns:
    decoded and cropped image.
  """
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float64)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
  ])
  image = tf.image.decode_and_crop_jpeg(
      image_bytes, crop_window, channels=channels)
  image = resize_image(image, image_size)

  return image


def center_crop(image, image_size, crop_padding):
  """Apply center crop on image.

  Args:
    image: tensor; Image tensor of shape [height, width, channels].
    image_size: int; Target size (both heigth and width).
    crop_padding: int; Amount of padding before crop.

  Returns:
    cropped image.
  """
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2

  image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                        padded_center_crop_size,
                                        padded_center_crop_size)
  bad = _at_least_x_are_equal(shape, tf.shape(image), 3)
  image = tf.cond(
      bad,
      lambda: tf.image.resize_with_crop_or_pad(image, image_size, image_size),
      lambda: resize_image(image, image_size))

  return image


def normalize_image(image, mean_rgb, stddev_rgb, channels=3):
  image -= tf.constant(mean_rgb, shape=[1, 1, channels], dtype=image.dtype)
  image /= tf.constant(stddev_rgb, shape=[1, 1, channels], dtype=image.dtype)
  return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def transform_image(image, data_augmentations, rand_augmentor, resolution,
                    crop_padding, channels):
  """Applies a series of transformations on the input image.

  Args:
    image: tf tensor; Image tensor with shape [h, v].
    data_augmentations: list(str); Name of data augmentations to apply.
    rand_augmentor: RandAugmentor; RandAugmentor object.
    resolution: int; Target image size.
    crop_padding: int; Size of padding when cropping.
    channels: int; Number of channels of the image.

  Returns:
    Transformer image.
  """
  data_augmentations = data_augmentations or []
  if 'center_crop' in data_augmentations:
    image = center_crop(image, resolution, crop_padding)
    image = tf.reshape(image, [resolution, resolution, channels])
  if 'random_reflect_crop' in data_augmentations:
    image = random_reflect_crop(image, resolution, crop_padding, channels)
    image = tf.reshape(image, [resolution, resolution, channels])
  if 'random_crop' in data_augmentations:
    image = random_crop(image, resolution)
    image = tf.reshape(image, [resolution, resolution, channels])
  if 'random_flip' in data_augmentations:
    image = tf.image.random_flip_left_right(image)
  if 'rand' in data_augmentations:
    image = rand_augmentor.distort(image)

  return image


def transform_image_bytes(image, data_augmentations, rand_augmentor, resolution,
                          crop_padding, channels):
  """Applies a series of transformations on the input image bytes.

  (when input image is not decoded).

  Args:
    image: tf tensor; Image bytes.
    data_augmentations: list(str); Name of data augmentations to apply.
    rand_augmentor: RandAugmentor; RandAugmentor object.
    resolution: int; Target image size.
    crop_padding: int; Size of padding when cropping.
    channels: int; Number of channels of the image.

  Returns:
    Transformer image.
  """
  data_augmentations = data_augmentations or []
  # Transformations applied on image bytes.
  if 'center_crop' in data_augmentations:
    image = decode_and_center_crop(
        image,
        image_size=resolution,
        crop_padding=crop_padding,
        channels=channels)
    image = tf.reshape(image, [resolution, resolution, channels])
  elif 'random_crop' in data_augmentations:
    image = decode_and_random_crop(
        image,
        image_size=resolution,
        crop_padding=crop_padding,
        channels=channels)
    image = tf.reshape(image, [resolution, resolution, channels])
  else:
    # If no cropping should be applied just decode the image.
    image = tf.io.decode_jpeg(image, channels=channels)

  # From here, transformations are applied on decoded image.
  if 'random_flip' in data_augmentations:
    image = tf.image.random_flip_left_right(image)
  if 'rand' in data_augmentations:
    image = rand_augmentor.distort(image)

  return image
