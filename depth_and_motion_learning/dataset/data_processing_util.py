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

"""Utility functions for data processing in struct2depth readers."""

from typing import Text

import tensorflow.compat.v1 as tf


def read_image_as_float_tensor(image_filepath):
  """Returns a 3-channel float Tensor of the specified image.

  Args:
    image_filepath: A local valid filepath. Supported types BMP, GIF (only
    first image is taken from an animated GIF), JPEG, and PNG,
  """
  # Read, decode, and normalize images.
  encoded_image = tf.io.read_file(image_filepath)
  # decode_image supports BMP, GIF, JPEG, or PNG by calling the appropriate
  # format decoding method. All decode_bmp, decode_jpeg and decode_png return
  # 3-D arrays [height, width, num_channels].
  # By default, decode_gif returns a 4-D array [num_frames, height, width, 3],
  # expand_animations=False will truncate animated GIF files to the first frame.
  # Thereby enforces a tensor rank=3.
  # Channels=3 deals with 'RGBA' format PNG by dropping the transparency mask.
  decoded_image = tf.image.decode_image(encoded_image, channels=3,
                                        expand_animations=False)
  # Help 'tf.keras.initializers.*' to infer shape.
  decoded_image.set_shape([None, None, 3])
  # Scaling is performed appropriately before casting.
  decoded_image = tf.image.convert_image_dtype(decoded_image, dtype=tf.float32)
  # Decoded_image range is [0.0, 1.0].
  return decoded_image


def read_image_grayscale(image_filepath):
  """Returns a 1-channel uint8 Tensor of the specified image.

  Args:
    image_filepath: A local valid filepath.
  """
  # Read, decode, and normalize images.
  image_string = tf.io.read_file(image_filepath)
  # Decode_image might return a 4-dimensional shape.
  image_grayscale = tf.image.decode_image(image_string, channels=1)
  # Enforces a 3-dimensional shape.
  image_grayscale.set_shape([None, None, 1])
  return image_grayscale


def read_image_validity_mask(image_filepath):
  """Returns a 1-channel binary Tensor(int32) of the specified image.

  Args:
    image_filepath: A local valid filepath.
  """
  validity_mask_uint8 = read_image_grayscale(image_filepath)
  # TPU does not support uint8-images, thus validity_mask is re-encoded as
  # int32-image.
  validity_mask_int = tf.cast(validity_mask_uint8, dtype=tf.int32)
  # validity_mask are used to compute the loss in valid pixels only.
  # validity_mask is converted to binary {0, 1} values to allow:
  # valid_loss_per_pixel = loss_per_pixel * validity_mask.
  validity_mask_int = tf.math.minimum(validity_mask_int, 1)
  return validity_mask_int


def crop_egomotion(egomotion, offset_height, offset_width, target_height,
                   target_width):
  """Transforms camera egomotion when the image is cropped.

  Args:
    egomotion: a 2-d transformation matrix.
    offset_height: amount of offset in y direction.
    offset_width: amount of offset in x direction.
    target_height: target height of images.
    target_width: target width of images.

  Returns:
    A 2-d transformation matrix.
  """
  del offset_height, offset_width, target_height, target_width  # unused
  return egomotion


def crop_intrinsics(intrinsics, offset_height, offset_width, target_height,
                    target_width):
  """Crops camera intrinsics based on target image dimensions and offset.

  Args:
    intrinsics: 1-d array containing w, h, fx, fy, x0, y0.
    offset_height: amount of offset in y direction.
    offset_width: amount of offset in x direction.
    target_height: target height of images.
    target_width: target width of images.

  Returns:
    A 1-d tensor containing the adjusted camera intrinsics.
  """
  with tf.name_scope('crop_intrinsics'):
    w, h, fx, fy, x0, y0 = tf.unstack(intrinsics)

    x0 -= tf.cast(offset_width, tf.float32)
    y0 -= tf.cast(offset_height, tf.float32)

    w = tf.cast(target_width, tf.float32)
    h = tf.cast(target_height, tf.float32)

    return tf.stack((w, h, fx, fy, x0, y0))


def crop_image(image, offset_height, offset_width, target_height, target_width):
  """Crops an image represented as a tensor.

  Args:
    image: an image represented as a (height, wdith, channels)-tensor.
    offset_height: amount of offset in y direction.
    offset_width: amount of offset in x direction.
    target_height: target height of images.
    target_width: target width of images.

  Returns:
    A cropped image represented as a (height, width, channels)-tensor.

  Raises:
    ValueError: Image tensor has incorrect rank.
  """
  with tf.name_scope('crop_image'):
    if image.shape.rank != 3:
      raise ValueError('Rank of endpoint is %d. Must be 3.' %
                       (image.shape.rank))
    out_img = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                            target_height, target_width)
    return out_img


def resize_egomotion(egomotion, target_size):
  """Transforms camera egomotion when the image is resized.

  Args:
    egomotion: a 2-d transformation matrix.
    target_size: target size, a tuple of (height, width).

  Returns:
    A 2-d transformation matrix.
  """
  del target_size  # unused
  return egomotion


def resize_intrinsics(intrinsics, target_size):
  """Transforms camera intrinsics when image is resized.

  Args:
    intrinsics: 1-d array containing w, h, fx, fy, x0, y0.
    target_size: target size, a tuple of (height, width).

  Returns:
    A 1-d tensor containing the adjusted camera intrinsics.
  """
  with tf.name_scope('resize_intrinsics'):
    w, h, fx, fy, x0, y0 = tf.unstack(intrinsics)

    def float_div(a, b):
      return tf.cast(a, tf.float32) / tf.cast(b, tf.float32)

    xfactor = float_div(target_size[1], w)
    yfactor = float_div(target_size[0], h)
    fx *= xfactor
    fy *= yfactor
    x0 *= xfactor
    y0 *= yfactor
    w = target_size[1]
    h = target_size[0]

    return tf.stack((w, h, fx, fy, x0, y0))


def resize_area(image, size):
  """Resizes an image represented as a tensor using the area method.

  Args:
    image: an image represented as a (height, width, channels)-tensor.
    size: A tuple ot two integers, the target (height, width).

  Returns:
    An image represented as a (height, wdith, channels)-tensor.
  """
  return _apply_on_one_image(tf.image.resize_area, image, size)


def resize_nearest_neighbor(image, size):
  """Resizes an image represented as a tensor using the nearest neighbor method.

  Args:
    image: an image represented as a (height, width, channels)-tensor.
    size: A tuple ot two integers, the target (height, width).

  Returns:
    An image represented as a (height, wdith, channels)-tensor.
  """

  return _apply_on_one_image(tf.image.resize_nearest_neighbor, image, size)


def flip_egomotion(egomotion):
  """Transforms camera egomotion when the image is flipped horizontally.

     The intrinsics matrix is ((fx, 0, x0), (0, fy, y0), (0, 0, 1)).
     Given a pixel (px, py, 1), the x coordinate is x = px * fx + 1.
     Now what if we flip the image along x? This maps px to w - 1 - px,
     where w is the image width. Therefore for the flipped image,
     we have x' = (w - px - 1) * fx + 1.
     Therefore x' = -x + (w - 1 - 2 * x0) / fx,
     if x0 = ((w - 1) / 2), that is, if the optical center is exactly at
     the center of the image, then indeed x' = -x, so we can just flip x.
     Otherwise there is a correction which is inrinsics-dependent:
     we'd have to add a small translation component to flip_mat, but we ignore
     this small correction for now.
  Args:
    egomotion: a 2-d transformation matrix.

  Returns:
    A 2-d transformation matrix.
  """
  with tf.name_scope('flip_egomotion'):
    flip_mat = tf.constant(
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=tf.float32)
    egomotion = tf.matmul(tf.matmul(flip_mat, egomotion), flip_mat)
    return egomotion


def flip_intrinsics(intrinsics):
  """Flips camera intrinsics when the image is flipped horizontally.

  Args:
    intrinsics: 1-d array containing w, h, fx, fy, x0, y0.

  Returns:
    A 1-d tensor containing the adjusted camera intrinsics.
  """
  with tf.name_scope('flip_intrinsics'):
    w, h, fx, fy, x0, y0 = tf.unstack(intrinsics)
    x0 = w - x0
    y0 = h - y0

    return tf.stack((w, h, fx, fy, x0, y0))


def flip_left_right(image):
  """Horizontally flips an image (left/right) represented as a tensor.

  Args:
    image: an image represented as a (height, wdith, channels)-tensor.

  Returns:
    A flipped image represented as a (height, wdith, channels)-tensor.
  """
  return _apply_on_one_image(tf.image.flip_left_right, image)


def _apply_on_one_image(fn, image, *args, **kwargs):
  """Makes a function that acts on one image (out of one that acts on a batch).

  Args:
    fn: A function that receives a batch of images as a first argument (rank 4),
      and other args and kwargs.
    image: A tensor of rank 3 (height, width, channels) representing an image.
    *args: Arguments to pass to fn
    **kwargs: Keyword arguments to pass to fn

  Returns:
    The result of `fn` when applied on `image`, after adding a batch dimension
    to `image` and removing it from the result.
  """

  with tf.name_scope('apply_on_one_image'):
    image = tf.convert_to_tensor(image)
    if image.shape.rank != 3:
      raise ValueError('Rank of endpoint is %d. Must be 3.' %
                       image.shape.rank)

    out_image = tf.expand_dims(image, axis=0)
    out_image = fn(out_image, *args, **kwargs)
    return tf.squeeze(out_image, axis=0)
